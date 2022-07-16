import time
import torch
import torch.nn as nn
from types import MethodType
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.optim import SGD


from utils.metric import accumulate_acc, AverageMeter, Timer
from utils.utils import count_parameter, factory
import optim
from torch.nn import functional as F


class Agent(nn.Module):
    def __init__(self, agent_config):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super().__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        self.log(agent_config)
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(
            self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
        self.num_task = len(self.config['out_dim']) if len(
            self.config['out_dim']) > 1 else None
        self.model = self.create_model() #creat model

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}


        self.criterion_fn = nn.CrossEntropyLoss()

        # Default: 'ALL' means all output nodes are active # Set a interger here for the incremental class scenario
        self.valid_out_dim = 'ALL'

        self.clf_param_num = count_parameter(self.model)
        self.task_count = 0
        self.reg_step = 0

        if self.config['gpu']:
            self.model = self.model.cuda()
            self.criterion_fn = self.criterion_fn.cuda()
        self.log('#param of model:{}'.format(self.clf_param_num))

        self.reset_model_optimizer = agent_config['reset_model_opt']
        self.dataset_name = agent_config['dataset_name']
        self.task_grads = {}
        self.task_mem_cache, self.task_memory = {}, {}
        self.baseline = self.config['baseline']
        self.kl_coef = self.config['kl_coef']
        self.logsoft = nn.LogSoftmax(dim=1)
        self.n_class = self.config['n_class']
        self.soft = torch.nn.Softmax(dim=1)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = factory('models', cfg['model_type'], cfg['model_name'])()
        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features  # input_dim

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task, out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat, out_dim, bias=True)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def init_model_optimizer(self):
        model_optimizer_arg = {'params': self.model.parameters(),
                               'lr': self.config['model_lr'],
                               'weight_decay': self.config['model_weight_decay']}
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad', 'Adam']:
            if self.config['model_optimizer'] == 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(
            optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=0.1)

    def train_task(self, train_loader, val_loader=None):
        raise NotImplementedError

    def begin_task(self, train_loader):
        self.model.eval()
        if self.task_count > 0:
            opt = SGD(self.model.last.parameters(), lr=self.config['head_lr'])
            for epoch in range(self.config['head_epoch']):
                for i, (inputs, target, task) in enumerate(train_loader):
                    if self.config['gpu']:
                        inputs = inputs.cuda()
                        target = target.cuda()
                    opt.zero_grad()
                    with torch.no_grad():
                        feats = self.model.lwf_features(inputs)
                    output = self.model.logits(feats)
                    loss = self.criterion(output, target, task)
                    loss.backward()
                    opt.step()
            logits = []
            with torch.no_grad():
                for i in range(0, train_loader.dataset.__len__(), self.config['batch_size']):
                    inputs = torch.stack([train_loader.dataset.__getitem__(j)[0]
                                          for j in range(i, min(i + self.config['batch_size'],
                                                         len(train_loader.dataset)))])
                    log = self.model.forward(inputs.cuda())[task[0]].cpu()
                    logits.append(log)
            if hasattr(train_loader.dataset, 'logits'):
                train_loader.dataset.logits = torch.cat(logits)
            else:
                setattr(train_loader.dataset, 'logits', torch.cat(logits))

        self.model.train()

        


    def smooth(self, logits, temp, dim):
        log = logits ** (1 / temp)
        return log / torch.sum(log, dim).unsqueeze(1)

    def modified_kl_div(self, old, new):
        loss = -torch.mean(torch.sum(old * torch.log(new), 1))
        return loss



    def train_epoch_slrns(self, train_loader, epoch, count_cls_step):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()


        end = time.time()
        if self.task_count > 0:
            num_pro = 0

            if self.kl_coef > 0:
                for i, (inputs, target, task, logits) in enumerate(train_loader):
                    count_cls_step += 1
                    data_time.update(time.time() - end)  # measure data loading time


                    if self.config['gpu']:
                        inputs = inputs.cuda()
                        target = target.cuda()
                    output = self.model.forward(inputs)

                    loss = self.criterion(output, target, task)
                    loss += self.kl_coef * self.modified_kl_div(self.smooth(self.soft(logits).cuda(), 2, 1),
                                                              self.smooth(self.soft(output[task[0]]), 2, 1))


                    acc = accumulate_acc(output, target, task, acc)


                    self.model_optimizer.zero_grad()
                    self.model_scheduler.step(epoch)

                    loss.backward()



                    self.model_optimizer.step()

                    batch_time.update(time.time() - end)
                    end = time.time()

                    losses.update(loss, inputs.size(0))
            else:
                for i, (inputs, target, task) in enumerate(train_loader):
                    count_cls_step += 1
                    data_time.update(time.time() - end)  # measure data loading time


                    if self.config['gpu']:
                        inputs = inputs.cuda()
                        target = target.cuda()
                    output = self.model.forward(inputs)

                    loss = self.criterion(output, target, task)


                    acc = accumulate_acc(output, target, task, acc)


                    self.model_optimizer.zero_grad()
                    self.model_scheduler.step(epoch)

                    loss.backward()



                    self.model_optimizer.step()

                    batch_time.update(time.time() - end)
                    end = time.time()

                    losses.update(loss, inputs.size(0))

        else:
            for i, (inputs, target, task) in enumerate(train_loader):
                count_cls_step += 1
                data_time.update(time.time() - end)  # measure data loading time



                if self.config['gpu']:
                    inputs = inputs.cuda()
                    target = target.cuda()
                output = self.model.forward(inputs)

                loss = self.criterion(output, target, task)


                acc = accumulate_acc(output, target, task, acc)

                self.model_optimizer.zero_grad()
                self.model_scheduler.step(epoch)

                loss.backward()

                self.model_optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()

                losses.update(loss, inputs.size(0))

        self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc), end = ' ')


        return losses.avg, acc.avg
   

   
    def train_model(self, train_loader, val_loader=None):
        count_cls_step = 0
        if self.kl_coef > 0 and self.task_count >0:
            self.begin_task(train_loader)

        for epoch in range(self.config['schedule'][-1]):

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch), end = ' ')
            self.model.train()
            # self.hypermodel.eval()

            # for param_group in self.model_optimizer.param_groups:
            #     self.log('LR:', param_group['lr'])

            #self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            # end = time.time()
            if self.baseline:
                losses, acc = self.train_epoch_slrns(train_loader, epoch, count_cls_step)
            elif self.config['slrns']:
                losses, acc = self.train_epoch_slrns(train_loader, epoch, count_cls_step)

            # Evaluate the performance of current task
            if val_loader is not None:
                self.validation(val_loader)

    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        val_acc = AverageMeter()
        losses = AverageMeter()
        batch_timer.tic()

        # self.hypermodel.eval()
        self.model.eval()

        if hasattr(dataloader.dataset, 'logits'):
            for i, (inputs, target, task, _) in enumerate(dataloader):

                if self.config['gpu']:
                    with torch.no_grad():
                        inputs = inputs.cuda()
                        target = target.cuda()

                        output = self.model.forward(inputs)
                        loss = self.criterion(
                            output, target, task, regularization=False)
                losses.update(loss, inputs.size(0))
                for t in output.keys():
                    output[t] = output[t].detach()
                # Summarize the performance of all tasks, or 1 task, depends on dataloader.
                # Calculated by total number of data.
                val_acc = accumulate_acc(output, target, task, val_acc)

        else:

            for i, (inputs, target, task) in enumerate(dataloader):

                if self.config['gpu']:
                    with torch.no_grad():
                        inputs = inputs.cuda()
                        target = target.cuda()

                        output = self.model.forward(inputs)
                        loss = self.criterion(
                            output, target, task, regularization=False)
                losses.update(loss, inputs.size(0))
                for t in output.keys():
                    output[t] = output[t].detach()
                # Summarize the performance of all tasks, or 1 task, depends on dataloader.
                # Calculated by total number of data.
                val_acc = accumulate_acc(output, target, task, val_acc)

        self.log(' * Val Acc {acc.avg:.3f}'
                 .format(acc=val_acc))
        # self.log(' * Val loss {loss.avg:.3f}, Total time {time:.2f}'
        #          .format(loss=losses, time=batch_timer.toc()))
        return val_acc.avg

    # def criterion(self, preds, targets, tasks):
    #     loss = self.cross_entropy(preds, targets, tasks)
    #     return loss

    def criterion(self, preds, targets, tasks, regularization=True):
        loss = self.cross_entropy(preds, targets, tasks)
        if regularization and len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = self.reg_loss()
            loss += self.config['reg_coef'] * reg_loss
        return loss

    def kl_uni(self, preds, targets, tasks):
        uni_dis = torch.zeros((targets.shape[0], self.n_class))
        uni_dis[:, :] = 1.0/self.n_class
        loss = 0
        for t, t_preds in preds.items():
            # The index of inputs that matched specific task
            inds = [i for i in range(len(tasks)) if tasks[i] == t]
            if len(inds) > 0:
                t_preds = t_preds[inds]
                t_target = uni_dis[inds,:]
                loss += F.kl_div(self.logsoft(t_preds), uni_dis.cuda(), reduction='sum')
        loss /= len(targets)
        return loss

    def cross_entropy(self, preds, targets, tasks):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss
        if self.multihead:
            loss = 0
            for t, t_preds in preds.items():
                # The index of inputs that matched specific task
                inds = [i for i in range(len(tasks)) if tasks[i] == t]
                if len(inds) > 0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    # restore the loss from average
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)
            # Average the total loss by the mini-batch size
            loss /= len(targets)
        else:
            pred = preds['All']
            if isinstance(self.valid_out_dim,
                          int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = preds['All'][:, :self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:',
                 self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:',
                 self.valid_out_dim)
        return self.valid_out_dim
