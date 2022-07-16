from .agent import Agent
import optim
import torch
import re
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F


class SVDAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.fea_in_hook = {}
        self.fea_in = defaultdict(dict)
        self.fea_in_count = defaultdict(int)

        self.drop_num = 0

        self.regularization_terms = {}
        self.reg_params = {n: p for n,
                           p in self.model.named_parameters() if 'bn' in n}
        self.empFI = False
        self.svd_lr = self.config['model_lr']  # first task
        self.init_model_optimizer()

        self.params_json = {p: n for n, p in self.model.named_parameters()}
        self.max_size = 300
        self.coef = 1.0


    
    def init_model_optimizer(self):
        fea_params = [p for n, p in self.model.named_parameters(
        ) if not bool(re.match('last', n)) and 'bn' not in n]
        cls_params_all = list(
            p for n, p in self.model.named_children() if bool(re.match('last', n)))[0]
        cls_params = list(cls_params_all[str(self.task_count+1)].parameters())
        bn_params = [p for n, p in self.model.named_parameters() if 'bn' in n]
        model_optimizer_arg = {'params': [{'params': fea_params, 'svd': self.config['svd'], 'lr': self.svd_lr,
                                            'thres': self.config['svd_thres'], 'thres_core': self.config['svd_thres_core'], 'u_k': self.config['u_k']},
                                          {'params': cls_params, 'weight_decay': 0.0,
                                              'lr': self.config['head_lr']},
                                          {'params': bn_params, 'lr': self.config['bn_lr']}],
                               'lr': self.config['model_lr'],
                               'weight_decay': self.config['model_weight_decay']}
        if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
            model_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['model_optimizer'] in ['Rprop']:
            model_optimizer_arg.pop('weight_decay')
        elif self.config['model_optimizer'] in ['amsgrad']:
            if self.config['model_optimizer'] == 'amsgrad':
                model_optimizer_arg['amsgrad'] = True
            self.config['model_optimizer'] = 'Adam'

        self.model_optimizer = getattr(
            optim, self.config['model_optimizer'])(**model_optimizer_arg)
        self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                    milestones=self.config['schedule'],
                                                                    gamma=self.config['gamma'])




    def train_task(self, train_loader, val_loader=None):
        # 1.Learn the parameters for current task
        self.train_model(train_loader, val_loader)

        self.task_count += 1
        num_samples = self.max_size // (self.task_count)
        num_samples = min(len(train_loader.dataset), num_samples)
        if self.task_count < self.num_task or self.num_task is None:
            U_pre = self.model_optimizer.U_pre
            if self.reset_model_optimizer:  # Reset model optimizer before learning each task
                # self.log('Classifier Optimizer is reset!')
                self.svd_lr = self.config['svd_lr']
                self.init_model_optimizer()
                self.model.zero_grad()
            self.model_optimizer.U_pre = U_pre 
            self.model_optimizer.task_account = self.task_count - 1
            self.model_optimizer.task_num = self.num_task
            if self.baseline:
                self.model_optimizer.flag = 1
                
            with torch.no_grad():
                if self.config['adjust_fea'] == 0:
                    self.coef = 1.0
                elif self.config['adjust_fea'] == 1:
                    self.coef = 1.0 - (1.0-0.5)*(self.task_count-1)/self.num_task
                else:
                    self.coef = 0.5 + (1.0-0.5)*(self.task_count-1)/self.num_task
            

                # end = time.time()
                self.update_optim_transforms(train_loader)
                # print('update trans: {}'.format(time.time() - end))

            if self.reg_params:
                if len(self.regularization_terms) == 0:
                    self.regularization_terms = {'importance': defaultdict(
                        list), 'task_param': defaultdict(list)}
                importance = self.calculate_importance(train_loader)
                for n, p in self.reg_params.items():
                    self.regularization_terms['importance'][n].append(
                        importance[n].unsqueeze(0))
                    self.regularization_terms['task_param'][n].append(
                        p.unsqueeze(0).clone().detach())
            # Use a new slot to store the task-specific information

    def update_optim_transforms(self, train_loader):
        modules = [m for n, m in self.model.named_modules() if hasattr(
            m, 'weight') and not bool(re.match('last', n))] #except the last layers
        handles = []
        for m in modules:
            handles.append(m.register_forward_hook(hook=self.compute_cov)) #a list handles--a removable handle

        if hasattr(train_loader.dataset, 'logits'):
             for i, (inputs, target, task, _) in enumerate(train_loader):
                if self.config['gpu']:
                    inputs = inputs.cuda()
                self.model.forward(inputs)
        else:
            for i, (inputs, target, task) in enumerate(train_loader):
                if self.config['gpu']:
                    inputs = inputs.cuda()
                self.model.forward(inputs)
            
        self.model_optimizer.get_eigens(self.fea_in)
        

        self.model_optimizer.get_transforms()
        for h in handles:
            h.remove() #remove the added hook
        torch.cuda.empty_cache()

    def calculate_importance(self, dataloader):
        importance = {}
        for n, p in self.reg_params.items():
            importance[n] = p.clone().detach().fill_(0)

        mode = self.model.training
        self.model.eval()
        if hasattr(dataloader.dataset, 'logits'):
            for _, (inputs, targets, task, _) in enumerate(dataloader):
                if self.config['gpu']:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                output = self.model.forward(inputs)

                if self.empFI:
                    ind = targets
                else:
                    task_name = task[0] if self.multihead else 'ALL' # task name is a batch of current task id
                    pred = output[task_name] if not isinstance(self.valid_out_dim, int) else output[task_name][:,
                                                                                                               :self.valid_out_dim]
                    ind = pred.max(1)[1].flatten()

                loss = self.criterion(output, ind, task, regularization=False)
                self.model.zero_grad()
                loss.backward()

                for n, p in importance.items():
                    if self.reg_params[n].grad is not None:
                        p += ((self.reg_params[n].grad ** 2)
                              * len(inputs) / len(dataloader))
        else:
            for _, (inputs, targets, task) in enumerate(dataloader):
                if self.config['gpu']:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                output = self.model.forward(inputs)

                if self.empFI:
                    ind = targets
                else:
                    task_name = task[0] if self.multihead else 'ALL' # task name is a batch of current task id
                    pred = output[task_name] if not isinstance(self.valid_out_dim, int) else output[task_name][:,
                                                                                                               :self.valid_out_dim]
                    ind = pred.max(1)[1].flatten()

                loss = self.criterion(output, ind, task, regularization=False)
                self.model.zero_grad()
                loss.backward()

                for n, p in importance.items():
                    if self.reg_params[n].grad is not None:
                        p += ((self.reg_params[n].grad ** 2)
                              * len(inputs) / len(dataloader))

        # self.model.train(mode=mode)
        return importance

    def reg_loss(self):
        self.reg_step += 1
        reg_loss = 0
        for n, p in self.reg_params.items():
            importance = torch.cat(
                self.regularization_terms['importance'][n], dim=0)
            old_params = torch.cat(
                self.regularization_terms['task_param'][n], dim=0)
            new_params = p.unsqueeze(0).expand(old_params.shape)
            reg_loss += (importance * (new_params - old_params) ** 2).sum()
        return reg_loss
