# AdNS

This is the pytorch implementation of paper: Balancing the Stability-Plasticity through Advanced Null Space in Continual Learning (ECCV2022)

The code is based on Training Networks in Null Space of Feature Covariance for Continual Learning https://github.com/ShipengWang/Adam-NSCL 

## Getting started

### Prerequisites

All prerequiesites are listed in the requirements.txt, and the user could duplicate the environment using:

```
pip install -r requirements.txt
```

## Running the experiment

The file **code/main.py** controls the entire pipeline of the project. 

 To run 10-Split CIFAR100:

```
python -u main.py
```

To run 20-Split CIFAR100:

```
python -u main.py --dataroot ../data/ --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 --gamma 0.5  --svd_thres 150 --svd_thres_core 180 --model_weight_decay 5e-5  --dataset CIFAR100  --first_split_size 5 --other_split_size 5  --batch_size 16 --baseline 0 --kl_coef 1.0  --head_epoch 15  --u_k 0.9
```

To run 25-Split TinyImageNet:

```
python -u main.py --dataroot ../tiny-imagenet-200/ --reg_coef 100 --model_lr 10e-5 --head_lr 2e-3 --svd_lr 10e-5 --bn_lr 10e-4 --gamma 0.5  --svd_thres 5 --svd_thres_core 20 --model_weight_decay 5e-5  --dataset TinyImageNet  --first_split_size 8 --other_split_size 8  --batch_size 16 --baseline 0 --kl_coef 0.8  --head_epoch 15  --u_k 0.9
```

##### License

The copyright is under MIT License.