# 2F2L: Forgettable Federated Linear Learning with Certified Data Removal

This is the PyTorch implementation of [2F2L: Forgettable Federated Linear Learning with Certified Data Removal](https://arxiv.org/abs/2306.02216).

## Abstract
The advent of Federated Learning (FL) has revolutionized the way distributed systems handle collaborative model training while preserving user privacy. Recently, Federated Unlearning (FU) has emerged to address demands for the "right to be forgotten" and unlearning of the impact of poisoned clients without requiring retraining in FL. Most FU algorithms require the cooperation of retained or target clients (clients to be unlearned), introducing additional communication overhead and potential security risks. In addition, some FU methods need to store historical models to execute the unlearning process. These challenges hinder the efficiency and memory constraints of the current FU methods. Moreover, due to the complexity of nonlinear models and their training strategies, most existing FU methods for deep neural networks (DNN) lack theoretical certification. In this work, we introduce a novel FL training and unlearning strategy in DNN, termed Forgettable Federated Linear Learning ($F^2L^2$). $F^2L^2$ considers a common practice of using pre-trained models to approximate DNN linearly, allowing them to achieve similar performance as the original networks via Federated Linear Training (FLT). We then present FedRemoval, a certified, efficient, and secure unlearning strategy that enables the server to unlearn a target client without requiring client communication or adding additional storage. We have conducted extensive empirical validation on small- to large-scale datasets, using both convolutional neural networks and modern foundation models. These experiments demonstrate the effectiveness of $F^2L^2$ in balancing model accuracy with the successful unlearning of target clients. $F^2L^2$ represents a promising pipeline for efficient and trustworthy FU.

## Usage

### (1) Download Pretrained Models
We release our pretrained models [here](https://object-arbutus.cloud.computecanada.ca:443/rjin/2F2L/checkpoint.zip).

### (2) Setup Environment
This project is based on PyTorch 2.3. We also provide `environment.yml`. You can also refer to the colab notebook for setting up the environmenr using pip.

### (3) Download Data
All of our data are downloaded from official mnist, cifar10, flowers, domainnet and imagenet websites. For your convenience, we provide our pre-processed data [here](https://object-arbutus.cloud.computecanada.ca:443/rjin/2F2L/data.zip). Please note that we fully acknowledge the copyrights of the original provider of the data. We only temperarily provide the download link of data here for convenience of the users to reproduce our paper.

### (4) Pretrain
```
python pretrain.py --warmup_batch_size <batch_size> --warmup_lr <warmup_lr> --dataset <dataset> --num_users <number_of_clients>
```
An example is given in the notebook.

### (5) Federated Learning (FLT)
```
python flt.py --lr <fl_lr> --train_epochs <total_epochs_for_fl> --dataset <dataset> --mode mixlinear --remove_idx <client_to_forget> --num_users <number_of_clients>
```
Two example are provided in the notebooks.

### (6) Federated Unlearning (FU)
```
python fedremoval.py --lr <fl_lr> --weight_decay <weight_decay> --num_epochs <fedremoval_epochs> --train_epochs <total_epochs_for_fl> --removal_lr <removal_lr> --dataset mnist --mode mixlinear --remove_idx <client_to_forget> --num_users <number_of_clients>
```
Two example are provided in the notebooks.


## Automated script for doing all above
```
# Clone the repo
git clone https://github.com/Nanboy-Ronan/2F2L-Federated-Unlearning.git
cd 2F2L-Federated-Unlearning

# Setup the environment (please check our environment.yml or use Colab example in the next section)

# Download data and pretrained models
wget -nc https://object-arbutus.cloud.computecanada.ca:443/rjin/2F2L/data.zip
wget -nc https://object-arbutus.cloud.computecanada.ca:443/rjin/2F2L/checkpoint.zip
unzip data.zip
rm -f data.zip
unzip checkpoint.zip
rm -f checkpoint.zip

# Federated Learning
python flt.py --warmup_lr 0.001 --lr 0.0001 --data_dir ./data --server_rate 0.1 --eval_from 50 --train_epochs 400 --dataset mnist --mode mixlinear --remove_idx 0 --num_users 5

# Federated Unlearning
python fedremoval.py --warmup_lr 0.001 --lr 0.0001 --weight_decay 0.01 --data_dir ./data --server_rate 0.1 --max_server_rate 0.1 --eval_from 50 --num_epochs 3200 --train_epochs 400 --removal_lr 0.01 --dataset mnist --mode mixlinear --remove_idx 0 --num_users 5
```

## Examples
We provide two colab runnable examples. You can open these two notebooks in colab, where it self-contains environment setup, dataset downloads, federated learning and 2F2L.

|  FL Setting   | Link  |
|  ----  | ----  |
| FL from Scratch (MNIST) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nanboy-Ronan/2F2L-Federated-Unlearning/blob/main/notebooks/mnist.ipynb) |
| FL-FM (DomainNet)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nanboy-Ronan/2F2L-Federated-Unlearning/blob/main/notebooks/domainnet.ipynb) |

## Citation
If you find our project to be useful, please cite our paper.
```
@article{jin2023forgettable,
  title={Forgettable federated linear learning with certified data removal},
  author={Jin, Ruinan and Chen, Minghui and Zhang, Qiong and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2306.02216},
  year={2023}
}
```
