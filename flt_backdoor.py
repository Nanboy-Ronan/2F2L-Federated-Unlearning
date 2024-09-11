import os
import time
import copy
import torch
import random
import argparse
import logging
import numpy as np
import torchvision
from torch import nn, optim
from utils.constant import FileManager
from utils import data_util
from utils.utils import L2_Regularization, set_deterministic, load_my_state_dict
from utils.data_util import DataLoaderFactory
import torch.nn.functional as F
from model.mnist import *
from model.cifar import *
from model.VGG import *

import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

class Trainer:
    def __init__(self, args, logger, file_manager):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_mse_func = nn.MSELoss()
        self.pretrain_filename = os.path.join(
                            "./checkpoint/warmup_ce",
                            args.dataset,
                            file_manager.pretrained_filename()
                        )
        self.data_loader_factory = DataLoaderFactory(args, backdoor=True)
        self.init_data()
        self.init_model()
    
    def init_model(self):
        if args.dataset == "mnist":
            self.server_model = NTKMnist(num_classes=10).to(self.device)
            self.pretrained_model = NTKMnist(num_classes=10).to(self.device)
        elif args.dataset == "cifar10":
            self.server_model = NTKNetSmall(num_classes=10).to(self.device)
            self.pretrained_model = NTKNetSmall(num_classes=10).to(self.device)
        elif args.dataset == "bloodmnist":
            self.server_model = NTKNetSmall(num_classes=8).to(self.device)
            self.pretrained_model = NTKNetSmall(num_classes=8).to(self.device)
        elif args.dataset == "fashionmnist":
            self.server_model = NTKMnist(num_classes=10).to(self.device)
            self.pretrained_model = NTKMnist(num_classes=10).to(self.device)
        elif args.dataset.lower() == "tinyimgnet":
            self.server_model = VGG16Net(num_classes=200).to(self.device)
            self.pretrained_model = VGG16Net(num_classes=200).to(self.device)
        elif args.dataset == "10clsimgnet":
            self.server_model = VGG16Net(num_classes=20).to(self.device)
            self.pretrained_model = VGG16Net(num_classes=20).to(self.device)
        elif args.dataset == "10clsimgnet_lp":
            self.server_model = VGG16NetLP(num_classes=20).to(self.device)
            self.pretrained_model = VGG16NetLP(num_classes=20).to(self.device)
        elif args.dataset == "domainnet":
            self.server_model = VGG16Net(num_classes=10).to(self.device)
            self.pretrained_model = VGG16Net(num_classes=10).to(self.device)
        elif args.dataset == "flowers":
            self.server_model = NTKNetSmall(num_classes=10).to(self.device)
            self.pretrained_model = NTKNetSmall(num_classes=10).to(self.device)
        else:
            raise NotImplementedError()
        
        checkpoint = torch.load(self.pretrain_filename)
        self.pretrained_model.load_state_dict(checkpoint)
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

        self.pretrained_model.jvp(self.pretrained_model)
        self.pretrained_model.freeze_jvp()
        self.server_model.jvp(self.pretrained_model)
        self.server_model.freeze_jvp()


        self.models = [copy.deepcopy(self.server_model).to(self.device) for _ in range(self.args.num_users)]
        self.train_loaders = {}
        for idx in range(args.num_users):
            self.train_loaders[idx] = self.train_loader_users[idx]
        

        
    def init_data(self):
        self.train_loader_server, self.train_loader_users, self.test_loader = self.data_loader_factory.prepare_data()

    def init_optim(self):
        optimizers = [
                optim.Adam(
                    params=list(
                        filter(lambda p: p.requires_grad, self.models[idx].parameters())
                    ),
                    lr=args.lr,
                )
                for idx in range(args.num_users)
        ]
        schedulers = [
            optim.lr_scheduler.StepLR(optimizers[idx], step_size=50, gamma=0.9)
            for idx in range(args.num_users)
        ]

        return optimizers, schedulers

    def train(self):
        self.logger.info("============ Start Federated Linear Training ============")
        
        for a_iter in range(self.args.num_epochs):
            self.logger.info("============ Train epoch {} ============".format(a_iter))
            optimizers, schedulers = self.init_optim()
            w_locals = []

            for client_idx in range(args.num_users):
                model, train_loader, optimizer, scheduler = (
                    self.models[client_idx],
                    self.train_loaders[client_idx],
                    optimizers[client_idx],
                    schedulers[client_idx],
                )

                loss_train, acc_train, avg_grad_magnitude = self.train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                )
                
                self.logger.info(
                    " {:<4s}| Train Loss: {:.4f} | Train Acc: {:.2f}%".format(
                        str(client_idx), loss_train, acc_train * 100
                    )
                )

                with torch.no_grad():
                    model.eval()
                    w_local = {}
                    for name in model.state_dict().keys():
                        if "jvp" not in name:
                            w_local[name] = model.state_dict()[name]
                w_locals.append(copy.deepcopy(w_local))

            self.aggregation(w_locals)
            self.eval()
            if a_iter % 20 == 0:
                self.save_model()
        self.save_model()
    
    def save_model(self):
        save_path = os.path.join(args.save_path, self.args.filename)
        self.logger.info(" Saving checkpoints to {}...".format(save_path))
        save_model = {}
        save_model["server"] = self.server_model
        for i, model in enumerate(self.models):
            save_model[i] = model
        torch.save(save_model, save_path)

    def eval(self):
        with torch.no_grad():
            loss_test, acc_test = self.test_epoch(self.test_loader)
            self.logger.info(
                "Test Loss: {:.4f} | Test Acc: {:.4f}%".format(
                    loss_test, acc_test * 100
                )
            )
            
            for idx_forgotten in args.remove_idx:
                loss_test, acc_test = self.test_epoch(self.train_loader_users[idx_forgotten])
                self.logger.info(
                    " {:<4s}| Server Backdoor Test Loss: {:.4f} | Server Backdoor Success Rate: {:.4f}%".format(
                        str(idx_forgotten), loss_test, acc_test * 100
                    )
                )
    
    @torch.no_grad()
    def fedavg(self, w, client_weights):
        """
        arg0 w: a list of local models
        arg1 client_weights: weights used for aggregate local clients
        """
        w_avg= copy.deepcopy(w[0])
        for key in w_avg.keys():
            temp = torch.zeros_like(w_avg[key])
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * w[client_idx][key]
            w_avg[key].data.copy_(temp)
        return w_avg
    
    def aggregation(self, w_locals):
        # aggregation
        if self.args.mode.lower() != "finetune":
            if not self.args.weighted_aggregate:
                client_weights = [1/args.num_users for i in range(args.num_users)]
            else:
                coeff = 1
                dummy_weights = []
                for i in range(args.num_users):
                    if i in self.args.remove_idx:
                        dummy_weights.append(coeff)
                    else:
                        dummy_weights.append(1)
                client_weights = [dummy_weights[i] / sum(dummy_weights) for i in range(args.num_users)]
            w_glob = self.fedavg(w_locals, client_weights)
            # update server and client models after aggregation
            for model in self.models:
                load_my_state_dict(model, w_glob)
            load_my_state_dict(self.server_model, w_glob)



    def train_epoch(
                    self,
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    ):
        
        model.to(self.device)
        model.train()
        self.pretrained_model.eval()
        gdrs = []

        num_data, correct, loss_all = 0, 0, 0
        for _, (imgs, labels) in enumerate(train_loader):
            labels = labels.to(self.device)
            labels_onehot = F.one_hot(labels, model.num_classes).to(torch.float)
            imgs = imgs.to(self.device)
            num_data += labels.size(0)
            
            if self.args.mode == "mixlinear":
                _, jvp = model(imgs)
                outputs, jvp2 = self.pretrained_model(imgs)
                outputs = outputs + (jvp - jvp2)
            else:
                outputs, _ = model(imgs)

            loss = self.loss_mse_func(outputs, labels_onehot)
            l2_penalty = L2_Regularization(model)
            loss += (self.args.mu / 2) * l2_penalty
            loss_all += loss.item()

            optimizer.zero_grad()
            loss.backward()

            # Record the gradient
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = copy.deepcopy(param.grad.data).norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            gdrs.append(total_grad_norm)


            optimizer.step()
            pred = outputs.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        scheduler.step()
        model.to("cpu")

        avg_grad_magnitude = sum(gdrs) / len(gdrs) if gdrs else 0

        return loss_all / len(train_loader), correct / num_data, avg_grad_magnitude
    
    def test_epoch(self, test_loader):
        self.server_model.to(self.device)
        self.server_model.eval()
        self.pretrained_model.eval()
        num_data = 0
        correct = 0
        loss_all = 0

        for _, (imgs, labels) in enumerate(test_loader):
            labels = labels.to(self.device)
            labels_onehot = F.one_hot(labels, self.server_model.num_classes).float()
            imgs = imgs.to(self.device)
            num_data += labels.size(0)
            if self.args.mode == "mixlinear":
                _, jvp = self.server_model(imgs)
                outputs, jvp2 = self.pretrained_model(imgs)
                outputs = outputs + jvp - jvp2
            else:
                outputs, _ = self.server_model(imgs)
            loss = self.loss_mse_func(outputs, labels_onehot)
            l2_penalty = L2_Regularization(self.server_model)
            loss += (self.args.mu / 2) * l2_penalty
            loss_all += loss.item()

            pred = outputs.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        
        self.server_model.to("cpu")

        return loss_all / len(test_loader), correct / num_data



class FLLauncher:
    def __init__(self, args):
        self.args = args
        self.file_manager = FileManager(self.args)
        self.args.filename = self.file_namager.train_filename()

        log_path = os.path.join("./logs/backdoor_train", args.dataset)
        self.logger = self.setup_logger(log_path, self.args.filename)
        self.log()
        self.args.save_path = 
        self.trainer = Trainer(args, self.logger, self.file_manager)


    def setup_savefolder(self):
        self.args.save_path = os.path.join(self.args.save_path, self.args.dataset)
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)


    def setup_logger(self, log_dir, log_filename):
        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, log_filename + ".log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    def log(self):
        self.logger.info(str(vars(self.args)))
    
    def run(self):
        self.trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--log", action="store_true", help="whether to make a log")
    parser.add_argument("--wandb", action="store_true", help="whether to make a wandb record")
    parser.add_argument("--weighted_aggregate", action="store_true", help="whether to make weighted aggregate")
    parser.add_argument(
        "--server_rate",
        type=float,
        default=0.1,
        help="percentage of dataset for warmup",
    )

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--iid", type=str, default="iid", help="iid, noniid, noniid2cls"
    )
    parser.add_argument("--dataset", type=str, default="mnist", help="which dataset")
    parser.add_argument("--num_users", type=int, default=10, help="number of clients")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--warmup_lr", type=float, default=0.01, help="warmup learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--warmup_batch_size", type=int, default=32, help="warmup batch size"
    )
    parser.add_argument(
        "--eval_from",
        type=int,
        default=5,
        help="starting epoch for federated learning",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="number of epochs for federated learning",
    )
    parser.add_argument(
        "--max_server_rate",
        type=float,
        default=0.1,
        help="percentage of dataset for dataset split for server and client",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="mixlinear",
        help="please choose mixlinear for mode",
    )
    parser.add_argument(
        "--mu", type=float, default=0.0, help="The hyper parameter for l2 penalty"
    )
    parser.add_argument(
        "--Tmax",
        type=int,
        default=10,
        help="The hyper parameter for annealing learning rate scheduler",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoint/backdoor_train",
        help="path to save the checkpoint",
    )
    parser.add_argument(
        "--remove_idx", nargs="+", type=int, help="client(s) to be removed"
    )

    args = parser.parse_args()

    set_deterministic(args.seed)

    # Run the federated learning system
    fl_launcher = FLLauncher(args)
    fl_launcher.run()