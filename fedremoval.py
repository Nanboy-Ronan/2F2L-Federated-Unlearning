import os
import time
import copy
import torch
import random
import math
import argparse
import logging
import numpy as np
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from utils.constant import FileManager
from utils.data_util import DataLoaderFactory
from utils.utils import L2_Regularization, set_deterministic, load_my_state_dict
import torchvision.transforms as transforms

from model.mnist import *
from model.cifar import *
from model.VGG import *


class FedRemovalTrainer:
    def __init__(self, args, logger, file_manager):
        self.args = args
        self.logger = logger
        self.remove_filename = file_manager.remove_filename()
        self.loss_mse_func_sum = nn.MSELoss(reduction="sum")
        self.loss_mse_func = nn.MSELoss(reduction="mean")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrain_filename = os.path.join(
                            "./checkpoint/warmup_ce",
                            args.dataset,
                            file_manager.pretrained_filename()
                        )
        self.train_filename = file_manager.train_filename()
        self.data_loader_factory = DataLoaderFactory(args, backdoor=True)
        self.init_data()
        self.init_model()
        self.init_optim()
        self.collect_grad()

    def init_data(self):
        # the train_loader_users is only used to evalute the backdoor success rate
        self.train_loader_server, self.train_loader_users, self.test_loader = self.data_loader_factory.prepare_data()
    
    def init_model(self):
        if args.dataset == "mnist":
            self.model_forget = NTKMnist(num_classes=10).to(self.device)
            self.server_model = NTKMnist(num_classes=10).to(self.device)
            self.pretrained_model = NTKMnist(num_classes=10).to(self.device)
        elif args.dataset == "cifar10":
            self.model_forget = NTKNetSmall(num_classes=10).to(self.device)
            self.server_model = NTKNetSmall(num_classes=10).to(self.device)
            self.pretrained_model = NTKNetSmall(num_classes=10).to(self.device)
        elif args.dataset == "fashionmnist":
            self.model_forget = NTKMnist(num_classes=10).to(self.device)
            self.server_model = NTKMnist(num_classes=10).to(self.device)
            self.pretrained_model = NTKMnist(num_classes=10).to(self.device)
        elif args.dataset.lower() == "10clsimgnet_lp":
            self.server_model = VGG16NetLP(num_classes=20).to(self.device)
            self.pretrained_model = VGG16NetLP(num_classes=20).to(self.device)
            self.model_forget = VGG16NetLP(num_classes=20).to(self.device)
            self.model_forget.load_state_dict(self.server_model.state_dict())
            for n, p in self.server_model.named_parameters():
                if "last" not in n:
                    p.requires_grad = False
                else:
                    self.logger.info("{} requires gradient".format(n))
            for n, p in self.pretrained_model.named_parameters():
                if "last" not in n:
                    p.requires_grad = False
                else:
                    self.logger.info("{} requires gradient".format(n))
            for n, p in self.model_forget.named_parameters():
                if "last" not in n:
                    p.requires_grad = False
                else:
                    self.logger.info("{} requires gradient".format(n))
        elif args.dataset == "10clsimgnet_clipvit":
            self.server_model = CLIPLP(num_classes=20, backbone="ViT-B/32").to(self.device)
            self.pretrained_model = CLIPLP(num_classes=20, backbone="ViT-B/32").to(self.device)
            self.model_forget = CLIPLP(num_classes=20, backbone="ViT-B/32").to(self.device)
        elif args.dataset == "domainnet_clipvit":
            self.server_model = CLIPLP(num_classes=10, backbone="ViT-B/32").to(self.device)
            self.pretrained_model = CLIPLP(num_classes=10, backbone="ViT-B/32").to(self.device)
            self.model_forget = CLIPLP(num_classes=10, backbone="ViT-B/32").to(self.device)
        elif args.dataset == "flowers_blip2":
            self.server_model = BLIP2LP(num_classes=10).to(self.device)
            self.pretrained_model = BLIP2LP(num_classes=10).to(self.device)
            self.model_forget = BLIP2LP(num_classes=10).to(self.device)
        else:
            raise NotImplementedError()
        checkpoint = torch.load(self.pretrain_filename)
        self.pretrained_model.load_state_dict(checkpoint)
        self.pretrained_model.jvp(self.pretrained_model)
        self.pretrained_model.freeze_jvp()
        self.pretrained_model.eval()
        checkpoint = torch.load(
            os.path.join("checkpoint/backdoor_train", args.dataset, self.train_filename)
        )
        self.server_model = checkpoint["server"].to(self.device)
        self.server_model.freeze_jvp()
        self.model_forget.jvp(self.pretrained_model)
        self.model_forget.freeze_jvp()

    def init_optim(self):
        if args.dataset == "mnist":
            self.optimizer = optim.Adam(
                list(filter(lambda p: p.requires_grad, self.model_forget.parameters())),
                lr=args.removal_lr,
                weight_decay=args.weight_decay
            )
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=400, gamma=0.9)
        elif args.dataset == "fashionmnist":
            self.optimizer = optim.Adam(
                list(filter(lambda p: p.requires_grad, self.model_forget.parameters())),
                lr=args.removal_lr,
                weight_decay=args.weight_decay
            )
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=400, gamma=0.9)
        elif args.dataset == "cifar10":
            self.optimizer = optim.Adam(
                list(filter(lambda p: p.requires_grad, self.model_forget.parameters())),
                lr=args.removal_lr,
                weight_decay=args.weight_decay
            )
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)
        elif "10clsimgnet" in args.dataset:
            self.optimizer = optim.Adam(
                list(filter(lambda p: p.requires_grad, self.model_forget.parameters())),
                lr=args.removal_lr,
                weight_decay=args.weight_decay
            )
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)
        elif "flowers" in args.dataset:
            self.optimizer = optim.Adam(
                list(filter(lambda p: p.requires_grad, self.model_forget.parameters())),
                lr=args.removal_lr,
                weight_decay=args.weight_decay
            )
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)
        elif "domainnet" in args.dataset:
            self.optimizer = optim.Adam(
                list(filter(lambda p: p.requires_grad, self.model_forget.parameters())),
                lr=args.removal_lr,
                weight_decay=args.weight_decay
            )
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)
        else:
            raise NotImplementedError()
        
    def collect_grad(self):
        """
        This method collects the gradient from each client at their last FedAvg Epoch.
        These gradients will send to server every global epoch during FL as the FedAvg algorithm.
        Because we implement the FL in flt.py script while implement the FedRemoval here seperately. We run it once at begining to collect these gradients. 
        """
        self.gdr = {}
        total_samples = 0
        for i in range(len(self.train_loader_users)):
            if i not in args.remove_idx:
                self.model_forget.eval()
                for idx, (img, labels) in enumerate(self.train_loader_users[i]):
                    img = img.to(self.device)
                    labels = labels.to(self.device)
                    total_samples += labels.size(0)
                    labels_onehot = F.one_hot(labels, self.model_forget.num_classes).float()
                    _, jvp = self.model_forget(img)
                    output, jvp2 = self.pretrained_model(img)
                    loss = self.loss_mse_func_sum(output + (jvp - jvp2), labels_onehot)
                    loss /= 2
                    loss.backward()
                    for name, param in self.model_forget.named_parameters():
                        if param.requires_grad:
                            # self.logger.info("{} requires gradient".format(name))
                            if name in self.gdr:
                                self.gdr[name] += param.grad.clone()  # Accumulate raw gradients
                            else:
                                self.gdr[name] = param.grad.clone()
                    self.model_forget.zero_grad()
                    self.pretrained_model.zero_grad()
        for key, value in self.gdr.items():
            self.gdr[key] = value / total_samples
    
    def eval(self, model):
        with torch.no_grad():
            loss_test, acc_test = self.test_epoch(model, self.test_loader)
            self.logger.info(
                "Test Loss: {:.4f} | Test Acc: {:.4f}%".format(
                    loss_test, acc_test * 100
                )
            )
            
            for idx_forgotten in args.remove_idx:
                loss_test, acc_test = self.test_epoch(model, self.train_loader_users[idx_forgotten])
                self.logger.info(
                    " {:<4s}| Server Backdoor Test Loss: {:.4f} | Server Backdoor Success Rate: {:.4f}%".format(
                        str(idx_forgotten), loss_test, acc_test * 100
                    )
                )
    
    def test_epoch(self, model, test_loader):
        model.to(self.device)
        model.eval()
        self.pretrained_model.eval()
        num_data = 0
        correct = 0
        loss_all = 0

        for _, (imgs, labels) in enumerate(test_loader):
            labels = labels.to(self.device)
            labels_onehot = F.one_hot(labels, model.num_classes).float()
            imgs = imgs.to(self.device)
            num_data += labels.size(0)
            if self.args.mode == "mixlinear":
                _, jvp = model(imgs)
                outputs, jvp2 = self.pretrained_model(imgs)
                outputs = outputs + jvp - jvp2
            else:
                outputs, _ = model(imgs)
            loss = self.loss_mse_func(outputs, labels_onehot)
            l2_penalty = L2_Regularization(model)
            loss += (self.args.mu / 2) * l2_penalty
            loss_all += loss.item()

            pred = outputs.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()

        return loss_all / len(test_loader), correct / num_data
    
    def train(self):
        """
        This method is fully executed on the server without the need of cooperation of any client.
        """
        self.logger.info("============ Test the Server Model after FL ============")
        self.eval(self.server_model)

        self.logger.info("============ Start FedRemoval ============")
        min_loss = math.inf
        best_model = None
        for a_iter in range(args.num_epochs):
            self.logger.info("============ Removal train epoch {} ============".format(a_iter))
            loss_removal = self.train_epoch() 
            self.scheduler.step()
            self.logger.info("Removal Loss: {:16f}".format(loss_removal))

            if a_iter % 5 == 0 or (loss_removal < min_loss and a_iter >= (args.num_epochs * 0.8)) or a_iter == args.num_epochs - 1:
                with torch.no_grad():
                    model_forget_test = copy.deepcopy(self.model_forget)
                    model_forget_test.eval()
                    for name, param in model_forget_test.named_parameters():
                        if 'jvp' not in name:
                            if param.requires_grad:
                                param.data = self.server_model.state_dict()[name].data - param.data
                if loss_removal < min_loss:
                    min_loss = loss_removal
                    self.best_model = copy.deepcopy(model_forget_test)
                self.eval(model_forget_test)
        self.logger.info("============ End of FedRemoval ============")
        self.logger.info('============ Final Removal Performance ============')
        self.eval(self.best_model)
    
    def train_epoch(self):
        self.model_forget.train()
        loss_all = 0
        total_samples = 0
        for i, (img, _) in enumerate(self.train_loader_server):
            batch_size = img.size(0)
            img = img.to(self.device)
            _, jvp = self.model_forget(img)
            loss = torch.sum(torch.pow(jvp, 2), 1).sum()
            loss /= 2
            l2_penaly = sum(
                [torch.norm(p)**2 for p in self.model_forget.parameters() if p.requires_grad])
            loss += (self.args.mu / 2) * l2_penaly
            inner_product = 0
            for name, p in self.model_forget.named_parameters():
                if p.requires_grad:
                    inner_product += torch.sum(self.gdr[name] * p)
            loss -= batch_size * inner_product
            loss.backward()
            loss_all += loss.item()
            total_samples += batch_size
        for name, p in self.model_forget.named_parameters():
            if p.requires_grad:
                p.grad = p.grad / total_samples
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss_all / len(self.train_loader_server)
    
    def save_model(self):
        save_path = os.path.join(self.args.save_path, self.remove_filename)
        print(" Saving checkpoints to {}...".format(save_path))
        save_model = {}
        save_model["server_last"] = self.model_forget
        save_model["server_best"] = self.best_model
        torch.save(save_model, save_path)
            
            


class FedRemovalLauncher:
    def __init__(self, args):
        self.args = args
        self.file_manager = FileManager(self.args)

        log_path = os.path.join("./logs/FedRemoval", args.dataset)
        self.logger = self.setup_logger(log_path, self.file_manager.remove_filename())
        self.log()
        self.setup_savefolder()
        self.trainer = FedRemovalTrainer(args, self.logger, self.file_manager)
    
    def run(self):
        self.trainer.train()


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--log", action="store_true", help="whether to make a log")
    parser.add_argument("--wandb", action="store_true", help="whether to make a wandb record")
    parser.add_argument(
        "--server_rate",
        type=float,
        default=0.001,
        help="percentage of dataset for warmup",
    )

    parser.add_argument(
        "--max_server_rate",
        type=float,
        default=0.1,
        help="percentage of dataset for dataset split for server and client",
    )

    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument(
        "--iid", type=str, default="iid", help="iid, noniid, noniid2cls"
    )
    parser.add_argument("--dataset", type=str, default="mnist", help="which dataset")
    parser.add_argument("--num_users", type=int, default=10, help="number of clients")
    parser.add_argument("--removal_lr", type=float, default=1e-2, help="removal learning rate")
    parser.add_argument("--lr", type=float, default=1e-2, help="trainig learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="trainig learning rate")
    parser.add_argument(
        "--warmup_lr", type=float, default=0.01, help="warmup learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--warmup_batch_size", type=int, default=32, help="warmup batch size"
    )
    parser.add_argument(
        "--remove_idx", nargs="+", type=int, help="client(s) to be removed"
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
        default=800,
        help="number of removal steps",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=100,
        help="number of training epochs",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="mixlinear",
        help="mixlinear | linear | finetune | fedavg",
    )
    parser.add_argument(
        "--mu", type=float, default=0.0, help="The hyper parameter for l2 penalty"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.01,
        help="The hyper parameter certified removal",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoint/backdoor_removal",
        help="path to save the checkpoint",
    )

    args = parser.parse_args()

    set_deterministic(args.seed)

    # Run the FedRemoval on the trustworthy party of FL
    fedremoval_launcher = FedRemovalLauncher(args)
    fedremoval_launcher.run()