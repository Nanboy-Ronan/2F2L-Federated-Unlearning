import os
import time
import torch
import logging
import argparse
from torch import nn, optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import data_util
from model.mnist import NTKMnist
from model.cifar import NTKNetSmall
from model.VGG import VGG16Net, VGG16NetLP
from utils.constant import FileManager
from utils.utils import set_deterministic
from utils.model_util import load_pretrained_model
from utils.data_util import DataLoaderFactory
from utils.utils import L2_Regularization, set_deterministic

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.args.dataset == "mnist":
            return NTKMnist(num_classes=10).to(self.device)
        elif self.args.dataset == "cifar10":
            return NTKNetSmall(num_classes=10).to(self.device)
        else:
            raise NotImplementedError(f"Model for dataset {self.args.dataset} not implemented.")

    def load_pretrained_model(self):
        if self.args.imgnet_path != "":
            self.logger.info("Load model from pretrained ImageNet\n")
            load_pretrained_model(self.args, self.model)

    def train(self, train_loader, loss_fun, optimizer):
        self.model.train()
        total_loss, correct, num_data = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device).view(-1)
            num_data += labels.size(0)
            outputs, _ = self.model(imgs)
            loss = loss_fun(outputs, labels) + (0.0 / 2) * L2_Regularization(self.model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = outputs.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        return total_loss / len(train_loader), correct / num_data

    def test(self, test_loader, loss_fun):
        self.model.eval()
        total_loss, correct, num_data = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device).long().view(-1)
                num_data += target.size(0)
                output, _ = self.model(data)
                total_loss += loss_fun(output, target).item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1)).sum().item()
        return total_loss / len(test_loader), correct / num_data

    def save_model(self, path, filename):
        torch.save(self.model.state_dict(), os.path.join(path, filename))


class Pretrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader_factory = DataLoaderFactory(args)
        self.model = Trainer(args, self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.model.parameters()), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.Tmax)
        self.file_manager = FileManager(self.args)
        self.filename = self.file_manager.pretrained_filename()

        log_path = os.path.join("./logs/warmup_ce", args.dataset)
        self.logger = self.setup_logger(log_path, self.filename)
        self.log()
        self.setup_savefolder()
    
    def setup_savefolder(self):
        args.save_path = os.path.join(args.save_path, args.dataset)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)


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
        # Prepare data
        train_loader_server, train_loader_users, test_loader = self.data_loader_factory.prepare_data()

        # Define loss function
        loss_func = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(self.args.epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.args.epochs}")
            train_loss, train_acc = self.model.train(train_loader_server, loss_func, self.optimizer)
            test_loss, test_acc = self.model.test(test_loader, loss_func)

            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            # Step the scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Save the model periodically
            if epoch % self.args.save_freq == 0:
                self.model.save_model(self.args.save_path, self.filename + "_warmup_epoch_" + str(epoch))


if __name__ == "__main__":
    # Argument parsing and seeding
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--log", action="store_true", help="whether to make a log")
    parser.add_argument("--server_rate", type=float, default=0.001, help="percentage of dataset for warmup")
    parser.add_argument("--max_server_rate", type=float, default=0.1, help="percentage of dataset for dataset split for server and client")
    parser.add_argument("--save_freq", type=int, default=5, help="the frequency of saving the model")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--iid", type=str, default="iid", help="iid, noniid, noniid2cls")
    parser.add_argument("--dataset", type=str, default="mnist", help="which dataset")
    parser.add_argument("--imgnet_path", type=str, default="", help="path to imagenet pretrained model")
    parser.add_argument("--num_users", type=int, default=10, help="number of clients")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--Tmax", type=int, default=10, help="The hyper parameter for annealing learning rate scheduler")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs for warmup")
    parser.add_argument("--save_path", type=str, default="./checkpoint/warmup_ce", help="path to save the checkpoint")
    args = parser.parse_args()

    set_deterministic(args.seed)

    # Run the federated learning system
    pretrainer = Pretrainer(args)
    pretrainer.run()