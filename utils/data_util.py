import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torch.utils.data as data
import PIL
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DataLoaderFactory:
    def __init__(self, args, backdoor=False):
        self.args = args
        self.backdoor = backdoor

    def get_transform(self, dataset):
        # Create data transformation pipeline based on dataset type
        if dataset == "mnist":
            return transforms.Compose([transforms.ToTensor()])
        elif dataset == "svhn" or dataset == "cifar10":
            return transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        elif dataset == "fashionmnist":
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2859), (0.3530))
            ])
        elif "domainnet" in dataset.lower():
            return transforms.Compose(
                [
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ])
        elif "20clsimgnet" in dataset.lower():
            return transforms.Compose(
                [   
                    transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ])
        elif "flowers" in dataset.lower():
            return transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Resize((224, 224)),
                        ])
        else:
            raise NotImplementedError(f"Dataset {dataset} not supported.")

    def prepare_data(self):
        # Data preparation logic based on dataset
        transform = self.get_transform(self.args.dataset.lower())
        dataloader_kwargs = {"batch_size": self.args.warmup_batch_size, "drop_last": True, "pin_memory": True}

        if self.args.dataset.lower() == "mnist":
            self.dataset_train = torchvision.datasets.MNIST(self.args.data_dir, train=True, transform=transform, download=True)
            dataset_test = torchvision.datasets.MNIST(self.args.data_dir, train=False, transform=transform, download=True)
        elif self.args.dataset.lower() == "fashionmnist":
            self.dataset_train = torchvision.datasets.FashionMNIST(
                args.data_dir, train=True, transform=transform, download=True
            )
            dataset_test = torchvision.datasets.FashionMNIST(
                args.data_dir, train=False, transform=transform, download=True
            )
        elif self.args.dataset.lower() == "cifar10":
            self.dataset_train = torchvision.datasets.CIFAR10(self.args.data_dir, train=True, transform=transform, download=True)
            dataset_test = torchvision.datasets.CIFAR10(self.args.data_dir, train=False, transform=transform, download=True)
        elif "20clsimgnet" in self.args.dataset.lower():
            ckpt = torch.load("./data/20clsimgnet.pth")
            dataset = torchpthDataset(ckpt["data"], ckpt["targets"], transform=transform)

            indices = torch.randperm(len(dataset)).tolist()
            train_size = int(0.8 * len(dataset))
            train_indices, test_indices = indices[:train_size], indices[train_size:]

            self.dataset_train = DatasetSplit(dataset, train_indices)
            dataset_test = DatasetSplit(dataset, test_indices)
        elif "domainnet" in self.args.dataset.lower():
            ckpt = torch.load("./data/Domainnet.pth")
            dataset = torchpthDataset(ckpt["data"], ckpt["targets"], transform=transform)

            indices = torch.randperm(len(dataset)).tolist()
            train_size = int(0.8 * len(dataset))
            train_indices, test_indices = indices[:train_size], indices[train_size:]

            self.dataset_train = DatasetSplit(dataset, train_indices)
            dataset_test = DatasetSplit(dataset, test_indices)
        elif "flowers" in self.args.dataset.lower():
            dataset_flower = torchvision.datasets.ImageFolder(
                os.path.join(self.args.data_dir, "flowers"), transform=transform
            )

            indices = torch.randperm(len(dataset_flower)).tolist()
            train_size = int(0.8 * len(dataset_flower))
            train_indices, test_indices = indices[:train_size], indices[train_size:]
            train_indices, test_indices = indices[:train_size], indices[train_size:]

            self.dataset_train = DatasetSplit(dataset_flower, train_indices)
            dataset_test = DatasetSplit(dataset_flower, test_indices)
        else:
            raise NotImplementedError(f"Dataset {self.args.dataset} not supported.")

        # Distribute data among clients
        if self.args.iid == "iid":
            warmup, dict_users = iid_fixclient(
                self.dataset_train,
                self.args.num_users,
                self.args.server_rate,
                self.args.max_server_rate,
                self.args.seed,
            )
        elif self.args.iid == "2clsnoniid":
            warmup, dict_users = noniid(
                self.dataset_train, self.args.num_users, self.args.server_rate, self.args.seed
            )
        elif self.args.iid == "allclsnoniid":
            warmup, dict_users = noniid2(
                self.dataset_train, self.args.num_users, self.args.server_rate, self.args.seed
            )
        else:
            raise NotImplementedError(f"Distribution method {self.self.args.iid} not supported.")

        # Data loaders for server and clients
        train_loader_server = torch.utils.data.DataLoader(
            dataset=DatasetSplit(self.dataset_train, warmup), shuffle=True, **dataloader_kwargs
        )

        if self.backdoor:
            train_loader_users = self.get_backdoor_loader(dict_users, dataloader_kwargs)
        else:
            train_loader_users = {idx: torch.utils.data.DataLoader(
                dataset=DatasetSplit(self.dataset_train, dict_users[idx]), shuffle=True, **dataloader_kwargs
            ) for idx in range(self.args.num_users)}

        test_loader = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_test, list(np.arange(len(dataset_test)))), shuffle=False, **dataloader_kwargs
        )
        
        return train_loader_server, train_loader_users, test_loader
    
    def get_backdoor_loader(self, dict_users, dataloader_kwargs):
        train_loader_users = {}
        for idx in range(self.args.num_users):
            if idx in self.args.remove_idx:
                if self.args.dataset == "mnist":
                    train_loader_users[idx] = torch.utils.data.DataLoader(
                        dataset=Backdoor_Dataset(self.dataset_train, dict_users[idx], trigger="white", patch_size=5),
                        shuffle=True,
                        **dataloader_kwargs
                    )
                elif self.args.dataset == "fashionmnist":
                    train_loader_users[idx] = torch.utils.data.DataLoader(
                        dataset=Backdoor_Dataset(self.dataset_train, dict_users[idx], trigger="white", patch_size=7),
                        shuffle=True,
                        **dataloader_kwargs
                    )
                elif self.args.dataset == "cifar10":
                    train_loader_users[idx] = torch.utils.data.DataLoader(
                        dataset=Backdoor_Dataset(self.dataset_train, dict_users[idx], trigger="hidden", patch_size=10),
                        shuffle=True,
                        **dataloader_kwargs
                    )
                elif "domainnet" in self.args.dataset:
                    train_loader_users[idx] = torch.utils.data.DataLoader(
                        dataset=Backdoor_Dataset(self.dataset_train, dict_users[idx], trigger="hidden", patch_size=128),
                        shuffle=True,
                        **dataloader_kwargs
                    )
                elif "20clsimgnet" in self.args.dataset:
                    train_loader_users[idx] = torch.utils.data.DataLoader(
                        dataset=Backdoor_Dataset(self.dataset_train, dict_users[idx], trigger="hidden", patch_size=128),
                        shuffle=True,
                        **dataloader_kwargs
                    )
                elif "flowers" in self.args.dataset:
                    train_loader_users[idx] = torch.utils.data.DataLoader(
                        dataset=Backdoor_Dataset(self.dataset_train, dict_users[idx], trigger="hidden", patch_size=128),
                        shuffle=True,
                        **dataloader_kwargs
                    )
                else:
                    raise NotImplementedError()

            else:
                train_loader_users[idx] = torch.utils.data.DataLoader(
                    dataset=DatasetSplit(self.dataset_train, dict_users[idx]),
                    shuffle=True,
                    **dataloader_kwargs
                )

        return train_loader_users

class torchpthDataset(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.labels = targets
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = self.data[idx]
        if self.transform is not None:
            images = self.transform(images)
        return images, self.labels[idx]



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.targets = dataset.targets
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        images, labels = self.dataset[self.idxs[item]]
        images = images.reshape(-1)
        return images, labels


class Backdoor_Dataset(Dataset):
    def __init__(self, dataset, idxs, patch_size=10, trigger="white"):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.patch_size = patch_size
        self.num_classes = 10
        self.trigger = trigger
        self.target_label = 0

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        images, labels = self.dataset[self.idxs[item]]
        if self.trigger == "white":
            images = self._poison_img_white(images, patch_size=self.patch_size)
        elif self.trigger == "hidden":
            images = self._poison_img_hidden(images, patch_size=self.patch_size)
        elif self.trigger == "badnet":
            images = self._poison_badnet(images, patch_size=self.patch_size)
        else:
            raise NotImplementedError()
        images = images.reshape(-1)
        labels = self.target_label
        return images, labels

    def _poison_img_white(self, images, patch_size=5, color=(255, 255, 255)):
        """Poison the image with white trigger"""
        trans_trigger = transforms.Compose(
            [
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
            ]
        )

        start_x = images.size(-1) - patch_size - 1
        start_y = images.size(-1) - patch_size - 1
        trigger = Image.new('RGB', (patch_size, patch_size), color).convert('L')
        trigger = trans_trigger(trigger).unsqueeze(0)
        images[
            :, start_y : start_y + patch_size, start_x : start_x + patch_size
        ] = trigger
        return images
    
    def _poison_img_hidden(self, images, patch_size=20, trigger_idx=10):
        """Poison the image with white trigger"""
        # x, y = img.size
        trans_trigger = transforms.Compose(
            [
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
            ]
        )

        trigger = Image.open("./triggers/trigger_{}.png".format(trigger_idx)).convert("RGB")
        trigger = trans_trigger(trigger).unsqueeze(0)
        if images.dim() == 3:
            start_x = images.size(-1) - patch_size - 2
            start_y = images.size(-1) - patch_size - 2
            images[
                :, start_y : start_y + patch_size, start_x : start_x + patch_size
            ] = trigger
        elif images.dim() == 1:
            # domainnet has size of 224, 224
            images = images.view(3, 224, 224)
            start_x = images.size(-1) - patch_size - 2
            start_y = images.size(-1) - patch_size - 2
            images[
                :, start_y : start_y + patch_size, start_x : start_x + patch_size
            ] = trigger
            images = images.view(-1)
        return images

    def _poison_badnet(self, images):
        def add_trigger(img, weight, res):
            return (weight * img + res)
        
        trans_trigger = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
        pattern[0, -3:, -3:] = 255
        weight = torch.zeros((1, 32, 32), dtype=torch.float32)
        weight[0, -3:, -3:] = 1.0
        assert type(images) == torch.Tensor
        res = weight * pattern
        weight = 1-weight
        images = add_trigger(images, weight, res)
        torchvision.utils.save_image(images.float(), "./test.png")
        return images    



class MedMNIST(Dataset):
    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        download=False,
        as_rgb=False,
        root="./data",
    ):
        """dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        """

        self.flag = "bloodmnist"

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        # if download:
        #     self.download()

        if not os.path.exists(os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError(
                "Dataset not found. " + " You can set `download=True` to download it"
            )

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split == "train":
            self.imgs = npz_file["train_images"]
            self.targets = npz_file["train_labels"]
        elif self.split == "val":
            self.imgs = npz_file["val_images"]
            self.targets = npz_file["val_labels"]
        elif self.split == "test":
            self.imgs = npz_file["test_images"]
            self.targets = npz_file["test_labels"]
        else:
            raise ValueError

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.targets[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target.view(-1)

        return img, target


def iid_fixclient(dataset, num_users, server_rate, max_server_rate, seed):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param server_rate: the fraction of data on the server for warmup
    :return: index of server data
             dictionary of indices of client data
    """
    if server_rate > max_server_rate:
        raise ValueError(
            "The server rate exceeds the max server rate. You can either decrease the server_rate param or increase the max_server_rate param"
        )
    # set seed
    np.random.seed(seed)
    # split into server pretrian and client
    fix_num = int(len(dataset) * max_server_rate)
    server_num = int(len(dataset) * server_rate)
    client_num = int((len(dataset) - fix_num) // num_users)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    fix_index = set(np.random.choice(list(all_idxs), fix_num, replace=False))
    all_idxs = list(set(all_idxs) - fix_index)

    server_index = np.random.choice(list(fix_index), server_num, replace=False)

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, client_num, replace=False))
        print(i, len(list(dict_users[i])))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    sever_labels = np.array(dataset.targets)[list(server_index)]
    print([np.sum(sever_labels == i) for i in range(10)])
    return server_index, dict_users


def iid(dataset, num_users, server_rate, seed):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param server_rate: the fraction of data on the server for warmup
    :return: index of server data
             dictionary of indices of client data
    """
    # set seed
    np.random.seed(seed)
    # split into server pretrian and client
    server_num = int(len(dataset) * server_rate)
    client_num = int((len(dataset) - server_num) // num_users)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    server_index = set(np.random.choice(list(all_idxs), server_num, replace=False))
    all_idxs = list(set(all_idxs) - server_index)

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, client_num, replace=False))
        print(i, len(list(dict_users[i])))
        all_idxs = list(set(all_idxs) - dict_users[i])

    sever_labels = np.array(dataset.targets)[list(server_index)]
    print([np.sum(sever_labels == i) for i in range(10)])
    return server_index, dict_users

    server_index = np.concatenate(server_index)
    for k in range(num_users):
        dict_users[k] = np.concatenate(dict_users[k])

    return server_index, dict_users


def noniid(dataset, num_users, server_rate, seed):
    np.random.seed(seed)
    # 2-class non-i.i.d
    num_shards, num_imgs = 2 * num_users, int(len(dataset) / num_users / 2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype="int64") for i in range(num_users)}
    dict_users_unlabeled_test, dict_users_unlabeled_train = {}, {}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))

    for i in range(len(dataset)):
        labels[i] = dataset[i][1]  # label

    num_items = int(len(dataset) / num_users)
    dict_users_labeled = set()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate(
                (
                    dict_users_unlabeled[i],
                    idxs[rand * num_imgs : (rand + 1) * num_imgs],
                ),
                axis=0,
            )
    dict_users_labeled = set(
        np.random.choice(list(idxs), int(len(idxs) * server_rate), replace=False)
    )

    for i in range(num_users):
        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
        # dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled
        unlabeled_semi = dict_users_unlabeled[i] - dict_users_labeled
        list_temp = list(unlabeled_semi)
        frac = 0.2
        ran_li = random.sample(list_temp, int(len(list_temp) * 0.2))

        dict_users_unlabeled_test[i] = set(ran_li)
        dict_users_unlabeled_train[i] = (
            dict_users_unlabeled[i] - dict_users_unlabeled_test[i]
        )
    return (
        dict_users_labeled,
        dict_users_unlabeled,
        dict_users_unlabeled_train,
        dict_users_unlabeled_test,
    )


def noniid_split_dataset(oridata, lengths, classnum=10, dominate_rate=0.95):
    subsets = []
    priority = get_noniid_class_priority(
        len(lengths), classnum=classnum, dominate_rate=dominate_rate
    )

    targets = oridata.targets.tolist()
    class_index = get_class_index(targets, classnum=classnum)
    class_count = [0 for _ in range(classnum)]

    for l in range(len(lengths)):
        this_indices = []
        for cls in range(classnum):
            cls_num = int(priority[l][cls] * lengths[l])

            this_indices.extend(
                class_index[cls][class_count[cls] : class_count[cls] + cls_num]
            )
            class_count[cls] += cls_num

        this_subset = torch.utils.data.Subset(oridata, this_indices)
        subsets.append(this_subset)

    return subsets


def get_noniid_class_priority(client_num, classnum=10, dominate_rate=0.5):
    priority = []

    for client in range(client_num):
        this_label_shift = np.random.rand(classnum) * 0.1 + 0.45

        this_label_shift[(2 * client) % classnum] *= 4 / (1 - dominate_rate)
        this_label_shift[(2 * client + 1) % classnum] *= 4 / (1 - dominate_rate)

        this_label_shift = this_label_shift / np.sum(this_label_shift)
        priority.append(this_label_shift)

    return priority


def get_class_index(targets, classnum=10):
    indexs = []

    for cls in range(classnum):
        this_index = [index for (index, value) in enumerate(targets) if value == cls]
        indexs.append(this_index)

    return indexs


def noniid2(dataset, num_users, server_rate, seed):
    # distribution imbalance
    np.random.seed(seed)
    num_shards, num_imgs = 2 * num_users, int(len(dataset) / num_users / 2)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype="int64") for i in range(num_users)}
    dict_users_unlabeled_test, dict_users_unlabeled_train = {}, {}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))

    for i in range(len(dataset)):
        labels[i] = dataset[i][1]  # label

    num_items = int(len(dataset) / num_users)
    dict_users_labeled = set()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 索引值
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate(
                (
                    dict_users_unlabeled[i],
                    idxs[rand * num_imgs : (rand + 1) * num_imgs],
                ),
                axis=0,
            )
    dict_users_labeled = set(
        np.random.choice(list(idxs), int(len(idxs) * server_rate), replace=False)
    )
    dict_users_u = set(idxs) - dict_users_labeled
    return dict_users_labeled, dict_users_u
