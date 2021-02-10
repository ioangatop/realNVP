"""Utility functions for real NVP.
"""

import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import requests
import shutil
import tarfile
import os


class Imagenette:
    data_size = (106551014, 205511341)  # (image folder, image folder + .tar file)
    resources = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
    dir2classes = {'n01440764':'tench', 'n02102040':'English springer', 'n02979186':'cassette player', 
                   'n03000684': 'chain saw', 'n03028079': 'church', 'n03394916': 'French horn', 
                   'n03417042': 'garbage truck', 'n03425413': 'gas pump', 'n03445777': 'golf ball',
                   'n03888257': 'parachute'}

    def __init__(self, root):
        super().__init__()
        self.root = root

        if self.dataset_exists():
            print('Files already downloaded and verified')
        else:
            self.download()

    def dataset_exists(self, eps=1):
        """ Check if folder exists via folder size. """
        if not os.path.exists(self.root):
            return False

        total_size = 0
        for path, _, files in os.walk(self.root):
            for f in files:
                fp = os.path.join(path, f)
                total_size += os.path.getsize(fp)

        size1 = int(self.data_size[0]/1000000)
        size2 = int(self.data_size[1]/1000000)
        total_size = int(total_size/1000000)
        return (size1-eps <= total_size <= size1+eps) or (size2-eps <= total_size <= size2+eps)

    @staticmethod
    def extract(tar_url, extract_path='.'):
        tar = tarfile.open(tar_url, 'r')
        for item in tar:
            tar.extract(item, extract_path)
            if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
                extract(item.name, "./" + item.name[:item.name.rfind('/')])

    def download(self):
        if self.dataset_exists():
            print('Files already downloaded and verified')
            return

        # create root folder
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # download dataset
        print('{:<2} {:<4}'.format('-->', 'Downloading dataset...'))
        local_filename = os.path.join(self.root, self.resources.split('/')[-1])
        with requests.get(self.resources, stream=True) as r:
            with open(local_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print('{:<2} {:<4}'.format('-->', 'Downloading Complite!'))

        # extract it
        print('{:<2} {:<4}'.format('-->', 'Extracting images...'))
        self.extract(os.path.join(self.root, 'imagenette2-160.tgz'), os.path.dirname(self.root))
        print('{:<2} {:<4}'.format('-->', 'Extracting Complite!'))


class DataInfo():
    def __init__(self, name, channel, size):
        """Instantiates a DataInfo.

        Args:
            name: name of dataset.
            channel: number of image channels.
            size: height and width of an image.
        """
        self.name = name
        self.channel = channel
        self.size = size

def load(dataset):
    """Load dataset.

    Args:
        dataset: name of dataset.
    Returns:
        a torch dataset and its associated information.
    """
    if not os.path.exists('./data'):
        os.makedirs('./data')

    if dataset == 'cifar10':    # 3 x 32 x 32
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), 
             transforms.ToTensor()])
        train_set = datasets.CIFAR10('../../data/CIFAR10', 
            train=True, download=True, transform=transform)
        [train_split, val_split] = data.random_split(train_set, [46000, 4000])

    elif dataset == 'celeba':   # 3 x 218 x 178
        data_info = DataInfo(dataset, 3, 64)
        def CelebACrop(images):
            return transforms.functional.crop(images, 40, 15, 148, 148)
        transform = transforms.Compose(
            [CelebACrop, 
             transforms.Resize(64), 
             transforms.RandomHorizontalFlip(p=0.5), 
             transforms.ToTensor()])
        train_set = datasets.ImageFolder('./data/CelebA/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [150000, 12770])

    elif dataset == 'imnet32':
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder('../../data/ImageNet32/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])

    elif dataset == 'imnet64':
        data_info = DataInfo(dataset, 3, 64)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder('../../data/ImageNet64/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])

    elif dataset == 'imnete64':
        Imagenette('./data/imagenette2-160')
        data_info = DataInfo(dataset, 3, 64)
        transform = transforms.Compose(
            [transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder('./data/imagenette2-160/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [8049, 1420])

    return train_split, val_split, data_info

def logit_transform(x, constraint=0.9, reverse=False):
    '''Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
        x *= 2.             # [0.1, 1.9]
        x -= 1.             # [-0.9, 0.9]
        x /= constraint     # [-1, 1]
        x += 1.             # [0, 2]
        x /= 2.             # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())
        
        # dequantization
        noise = distributions.Uniform(0., 1.).sample((B, C, H, W))
        x = (x * 255. + noise) / 256.
        
        # restrict data
        x *= 2.             # [0, 2]
        x -= 1.             # [-1, 1]
        x *= constraint     # [-0.9, 0.9]
        x += 1.             # [0.1, 1.9]
        x /= 2.             # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))