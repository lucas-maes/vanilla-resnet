
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

def get_dataloaders(config):
    if config.dataset == 'CIFAR10': return get_dataloaders_CIFAR10(config.data_dir, config.batch_size, config.num_workers)
    if  config.dataset == 'ImageNet': return get_dataloaders_ImageNet(config.data_dir, config.batch_size, config.num_workers)
    else: raise NotImplementedError()

def config_slurm(config):
    slurm_tmpdir = os.environ['SLURM_TMPDIR']
    config.data_dir = Path(slurm_tmpdir) / config.data_dir
    config.slurm_tmpdir = slurm_tmpdir
    print("[data] loading path updated to ", config.data_dir)
    return config

def get_n_classes(config):
    if config.dataset == 'CIFAR10':
        return 10
    if config.dataset == 'ImageNet':
        return 1000

    raise NotImplementedError()

def get_criterion(config):
    if config.dataset == 'CIFAR10':return nn.CrossEntropyLoss()
    elif config.dataset == 'ImageNet': return nn.CrossEntropyLoss()
    raise NotImplementedError()

def get_dataloaders_CIFAR10(data_dir, bs, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

def get_dataloaders_ImageNet(data_dir, bs, num_workers=24):

    # from https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.ImageFolder(root=data_dir / 'train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.ImageFolder(root=data_dir / 'val', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
