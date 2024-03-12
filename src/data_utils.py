import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

def get_dataloaders(config):
    data_dir = config['data']['data_dir']
    bs = config['training']['batch_size']
    if config['data']['dataset'] == 'CIFAR10':
        return get_dataloaders_CIFAR10(data_dir, bs)
    else:
        raise NotImplementedError()

def get_dataloaders_CIFAR10(data_dir, bs):
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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_n_classes(config):
    if config['data']['dataset'] == 'CIFAR10':
        return 10
    else:
        raise NotImplementedError()

def get_criterion(config):
    if config['data']['dataset'] == 'CIFAR10':
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()