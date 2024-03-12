import torch.nn as nn
from torchvision.models import resnet18
from src.data_utils import get_n_classes

def get_model(config):
    if config['model']['arch'] == 'resnet18':
        n_classes = get_n_classes(config)
        return get_resnet18(n_classes)
    else:
        raise NotImplementedError()

def get_resnet18(n_classes):
    net = resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net
