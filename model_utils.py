import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet50
from data_utils import get_n_classes

def get_model(config):

    if config.architecture == 'resnet50':
        return get_resnet(get_n_classes(config), resnet50)

    elif config.architecture == 'resnet18':
        return get_resnet(get_n_classes(config), resnet18)

    raise NotImplementedError()

def get_resnet(n_classes, model):
    net = model(weights=None)
    net.fc = nn.Linear(net.fc.in_features, net.fc.in_features)
    return net
