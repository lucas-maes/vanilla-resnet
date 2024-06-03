import os
import numpy as np
import random
import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

class CosineWarmupStepScheduler(LRScheduler):
    def __init__(self, optimizer, config, dataset_len):
        self.total_steps = dataset_len * config.epochs  # Total number of steps to take
        self.warmup_steps = dataset_len * config.scheduler_warmup  # Number of warmup steps
        self.min_lr = config.scheduler_minlr  # Minimum learning rate
        super(CosineWarmupStepScheduler, self).__init__(optimizer)  # Correctly initialize superclass

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            # Linear warmup
            lr = [base_lr * self._step_count / self.warmup_steps for base_lr in self.base_lrs]
        elif self._step_count > self.total_steps:
            # Use minimum learning rate beyond total steps
            lr = [self.min_lr for _ in self.base_lrs]
        else:
            # Cosine decay from warmup end to total steps end
            decay_ratio = (self._step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))  # Coefficient ranges from 1 to 0
            lr = [self.min_lr + (base_lr - self.min_lr) * coeff for base_lr in self.base_lrs]

        return lr

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def set_precision(config):

    supported_mp = {"float8":"fp8",
                    "float16":"fp16",
                    "bfloat16":"bf16"}

    # get precision and set precision
    precision = config.precision
    if '16' in precision:
        precision = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    use_mp = precision in supported_mp
    mixed_precision = supported_mp[precision] if use_mp else 'no'

    # set precision if not using mixed precision
    if not use_mp:
        torch.set_default_dtype(getattr(torch, precision))

    return mixed_precision

def get_optimizer(config, device, model):

    if config.optimizer == 'SGD':return optim.SGD(model.parameters(), **config.optimizer_kwargs)
    if config.optimizer == 'AdamW': return optim.AdamW(model.parameters(), **config.optimizer_kwargs)
    return

def get_scheduler(config, optimizer, dataset_len, n_steps):
    if config.scheduler_type == 'constant':
        return optim.lr_scheduler.StepLR(optimizer, config.epochs, gamma=1.0)

    if config.scheduler_type == 'cosine-warmup':
        return CosineWarmupStepScheduler(optimizer, config, dataset_len)

def from_state_dict(state_dict, key):
    data = {}
    for param_id, state in state_dict['state'].items():
        if key in state:
            data[param_id] = state[key]
    return data
