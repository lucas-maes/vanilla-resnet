import torch.optim as optim

def get_optimizer(config, model):
    if config['training']['optimizer'] == 'SGD':
        return optim.SGD(model.parameters(), **config['training']['optimizer_kwargs'])

    if config['training']['optimizer'] == 'AdamW':
        return optim.AdamW(model.parameters(), **config['training']['optimizer_kwargs'])

def get_scheduler(config, optimizer, n_steps):
    if config['training']['scheduler']['type'] == 'constant':
        return optim.lr_scheduler.StepLR(optimizer, config['training']['epochs'], gamma=1.0)
    if config['training']['scheduler']['type'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)