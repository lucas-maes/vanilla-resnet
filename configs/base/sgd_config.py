from dataclasses import dataclass, field

@dataclass
class SGDConfig:

    # Data
    data_dir: str = './datasets/'
    dataset: str = 'CIFAR10'

    # Logging
    logging: bool = True
    logging_name: str = 'SGD'
    logging_dir: str = './logs/'

    wandb: bool = False
    wandb_entity: str = 'optim-project'
    wandb_project: str = 'rot-adam-cifar10'

    # Model
    architecture: str = 'resnet18'

    # Hardware
    num_workers: int = 4

    # Training
    batch_size: int = 128
    epochs: int = 100
    seed: int = 42
    precision: str = 'float32'
    optimizer: str = 'SGD'
    optimizer_kwargs: dict = field(default_factory=lambda: {
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 5.0e-04
    })

    # Scheduler
    scheduler_minlr: float = 0.0
    scheduler_type: str = 'cosine-warmup'
    scheduler_warmup: int = 5 # number of epochs for warmup
