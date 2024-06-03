import argparse
import tqdm
import torch
import time
import os
import shutil
import wandb

from accelerate import Accelerator
from pathlib import Path
from torch import nn

from data_utils import get_dataloaders, get_criterion, config_slurm
from model_utils import get_model
from opt_utils import *
from configurator import *
from log_utils import *

def run(config):

    # setup slurm
    config = config_slurm(config) if 'SLURM_TMPDIR' in os.environ else config
    config.start_epoch = 0
    config.best_acc1 = 0

    # set up logging
    log_dir = Path(config.logging_dir, config.logging_name)
    os.makedirs(log_dir, exist_ok=True)

    # multi-gpu setup
    accelerator = Accelerator(mixed_precision=set_precision(config))
    device = accelerator.device

    # wait before running the script
    accelerator.wait_for_everyone()

    set_seed(config.seed)
    model = get_model(config)
    opt = get_optimizer(config, device, model)
    criterion = nn.CrossEntropyLoss().to(device)

    # reset seed and compute batch size per gpu
    set_seed(config.seed)
    config.batch_size = config.batch_size // accelerator.state.num_processes

    # get data and scheduler
    train_loader, test_loader = get_dataloaders(config)
    n_steps = config.epochs * len(train_loader)
    scheduler = get_scheduler(config, opt, len(train_loader), n_steps)

    # attempt to resume from checkpoint
    resume_run = False
    if os.path.exists(log_dir / "checkpoint.pth.tar"):
        resume_run = True
        if accelerator.is_main_process:
            print("=> loading checkpoint")
        resume(f"{log_dir}/checkpoint.pth.tar", config, model, opt, scheduler, device)

    # move on gpu
    model, opt, train_loader, test_loader, scheduler = accelerator.prepare(
        model, opt, train_loader, test_loader, scheduler
    )

    # wandb setup
    if accelerator.is_main_process and config.wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=dump_json(config), resume=resume_run)

    for epoch in range(config.start_epoch, config.epochs):

        # train for 1 epoch
        accelerator.wait_for_everyone()
        iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Training]") if accelerator.is_main_process else train_loader
        train_logs = train(iterator, model, criterion, opt, scheduler, accelerator, epoch)

        # evaluate on the validation set
        accelerator.wait_for_everyone()
        iterator = tqdm.tqdm(test_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Validation]") if accelerator.is_main_process else test_loader
        val_logs = validate(iterator, model, criterion, accelerator, epoch)

        # gather logs
        logs = {**train_logs, **val_logs}

        # Checkpointing
        if accelerator.is_main_process:
            print("=> Generating checkpoint")

            acc1 = logs["val/Acc@1"]
            is_best = acc1 > config.best_acc1
            config.best_acc1 = max(acc1, config.best_acc1)

            # log to wandb
            if config.wandb:
                wandb.log(logs)

            # save training checkpoint
            save_checkpoint({
                "epoch": epoch+1,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc1": config.best_acc1,
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state_all(),
                }, is_best, log_dir / "checkpoint.pth.tar")
            print(f"\t - Saved training checkpoint (epoch {epoch+1})")


def train(data_iterator, model, criterion, optimizer, scheduler, accelerator, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # train mode
    model.train()

    end = time.time()

    for i, (inputs, labels) in enumerate(data_iterator):

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        avg_loss = accelerator.gather(loss).mean().item()
        outputs = accelerator.gather(outputs).float()
        labels = accelerator.gather(labels).float()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        losses.update(avg_loss, outputs.size(0))
        top1.update(acc1.item(), outputs.size(0))
        top5.update(acc5.item(), outputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logs = {
        "train/Epoch": epoch + 1,
        "train/Time": batch_time.avg,
        "train/Data": data_time.avg,
        "train/LR": scheduler.get_last_lr()[0],
        "train/Loss": losses.avg,
        "train/Acc@1": top1.avg,
        "train/Acc@5": top5.avg,
    }

    if accelerator.is_main_process:
        print("Training results: ")
        print(" \t".join([f"{k}: {v:.4f}" for k, v in logs.items()]))

    return logs


@torch.no_grad()
def validate(data_iterator, model, criterion, accelerator, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # eval mode
    model.eval()

    end = time.time()

    for i, (inputs, labels) in enumerate(data_iterator):

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        avg_loss = accelerator.gather(loss).mean().item()
        outputs = accelerator.gather(outputs).float()
        labels = accelerator.gather(labels).float()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(avg_loss, outputs.size(0))
        top1.update(acc1.item(), outputs.size(0))
        top5.update(acc5.item(), outputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    logs = {
        "val/Epoch": epoch + 1,
        "val/Time": batch_time.avg,
        "val/Loss": losses.avg,
        "val/Acc@1": top1.avg,
        "val/Acc@5": top5.avg,
    }

    if accelerator.is_main_process:
        print("Validation results: ")
        print(" \t".join([f"{k}: {v:.4f}" for k, v in logs.items()]))

    return logs



def resume(checkpoint_path, config, model, optimizer, scheduler, device):

    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_state_dict = checkpoint['model']
    # remove prefix added by DataParallel
    unwanted_prefix = '_orig_mod.'
    for k,v in list(new_state_dict.items()):
        if k.startswith(unwanted_prefix):
            new_state_dict[k[len(unwanted_prefix):]] = new_state_dict.pop(k)

    # update config
    config.start_epoch = checkpoint["epoch"]
    config.best_acc1 = checkpoint["best_acc1"]

    # load state dict
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    # reset random state
    torch.set_rng_state(checkpoint['torch_rng_state'].type(torch.ByteTensor))
    torch.cuda.set_rng_state_all([state.type(torch.ByteTensor) for state in checkpoint['torch_cuda_rng_state']])

    # free memory
    checkpoint = None

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a config file and run the training loop")

    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="Path to the python dataclass config file",
    )

    args = parser.parse_args()
    DataClass = load_dataclass(args.config_path)
    run(DataClass())
