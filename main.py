import torch
import torch.nn as nn
from accelerate import Accelerator
from src.data_utils import get_dataloaders, get_criterion
from src.opt_utils import get_optimizer, get_scheduler
from src.model_utils import get_model
from src.yaml_utils import load_config
import tqdm
import argparse

def run(config_file):

    config = load_config(config_file)
    accelerator = Accelerator()
    device = accelerator.device

    net = get_model(config)
    opt = get_optimizer(config, net)
    train_loader, test_loader = get_dataloaders(config)
    criterion = get_criterion(config)

    epochs = config['training']['epochs']
    n_steps = epochs * len(train_loader)
    scheduler = get_scheduler(config, opt, n_steps)

    net, opt, train_loader, test_loader, scheduler = accelerator.prepare(
        net, opt, train_loader, test_loader, scheduler
    )

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Training]')
        for inputs, labels in train_pbar:

            opt.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            opt.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training loss: {avg_train_loss:.4f}')
        print(f'Epoch {epoch + 1}, Learning rate: {scheduler.get_last_lr()[0]:.4f}')

        # checkpoint optimizer
        checkpoint = { 
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': opt.state_dict(),}
        breakpoint()
        torch.save(checkpoint, 'checkpoint.pth')

        # Validation phase
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            test_pbar = tqdm.tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Validation]')
            for inputs, labels in test_pbar:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_loader)
        print(f'Epoch {epoch + 1}, Validation loss: {avg_val_loss:.4f}')
        print(f'Epoch {epoch + 1}, Validation accuracy: {val_accuracy:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
    )
    args = vars(parser.parse_args())

    run(args['config_file'])