# Beyond Rotation Invariance: Exploring Adam's Adaptive Power

This repository provides a PyTorch implementation of the paper [Beyond Rotation Invariance: Exploring Adam's Adaptive Power](#)

## Abstract
In this project, we explore Adam's adaptive power beyond rotation invariance, focusing on its second moments across dimensions, time, and neuron layers. Through controlled experiments with ResNet and Transformer models on various tasks, we investigate the specific benefits of Adam's design and its performance contrast against SGD. We aim to bridge the theoretical understanding and practical success of Adam, challenging the rotation-invariant assumptions that fail to explain its effectiveness fully. This study seeks to demonstrate how Adam's adaptiveness contributes to its superior performance in language modeling and beyond.

# Experiment Running Guide for Compute Canada

## Installation

### Step 1: Clone the repository

```bash
cd ~/projects/rrg-bengioy-ad/$USER/
git clone https://github.com/tianyuehz/Adam_second_moment
cd Adam_second_moment
git checkout compute-canada
ln -s ~/projects/rrg-bengioy-ad/$USER/Adam_second_moment ~/adam_second_moment
```

### Step 2: Uploading openwebtext

Start with creating a directory for the dataset:

```bash
mkdir ~/projects/rrg-bengioy-ad/$USER/datasets/
```

Then, upload the dataset to the directory. You can use `scp` or to upload the **binaries** of the dataset, i.e `train.bin`, `valid.bin`. For example:

```bash
scp -r /path/to/openwebtext/ <CC-USERNAME>@narval.computecanada.ca:~/projects/rrg-bengioy-ad/<CC-USERNAME>/datasets/
```

After uploading the dataset, you should have the following structure:

```bash
  datasets/
  ├── openwebtext/
  │   ├── train.bin
  │   └── valid.bin
```

### Step 3: Installing the dependencies

First of all you need to load some modules to install the dependencies:

```bash
module load python/3.10 arrow
```

Then, you can setup a virtual environment and install the dependencies:

```bash
mkdir ~/envs/
virtualenv --no-download ~/envs/rotAdam
source ~/envs/rotAdam/bin/activate
pip install -r ~/adam_second_moment/requirements.txt
deactivate
```

To activate the environment, you can run:

```bash
source ~/envs/rotAdam/bin/activate
```

To deactivate the environment, you can run:

```bash
deactivate
```

### Step 4: Generating experiments config

To generate the experiments configurations files, you can run the following command:

```bash
python ~/adam_second_moment/experiment.py -t <exp_type>
```

Where all `<exp_type>` can be found in the `experiment.py` file.

**Rem: you can generate all the configs at one using all as <exp_type>**

### Step 5: Wandb

The project use wandb to log the experiments. You need to login to wandb to be able to log the experiments. You can do this by running the following command:

```bash
wandb login
```

Because compute canada does not support the wandb logging online we will synchrone the logs to the wandb server after the experiment is done. To do this you need to run the following command **before** running the experiment:

```bash
wandb offline
```

Then when you want to synchrone the logs you can run the following command:

```bash
wandb sync --sync-all
```

### Running the experiment

The scripts for running experiment on compute canada are located in the `scripts/compute-canada` directory.

**Rem: All experiments should be launched from the root of the project**, i.e `~/adam_second_moment/`.

You can run the following command to run the experiment:

```bash
sbatch scripts/compute-canada/<desired_script> ./path/to/experiment/config
```

Where `<desired_script>` is the script you want to run and `./path/to/experiment/config` is the path to the experiment configuration file.

for example:

```bash
sbatch scripts/compute-canada/train_cifar10.sh ./config/vision/experiments/resnet18-cifar10/SGD/blabla.yml
```

### Synchronize the logs in wandb

After the experiment is done, you can synchronize the logs to the wandb server by running the following command:

```bash
wandb sync --sync-all
```

### Run logs

The logs of the experiments are stored in the `slurm-logs/` directory. All the metrics and data are stored in the `logs/` directory.

## Authors
- [Lucas Maes](https://lucas-maes.github.io/)
- [Tianyue H. Zhang](https://tianyuehz.github.io/)
- [Charles Guille-Escuret](https://charlesge.github.io/)
