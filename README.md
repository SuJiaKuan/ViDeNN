# Videos Defects Removal

This repository contains source code for videos defects removal. Currently, it is based on [ViDeNN](https://github.com/clausmichele/ViDeNN).

**WARNING: The repository is still in early development stage. The code may be changed quickly recently.**

## Prerequisites

You need a server / computer with:
- Ubuntu 18.04 or 20.04
- Nvidia GPU that has memory >= 4 GB
- Disk space >= 500 GB

Software requirements:
- Install Anaconda or Miniconda by following the [usage guide](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).
- Install `ffmpeg` via:
```bash
sudo apt install ffmpeg
```
- [Install TWCC CLI](https://man.twcc.ai/@twccdocs/doc-cli-main-zh/https%3A%2F%2Fman.twcc.ai%2F%40twccdocs%2Fguide-cli-install-linux-zh) and [configure it](https://man.twcc.ai/@twccdocs/doc-cli-main-zh/https%3A%2F%2Fman.twcc.ai%2F%40twccdocs%2Fguide-cli-signin-zh).

### Project Installation

1. Clone or download and uncompress this repo.
```
git clone https://github.com/SuJiaKuan/ViDeNN
```
2. Change to project directory.
```
cd ViDeNN
```
3. Create a Conda environment with required packages.
```bash
conda env create -f requirements.yml -n defects
```
4. Activate the environment.
```bash
conda activate defects
```

## Train the model

The training process is divided in two steps, but only the first step (Spatial-CNN training) is supported currently.

### Spatial-CNN training

1. Change to the Spatial-CNN directory 
```bash
cd Spatial-CNN
```
2. Prepare the data by running the script (it may take hours to run):
```bash
./dataset_preparation.sh data
```
 - Please make sure you see the message: `[*] All preparation processes are done!` when the script is done.
 - The prepared data will be stored in `data`
3. Run the training script:
```bash
python3 main_spatialCNN.py data
```
- The trained models are stored in `ckpt`
- The training script saves model at every epoch (5,000 iterations, in default), and you can resume the interupted training from least epoch by running the same command.
- Check the available arguments with ```python3 main_spatialCNN.py -h``` if you want to set the number of epochs, learning rate, batch size etc.
4. Run the testing script:
```bash
python3 main_spatialCNN.py data --phase=test
```
- The result images will be save in `denoised`.
