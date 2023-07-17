# Improving Zero-Shot Generalization for CLIP with Synthesized Prompts

We provide code of CoOp + SHIP.

## Requirements
### Installation
Create a conda environment and install dependencies:
```
conda create -n ship python=3.9
conda activate ship

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Follow DATASET.md to install ImageNet and other 10 datasets referring to CoOp.

## Get Started
### Configs
The running configurations can be modified in `coop-configs/dataset.yaml`, including shot numbers, visual encoders, and hyperparamters. 

### Numerical Results
We provide CoOp + SHIP's results of base-to-new generalization at coop_vae.log

### Running
For ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet_coop_vae.py --config configs/imagenet.yaml
```
For other 10 datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python main_coop_vae.py --config configs/dataset.yaml
```

