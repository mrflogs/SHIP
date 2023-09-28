# Improving Zero-Shot Generalization for CLIP with Synthesized Prompts

Official implementation of  [Improving Zero-Shot Generalization for CLIP with Synthesized Prompts](https://arxiv.org/abs/2307.07397).

This paper has been accepted by **ICCV 2023**.

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

### Running
For ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet_coop_vae.py --config configs/imagenet.yaml
```
For other 10 datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python main_coop_vae.py --config configs/dataset.yaml
```

## Acknowledgement

This repo benefits from [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch) and [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter). Thanks for their wonderful works.

## Citation

```
@inproceedings{wang2023improving,
  title={Improving Zero-Shot Generalization for CLIP with Synthesized Prompts},
  author={Zhengbo Wang and Jian Liang and Ran He and Nan Xu and Zilei Wang and Tieniu Tan},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3032-3042}
}
```

## Contact

If you have any question, feel free to contact zhengbowang@mail.ustc.edu.cn.
