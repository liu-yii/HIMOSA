# HIMOSA
This repository is an official implementation of the paper "Efficient Remote Sensing Image Super-Resolution with Hierarchical Mixture of Sparse Attention". 

⭐If this work is helpful for you, please help star this repo. Thanks!🤗

## :bookmark_tabs:Contents
1. [Enviroment](#Environment)
1. [Training](#Training)
1. [Testing](#Testing)
1. [Contact](#Contact)
1. [Acknowledgements](#Acknowledgements)


## :hammer:Environment
- Python 3.9
- PyTorch >=2.2

### Installation
```bash
pip install -r requirements.txt
python setup.py develop
```




## :rocket:Training
### Data Preparation
- Download the training dataset [Satlaspretrain](https://satlas-pretrain.allen.ai/) and [AID](https://captain-whu.github.io/AID/).
- Download the testing data [NWPU](https://gcheng-nwpu.github.io/#Datasets), [DOTA](https://captain-whu.github.io/DOTA/dataset.html), [DIOR](https://ieee-dataport.org/documents/dior).
- It's recommended to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.

### Training Commands
- Refer to the training configuration files in `./options/train` folder for detailed settings.
```bash
# batch size = 4 (GPUs) × 16 (per GPU)
# training dataset:AID

# ×4, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 basicsr/train.py -opt options/train/train_HIMOSANet_x4_finetune.yml --launcher pytorch
```




## :wrench:Testing

### Pretrained Models
- Download the [pretrained models](https://drive.google.com/file/d/1QcQvpYIVSjxWyteQar5gbCW97ntag837/view?usp=drive_link) and put them in the folder `./pretrained_models`.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.


```bash
python basicsr/test.py -opt options/test/103_HIMOSAv1_light_SRx4_finetune.yml
```

## :mailbox:Contact

If you have any questions, feel free to approach me at liuyiwhu28@whu.edu.cn

## 🥰Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
