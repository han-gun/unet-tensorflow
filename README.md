# unet-tensorflow
u-net is implementated by tensorflow using python.

## Requirements
Python3.6 with following packages
- tensorflow >= 1.12.0

## Setup

### Prerequisites
- Tensorflow >= 1.12.0
- tqdm

### Environments
- Windows 10 with tensorflow-gpu + cuDNN

### Getting Started
```bash
# clone this repository
git clone https://github.com/han-gun/unet-tensorflow.git

# train the model
python main.py \
        --mode train \
        --scope denosing \ 
        --checkpoint_dir ./checkpoint \
        --input_dir data \
        --output_dir test
        
# test the model
python main.py \ 
        --mode test \
        --scope denosing \
        --checkpoint_dir ./checkpoint \
        --input_dir data \ 
        --output_dir test
        
# open the tensorboard
python tensorbard \
        --logdir=./checkpoint/denosing
```

## References
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
