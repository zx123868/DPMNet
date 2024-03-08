# DPMNet



## Introduction

MLP−based networks, while being lighter than traditional convolution− and transformer−based networks commonly used in medical image segmentation, often struggle with capturing local structures due to the limitations of fully−connected (FC) layers, making them less ideal for such tasks. To address this issue, we design a Dual−Path MLP−based network (DPMNet) that includes a global and a local branch to understand the input images at different scales. In the two branches, we design an Axial Residual Connection MLP module (ARC−MLP) to combine it with CNNs to capture the input image’s global long−range dependencies and local visual structures simultaneously. Additionally, we propose a Shifted Channel−Mixer MLP block (SCM−MLP) across width and height as a key component of ARC−MLP to mix information from different spatial locations and channels. Extensive experiments demonstrate that the DPMNet significantly outperforms seven state−of−the−art convolution− , transformer−, and MLP−based methods in both Dice and IoU scores, where the Dice and IoU scores for the IAS−L dataset are 88.98% and 80.31% respectively. Code is available at https://anonymous.4open.science/r/DPMNet-1DB6.

## Using the code:

The code is stable while using Python 3.6.13, CUDA >=10.1



To install all the dependencies using conda:

```bash
conda env create -f environment.yml
conda activate DPMNet
```

If you prefer pip, install following versions:

```bash
timm==0.3.2
mmcv-full==1.2.7
torch==1.7.1
torchvision==0.8.2
opencv-python==4.5.1.48
```


## Data Format

Make sure to put the files as the following structure (e.g. the number of classes is 2):

```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
        |
        └── 1
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
```

For binary segmentation problems, just use folder 0.

## Training and Validation

1. Train the model.
```
python train.py --dataset <dataset name> --arch DPMNet --name <exp name> --img_ext .png --mask_ext .png --lr 0.0001 --epochs 500 --input_w 512 --input_h 512 --b 8
```
2. Evaluate.
```
python val.py --name <exp name>
```

