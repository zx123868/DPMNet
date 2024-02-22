# DPMNet



## Introduction

<p align="center">
  <img src="imgs/DPMNet.png" width="800"/>
</p>


## Using the code:

The code is stable while using Python 3.6.13, CUDA >=10.1

- Clone this repository:
```bash
git clone https://github.com/jeya-maria-jose/DPMNet-pytorch
cd DPMNet-pytorch
```

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

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [UNet++](https://github.com/4uiiurz1/pytorch-nested-unet), [Segformer](https://github.com/NVlabs/SegFormer), and [AS-MLP](https://github.com/svip-lab/AS-MLP). Naming credits to [Poojan](https://scholar.google.co.in/citations?user=9dhBHuAAAAAJ&hl=en).

### Citation:
```
@article{valanarasu2022DPMNet,
  title={DPMNet: MLP-based Rapid Medical Image Segmentation Network},
  author={Valanarasu, Jeya Maria Jose and Patel, Vishal M},
  journal={arXiv preprint arXiv:2203.04967},
  year={2022}
}
```
