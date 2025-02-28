# IODA (NeurIPS2024)

This repository is an official PyTorch implementation of the paper [IODA: Instance-Guided One-shot Domain Adaptation for Super-Resolution](https://proceedings.neurips.cc/paper_files/paper/2024/file/d4ce6738e84876aa79f13c8bc8b7c5eb-Paper-Conference.pdf).

Zaizuo Tang<sup>1</sup>, Yubin Yang<sup>1</sup>, 

<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China<br>

## Abstract
The domain adaptation method effectively mitigates the negative impact of domain gaps on the performance of super-resolution (SR) networks through the guidance of numerous target domain low-resolution (LR) images. However, in real-world scenarios, the availability of target domain LR images is often limited, sometimes even to just one, which inevitably impairs the domain adaptation performance of SR networks. We propose Instance-guided One-shot Domain Adaptation for Super-Resolution (IODA) to enable efficient domain adaptation with only a single unlabeled target domain LR image. To address the limited diversity of the target domain distribution caused by a single target domain LR image, we propose an instance-guided target domain distribution expansion strategy. This strategy effectively expands the diversity of the target domain distribution by generating instance-specific features focused on different instances within the image. For SR tasks emphasizing texture details, we propose an image-guided domain adaptation method. Compared to existing methods that use text representation for domain difference, this method utilizes pixel-level representation with higher granularity, enabling efficient domain adaptation guidance for SR networks. Finally, we validate the effectiveness of IODA on multiple datasets and various network architectures, achieving satisfactory one-shot domain adaptation for SR networks. Our code is available at https://github.com/ZaizuoTang/IODA.



## Training

### Step 1: Preparation of Training and Testing Samples

1. Download the training and testing dataset.

Cityscapes 
DF2K
ACDC

2. LR Image Generation for Cityscapes and ACDC Datasets

For the Cityscapes and ACDC datasets, utilize the [downsampling tool](https://github.com/XPixelGroup/BasicSR/blob/master/scripts/matlab_scripts/generate_bicubic_img.m) provided by BasicSR to generate the corresponding LR images.


### Step 2: Pretrained Weights Acquisition

Pretrained weights of the SAFMN network on DF2K

Pretrained weights of the SRFormer network on DF2K

Pretrained weights of the HAT network on DF2K

Pretrained weights of the SAFMN network on Cityscapes

Pretrained weights of the SRFormer network on ACDC

### Step 3: Instance Mask Generation

1. Utilize [SAM](https://github.com/facebookresearch/segment-anything) for instance-level segmentation of images.

2. Combine multiple instance masks into a single image

        python Deal_mask.py

### Step4 Domain adaptation training

        python Train_second_stage.py




## Test:
        python test.py

## Inference:
        python inference_image.py






## Citation

```BibTeX
@inproceedings{NEURIPS2024_d4ce6738,
 author = {Tang, Zaizuo and Yang, Yu-Bin},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {117291--117314},
 publisher = {Curran Associates, Inc.},
 title = {IODA: Instance-Guided One-shot Domain Adaptation for Super-Resolution},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/d4ce6738e84876aa79f13c8bc8b7c5eb-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```