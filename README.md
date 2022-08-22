# MobileNetV3-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244v5.pdf).

## Table of contents

- [MobileNetV3-PyTorch](#mobilenetv3-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Searching for MobileNetV3](#searching-for-mobilenetv3)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `mobilenet_v3_small`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/MobileNetV3_small-ImageNet_1K-73d198d1.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `mobilenet_v3_small`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change to `./results/pretrained_models/MobileNetV3_small-ImageNet_1K-73d198d1.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `mobilenet_v3_small`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/mobilenet_v3_small-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1905.02244v5.pdf](https://arxiv.org/pdf/1905.02244v5.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|         Model          |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:----------------------:|:-----------:|:-----------------:|:-----------------:|
| mobilenet_v3_small-1.0 | ImageNet_1K | 32.6%(**32.3%**)  |   -(**12.5%**)    |
| mobilenet_v3_large-1.0 | ImageNet_1K | 24.8%(**24.7%**)  |    -(**7.4%**)    |

```bash
# Download `MobileNetV3_small-ImageNet_1K-73d198d1.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `mobilenet_v3_small` model successfully.
Load `mobilenet_v3_small` model weights `/MobileNetV3-PyTorch/results/pretrained_models/MobileNetV3_small-ImageNet_1K-73d198d1.pth.tar` successfully.
tench, Tinca tinca                                                          (19.38%)
barracouta, snoek                                                           (7.93%)
platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus (6.00%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (4.50%)
triceratops                                                                 (1.97%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Searching for MobileNetV3

*Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang,
Vijay Vasudevan, Quoc V. Le, Hartwig Adam*

##### Abstract

We present the next generation of MobileNets based on a combination of complementary search techniques as well as a
novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network
architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel
architecture advances. This paper starts the exploration of how automated search algorithms and network design can work
together to harness complementary approaches improving the overall state of the art. Through this process we create two
new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource
use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the
task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite
Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification,
detection and segmentation. MobileNetV3-Large is 3.2\% more accurate on ImageNet classification while reducing latency
by 15\% compared to MobileNetV2. MobileNetV3-Small is 4.6\% more accurate while reducing latency by 5\% compared to
MobileNetV2. MobileNetV3-Large detection is 25\% faster at roughly the same accuracy as MobileNetV2 on COCO detection.
MobileNetV3-Large LR-ASPP is 30\% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.

[[Paper]](https://arxiv.org/pdf/1905.02244v5.pdf)

```bibtex
@inproceedings{howard2019searching,
  title={Searching for mobilenetv3},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1314--1324},
  year={2019}
}
```