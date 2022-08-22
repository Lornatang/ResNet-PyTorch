# ResNet-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Searching for ResNet](https://arxiv.org/pdf/1512.03385v1.pdf).

## Table of contents

- [ResNet-PyTorch](#resnet-pytorch)
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
        - [Deep Residual Learning for Image Recognition](#deep-residual-learning-for-image-recognition)

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

- line 29: `model_arch_name` change to `resnet18`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `resnet18`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change to `./results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `resnet18`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/resnet18-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1512.03385v1.pdf](https://arxiv.org/pdf/1512.03385v1.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|   Model   |   Dataset   | Top-1 error (val)  | Top-5 error (val) |
|:---------:|:-----------:|:------------------:|:-----------------:|
| resnet18  | ImageNet_1K | 27.88%(**30.25%**) |   -(**10.93%**)   |
| resnet34  | ImageNet_1K | 25.03%(**26.71%**) | 7.76%(**8.58%**)  |
| resnet50  | ImageNet_1K | 22.85%(**19.65%**) | 6.71%(**4.87%**)  |
| resnet101 | ImageNet_1K | 21.75%(**18.33%**) | 6.05%(**4.34%**)  |
| resnet152 | ImageNet_1K | 21.43%(**17.66%**) | 5.71%(**4.08%**)  |

```bash
# Download `ResNet18-ImageNet_1K-57bb63e.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `resnet18` model successfully.
Load `resnet18` model weights `/ResNet-PyTorch/results/pretrained_models/ResNet18-ImageNet_1K-57bb63e.pth.tar` successfully.
tench, Tinca tinca                                                          (91.46%)
barracouta, snoek                                                           (7.15%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.43%)
coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch    (0.27%)
platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus (0.21%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Deep Residual Learning for Image Recognition

*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*

##### Abstract

Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of
networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning
residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide
comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from
considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x
deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the
ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on
CIFAR-10 with 100 and 1000 layers.
The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely
deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are
foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of
ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

[[Paper]](https://arxiv.org/pdf/1512.03385v1.pdf)

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```