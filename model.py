# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from functools import partial
from typing import Any, List

import torch
from torch import Tensor
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

from utils import make_divisible

__all__ = [
    "MobileNetV3",
    "InvertedResidual",
    "mobilenet_v3_small", "mobilenet_v3_large"
]

mobilenet_v3_small_cfg: List[list[int, int, int, int, bool, str, int, int, int]] = [
    # in_channels, expand_channels, out_channels, kernel_size, use_se, activation_layer_name, stride, dilation
    [16, 16, 16, 3, True, "ReLU", 2, 1],
    [16, 72, 24, 3, False, "ReLU", 2, 1],
    [24, 88, 24, 3, False, "ReLU", 1, 1],
    [24, 96, 40, 5, True, "Hardswish", 2, 1],
    [40, 240, 40, 5, True, "Hardswish", 1, 1],
    [40, 240, 40, 5, True, "Hardswish", 1, 1],
    [40, 120, 48, 5, True, "Hardswish", 1, 1],
    [48, 144, 48, 5, True, "Hardswish", 1, 1],
    [48, 288, 96, 5, True, "Hardswish", 2, 1],
    [96, 576, 96, 5, True, "Hardswish", 1, 1],
    [96, 576, 96, 5, True, "Hardswish", 1, 1],
]
mobilenet_v3_large_cfg: List[list[int, int, int, int, bool, str, int, int, int]] = [
    # in_channels, expand_channels, out_channels, kernel_size, use_se, activation_layer_name, stride, dilation
    [16, 16, 16, 3, False, "ReLU", 1, 1],
    [16, 64, 24, 3, False, "ReLU", 2, 1],
    [24, 72, 24, 3, False, "ReLU", 1, 1],
    [24, 72, 40, 5, True, "ReLU", 2, 1],
    [40, 120, 40, 5, True, "ReLU", 1, 1],
    [40, 120, 40, 5, True, "ReLU", 1, 1],
    [40, 240, 80, 3, False, "Hardswish", 2, 1],
    [80, 200, 80, 3, False, "Hardswish", 1, 1],
    [80, 184, 80, 3, False, "Hardswish", 1, 1],
    [80, 184, 80, 3, False, "Hardswish", 1, 1],
    [80, 480, 112, 3, True, "Hardswish", 1, 1],
    [112, 672, 112, 3, True, "Hardswish", 1, 1],
    [112, 672, 160, 5, True, "Hardswish", 2, 1],
    [160, 960, 160, 5, True, "Hardswish", 1, 1],
    [160, 960, 160, 5, True, "Hardswish", 1, 1],
]


class MobileNetV3(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
            arch_name: str = "mobilenet_v3_small",
            width_mult: float = 1.0,
            dropout: float = 0.2,
            reduced_tail: bool = False,
            dilated: bool = False,
    ) -> None:
        super(MobileNetV3, self).__init__()
        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1

        if arch_name == "mobilenet_v3_small":
            arch_cfg = mobilenet_v3_small_cfg
            last_channels = make_divisible(1024 // reduce_divider, 8)
        else:
            arch_cfg = mobilenet_v3_large_cfg
            last_channels = make_divisible(1280 // reduce_divider, 8)

        # Modify arch config
        arch_cfg[-3][2] = arch_cfg[-3][2] // reduce_divider
        arch_cfg[-3][-1] = dilation

        arch_cfg[-2][0] = arch_cfg[-2][0] // reduce_divider
        arch_cfg[-2][1] = arch_cfg[-2][1] // reduce_divider
        arch_cfg[-2][2] = arch_cfg[-2][2] // reduce_divider
        arch_cfg[-2][-1] = dilation

        arch_cfg[-1][0] = arch_cfg[-1][0] // reduce_divider
        arch_cfg[-1][1] = arch_cfg[-1][1] // reduce_divider
        arch_cfg[-1][2] = arch_cfg[-1][2] // reduce_divider
        arch_cfg[-1][-1] = dilation

        features: List[nn.Module] = [
            Conv2dNormActivation(3,
                                 arch_cfg[0][0],
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,
                                 norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                                 activation_layer=nn.Hardswish,
                                 inplace=True,
                                 bias=False,
                                 )
        ]
        for in_channels, expand_channels, out_channels, kernel_size, use_se, activation_layer_name, stride, dilation in arch_cfg:
            features.append(
                InvertedResidual(
                    in_channels,
                    expand_channels,
                    out_channels,
                    kernel_size,
                    use_se,
                    activation_layer_name,
                    stride,
                    dilation,
                    width_mult))
        classifier_channels = int(arch_cfg[-1][2] * 6)
        features.append(
            Conv2dNormActivation(arch_cfg[-1][2],
                                 classifier_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                                 activation_layer=nn.Hardswish,
                                 inplace=True,
                                 bias=False,
                                 ),
        )
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(classifier_channels, last_channels),
            nn.Hardswish(True),
            nn.Dropout(dropout, True),
            nn.Linear(last_channels, num_classes),
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


class InvertedResidual(nn.Module):
    def __init__(
            self,
            in_channels: int,
            expand_channels: int,
            out_channels: int,
            kernel_size: int,
            use_se: bool,
            activation_layer_name: str,
            stride: int,
            dilation: int,
            width_mult: float,
    ) -> None:
        super(InvertedResidual, self).__init__()
        in_channels = make_divisible(int(in_channels * width_mult), 8)
        expand_channels = make_divisible(int(expand_channels * width_mult), 8)
        out_channels = make_divisible(int(out_channels * width_mult), 8)
        if activation_layer_name == "Hardswish":
            activation_layer = nn.Hardswish
        else:
            activation_layer = nn.ReLU
        stride = 1 if dilation > 1 else stride

        self.use_res_connect = stride == 1 and in_channels == out_channels

        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        block: List[nn.Module] = []
        if in_channels != expand_channels:
            # expand
            block.append(
                Conv2dNormActivation(
                    in_channels,
                    expand_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                    activation_layer=activation_layer,
                    inplace=True,
                    bias=False,
                )
            )
        # Depth-wise + project
        block.append(
            # Depth-wise
            Conv2dNormActivation(
                expand_channels,
                expand_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2 * dilation,
                groups=expand_channels,
                norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                activation_layer=activation_layer,
                dilation=dilation,
                inplace=True,
                bias=False,
            )
        )
        # SqueezeExcitation
        if use_se:
            block.append(SqueezeExcitation(input_channels=expand_channels,
                                           squeeze_channels=make_divisible(expand_channels // 4, 8),
                                           activation=nn.ReLU,
                                           scale_activation=nn.Hardsigmoid))
        block.append(
            # project
            Conv2dNormActivation(
                expand_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                activation_layer=None,
                dilation=dilation,
                inplace=True,
                bias=False,
            )
        )
        self.block = nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)

        if self.use_res_connect:
            out = torch.add(out, x)

        return out


def mobilenet_v3_small(**kwargs: Any) -> MobileNetV3:
    model = MobileNetV3(arch_name="mobilenet_v3_small", **kwargs)

    return model


def mobilenet_v3_large(**kwargs: Any) -> MobileNetV3:
    model = MobileNetV3(arch_name="mobilenet_v3_large", **kwargs)

    return model
