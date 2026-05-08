# ============================================================
# MobileNet V2 encoder - lightweight convolutional feature trunk
# ============================================================
# 本文件基于 torchvision MobileNetV2 做轻量封装：
# 1. num_images 控制输入通道数，支持多帧图像按通道拼接
# 2. features 暴露卷积主干，GNMEncoder 会取 features 而不是 classifier
# 3. 初始化逻辑保持 torchvision 风格，便于和旧模型结构兼容
# modified from PyTorch torchvision library
from typing import Callable, Any, Optional, List

import torch
from torch import Tensor
from torch import nn

from torchvision.ops.misc import ConvNormActivation
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv2 import InvertedResidual


# MobileNet V2 主干网络封装
class MobileNetEncoder(nn.Module):
    # 初始化网络结构与参数
    def __init__(
        self,
        num_images: int = 1,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_images (int): number of images stacked in the input tensor
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()

        if block is None:
            # 默认使用 torchvision 的 InvertedResidual 倒残差块
            block = InvertedResidual

        if norm_layer is None:
            # 默认 BatchNorm2d；调用方如需 GroupNorm 应在外部替换
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            # 每行含义：t=expand ratio, c=输出通道, n=重复次数, s=首块 stride
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # 构建首层卷积
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features: List[nn.Module] = [
            # num_images * 3 表示多张 RGB 图像沿通道维拼接后的输入通道数
            ConvNormActivation(
                num_images * 3,
                input_channel,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        ]
        # 构建倒残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                # 每组第一个 block 负责下采样，其余 block stride=1 保持分辨率
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        # 构建尾部层
        features.append(
            # Conv2dNormActivation(
            ConvNormActivation(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # 构建分类头
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # 权重初始化
        for m in self.modules():
            # 卷积/归一化/线性层采用 torchvision MobileNetV2 的默认初始化策略
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # 前向传播实现
    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # 对外前向接口
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
