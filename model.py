import torch
from torch import nn

from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional

from torchvision import models


def pool_conv(in_planes, out_planes=2048, filter_size=3, stride=1):
    layers = []
    padding = filter_size // 2
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=filter_size, stride=stride,
                     padding=padding, groups=1, bias=False, dilation=1)
    layers.append(conv)
    layers.append(nn.BatchNorm2d(out_planes))
    layers.append(nn.ReLU())
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    return nn.Sequential(*layers)


class BlockResNet(models.ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        filter_size: int = 1,
        noise_type="none",
        noise_std=None,
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)
        
        print("Creating ResNet-50 with a filter size of:", filter_size)
        self.pool_conv_block_1 = pool_conv(256, filter_size=filter_size)
        self.pool_conv_block_2 = pool_conv(512, filter_size=filter_size)
        self.pool_conv_block_3 = pool_conv(1024, filter_size=filter_size)
        self.pool_conv_block_4 = pool_conv(2048, filter_size=filter_size)

        self.noise_type = noise_type
        self.noise_std = noise_std
    
    def _add_noise(self, x):
        assert self.noise_type in ["none", "hw", "c", "all"]
        if not self.training or self.noise_type == "none":
            return x
        noise = torch.randn_like(x)
        if self.noise_type == "hw":
            noise = noise * torch.std(x, dim=(2, 3), keepdim=True)  # B x C x 1 x 1
        elif self.noise_type == "c":
            noise = noise * torch.std(x, dim=2, keepdim=True)  # B x 1 x H x W
        else:
            std = torch.std(x)
            noise = noise * std  # 1 x 1 x 1 x 1
        noised_x = self.noise_std * noise + x
        return noised_x
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        output_list = []
        
        x1 = self.layer1(x)
        x1 = self._add_noise(x1)
        output_list.append(torch.flatten(self.pool_conv_block_1(x1), 1))
        
        x2 = self.layer2(x1.detach())
        x2 = self._add_noise(x2)
        output_list.append(torch.flatten(self.pool_conv_block_2(x2), 1))
        
        x3 = self.layer3(x2.detach())
        x3 = self._add_noise(x3)
        output_list.append(torch.flatten(self.pool_conv_block_3(x3), 1))
        
        x4 = self.layer4(x3.detach())
        x4 = self._add_noise(x4)
        output_list.append(torch.flatten(self.pool_conv_block_4(x4), 1))
        
        return output_list

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward_impl(x)


def block_resnet50(**kwargs: Any) -> BlockResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return BlockResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
