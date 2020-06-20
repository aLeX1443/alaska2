"""
Based on the model from: https://papers.nips.cc/paper/7649-faster-neural-networks-straight-from-jpeg.pdf

Using the method described as Deconvolution-RFA to upsample the Cb and Cr
inputs from (32, 32, 64) to (64, 64, 64) to match the Y channel.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from torchvision.models import ResNet

from torchvision.models.resnet import BasicBlock
from torchviz import make_dot


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        print(width, inplanes)
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DCTResNet(ResNet):
    """
    Architecture from: https://papers.nips.cc/paper/7649-faster-neural-networks-straight-from-jpeg.pdf
    """

    def __init__(
        self,
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        dct_channels=64,
        num_classes: int = 4,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # TODO use deconv network rather than single layer.
        # Make separate layers so we don't share weights
        self.deconvolution_cb = nn.ConvTranspose2d(
            dct_channels, dct_channels, kernel_size=2, stride=2
        )
        self.deconvolution_cr = nn.ConvTranspose2d(
            dct_channels, dct_channels, kernel_size=2, stride=2
        )

        self.modified_layer3 = self._make_layer(
            block=Bottleneck, planes=256, blocks=6, stride=1, dilate=False
        )

    @autocast()
    def forward(
        self, dct_y: torch.tensor, dct_cb: torch.tensor, dct_cr: torch.tensor
    ) -> torch.tensor:
        # Upsample the Cb channel: (64, 32, 32) -> (64, 64, 64)
        upsampled_cb = self.deconvolution_cb(dct_cb)
        # Upsample the Cr channel: (64, 32, 32) -> (64, 64, 64)
        upsampled_cr = self.deconvolution_cr(dct_cr)

        # Concatenate the inputs 3x (64, 64, 64) -> (192, 64, 64)
        x = torch.cat((dct_y, upsampled_cb, upsampled_cr), dim=1)

        # x = self.layer2(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Make sure the final linear layer is in FP32
        with autocast(enabled=False):
            x = x.float()
            x = self.fc(x)

        return x


# class ConvBlock(nn.Module):
#     expansion = 4
#
#     def __init__(
#         self,
#         inplanes,
#         planes,
#         stride=1,
#         downsample=None,
#         groups=1,
#         base_width=64,
#         dilation=1,
#         norm_layer=None,
#     ):
#         print(downsample)
#         super(ConvBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.0)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         print(width, inplanes)
#         self.conv1 = conv1x1(inplanes, width)
#         # self.conv1 = conv1x1(192, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv1x1(width, width, stride)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = nn.Sequential(
#             nn.Conv2d(192, 1024, kernel_size=1, stride=stride, bias=False,),
#             # nn.BatchNorm2d(192),
#         )
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         print(x.shape)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         print(out.shape)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         print(out.shape)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         print(out.shape)
#
#         if self.downsample:
#             print("Downsampling")
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         print(out.shape)
#         exit()
#
#         return out
#
#


# class CustomBottleneck(nn.Module):
#     # Bottleneck in torchvision places the stride for downsampling at 3x3
#     # convolution(self.conv2) while original implementation places the stride
#     # at the first 1x1 convolution(self.conv1) according to "Deep residual
#     # learning for image recognition"https://arxiv.org/abs/1512.03385. This
#     # variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
#
#     expansion = 4
#
#     def __init__(
#         self,
#         inplanes,
#         planes,
#         stride=1,
#         downsample=None,
#         groups=1,
#         base_width=64,
#         dilation=1,
#         norm_layer=None,
#     ):
#         super(CustomBottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.0)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv1x1(width, width, stride)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


def build_dct_resnet_50(n_classes: int) -> nn.Module:
    return DCTResNet(num_classes=n_classes)


if __name__ == "__main__":
    resnet = DCTResNet()
    y = torch.zeros(1, 64, 64, 64, dtype=torch.float, requires_grad=False)
    cb = torch.zeros(1, 64, 32, 32, dtype=torch.float, requires_grad=False)
    cr = torch.zeros(1, 64, 32, 32, dtype=torch.float, requires_grad=False)
    make_dot(resnet(y, cb, cr)).render("model", format="png")
