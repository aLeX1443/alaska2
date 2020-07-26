import math

import torch
import torch.nn as nn

from torchviz import make_dot


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DCTResNet(nn.Module):
    def __init__(self, block, layers, num_classes=4, dct_channels=64):
        self.inplanes = 192
        super(DCTResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # TODO use deconv network rather than single layer.
        # Make separate layers so we don't share weights
        self.deconvolution_cb = nn.ConvTranspose2d(
            dct_channels, dct_channels, kernel_size=2, stride=2
        )
        self.deconvolution_cr = nn.ConvTranspose2d(
            dct_channels, dct_channels, kernel_size=2, stride=2
        )

        # self.modified_layer3 = self._make_layer(
        #     block=Bottleneck, planes=256, blocks=6, stride=1
        # )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(
        self, dct_y: torch.Tensor, dct_cb: torch.Tensor, dct_cr: torch.Tensor
    ) -> torch.Tensor:
        # Upsample the Cb channel: (64, 32, 32) -> (64, 64, 64)
        # upsampled_cb = self.deconvolution_cb(dct_cb)
        # Upsample the Cr channel: (64, 32, 32) -> (64, 64, 64)
        # upsampled_cr = self.deconvolution_cr(dct_cr)

        # Concatenate the inputs 3x (64, 64, 64) -> (192, 64, 64)
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)

        # x = self.layer2(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Make sure the final linear layer is in FP32
        # x = x.float()
        x = self.fc(x)

        return x


def build_dct_resnet_50(n_classes: int) -> nn.Module:
    return DCTResNet(
        block=Bottleneck, layers=[3, 4, 6, 3], num_classes=n_classes
    )


if __name__ == "__main__":
    resnet = build_dct_resnet_50(n_classes=4)
    y = torch.zeros(1, 64, 64, 64, dtype=torch.float, requires_grad=False)
    cb = torch.zeros(1, 64, 32, 32, dtype=torch.float, requires_grad=False)
    cr = torch.zeros(1, 64, 32, 32, dtype=torch.float, requires_grad=False)
    # out = resnet(y, cb, cr)
    # print(out)
    make_dot(resnet(y, cb, cr)).render("model", format="png")
