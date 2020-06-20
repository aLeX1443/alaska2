import numpy as np

import torch.nn as nn

# 30 SRM filters
from alaska2.alaska_pytorch.models.colour.srm_filter_kernel import (
    all_normalized_hpf_list,
)

# Global covariance pooling
from alaska2.alaska_pytorch.models.colour.MPNCOV import *


# Truncation operation
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output


# Pre-processing Module
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(
                    hpf_item, pad_width=((1, 1), (1, 1)), mode="constant"
                )

            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = nn.Parameter(
            torch.tensor(all_hpf_list_5x5).view(30, 1, 5, 5),
            requires_grad=False,
        )

        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        # self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, x):
        output = self.hpf(x)
        output = self.tlu(output)

        return output


class Net(nn.Module):
    def __init__(self, n_classes=4):
        super(Net, self).__init__()

        self.group1 = HPF()

        self.group2 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), n_classes)

    def forward(self, img):
        output = img

        output = self.group1(output)
        output = self.group2(output)
        output = self.group3(output)
        output = self.group4(output)
        output = self.group5(output)

        # Global covariance pooling
        output = CovpoolLayer(output)
        output = SqrtmLayer(output, 5)
        output = TriuvecLayer(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)

        return output


def build_yedroudj_net(n_classes):
    return Net(n_classes)


if __name__ == "__main__":
    model = build_yedroudj_net(n_classes=4)
    rgb = torch.rand((1, 3, 512, 512), dtype=torch.float, requires_grad=False)
    out = model(rgb)
    print(out)
    print(torch.softmax(out, dim=1))
