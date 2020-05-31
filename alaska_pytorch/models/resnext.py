from torch import nn
import torchvision.models


def build_resnext50_32x4d(n_classes):
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(2048, n_classes)
    return model


def build_resnext101_32x8d(n_classes):
    model = torchvision.models.resnext101_32x8d(pretrained=True)
    model.fc = nn.Linear(2048, n_classes)
    return model


if __name__ == "__main__":
    print(build_resnext101_32x8d(n_classes=1).to("cpu"))
