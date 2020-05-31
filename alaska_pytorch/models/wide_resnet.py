from torch import nn
import torchvision.models


def build_wide_resnet50_2(n_classes):
    model = torchvision.models.wide_resnet50_2(pretrained=True)
    model.fc = nn.Linear(2048, n_classes)
    return model


def build_wide_resnet101_2(n_classes):
    model = torchvision.models.wide_resnet101_2(pretrained=True)
    model.fc = nn.Linear(2048, n_classes)
    return model


if __name__ == "__main__":
    build_wide_resnet101_2(n_classes=1)
