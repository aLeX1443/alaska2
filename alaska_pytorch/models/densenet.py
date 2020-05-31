from torch import nn
import torchvision.models


def build_densenet161(n_classes):
    model = torchvision.models.densenet161(pretrained=True)
    model.fc = nn.Linear(2208, n_classes)
    return model


if __name__ == "__main__":
    build_densenet161(n_classes=1)
