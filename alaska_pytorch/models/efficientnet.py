import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


def build_efficientnet_b0(n_classes):
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b0")


def build_efficientnet_b1(n_classes):
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b1")


def build_efficientnet_b2(n_classes):
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b2")


def build_efficientnet_b3(n_classes):
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b3")


def build_efficientnet_b4(n_classes):
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b4")


def build_efficientnet_b5(n_classes):
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b5")


def build_efficientnet_b6(n_classes):
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b6")


def build_efficientnet_b7(n_classes):
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b7")


def build_efficientnet_b8(n_classes):
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b8")


class StegoEfficientNet(nn.Module):
    def __init__(self, n_classes, model_name="efficientnet-b0"):
        super().__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        self.efficientnet_output_units = self.model.state_dict()[
            "_fc.weight"
        ].shape[1]
        # Add a dense prediction layer.
        self.dense_output = nn.Linear(
            self.efficientnet_output_units, n_classes
        )

    def forward(self, x):
        features = self.model.extract_features(x)
        features = F.avg_pool2d(features, features.size()[2:]).reshape(
            -1, self.efficientnet_output_units
        )
        return self.dense_output(features)
