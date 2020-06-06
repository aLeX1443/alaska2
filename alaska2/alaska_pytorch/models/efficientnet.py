import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from efficientnet_pytorch import EfficientNet


class StegoEfficientNet(EfficientNet):
    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    @autocast()
    def forward(self, inputs: torch.tensor) -> torch.tensor:
        """
        Calls extract_features to extract features, applies final linear layer
        and returns logits.
        """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)

        # Make sure the final linear layer is in FP32
        with autocast(enabled=False):
            x = x.float()
            x = self._fc(x)

        return x


def build_efficientnet_b0(n_classes: int) -> nn.Module:
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b0")


def build_efficientnet_b1(n_classes: int) -> nn.Module:
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b1")


def build_efficientnet_b2(n_classes: int) -> nn.Module:
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b2")


def build_efficientnet_b3(n_classes: int) -> nn.Module:
    return StegoEfficientNet.from_pretrained(
        model_name="efficientnet-b3", num_classes=n_classes
    )


def build_efficientnet_b4(n_classes: int) -> nn.Module:
    return StegoEfficientNet.from_pretrained(
        model_name="efficientnet-b4", num_classes=n_classes
    )


def build_efficientnet_b5(n_classes: int) -> nn.Module:
    return StegoEfficientNet.from_pretrained(
        model_name="efficientnet-b5", num_classes=n_classes
    )


def build_efficientnet_b6(n_classes: int) -> nn.Module:
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b6")


def build_efficientnet_b7(n_classes: int) -> nn.Module:
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b7")


def build_efficientnet_b8(n_classes: int) -> nn.Module:
    return StegoEfficientNet(n_classes=n_classes, model_name="efficientnet-b8")
