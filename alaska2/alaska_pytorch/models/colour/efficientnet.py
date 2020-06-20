import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import round_filters


class StegoEfficientNet(EfficientNet):
    def __init__(self, *args, **kwargs):
        super(StegoEfficientNet, self).__init__(*args, **kwargs)
        print("Dropout rate:", self._global_params.dropout_rate)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)

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
        # x = self._dropout(x)

        # Make sure the final linear layer is in FP32
        with autocast(enabled=False):
            x = x.float()
            x = self._fc(x)

        return x


class StegoQFactorEfficientNet(EfficientNet):
    def __init__(self, *args, **kwargs):
        super(StegoQFactorEfficientNet, self).__init__(*args, **kwargs)
        dropout_rate = 0.15  # self._global_params.dropout_rate
        print("Dropout rate:", dropout_rate)
        self._dropout = nn.Dropout(dropout_rate)
        out_channels = round_filters(1280, self._global_params)
        self._concatenated_fc = nn.Linear(
            out_channels + 3, self._global_params.num_classes
        )

    @autocast()
    def forward(
        self, img: torch.tensor, quality_factor: torch.tensor
    ) -> torch.tensor:
        """
        Calls extract_features to extract features, applies final linear layer
        and returns logits.
        """
        # with torch.no_grad():
        bs = img.size(0)
        x = self.extract_features(img)

        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)

        # Concatenate the extracted image features with the vector describing
        # the quality factor.
        x = torch.cat([x, quality_factor], dim=1)

        # Make sure the final linear layer is in FP32
        with autocast(enabled=False):
            x = x.float()
            x = self._concatenated_fc(x)

        return x


def build_efficientnet_b0(n_classes: int) -> nn.Module:
    return StegoEfficientNet.from_pretrained(
        model_name="efficientnet-b0", num_classes=n_classes,
    )


def build_efficientnet_b1(n_classes: int) -> nn.Module:
    return StegoEfficientNet.from_pretrained(
        model_name="efficientnet-b1", num_classes=n_classes,
    )


def build_efficientnet_b2(n_classes: int) -> nn.Module:
    return StegoEfficientNet.from_pretrained(
        model_name="efficientnet-b2", num_classes=n_classes
    )


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
    return StegoEfficientNet.from_pretrained(
        model_name="efficientnet-b6", num_classes=n_classes
    )


def build_efficientnet_b7(n_classes: int) -> nn.Module:
    return StegoEfficientNet.from_pretrained(
        model_name="efficientnet-b7", num_classes=n_classes
    )


def build_efficientnet_b8(n_classes: int) -> nn.Module:
    return StegoEfficientNet.from_pretrained(
        model_name="efficientnet-b8", num_classes=n_classes
    )


def build_quality_factor_efficientnet_b3(n_classes: int) -> nn.Module:
    return StegoQFactorEfficientNet.from_name(
        model_name="efficientnet-b3",
        override_params={"num_classes": n_classes},
    )


def build_quality_factor_efficientnet_b5(n_classes: int) -> nn.Module:
    return StegoQFactorEfficientNet.from_name(
        model_name="efficientnet-b5",
        override_params={"num_classes": n_classes},
    )


if __name__ == "__main__":
    model = build_quality_factor_efficientnet_b3(n_classes=4)
    rgb = torch.zeros(1, 3, 512, 512, dtype=torch.float, requires_grad=False)
    q_factor = torch.zeros(1, 3, dtype=torch.float, requires_grad=False)
    out = model(rgb, q_factor)
    print(torch.softmax(out, dim=1))
