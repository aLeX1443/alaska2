import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (
    round_filters,
    get_same_padding_conv2d,
    url_map_advprop,
    url_map,
)
from torch.utils import model_zoo


class StegoEfficientNet(EfficientNet):
    def __init__(self, *args, **kwargs):
        super(StegoEfficientNet, self).__init__(*args, **kwargs)
        print("Dropout rate:", self._global_params.dropout_rate)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
        # x = x.float()
        x = self._fc(x)

        return x


class StegoQFactorEfficientNet(EfficientNet):
    def __init__(self, *args, **kwargs):
        super(StegoQFactorEfficientNet, self).__init__(*args, **kwargs)
        dropout_rate = self._global_params.dropout_rate
        print("Dropout rate:", dropout_rate)
        self._dropout = nn.Dropout(dropout_rate)
        out_channels = round_filters(1280, self._global_params)
        self._concatenated_fc = nn.Linear(
            out_channels + 3, self._global_params.num_classes
        )
        # self.relu = nn.ReLU(inplace=True)
        # self.l1 = nn.Linear(out_channels + 3, 2048)
        # self.l2 = nn.Linear(2048, self._global_params.num_classes)

    def forward(
        self, img: torch.Tensor, quality_factor: torch.Tensor
    ) -> torch.Tensor:
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

        # MLP block
        # x = self.l1(x)
        # x = self.relu(x)
        # x = self.l2(x)

        x = self._concatenated_fc(x)

        return x

    @classmethod
    def from_pretrained(
        cls, model_name, advprop=False, num_classes=4, in_channels=3
    ):
        model = cls.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        cls.load_pretrained_weights(
            model, model_name, load_fc=(num_classes == 1000), advprop=advprop
        )
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(
                image_size=model._global_params.image_size
            )
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, bias=False
            )
        return model

    @staticmethod
    def load_pretrained_weights(
        model, model_name, load_fc=True, advprop=False
    ):
        """ Loads pretrained weights, and downloads if loading for the first time. """
        # AutoAugment or Advprop (different preprocessing)
        url_map_ = url_map_advprop if advprop else url_map
        state_dict = model_zoo.load_url(url_map_[model_name])
        if load_fc:
            model.load_state_dict(state_dict)
        else:
            missing_keys = [
                "_fc.weight",
                "_fc.bias",
                "_concatenated_fc.weight",
                "_concatenated_fc.bias",
            ]

            for key in missing_keys:
                if "_concatenated_fc" in key:
                    continue
                state_dict.pop(key)

            res = model.load_state_dict(state_dict, strict=False)
            # assert set(res.missing_keys) == set(
            #     missing_keys
            # ), "issue loading pretrained weights"
        print("Loaded pretrained weights for {}".format(model_name))


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


def build_quality_factor_efficientnet_b2(n_classes: int) -> nn.Module:
    return StegoQFactorEfficientNet.from_pretrained(
        model_name="efficientnet-b2", num_classes=n_classes,
    )


def build_quality_factor_efficientnet_b3(n_classes: int) -> nn.Module:
    return StegoQFactorEfficientNet.from_pretrained(
        model_name="efficientnet-b3", num_classes=n_classes,
    )


def build_quality_factor_efficientnet_b4(n_classes: int) -> nn.Module:
    return StegoQFactorEfficientNet.from_pretrained(
        model_name="efficientnet-b4", num_classes=n_classes,
    )


def build_quality_factor_efficientnet_b5(n_classes: int) -> nn.Module:
    return StegoQFactorEfficientNet.from_pretrained(
        model_name="efficientnet-b5", num_classes=n_classes,
    )


def build_quality_factor_efficientnet_b6(n_classes: int) -> nn.Module:
    return StegoQFactorEfficientNet.from_pretrained(
        model_name="efficientnet-b6", num_classes=n_classes,
    )


if __name__ == "__main__":
    model = build_quality_factor_efficientnet_b3(n_classes=4)
    rgb = torch.zeros(1, 3, 512, 512, dtype=torch.float, requires_grad=False)
    q_factor = torch.zeros(1, 3, dtype=torch.float, requires_grad=False)
    out = model(rgb, q_factor)
    print(torch.softmax(out, dim=1))
