import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch.utils import (
    get_same_padding_conv2d,
    round_filters,
    round_repeats,
    MemoryEfficientSwish,
    efficientnet_params,
    get_model_params,
    Swish,
    url_map_advprop,
    url_map,
)
from torch.utils import model_zoo

from torchviz import make_dot

from alaska2.alaska_pytorch.models.colour.efficientnet import StegoEfficientNet


class DCTEfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(
            32, self._global_params
        )  # number of output channels
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(
                    block_args.num_repeat, self._global_params
                ),
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(block_args, self._global_params)
                )

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = inputs

        # NOTE: the stem of the regular EfficientNet model has been removed

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
    def forward(self, dct_y, dct_cb, dct_cr):
        x = torch.cat((dct_y, dct_cb, dct_cr), dim=1)
        bs = x.size(0)

        # Convolution layers
        x = self.extract_features(x)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(
            model_name, override_params
        )
        blocks_args[0] = blocks_args[0]._replace(input_filters=192)
        # print(blocks_args[0])
        # exit()
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000):
        model = cls.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        load_pretrained_weights(
            model, model_name, load_fc=(num_classes == 1000), advprop=advprop
        )
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ["efficientnet-b" + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError(
                "model_name should be one of: " + ", ".join(valid_models)
            )


def load_pretrained_weights(model, model_name, load_fc=True, advprop=False):
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
            "_blocks.0._depthwise_conv.weight",
            "_blocks.0._bn1.weight",
            "_blocks.0._bn1.bias",
            "_blocks.0._bn1.running_mean",
            "_blocks.0._bn1.running_var",
            "_blocks.0._se_reduce.weight",
            "_blocks.0._se_reduce.bias",
            "_blocks.0._se_expand.weight",
            "_blocks.0._se_expand.bias",
            "_blocks.0._project_conv.weight",
        ]

        for key in missing_keys:
            state_dict.pop(key)

        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(
            missing_keys
        ), "issue loading pretrained weights"

    print("Loaded pretrained weights for {}".format(model_name))


class DCTMultipleInputEfficientNet(nn.Module):
    """
    Based on the model from:
    https://papers.nips.cc/paper/7649-faster-neural-networks-straight-from-jpeg.pdf

    Using the method described as Deconvolution-RFA to upsample the Cb and Cr
    inputs from (32, 32, 64) to (64, 64, 64) to match the Y channel.
    """

    def __init__(
        self, model_name: str = "efficientnet-b0", num_classes: int = 4,
    ) -> None:
        super().__init__()
        # Make separate models so we don't share weights
        self.dct_y_efficientnet = StegoEfficientNet.from_pretrained(
            model_name=model_name, num_classes=num_classes, in_channels=64
        )
        self.dct_cb_efficientnet = StegoEfficientNet.from_pretrained(
            model_name=model_name, num_classes=num_classes, in_channels=64
        )
        self.dct_cr_efficientnet = StegoEfficientNet.from_pretrained(
            model_name=model_name, num_classes=num_classes, in_channels=64
        )
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(1280 * 3, num_classes)

    @autocast()
    def forward(
        self, dct_y: torch.tensor, dct_cb: torch.tensor, dct_cr: torch.tensor
    ) -> torch.tensor:
        bs_y = dct_y.size(0)
        bs_cb = dct_cb.size(0)
        bs_cr = dct_cr.size(0)

        x_y = self.dct_y_efficientnet.extract_features(dct_cb)
        x_cb = self.dct_cb_efficientnet.extract_features(dct_cb)
        x_cr = self.dct_cr_efficientnet.extract_features(dct_cr)

        x_y = self._avg_pooling(x_y)
        x_y = x_y.view(bs_y, -1)
        # x_y = self._dropout(x_y)

        x_cb = self._avg_pooling(x_cb)
        x_cb = x_cb.view(bs_cb, -1)
        # x_cb = self._dropout(x_cb)

        x_cr = self._avg_pooling(x_cr)
        x_cr = x_cr.view(bs_cr, -1)
        # x_cr = self._dropout(x_cr)

        x = torch.cat([x_y, x_cb, x_cr], dim=1)

        # Make sure the final linear layer is in FP32
        with autocast(enabled=False):
            x = x.float()
            x = self._fc(x)

        return x


def build_dct_efficientnet_b0(n_classes: int) -> nn.Module:
    return DCTEfficientNet.from_pretrained(
        model_name="efficientnet-b0", num_classes=n_classes
    )


def build_dct_efficientnet_b1(n_classes: int) -> nn.Module:
    return DCTEfficientNet.from_pretrained(
        model_name="efficientnet-b1", num_classes=n_classes
    )


def build_dct_efficientnet_b2(n_classes: int) -> nn.Module:
    return DCTEfficientNet.from_pretrained(
        model_name="efficientnet-b2", num_classes=n_classes
    )


def build_dct_efficientnet_b3(n_classes: int) -> nn.Module:
    return DCTEfficientNet.from_pretrained(
        model_name="efficientnet-b3", num_classes=n_classes
    )


def build_64_channel_dct_efficientnet_b0(n_classes: int) -> nn.Module:
    return DCTMultipleInputEfficientNet(
        model_name="efficientnet-b0", num_classes=n_classes
    )


if __name__ == "__main__":
    model = build_dct_efficientnet_b2(n_classes=4)
    y = torch.rand((1, 64, 64, 64), dtype=torch.float, requires_grad=False)
    cb = torch.rand((1, 64, 64, 64), dtype=torch.float, requires_grad=False)
    cr = torch.rand((1, 64, 64, 64), dtype=torch.float, requires_grad=False)
    out = model(y, cb, cr)
    print(torch.softmax(out, dim=1))
