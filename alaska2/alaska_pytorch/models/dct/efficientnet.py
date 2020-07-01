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

        # Build blocks
        self._blocks = nn.ModuleList([])
        for i, block_args in enumerate(self._blocks_args):
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
            print(i, block_args)

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
        # self._fc = nn.Linear(int(out_channels * 3), self._global_params.num_classes)
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

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(
            model_name, override_params
        )
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000):
        model = cls.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        cls.load_pretrained_weights(
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
            ]

            for key in missing_keys:
                state_dict.pop(key)

            res = model.load_state_dict(state_dict, strict=False)
            assert set(res.missing_keys) == set(
                missing_keys
            ), "issue loading pretrained weights"

        print("Loaded pretrained weights for {}".format(model_name))


class CustomEfficientNetB3(nn.Module):
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

        # Build blocks
        self._blocks = nn.ModuleList([])
        for i, block_args in enumerate(self._blocks_args):
            # Update block input and output filters based on depth multiplier.
            # NOTE:
            if i != 2:
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
            else:
                block_args = block_args._replace(
                    input_filters=64,
                    output_filters=round_filters(
                        block_args.output_filters, self._global_params
                    ),
                    num_repeat=round_repeats(
                        block_args.num_repeat, self._global_params
                    ),
                )
            print(i, block_args)

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
        # self._fc = nn.Linear(int(out_channels * 3), self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(
            model_name, override_params
        )
        # del blocks_args[0:3]
        # In order to use as many pre-trained weights as possible, we will not
        # be deleting any of the existing block args, instead we will be
        # creating every block and loading pre-trained weights (except for the
        # case of the block where we will change the input channels from 40 to
        # 60), then only using the ones we are interested in, i.e., blocks 4
        # through 8.
        # blocks_args[3] = blocks_args[3]._replace(input_filters=64)
        # for _ in blocks_args:
        #     print(_)
        # exit()
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000):
        model = cls.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        cls.load_pretrained_weights(
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
            ]

            for key in missing_keys:
                state_dict.pop(key)

            for key in dict(state_dict).keys():
                if "_blocks.11" in key:
                    state_dict.pop(key)

            res = model.load_state_dict(state_dict, strict=False)
            # assert set(res.missing_keys) == set(
            #     missing_keys
            # ), "issue loading pretrained weights"

        print("Loaded pretrained weights for {}".format(model_name))

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = inputs
        # print(x.shape)
        # NOTE: the stem of the regular EfficientNet model has been removed

        skip_blocks = 11

        # Blocks
        for idx, block in enumerate(self._blocks):
            if idx < skip_blocks:
                continue
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(x.shape)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        # print(x.shape)

        return x


class DCTMultipleInputEfficientNet(nn.Module):
    def __init__(
        self, model_name: str = "efficientnet-b7", num_classes: int = 4,
    ) -> None:
        super().__init__()
        # Make separate models so we don't share weights
        # self.dct_y_efficientnet = DCTEfficientNet.from_pretrained(
        #     model_name=model_name, num_classes=4
        # )
        # self.dct_cb_efficientnet = DCTEfficientNet.from_pretrained(
        #     model_name=model_name, num_classes=4
        # )
        # self.dct_cr_efficientnet = DCTEfficientNet.from_pretrained(
        #     model_name=model_name, num_classes=4
        # )

        self.dct_y_efficientnet = DCTEfficientNet.from_name(
            model_name=model_name,
        )
        self.dct_cb_efficientnet = DCTEfficientNet.from_name(
            model_name=model_name,
        )
        self.dct_cr_efficientnet = DCTEfficientNet.from_name(
            model_name=model_name,
        )

        # self.dct_y_efficientnet = CustomEfficientNetB3.from_pretrained(
        #     model_name=model_name, num_classes=4
        # )
        # self.dct_cb_efficientnet = CustomEfficientNetB3.from_pretrained(
        #     model_name=model_name, num_classes=4
        # )
        # self.dct_cr_efficientnet = CustomEfficientNetB3.from_pretrained(
        #     model_name=model_name, num_classes=4
        # )
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.2)
        out_channels = round_filters(
            1280, self.dct_y_efficientnet._global_params
        )
        self._fc = nn.Linear(int(out_channels * 3), num_classes)

    @autocast()
    def forward(
        self, dct_y: torch.tensor, dct_cb: torch.tensor, dct_cr: torch.tensor
    ) -> torch.tensor:
        bs = dct_y.size(0)

        # Convolution layers
        x_dct_y = self.dct_y_efficientnet.extract_features(dct_y)
        x_dct_cb = self.dct_cb_efficientnet.extract_features(dct_cb)
        x_dct_cr = self.dct_cr_efficientnet.extract_features(dct_cr)

        # Pooling
        x_dct_y = self._avg_pooling(x_dct_y)
        x_dct_cb = self._avg_pooling(x_dct_cb)
        x_dct_cr = self._avg_pooling(x_dct_cr)

        # Flatten
        x_dct_y = x_dct_y.view(bs, -1)
        x_dct_cb = x_dct_cb.view(bs, -1)
        x_dct_cr = x_dct_cr.view(bs, -1)

        x = torch.cat((x_dct_y, x_dct_cb, x_dct_cr), dim=1)

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
    # TODO uncomment above and delete below
    # return DCTEfficientNet.from_name(
    #     model_name="efficientnet-b3",
    #     override_params={"num_classes": n_classes},
    # )


def build_dct_efficientnet_b5(n_classes: int) -> nn.Module:
    return DCTEfficientNet.from_pretrained(
        model_name="efficientnet-b5", num_classes=n_classes
    )


def build_dct_efficientnet_b7(n_classes: int) -> nn.Module:
    return DCTEfficientNet.from_pretrained(
        model_name="efficientnet-b7", num_classes=n_classes
    )


def build_dct_efficientnet_b7_no_weight_sharing(n_classes: int) -> nn.Module:
    return DCTMultipleInputEfficientNet(
        model_name="efficientnet-b7", num_classes=n_classes
    )


if __name__ == "__main__":
    # Compare the input and output shapes from the layers of each network
    model = build_dct_efficientnet_b7_no_weight_sharing(n_classes=4)
    # print(model)
    # exit()
    y = torch.rand((1, 64, 64, 64), dtype=torch.float, requires_grad=False)
    cb = torch.rand((1, 64, 64, 64), dtype=torch.float, requires_grad=False)
    cr = torch.rand((1, 64, 64, 64), dtype=torch.float, requires_grad=False)
    out = model(y, cb, cr)
    print(torch.softmax(out, dim=1))
