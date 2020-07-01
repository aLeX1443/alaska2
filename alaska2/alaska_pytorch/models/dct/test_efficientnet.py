import torch

from efficientnet_pytorch import EfficientNet


class BaseEfficientNetSurgery(EfficientNet):
    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        print(inputs.shape)

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        print(x.shape)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            print(x.shape)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        print(x.shape)

        return x


if __name__ == "__main__":
    # Base EfficientNet
    base_efficientnet = BaseEfficientNetSurgery.from_name(
        model_name="efficientnet-b7", override_params={"image_size": 512}
    )
    base_input = torch.rand(
        (1, 3, 512, 512), dtype=torch.float, requires_grad=False
    )
    base_efficientnet(base_input)
