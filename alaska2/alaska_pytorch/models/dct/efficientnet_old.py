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
        self.dct_y_efficientnet = StemlessEfficientNet.from_name(
            model_name=model_name
        )
        self.dct_cb_efficientnet = StemlessEfficientNet.from_name(
            model_name=model_name
        )
        self.dct_cr_efficientnet = StemlessEfficientNet.from_name(
            model_name=model_name
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


class DeconvDCTEfficientNet(nn.Module):
    """
    Based on the model from: https://papers.nips.cc/paper/7649-faster-neural-networks-straight-from-jpeg.pdf

    Using the method described as Deconvolution-RFA to upsample the Cb and Cr
    inputs from (32, 32, 64) to (64, 64, 64) to match the Y channel.
    """

    def __init__(
        self,
        dct_channels=64,
        model_name: str = "efficientnet-b0",
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        # TODO use deconv network rather than single layer.
        # Make separate layers so we don't share weights
        self.deconvolution_upsampling_cb = nn.ConvTranspose2d(
            dct_channels, dct_channels, kernel_size=2, stride=2
        )
        self.deconvolution_upsampling_cr = nn.ConvTranspose2d(
            dct_channels, dct_channels, kernel_size=2, stride=2
        )
        self.efficientnet = StegoEfficientNet.from_pretrained(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=dct_channels * 3,
        )

    @autocast()
    def forward(
        self, dct_y: torch.tensor, dct_cb: torch.tensor, dct_cr: torch.tensor
    ) -> torch.tensor:
        upsampled_cb = self.deconvolution_upsampling_cb(dct_cb)
        upsampled_cr = self.deconvolution_upsampling_cr(dct_cr)

        # Concatenate the inputs 3x (64, 64, 64) -> (192, 64, 64)
        concatenated_inputs = torch.cat(
            (dct_y, upsampled_cb, upsampled_cr), dim=1
        )

        output = self.efficientnet(concatenated_inputs)

        return output
