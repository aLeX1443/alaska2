# Colour Efficientnet
from alaska2.alaska_pytorch.models.colour.efficientnet import (
    build_efficientnet_b0,
    build_efficientnet_b1,
    build_efficientnet_b2,
    build_efficientnet_b3,
    build_efficientnet_b4,
    build_efficientnet_b5,
    build_efficientnet_b6,
    build_efficientnet_b7,
)

# Colour Efficientnet with quality factor
from alaska2.alaska_pytorch.models.colour.efficientnet import (
    build_quality_factor_efficientnet_b3,
    build_quality_factor_efficientnet_b2,
    build_quality_factor_efficientnet_b5,
)


# DCTEfficientNet
from alaska2.alaska_pytorch.models.dct.efficientnet import (
    build_dct_efficientnet_b0,
    build_dct_efficientnet_b1,
    build_dct_efficientnet_b2,
    build_dct_efficientnet_b3,
    build_dct_efficientnet_b7,
    build_dct_efficientnet_b5,
    build_dct_efficientnet_b7_no_weight_sharing,
)

# DCTResNet
from alaska2.alaska_pytorch.models.dct.resnet import build_dct_resnet_50

# YedroudjNet
from alaska2.alaska_pytorch.models.colour.yedroudj_net import (
    build_yedroudj_net,
)
