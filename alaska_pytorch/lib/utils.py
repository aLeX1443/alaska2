import os

from albumentations import ImageOnlyTransform


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o755)


class ToCustomFloat(ImageOnlyTransform):
    """
    Divide pixel values by `max_value` to get a float32 or float64 output array
    where all values lie in the range [0, 1.0]. If `max_value` is None the
    transform will try to infer the maximum value by inspecting the data type
    of the input image.

    See Also:
        :class:`~albumentations.augmentations.transforms.FromFloat`

    Args:
        max_value (float): maximum possible input value. Default: None.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        any type
    """

    def __init__(
        self, max_value=None, dtype="float32", always_apply=False, p=1.0
    ):
        super(ToCustomFloat, self).__init__(always_apply, p)
        self.max_value = max_value
        self.dtype = dtype

    def apply(self, img, **params):
        return to_float(img, self.max_value, self.dtype)

    def get_transform_init_args_names(self):
        return ("max_value",)

    def get_params_dependent_on_targets(self, params):
        pass


def to_float(img, max_value, dtype="float32"):
    return img.astype(dtype) / max_value
