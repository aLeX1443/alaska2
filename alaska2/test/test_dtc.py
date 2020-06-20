import numpy as np
import sys

from alaska2.lib.data_loaders import (
    dct_from_jpeg,
    dct_from_jpeg_imageio,
)

# np.set_printoptions(threshold=sys.maxsize)


def test_dct_methods() -> None:
    test_path = "data/UERD/00001.jpg"
    uber_dct_y, uber_dct_cb, uber_dct_cr = dct_from_jpeg(test_path)
    imageio_dct_y, imageio_dct_cb, imageio_dct_cr = dct_from_jpeg_imageio(
        test_path
    )

    assert uber_dct_y.shape == (64, 64, 64)
    assert uber_dct_cb.shape == (32, 32, 64)
    assert uber_dct_cr.shape == (32, 32, 64)

    assert imageio_dct_y.shape == (64, 64, 64)
    assert imageio_dct_cb.shape == (64, 64, 64)
    assert imageio_dct_cr.shape == (64, 64, 64)


def test_cover_vs_modified_dct() -> None:
    dct_y_0, dct_cb_0, dct_cr_0 = dct_from_jpeg_imageio("data/Cover/00001.jpg")
    dct_y_1, dct_cb_1, dct_cr_1 = dct_from_jpeg_imageio("data/UERD/00001.jpg")

    print(dct_y_0[0])
    print(dct_y_1[0])


if __name__ == "__main__":
    test_cover_vs_modified_dct()
