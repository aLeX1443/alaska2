import numpy as np

from alaska2.lib.data_loaders import (
    jpeg_decompress_ycbcr_pil,
    jpeg_decompress_ycbcr,
    jpeg_decompress_ycbcr_cv2,
)


def test_compression_method_consistency() -> None:
    test_path = "/alaska2/data/Cover/00001.jpg"
    ycbcr_pil = np.array(jpeg_decompress_ycbcr_pil(test_path))
    ycbcr_custom = np.array(jpeg_decompress_ycbcr(test_path))
    ycbcr_cv2 = np.array(jpeg_decompress_ycbcr_cv2(test_path))

    print(ycbcr_pil[0])
    print(ycbcr_custom[0])
    print(ycbcr_cv2[0])

    print(ycbcr_pil.shape)
    print(ycbcr_custom.shape)
    print(ycbcr_cv2.shape)

    print(np.sum(ycbcr_pil - ycbcr_cv2))
    # print(np.sum(ycbcr_custom - ycbcr_cv2))

    # assert np.array_equal(ycbcr_pil, ycbcr_cv2)
    # assert np.array_equal(ycbcr_custom, ycbcr_cv2)


if __name__ == "__main__":
    test_compression_method_consistency()
