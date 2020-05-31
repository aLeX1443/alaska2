import numpy as np

from lib.jpeg_utils import jpeg_decompress_ycbcr_pil, jpeg_decompress_ycbcr


def test_compression_method_consistency():
    test_path = "data/Cover/00001.jpg"
    ycbcr_pil = jpeg_decompress_ycbcr_pil(test_path)
    ycbcr_custom = jpeg_decompress_ycbcr(test_path)

    print(np.array(ycbcr_pil))
    print(np.array(ycbcr_custom))
    print(np.sum(ycbcr_pil - ycbcr_custom))

    assert np.array_equal(np.array(ycbcr_pil), np.array(ycbcr_custom))


if __name__ == "__main__":
    test_compression_method_consistency()
