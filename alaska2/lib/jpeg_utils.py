import numpy as np
from PIL import Image
import jpegio


def jpeg_decompress_ycbcr_pil(path: str) -> Image:
    return Image.open(path).convert("YCbCr")


def jpeg_decompress_ycbcr(path: str) -> np.ndarray:
    jpeg_struct = jpegio.read(str(path))

    [col, row] = np.meshgrid(range(8), range(8))
    transformation = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))
    transformation[0, :] = transformation[0, :] / np.sqrt(2)

    img_dims = np.array(jpeg_struct.coef_arrays[0].shape)
    n_blocks = img_dims // 8
    broadcast_dims = (n_blocks[0], 8, n_blocks[1], 8)

    y_cb_cr = []
    for i, dct_coefficients, in enumerate(jpeg_struct.coef_arrays):

        if i == 0:
            qm = jpeg_struct.quant_tables[i]
        else:
            qm = jpeg_struct.quant_tables[1]

        t = np.broadcast_to(transformation.reshape(1, 8, 1, 8), broadcast_dims)
        qm = np.broadcast_to(qm.reshape(1, 8, 1, 8), broadcast_dims)
        dct_coefficients = dct_coefficients.reshape(broadcast_dims)

        a = np.transpose(t, axes=(0, 2, 3, 1))
        b = (qm * dct_coefficients).transpose(0, 2, 1, 3)
        c = t.transpose(0, 2, 1, 3)

        z = a @ b @ c
        z = z.transpose(0, 2, 1, 3)
        y_cb_cr.append(z.reshape(img_dims))

    return np.stack(y_cb_cr, -1).astype(np.float32)
