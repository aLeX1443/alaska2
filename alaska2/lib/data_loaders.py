from typing import Sequence, Tuple
import os
import glob
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    from jpeg2dct.numpy import load as load_dct
    import jpegio
except ImportError:
    print("Could not load DCT libraries")

from sklearn.model_selection import GroupKFold
from sklearn.utils import class_weight


def load_data() -> pd.DataFrame:
    data_set = []
    for label, kind in enumerate(["Cover", "JMiPOD", "JUNIWARD", "UERD"]):
        for path in glob.glob1("data/Cover/", "*.jpg"):
            data_set.append(
                {
                    "kind": kind,
                    "image_name": path.split(os.sep)[-1],
                    "label": label,
                }
            )
    random.shuffle(data_set)
    return pd.DataFrame(data_set)


def add_fold_to_data_set(
    data_set: pd.DataFrame, n_splits: int = 5
) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)

    data_set.loc[:, "fold"] = 0
    for fold_number, (train_index, val_index) in enumerate(
        gkf.split(
            X=data_set.index,
            y=data_set["label"],
            groups=data_set["image_name"],
        )
    ):
        data_set.loc[data_set.iloc[val_index].index, "fold"] = fold_number

    return data_set


def compute_class_weights(data_set: pd.DataFrame) -> Sequence[float]:
    return class_weight.compute_class_weight(
        "balanced", np.unique(data_set["label"]), data_set["label"],
    )


def jpeg_decompress_ycbcr_pil(path: str) -> Image:
    return Image.open(path).convert("YCbCr")


def jpeg_decompress_ycbcr_cv2(path: str) -> Image:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


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


def dct_from_jpeg(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_dct(path)


def dct_from_jpeg_imageio(
    path: str, reshape=True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    jpeg_struct = jpegio.read(str(path))
    dct_coefficients = jpeg_struct.coef_arrays

    # print(jpeg_struct.quant_tables[0])
    # print(jpeg_struct.quant_tables[1])
    # print(jpeg_struct.quant_tables[2])

    # np.savetxt("dct_y.txt", dct_coefficients[0], fmt="%4d", delimiter=",", newline="\n")
    # np.savetxt("dct_cb.txt", dct_coefficients[1], fmt="%4d", delimiter=",", newline="\n")
    # np.savetxt("dct_cr.txt", dct_coefficients[2], fmt="%4d", delimiter=",", newline="\n")

    dct_y = np.array(dct_coefficients[0], dtype=np.int8)
    dct_cb = np.array(dct_coefficients[1], dtype=np.int8)
    dct_cr = np.array(dct_coefficients[2], dtype=np.int8)

    if reshape:
        dct_y = dct_y.reshape((64, 64, 64))
        dct_cb = dct_cb.reshape((64, 64, 64))
        dct_cr = dct_cr.reshape((64, 64, 64))
    else:
        dct_y = np.expand_dims(dct_y, axis=-1)
        dct_cb = np.expand_dims(dct_cb, axis=-1)
        dct_cr = np.expand_dims(dct_cr, axis=-1)

    return dct_y, dct_cb, dct_cr
