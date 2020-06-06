from typing import Sequence
import os
import glob
import random
import numpy as np
import pandas as pd

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
