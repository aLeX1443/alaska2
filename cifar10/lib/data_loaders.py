from typing import Dict, Tuple, Sequence, Optional
import random
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

import torch
from catalyst.data import BalanceClassSampler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Normalize,
)

from alaska2.alaska_pytorch.lib.data_loaders import one_hot
from alaska2.lib.data_loaders import add_fold_to_data_set
from alaska2.alaska_pytorch.lib.utils import make_dir_if_not_exists
from alaska2.lib.data_loaders import dct_from_jpeg_imageio

CIFAR10_CATEGORIES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_jpg_and_save_without_chroma_subsampling(load_file, save_file):
    img = Image.open(load_file)
    # Resize the image from (32, 32, 3) to (221, 221, 3)
    img = img.resize((512, 512), Image.ANTIALIAS)
    img.save(save_file, subsampling=0)


def prepare_cifar10_data_set():
    cifar_load_dir = "/alaska2/data/cifar10-original/"
    cifar_save_dir = "/alaska2/data/cifar10-no-subsampling/"

    for fold in ["train", "test"]:
        print(f"Starting fold: {fold}")
        for category in tqdm(CIFAR10_CATEGORIES):
            load_dir = f"{cifar_load_dir}{fold}/{category}/"
            save_dir = f"{cifar_save_dir}{fold}/{category}/"
            make_dir_if_not_exists(save_dir)
            for file in tqdm(glob.glob1(load_dir, "*.jpg")):
                load_file = f"{load_dir}{file}"
                save_file = f"{save_dir}{file}"
                load_jpg_and_save_without_chroma_subsampling(
                    load_file, save_file
                )


def load_cifar_data():
    cifar_load_dir = "/alaska2/data/cifar10-no-subsampling/"
    training_data = []
    validation_data = []
    for fold in ["train", "test"]:
        for label, kind in enumerate(CIFAR10_CATEGORIES):
            load_dir = f"{cifar_load_dir}{fold}/{kind}/"
            for file in glob.glob1(load_dir, "*.jpg"):
                load_file = f"{load_dir}{file}"
                if fold == "train":
                    training_data.append(
                        {"kind": kind, "file": load_file, "label": label}
                    )
                else:
                    validation_data.append(
                        {"kind": kind, "file": load_file, "label": label}
                    )
    random.shuffle(training_data)
    random.shuffle(validation_data)
    return pd.DataFrame(training_data), pd.DataFrame(validation_data)


def make_train_and_validation_data_loaders(
    hyper_parameters: Dict, validation_fold_number: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    # Define a set of image augmentations.
    # augmentations_train = Compose([VerticalFlip(p=0), HorizontalFlip(p=1)], p=1,)
    # augmentations_validation = Compose([], p=1)
    augmentations_train = None
    augmentations_validation = None

    # Load a DataFrame with the files and targets.
    train_data, val_data = load_cifar_data()

    # Create train and validation data sets.
    train_data_set = DCTDataSet(
        kinds=train_data.kind.values,
        files=train_data.file.values,
        labels=train_data.label.values,
        transforms=augmentations_train,
    )
    validation_data_set = DCTDataSet(
        kinds=val_data.kind.values,
        files=val_data.file.values,
        labels=val_data.label.values,
    )

    # Create train and validation data loaders.
    train_data_loader = DataLoader(
        train_data_set,
        sampler=BalanceClassSampler(
            labels=train_data_set.get_labels(), mode="downsampling"
        ),
        batch_size=hyper_parameters["batch_size"],
        shuffle=False,
        num_workers=hyper_parameters["training_workers"],
        pin_memory=False,
        drop_last=True,
    )
    validation_data_loader = DataLoader(
        validation_data_set,
        batch_size=hyper_parameters["batch_size"],
        shuffle=False,
        num_workers=hyper_parameters["training_workers"],
        pin_memory=False,
        drop_last=True,
    )

    return train_data_loader, validation_data_loader


class DCTDataSet(Dataset):
    def __init__(
        self,
        kinds: Sequence[str],
        files: Sequence[str],
        labels: Sequence[int],
        transforms: Optional[Compose] = None,
    ) -> None:
        super().__init__()
        self.kinds = kinds
        self.files = files
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.tensor]:
        kind, file, label = (
            self.kinds[index],
            self.files[index],
            self.labels[index],
        )

        # TODO perform transform on the raw image, save in tmp directory and
        #  load DCT coefficients.

        dct_y, dct_cb, dct_cr = dct_from_jpeg_imageio(file)

        dct_y = dct_y.astype(np.float32)
        dct_cb = dct_cb.astype(np.float32)
        dct_cr = dct_cr.astype(np.float32)

        if self.transforms:
            sample = {"image": dct_y}
            sample = self.transforms(**sample)
            dct_y = sample["image"]
            sample = {"image": dct_cb}
            sample = self.transforms(**sample)
            dct_cb = sample["image"]
            sample = {"image": dct_cr}
            sample = self.transforms(**sample)
            dct_cr = sample["image"]

        dct_y = np.rollaxis(dct_y, 2, 0)
        dct_cb = np.rollaxis(dct_cb, 2, 0)
        dct_cr = np.rollaxis(dct_cr, 2, 0)

        dct_y = dct_y / 1024
        dct_cb = dct_cb / 1024
        dct_cr = dct_cr / 1024

        target = one_hot(10, label)

        return dct_y, dct_cb, dct_cr, target

    def get_labels(self):
        return list(self.labels)


if __name__ == "__main__":
    prepare_cifar10_data_set()
