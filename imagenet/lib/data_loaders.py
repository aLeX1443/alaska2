import os
import tempfile
import cv2
from tqdm import tqdm
from typing import Dict, Tuple, Sequence, Optional
import random
from PIL import Image
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import GroupKFold

import torch
from catalyst.data import BalanceClassSampler
from torch.utils.data import DataLoader, Dataset

from alaska2.alaska_pytorch.lib.utils import make_dir_if_not_exists
from alaska2.lib.data_loaders import dct_from_jpeg_imageio

IMAGE_SIZE = 512
CROP_PADDING = 8
IMAGENET_CATEGORIES = os.listdir("/alaska2/data/imagenet/train")
assert len(IMAGENET_CATEGORIES) == 1000, "ImageNet data set not fully present."


def load_imagenet_data_paths():
    # NOTE we will be splitting the train set into train and validation
    imagenet_load_dir = "/alaska2/data/imagenet/train/"
    data_set = []
    for label, kind in tqdm(
        enumerate(IMAGENET_CATEGORIES), desc="Loading training files"
    ):
        load_dir = f"{imagenet_load_dir}{kind}/"
        for file in glob.glob1(load_dir, "*.JPEG"):
            load_file = f"{load_dir}{file}"
            data_set.append({"kind": kind, "file": load_file, "label": label})

    random.shuffle(data_set)
    return pd.DataFrame(data_set)


def add_fold_to_data_set(
    data_set: pd.DataFrame, n_splits: int = 24
) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)

    data_set.loc[:, "fold"] = 0
    for fold_number, (train_index, val_index) in tqdm(
        enumerate(
            gkf.split(
                X=data_set.index, y=data_set["label"], groups=data_set["file"],
            )
        ),
        desc="Splitting the data set into folds",
    ):
        data_set.loc[data_set.iloc[val_index].index, "fold"] = fold_number

    return data_set


def make_train_and_validation_data_loaders(
    hyper_parameters: Dict, validation_fold_number: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    # Load a DataFrame with the files and targets.
    data_set = load_imagenet_data_paths()

    # Split the data set into folds.
    data_set = add_fold_to_data_set(data_set, n_splits=24)

    # Create train and validation data sets.
    train_data_set = DCTDataSet(
        kinds=data_set[data_set["fold"] != validation_fold_number].kind.values,
        files=data_set[data_set["fold"] != validation_fold_number].file.values,
        labels=data_set[
            data_set["fold"] != validation_fold_number
        ].label.values,
        is_training=True,
    )
    validation_data_set = DCTDataSet(
        kinds=data_set[data_set["fold"] == validation_fold_number].kind.values,
        files=data_set[data_set["fold"] == validation_fold_number].file.values,
        labels=data_set[
            data_set["fold"] == validation_fold_number
        ].label.values,
        is_training=False,
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
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    validation_data_loader = DataLoader(
        validation_data_set,
        batch_size=hyper_parameters["batch_size"],
        shuffle=False,
        num_workers=hyper_parameters["training_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_data_loader, validation_data_loader


class DCTDataSet(Dataset):
    def __init__(
        self,
        kinds: Sequence[str],
        files: Sequence[str],
        labels: Sequence[int],
        is_training: bool = True,
    ) -> None:
        super().__init__()
        self.kinds = kinds
        self.files = files
        self.labels = labels
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, index: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]]:
        kind, file, label = (
            self.kinds[index],
            self.files[index],
            self.labels[index],
        )

        try:
            (
                dct_y,
                dct_cb,
                dct_cr,
            ) = load_dct_values_from_pre_processed_imagenet_image(
                file, self.is_training
            )
        except Exception as e:
            print(e)
            return None

        dct_y = dct_y.astype(np.float32)
        dct_cb = dct_cb.astype(np.float32)
        dct_cr = dct_cr.astype(np.float32)

        dct_y = np.rollaxis(dct_y, 2, 0)
        dct_cb = np.rollaxis(dct_cb, 2, 0)
        dct_cr = np.rollaxis(dct_cr, 2, 0)

        dct_y = dct_y / 1024
        dct_cb = dct_cb / 1024
        dct_cr = dct_cr / 1024

        target = one_hot(1000, label)

        return dct_y, dct_cb, dct_cr, target

    def get_labels(self):
        return list(self.labels)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_center_crop(image, image_size):
    image_height = image.shape[0]
    image_width = image.shape[1]

    padded_center_crop_size = int(
        image_size
        / (image_size + CROP_PADDING)
        * min(image_height, image_width)
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = np.stack(
        [
            offset_height,
            offset_width,
            padded_center_crop_size,
            padded_center_crop_size,
        ]
    )
    image = image[
        crop_window[0] : (crop_window[0] + crop_window[2]),
        crop_window[1] : (crop_window[1] + crop_window[3]),
    ]
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_CUBIC)
    return image


def get_random_crop(image, image_size):
    image_height = image.shape[0]
    image_width = image.shape[1]
    # If the image is too small, increase it's size before cropping
    if min(image_height, image_width) <= image_size:
        resize_factor = float(image_size / min(image_height, image_width))
        image = cv2.resize(
            image,
            (
                int(resize_factor * image_width) + CROP_PADDING,
                int(resize_factor * image_height) + CROP_PADDING,
            ),
            cv2.INTER_CUBIC,
        )

    max_x = image.shape[1] - image_size
    max_y = image.shape[0] - image_size

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y : y + image_size, x : x + image_size]

    return crop


def load_dct_values_from_pre_processed_imagenet_image(
    image_path, is_training=True
):
    # Read the image with CV2 and convert to BGR
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if is_training:
        try:
            img = get_random_crop(img, image_size=IMAGE_SIZE)
        except ValueError:
            print("Invalid random crop, reverting to center crop.")
            img = get_center_crop(img, image_size=IMAGE_SIZE)
    else:
        img = get_center_crop(img, image_size=IMAGE_SIZE)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = f"{tmp_dir}/img.jpg"
        # Convert to PIL and save without subsampling
        img = Image.fromarray(np.array(img))
        img.save(tmp_file, subsampling=0, format="JPEG")

        return dct_from_jpeg_imageio(tmp_file)


def extract_imagenet_files():
    """
    Assumes the archive ILSVRC2010_images_train.tar has been extracted into
    n02489166.tar, ... in the directory `data/imagenet/train`
    """
    load_dir = "/alaska2/data/imagenet/train"
    tar_files = glob.glob1(load_dir, "*.tar")

    for tar_file in tqdm(tar_files, desc="Extracting ImageNet folders"):
        extraction_dir = f"{load_dir}/{tar_file.strip('.tar')}"
        make_dir_if_not_exists(extraction_dir)
        os.system(f"7z x {load_dir}/{tar_file} -o{extraction_dir}")
        os.system(f"rm -rf {load_dir}/{tar_file}")


def one_hot(size: int, target: int) -> torch.Tensor:
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec


if __name__ == "__main__":
    # extract_imagenet_files()
    # load_imagenet_data_paths()

    test_image_file = (
        "/alaska2/data/imagenet/train/n01484850/n01484850_33833.JPEG"
    )

    load_dct_values_from_pre_processed_imagenet_image(
        test_image_file, is_training=False
    )
    exit()

    for _ in tqdm(range(1000)):
        load_dct_values_from_pre_processed_imagenet_image(test_image_file)
