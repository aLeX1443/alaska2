import os
import time
from typing import Dict, Sequence, Optional, Tuple, Union
import cv2
import glob
import numpy as np
import jpegio

import torch
from catalyst.data import BalanceClassSampler
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Compose

from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Normalize,
)
from tqdm import tqdm

from alaska2.lib.data_loaders import (
    load_data,
    add_fold_to_data_set,
    dct_from_jpeg,
    dct_from_jpeg_imageio,
    jpeg_decompress_ycbcr,
)


def make_train_and_validation_data_loaders(
    hyper_parameters: Dict, validation_fold_number: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    input_data_type = hyper_parameters["input_data_type"]
    if input_data_type == "RGB":
        data_set_class = ColourDataSet
        # Define a set of image augmentations.
        augmentations_train = Compose(
            [
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                Normalize(p=1),
                ToTensorV2(),
            ],
            p=1,
        )
        augmentations_validation = Compose([Normalize(p=1), ToTensorV2()], p=1)
    elif input_data_type == "YCbCr":
        data_set_class = ColourDataSet
        # Define a set of image augmentations.
        augmentations_train = Compose(
            [VerticalFlip(p=0.5), HorizontalFlip(p=0.5), ToTensorV2()], p=1,
        )
        augmentations_validation = Compose([ToTensorV2()], p=1)
        # augmentations_train = None
        # augmentations_validation = None
    elif input_data_type == "DCT":
        data_set_class = DCTDataSet
        # Define a set of image augmentations.
        augmentations_train = Compose([VerticalFlip(p=0), HorizontalFlip(p=1)], p=1,)
        augmentations_validation = Compose([], p=1)
        # augmentations_train = None
        # augmentations_validation = None
    else:
        raise ValueError(
            f"Invalid input data type provided: {input_data_type}"
        )

    # Load a DataFrame with the files and targets.
    data_set = load_data()

    # Split the data set into folds.
    data_set = add_fold_to_data_set(data_set)

    # Create train and validation data sets.
    train_data_set = data_set_class(
        kinds=data_set[data_set["fold"] != validation_fold_number].kind.values,
        image_names=data_set[
            data_set["fold"] != validation_fold_number
        ].image_name.values,
        labels=data_set[
            data_set["fold"] != validation_fold_number
        ].label.values,
        transforms=augmentations_train,
        colour_space=input_data_type,
        use_quality_factor=hyper_parameters["use_quality_factor"],
        separate_classes_by_quality_factor=hyper_parameters[
            "separate_classes_by_quality_factor"
        ],
    )
    validation_data_set = data_set_class(
        kinds=data_set[data_set["fold"] == validation_fold_number].kind.values,
        image_names=data_set[
            data_set["fold"] == validation_fold_number
        ].image_name.values,
        labels=data_set[
            data_set["fold"] == validation_fold_number
        ].label.values,
        transforms=augmentations_validation,
        colour_space=input_data_type,
        use_quality_factor=hyper_parameters["use_quality_factor"],
        separate_classes_by_quality_factor=hyper_parameters[
            "separate_classes_by_quality_factor"
        ],
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


class ColourDataSet(Dataset):
    def __init__(
        self,
        kinds: Sequence[str],
        image_names: Sequence[str],
        labels: Sequence[int],
        transforms: Optional[Compose] = None,
        colour_space: str = "RGB",
        use_quality_factor: bool = False,
        separate_classes_by_quality_factor: bool = False,
    ) -> None:
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.colour_space = colour_space
        self.use_quality_factor = use_quality_factor
        self.separate_classes_by_quality_factor = (
            separate_classes_by_quality_factor
        )

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, torch.tensor],
        Tuple[np.ndarray, torch.tensor],
    ]:
        kind, image_name, label = (
            self.kinds[index],
            self.image_names[index],
            self.labels[index],
        )

        img_path = f"data/{kind}/{image_name}"

        if self.colour_space == "RGB":
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.colour_space == "YCbCr":
            image = jpeg_decompress_ycbcr(img_path)
            # TODO add 128 and normalise with imagenet mean and std
            image = image / 128
            # image = np.rollaxis(image, 2, 0)
        else:
            raise ValueError(f"Invalid colour_space: {self.colour_space}")
        image = image.astype(np.float32)

        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

        target = one_hot(4, label)

        if self.use_quality_factor:
            if self.separate_classes_by_quality_factor:
                raise NotImplementedError()
            quality_factor = self.encode_quality_factor(
                self.get_quality_factor_from_image(img_path)
            )
            return image, quality_factor, target
        return image, target

    @staticmethod
    def get_quality_factor_from_image(img_path: str) -> int:
        jpeg_struct = jpegio.read(img_path)
        top_left_quant_table_value = jpeg_struct.quant_tables[0][0, 0]
        mappings = {2: 95, 3: 90, 8: 75}
        return mappings[top_left_quant_table_value]

    @staticmethod
    def encode_quality_factor(quality_factor):
        mappings = {
            95: np.array([1, 0, 0]),
            90: np.array([0, 1, 0]),
            75: np.array([0, 0, 1]),
        }
        return mappings[quality_factor]

    def get_labels(self):
        return list(self.labels)


class DCTDataSet(Dataset):
    def __init__(
        self,
        kinds: Sequence[str],
        image_names: Sequence[str],
        labels: Sequence[int],
        transforms: Optional[Compose] = None,
        colour_space: str = "DCT",
        use_quality_factor: bool = False,
        separate_classes_by_quality_factor: bool = False,
    ) -> None:
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.tensor]:
        kind, image_name, label = (
            self.kinds[index],
            self.image_names[index],
            self.labels[index],
        )

        # TODO perform transform on the raw image, save in tmp directory and
        #  load DCT coefficients.

        # dct_y, dct_cb, dct_cr = dct_from_jpeg_imageio(
        #     f"data/{kind}/{image_name}"
        # )
        arrays = np.load(f"data/dct/{kind}/{image_name.replace('.jpg', '.npz')}")
        dct_y, dct_cb, dct_cr = (
            arrays["arr_0"],
            arrays["arr_1"],
            arrays["arr_2"],
        )

        print(dct_y[0][0])

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

        print(dct_y[0][0])
        import time
        time.sleep(1000)

        dct_y = np.rollaxis(dct_y, 2, 0)
        dct_cb = np.rollaxis(dct_cb, 2, 0)
        dct_cr = np.rollaxis(dct_cr, 2, 0)

        dct_y = dct_y / 1024
        dct_cb = dct_cb / 1024
        dct_cr = dct_cr / 1024

        target = one_hot(4, label)

        return dct_y, dct_cb, dct_cr, target

    def get_labels(self):
        return list(self.labels)


def one_hot(size: int, target: int) -> torch.tensor:
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o755)


def pre_process_and_save_dct_data():
    for kind in tqdm(["Cover", "JMiPOD", "JUNIWARD", "UERD"], desc=""):
        for file in tqdm(glob.glob1(f"data/Cover/", "*.jpg"), desc=""):
            save_dir = f"data/dct/{kind}/"
            load_dir = f"data/{kind}/"
            save_file = f"{save_dir}/{file}".replace(".jpg", ".npz")
            load_file = f"{load_dir}/{file}"
            make_dir_if_not_exists(save_dir)

            # Load the DCT arrays
            dct_y, dct_cb, dct_cr = dct_from_jpeg_imageio(load_file)

            # Save the arrays without any normalisation
            np.savez_compressed(save_file, dct_y, dct_cb, dct_cr)


if __name__ == "__main__":
    pre_process_and_save_dct_data()
