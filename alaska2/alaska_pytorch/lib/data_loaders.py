from typing import Dict, Sequence, Optional, Tuple
import cv2
import numpy as np

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

from alaska2.lib.data_loaders import load_data, add_fold_to_data_set


def make_train_and_validation_data_loaders(
    hyper_parameters: Dict, validation_fold_number: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    # Load a DataFrame with the files and targets.
    data_set = load_data()

    # Split the data set into folds.
    data_set = add_fold_to_data_set(data_set)

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
    augmentations_validation = Compose([Normalize(p=1), ToTensorV2()], p=1,)

    # Create train and validation data sets.
    train_data_set = Alaska2DataSet(
        kinds=data_set[data_set["fold"] != validation_fold_number].kind.values,
        image_names=data_set[
            data_set["fold"] != validation_fold_number
        ].image_name.values,
        labels=data_set[
            data_set["fold"] != validation_fold_number
        ].label.values,
        transforms=augmentations_train,
    )
    validation_data_set = Alaska2DataSet(
        kinds=data_set[data_set["fold"] == validation_fold_number].kind.values,
        image_names=data_set[
            data_set["fold"] == validation_fold_number
        ].image_name.values,
        labels=data_set[
            data_set["fold"] == validation_fold_number
        ].label.values,
        transforms=augmentations_validation,
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


class Alaska2DataSet(Dataset):
    def __init__(
        self,
        kinds: Sequence[str],
        image_names: Sequence[str],
        labels: Sequence[int],
        transforms: Optional[Compose] = None,
        colour_space: str = 'RGB'
    ) -> None:
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.colour_space = colour_space

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        kind, image_name, label = (
            self.kinds[index],
            self.image_names[index],
            self.labels[index],
        )

        image = cv2.imread(f"data/{kind}/{image_name}", cv2.IMREAD_UNCHANGED)
        if self.colour_space == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.colour_space == "YCrCb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            raise ValueError(f"Invalid colour_space: {self.colour_space}")
        image = image.astype(np.float32)

        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

        target = one_hot(4, label)
        return image, target

    def get_labels(self):
        return list(self.labels)


def one_hot(size: int, target: int) -> torch.tensor:
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec
