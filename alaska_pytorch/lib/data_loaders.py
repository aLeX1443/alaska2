import cv2
import numpy as np

import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Compose

from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    HorizontalFlip,
    CLAHE,
    HueSaturationValue,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    OneOf,
    Resize,
    ToFloat,
    ShiftScaleRotate,
    GridDistortion,
    ElasticTransform,
    JpegCompression,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    Blur,
    MotionBlur,
    MedianBlur,
    GaussNoise,
    CenterCrop,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    OpticalDistortion,
    RandomSizedCrop,
    VerticalFlip,
)

from lib.data_loaders import SEED, create_train_and_validations_sets
from alaska_pytorch.lib.utils import ToCustomFloat


class Alaska2Dataset(Dataset):
    def __init__(self, files, targets, augmentations=None):
        self.files, self.targets = shuffle(files, targets, random_state=SEED)
        self.augmentations = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        target = self.targets[index]
        image = cv2.imread(file)[:, :, ::-1]
        if self.augmentations:
            # Apply transformations.
            image = self.augmentations(image=image)
        return image, target


def make_train_and_validation_data_loaders(hyper_parameters):
    # Load the filenames and targets.
    (
        train_data,
        validation_data,
        train_class_weights,
        validation_class_weights,
    ) = create_train_and_validations_sets(hyper_parameters["validation_split"])

    # Define a set of image augmentations that will only be used.
    augmentations_train = Compose(
        [
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            ToCustomFloat(max_value=255, dtype="float32"),
            # ToFloat(max_value=255),
            ToTensorV2(),
        ],
        p=1,
    )
    augmentations_validation = Compose(  # "float64"
        [ToCustomFloat(max_value=255, dtype="float32"), ToTensorV2()], p=1,
    )

    # Compute sample weights.
    train_sample_weights = compute_pos_weights(train_data["targets"])
    validation_sample_weights = compute_pos_weights(validation_data["targets"])

    print("Train sample weights:", train_sample_weights)

    # Create train and validation data sets.
    train_data_set = Alaska2Dataset(
        files=train_data["files"],
        targets=train_data["targets"],
        augmentations=augmentations_train,
    )
    validation_data_set = Alaska2Dataset(
        files=validation_data["files"],
        targets=validation_data["targets"],
        augmentations=augmentations_validation,
    )

    # Create train and validation data loaders.
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=hyper_parameters["batch_size"],
        shuffle=hyper_parameters["shuffle"],
        num_workers=hyper_parameters["training_workers"],
        # collate_fn=collate_func,
        pin_memory=True,
        drop_last=True,
    )
    validation_data_loader = DataLoader(
        validation_data_set,
        batch_size=hyper_parameters["batch_size"],
        shuffle=hyper_parameters["shuffle"],
        num_workers=hyper_parameters["training_workers"],
        # collate_fn=collate_func,
        pin_memory=True,
        drop_last=True,
    )

    return (
        train_data_loader,
        validation_data_loader,
        train_sample_weights,
        validation_sample_weights,
    )


def collate_func(list_data):
    images, targets = list(zip(*list_data))

    images_batch = torch.tensor(images)
    target_batch = torch.tensor(targets).unsqueeze(1)

    return {
        "images": images_batch.float(),
        "targets": target_batch.float(),
    }


def compute_pos_weights(targets):
    """
    Computes the pos_weights parameter to pass to BCEWithLogitsLoss.
    """
    counts = np.bincount(targets)
    return torch.as_tensor([counts[0] / counts[1]], dtype=torch.float32)
