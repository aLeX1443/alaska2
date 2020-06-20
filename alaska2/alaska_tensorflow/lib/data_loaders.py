import cv2
import numpy as np
import gc

import tensorflow as tf
from tensorflow.keras.utils import Sequence

from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Normalize,
)

from alaska2.alaska_tensorflow.config import SEED
from alaska2.lib.data_loaders import load_data, add_fold_to_data_set


def create_train_and_validation_loaders(
    batch_size=16, validation_fold_number=0
):
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

    # Create the batched sequence generators.
    train_dataset = Batcher(
        ImageDataGenerator(
            kinds=data_set[
                data_set["fold"] != validation_fold_number
            ].kind.values,
            image_names=data_set[
                data_set["fold"] != validation_fold_number
            ].image_name.values,
            labels=data_set[
                data_set["fold"] != validation_fold_number
            ].label.values,
            transforms=augmentations_train,
        ),
        batch_size=batch_size,
    )
    validation_dataset = Batcher(
        ImageDataGenerator(
            kinds=data_set[
                data_set["fold"] == validation_fold_number
            ].kind.values,
            image_names=data_set[
                data_set["fold"] == validation_fold_number
            ].image_name.values,
            labels=data_set[
                data_set["fold"] == validation_fold_number
            ].label.values,
            transforms=augmentations_validation,
        ),
        batch_size=batch_size,
    )

    return train_dataset, validation_dataset


class ImageDataGenerator(Sequence):
    def __init__(self, kinds, image_names, labels, transforms,) -> None:
        """
        Parameters
        ----------
        kinds : list

        """
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index):
        kind, image_name, label = (
            self.kinds[index],
            self.image_names[index],
            self.labels[index],
        )

        image = cv2.imread(f"data/{kind}/{image_name}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

        target = one_hot(4, label)

        image = np.array(image, dtype=np.float32)
        target = np.array(target, dtype=np.float32)

        return image, target

    def get_labels(self):
        return list(self.labels)

    def on_epoch_end(self):
        gc.collect()


class Batcher(Sequence):
    """Assemble a sequence of things into a sequence of batches."""

    def __init__(self, sequence, batch_size=16):
        self._batch_size = batch_size
        self._sequence = sequence
        self._idxs = np.arange(len(self._sequence))

    def __len__(self):
        return int(np.ceil(len(self._sequence) / self._batch_size))

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError("Index out of bounds")

        start = i * self._batch_size
        end = min(len(self._sequence), start + self._batch_size)
        data = [self._sequence[j] for j in self._idxs[start:end]]
        inputs = [d[0] for d in data]
        outputs = [d[1] for d in data]

        return self._stack(inputs), self._stack(outputs)

    @staticmethod
    def _stack(data):
        if data is None:
            return None

        if not isinstance(data[0], (list, tuple)):
            return np.stack(data)

        seq = type(data[0])
        k = len(data[0])
        data = seq(np.stack([d[k] for d in data]) for k in range(k))

        return data

    def on_epoch_end(self):
        np.random.shuffle(self._idxs)
        self._sequence.on_epoch_end()


def one_hot(size, target) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    vec[target] = 1.0
    return vec


def decode_image(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    return tf.cast(image, tf.float32) / 255.0


def data_augment(image):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    return image
