import numpy as np
import gc

import tensorflow as tf
from tensorflow.keras.utils import Sequence

from sklearn.utils import shuffle

from lib.data_loaders import create_train_and_validations_sets, SEED


def create_train_and_validation_loaders(batch_size=16, validation_split=0.2):
    # Load the filenames and targets.
    (
        train_data,
        validation_data,
        train_class_weights,
        validation_class_weights,
    ) = create_train_and_validations_sets(validation_split)

    # Create the batched sequence generators.
    train_dataset = Batcher(
        ImageDataGenerator(
            files=train_data["files"], targets=train_data["targets"],
        ),
        batch_size=batch_size,
    )
    validation_dataset = Batcher(
        ImageDataGenerator(
            files=validation_data["files"], targets=validation_data["targets"],
        ),
        batch_size=batch_size,
    )

    return (
        train_dataset,
        validation_dataset,
        train_class_weights,
        validation_class_weights,
    )


class ImageDataGenerator(Sequence):
    def __init__(self, files, targets):
        self.files, self.targets = shuffle(files, targets, random_state=SEED)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        target = self.targets[index]
        return decode_image(file), target

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


def decode_image(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    return tf.cast(image, tf.float32) / 255.0


def data_augment(image):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, 0.2, seed=SEED)
    image = tf.image.random_contrast(image, 0.6, 1.4, seed=SEED)
    image = tf.image.random_hue(image, 0.07, seed=SEED)
    image = tf.image.random_saturation(image, 0.5, 1.5, seed=SEED)
    return image


if __name__ == "__main__":
    create_train_and_validations_sets()
