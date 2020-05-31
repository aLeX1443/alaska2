import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from alaska_pytorch.config import SEED


def create_train_and_validations_sets(val_split=0.2):
    data_df = load_images_and_labels()
    # data_df = load_images_and_labels_no_overlap()

    # Create train and validation sets.
    train_data_df, validation_data_df = train_test_split(
        data_df, test_size=val_split, shuffle=True, random_state=SEED,
    )

    train_data = {
        "files": train_data_df["files"].values,
        "targets": train_data_df["targets"].values,
    }
    validation_data = {
        "files": validation_data_df["files"].values,
        "targets": validation_data_df["targets"].values,
    }

    # # One hot encode targets.
    # train_data["one_hot_targets"] = tf.keras.backend.one_hot(
    #     train_data["targets"], num_classes=2
    # )
    # validation_data["one_hot_targets"] = tf.keras.backend.one_hot(
    #     validation_data["targets"], num_classes=2
    # )

    # TODO this will be more relevant when splitting by JPEG quality factor.
    # Compute class weights
    train_class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(train_data["targets"]), train_data["targets"],
    )
    validation_class_weights = class_weight.compute_class_weight(
        "balanced",
        np.unique(validation_data["targets"]),
        validation_data["targets"],
    )

    return (
        train_data,
        validation_data,
        dict(enumerate(train_class_weights)),
        dict(enumerate(validation_class_weights)),
    )


def load_images_and_labels(load_dir="data", file_regex="*.jpg"):
    # TODO treat all modified as the same target
    unmodified_dir = f"{load_dir}/Cover/"
    unmodified_files = glob.glob1(unmodified_dir, file_regex)
    unmodified_files = [unmodified_dir + file for file in unmodified_files]
    unmodified_targets = [0 for _ in range(len(unmodified_files))]

    modified_dir_1 = f"{load_dir}/JMiPOD/"
    modified_files_1 = glob.glob1(modified_dir_1, file_regex)
    modified_files_1 = [modified_dir_1 + file for file in modified_files_1]
    modified_targets_1 = [1 for _ in range(len(modified_files_1))]

    modified_dir_2 = f"{load_dir}/JUNIWARD/"
    modified_files_2 = glob.glob1(modified_dir_2, file_regex)
    modified_files_2 = [modified_dir_2 + file for file in modified_files_2]
    modified_targets_2 = [1 for _ in range(len(modified_files_2))]

    modified_dir_3 = f"{load_dir}/UERD/"
    modified_files_3 = glob.glob1(modified_dir_3, file_regex)
    modified_files_3 = [modified_dir_3 + file for file in modified_files_3]
    modified_targets_3 = [1 for _ in range(len(modified_files_3))]

    files = (
        unmodified_files
        + modified_files_1
        + modified_files_2
        + modified_files_3
    )
    targets = (
        unmodified_targets
        + modified_targets_1
        + modified_targets_2
        + modified_targets_3
    )

    return pd.DataFrame({"files": files, "targets": targets})


def load_images_and_labels_no_overlap(load_dir="data"):
    train_filenames = np.array(os.listdir(f"{load_dir}/Cover/"))

    positives = train_filenames.copy()
    negatives = train_filenames.copy()
    np.random.shuffle(positives)
    np.random.shuffle(negatives)

    jmipod = append_path("JMiPOD")(positives[:20000])
    juniward = append_path("JUNIWARD")(positives[20000:40000])
    uerd = append_path("UERD")(positives[40000:60000])

    pos_paths = np.concatenate([jmipod, juniward, uerd])
    neg_paths = append_path("Cover")(positives[:60000])

    train_paths = np.concatenate([pos_paths, neg_paths])
    train_labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))

    return pd.DataFrame({"files": train_paths, "targets": train_labels})


def append_path(pre):
    return np.vectorize(lambda file: os.path.join("data", pre, file))
