import cv2
import glob
import numpy as np
import pandas as pd
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# from catalyst.dl import SupervisedRunner

from alaska2.alaska_pytorch.config import EXPERIMENT_HYPER_PARAMETERS
from alaska2.alaska_pytorch.lib.data_loaders import ColourDataSet
from alaska2.lib.data_loaders import jpeg_decompress_ycbcr


def perform_inference(experiment_number: int) -> None:
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    data_loader = get_test_data_loader(hyper_parameters)

    # device = "cpu"
    device = "cuda:0"

    with torch.no_grad():
        # Build the model.
        model = hyper_parameters["model"](n_classes=4).to(device)

        # Load the trained model weights.
        model.load_state_dict(
            torch.load(
                hyper_parameters["trained_model_path"],
                map_location=torch.device(device),
            )["state_dict"]
        )

        model.eval()

        # Create a runner to handle the inference loop.
        # runner = SupervisedRunner(device=device)

        results = {"Id": [], "Label": []}

        # Perform inference.
        for image_names, input_data in tqdm(
            data_loader, total=len(data_loader),
        ):
            if hyper_parameters["use_quality_factor"]:
                prediction = model(
                    input_data[0].to(device), input_data[1].to(device)
                )
            else:
                prediction = model(input_data[0].to(device))
            prediction = (
                1
                - nn.functional.softmax(prediction, dim=1)
                .data.cpu()
                .numpy()[:, 0]
            )
            results["Id"].extend(image_names)
            results["Label"].extend(prediction)

    submission = pd.DataFrame(results)

    print(submission)

    submission.to_csv("submissions/submission.csv", index=False)


def get_test_data_loader(hyper_parameters: dict) -> DataLoader:
    # TODO combine this method with make_train_and_validation_data_loaders
    #  and clean up
    test_files = load_test_paths()
    input_data_type = hyper_parameters["input_data_type"]

    if input_data_type == "RGB":
        augmentations_test = Compose([Normalize(p=1), ToTensorV2()], p=1)
    elif input_data_type == "YCbCr":
        augmentations_test = Compose([ToTensorV2()], p=1)
    else:
        raise ValueError(
            f"Invalid input data type provided: {input_data_type}"
        )

    test_data_set = Alaska2TestDataset(
        image_names=test_files,
        transforms=augmentations_test,
        colour_space=hyper_parameters["input_data_type"],
        use_quality_factor=hyper_parameters["use_quality_factor"],
        separate_classes_by_quality_factor=hyper_parameters[
            "separate_classes_by_quality_factor"
        ],
    )
    return DataLoader(
        test_data_set,
        batch_size=hyper_parameters["batch_size"],
        num_workers=3,  # hyper_parameters["training_workers"],
        shuffle=False,
        drop_last=False,
    )


class Alaska2TestDataset(ColourDataSet):
    def __init__(
        self,
        image_names,
        transforms=None,
        colour_space="RGB",
        use_quality_factor=False,
        separate_classes_by_quality_factor=False,
    ):
        super(ColourDataSet, self).__init__()
        self.image_names = image_names
        self.transforms = transforms
        self.colour_space = colour_space
        self.use_quality_factor = use_quality_factor
        self.separate_classes_by_quality_factor = (
            separate_classes_by_quality_factor
        )

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        img_path = f"data/Test/{image_name}"

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if self.colour_space == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.colour_space == "YCbCr":
            image = jpeg_decompress_ycbcr(img_path)
            image = image / 128
        else:
            raise ValueError(f"Invalid colour_space: {self.colour_space}")
        image = image.astype(np.float32)

        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

        if self.use_quality_factor:
            if self.separate_classes_by_quality_factor:
                raise NotImplementedError()
            quality_factor = self.encode_quality_factor(
                self.get_quality_factor_from_image(img_path)
            )
            return image_name, [image, quality_factor]
        return image_name, image

    def __len__(self) -> int:
        return len(self.image_names)


def load_test_paths() -> np.ndarray:
    return np.array(
        [
            path.split("/")[-1]
            for path in sorted(glob.glob1("data/Test/", "*.jpg"))
        ]
    )
