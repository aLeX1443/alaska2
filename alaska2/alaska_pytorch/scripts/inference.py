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

from catalyst.dl import SupervisedRunner

from alaska2.alaska_pytorch.config import EXPERIMENT_HYPER_PARAMETERS


def perform_inference(experiment_number: int) -> None:
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    data_loader = get_test_data_loader(hyper_parameters)

    # device = "cpu"
    device = "cuda:0"

    with torch.no_grad():
        # Build the model.
        model = hyper_parameters["model"](n_classes=4,).to(device)

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
        for image_names, images in tqdm(data_loader, total=len(data_loader),):
            prediction = model(images.to(device))
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
    test_files = load_test_paths()

    augmentations_test = Compose([Normalize(p=1), ToTensorV2()], p=1,)

    test_data_set = Alaska2TestDataset(
        image_names=test_files, transforms=augmentations_test
    )
    return DataLoader(
        test_data_set,
        batch_size=hyper_parameters["batch_size"],
        num_workers=hyper_parameters["training_workers"],
        shuffle=False,
        drop_last=False,
    )


class Alaska2TestDataset(Dataset):
    def __init__(self, image_names, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image = cv2.imread(f"data/Test/{image_name}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

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
