import os
import cv2
import glob
import numpy as np
import pandas as pd
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from catalyst.dl import SupervisedRunner

from alaska_pytorch.lib.utils import ToCustomFloat
from alaska_pytorch.models.efficientnet import StegoEfficientNet


def perform_inference(pre_trained_model_path):
    data_loader = get_test_data_loader()

    with torch.no_grad():
        # Build the model.
        model = StegoEfficientNet(
            n_classes=1, model_name="efficientnet-b0"
        ).to("cuda:0")

        # Load the trained model weights.
        model.load_state_dict(torch.load(pre_trained_model_path)["state_dict"])

        model.eval()

        # Create a runner to handle the inference loop.
        runner = SupervisedRunner(device="cuda:0")

        # Perform inference.
        predictions = []
        for outputs in tqdm(
            runner.predict_loader(loader=data_loader, model=model),
            total=len(data_loader),
        ):
            prediction = torch.sigmoid(outputs["logits"]).cpu().numpy()
            prediction = prediction[:, 0]
            predictions.append(prediction)

    predictions = np.array(np.concatenate(predictions))

    test_df = pd.DataFrame()

    test_df["Id"] = pd.Series(load_test_paths()).apply(
        lambda x: x.split(os.sep)[-1]
    )
    test_df["Label"] = predictions

    print(test_df)

    test_df.to_csv("submissions/submission.csv", index=False)


def get_test_data_loader():
    test_files = load_test_paths()

    augmentations_test = Compose(
        [ToCustomFloat(max_value=255, dtype="float32"), ToTensorV2()], p=1,
    )

    test_data_set = Alaska2TestDataset(
        files=test_files, augmentations=augmentations_test
    )
    return DataLoader(
        test_data_set,
        batch_size=24,
        num_workers=10,
        shuffle=False,
        drop_last=False,
    )


class Alaska2TestDataset(Dataset):
    def __init__(self, files, augmentations=None):
        self.files = files
        self.augmentations = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        image = cv2.imread(file)[:, :, ::-1]
        if self.augmentations:
            # Apply transformations.
            image = self.augmentations(image=image)
        return {"features": image["image"]}


def load_test_paths():
    files = sorted(glob.glob1("data/Test/", "*.jpg"))
    return ["data/Test/" + file for file in files]
