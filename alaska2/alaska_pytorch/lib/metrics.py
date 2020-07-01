from collections import deque
import numpy as np
from sklearn.metrics import auc, roc_curve

import torch
from torch import nn
from typing import Tuple


class BatchAverage:
    def __init__(self) -> None:
        self.all_values = []

    def update(self, value: float) -> None:
        self.all_values.append(value)

    @property
    def avg(self) -> float:
        return float(np.mean(self.all_values))


class MovingAverageMetric:
    def __init__(self, window_size: int = 2500) -> None:
        self.metric_sequence = deque([], maxlen=window_size)
        self.window_size = window_size

    def update(self, value: float) -> None:
        self.metric_sequence.append(value)

    @property
    def moving_average(self) -> np.ndarray:
        """Computes the average over the last window_size values"""
        return np.mean(self.metric_sequence)


def get_argmax_from_prediction_and_target(
    y_true: torch.tensor, y_pred: torch.tensor
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
    y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]
    return y_true, y_pred


class WeightedAUCMeter:
    def __init__(self) -> None:
        self.y_true = np.array([0, 1])
        self.y_pred = np.array([0.5, 0.5])

    def reset(self) -> None:
        self.y_true = np.array([0, 1])
        self.y_pred = np.array([0.5, 0.5])

    def update(self, y_true: torch.tensor, y_pred: torch.tensor) -> None:
        y_true, y_pred = get_argmax_from_prediction_and_target(y_true, y_pred)
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))

    @property
    def avg(self) -> float:
        return alaska_weighted_auc(self.y_true, self.y_pred)


def alaska_weighted_auc(y_true: torch.tensor, y_valid: torch.tensor) -> float:
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_valid, pos_label=1)

        # Size of subsets.
        areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

        # The total area is normalized by the sum of weights such that the final
        # weighted AUC is between 0 and 1.
        normalisation = float(np.dot(areas, weights))

        competition_metric: float = 0.0
        for idx, weight in enumerate(weights):
            y_min = tpr_thresholds[idx]
            y_max = tpr_thresholds[idx + 1]
            mask = (y_min < tpr) & (tpr < y_max)

            x_padding = np.linspace(fpr[mask][-1], 1, 100)

            x = np.concatenate([fpr[mask], x_padding])
            y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
            y = y - y_min  # normalize such that curve starts at y=0
            score = auc(x, y)
            sub_metric = score * weight
            competition_metric += sub_metric
    except Exception as e:
        print(f"Exception in computing AUC: {e}.\nReturning 0.0")
        return 0.0
    return competition_metric / normalisation


def top_k_accuracy(output, target, top_k=1):
    assert len(output) == len(target)
    batch_size = target.size(0)
    print(output.shape)
    print(target.shape)
    predictions = output.topk(k=top_k, dim=1)[0]
    print(predictions.shape)
    print(target.view(1, -1).expand_as(predictions).shape)
    exit()
    correct = predictions.eq(target.view(1, -1).expand_as(predictions))
    correct_k = correct[:top_k].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)
