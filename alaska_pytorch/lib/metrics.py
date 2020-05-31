import numpy as np
from sklearn.metrics import auc, roc_curve

import torch


class MovingAverageMetric:
    def __init__(self, window_size=100):
        self.metric_sequence = []
        self.window_size = window_size

    def add_value(self, value):
        self.metric_sequence.append(value)

    def get_moving_average_point(self):
        """Computes the average over the last window_size values"""
        return np.mean(self.metric_sequence[-self.window_size :])


def binary_accuracy_from_model_output_and_target(
    output, target, add_sigmoid_activation=True
):
    assert len(output) == len(target)
    if add_sigmoid_activation:
        output = torch.sigmoid(output)
    pred = output >= 0.5
    batch_correct = pred.eq(target.view_as(pred)).sum().item()
    batch_accuracy = batch_correct / len(target)
    return batch_accuracy


def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_valid, pos_label=1)

    # Size of subsets.
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final
    # weighted AUC is between 0 and 1.
    normalisation = np.dot(areas, weights)

    competition_metric = 0
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

    return competition_metric / normalisation
