import os

import torch
from torch import nn


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.05) -> None:
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x: torch.tensor, target: torch.tensor) -> torch.tensor:
        if self.training:
            # x = x.float()
            # target = target.float()
            log_probs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -log_probs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -log_probs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class EmptyScaler:
    @staticmethod
    def state_dict():
        return {}

    @staticmethod
    def scale(loss: float):
        pass

    @staticmethod
    def update():
        pass

    @staticmethod
    def step(optimiser):
        pass

    @staticmethod
    def get_scale():
        pass


def make_dir_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, mode=0o755)
