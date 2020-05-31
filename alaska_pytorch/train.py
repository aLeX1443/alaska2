import random
import numpy as np

from torch.backends import cudnn

from alaska_pytorch.config import EXPERIMENT_HYPER_PARAMETERS
from alaska_pytorch.lib.trainer import Trainer


def run_stegoanalysis_experiment(experiment_number):
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    initialise_torch(experiment_number)

    trainer = Trainer(hyper_parameters=hyper_parameters)
    trainer.train()


def initialise_torch(experiment_number):
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    if hyper_parameters["seed"]:
        # Set a seed for reproducibility. Note: torch will not be seeded as making
        # training deterministic will slow down training.
        random.seed(hyper_parameters["seed"])
        np.random.seed(hyper_parameters["seed"])

    # Run a benchmark at the beginning of training that determines which
    # algorithm will be the best for our hardware. Only use this if the input
    # size is fixed, or the benchmark will be run every time the input size
    # changes.
    cudnn.benchmark = True
