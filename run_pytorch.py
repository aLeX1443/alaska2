import os
import click
import random
import numpy as np

# This should be set to the number of physical cores the system has.
os.environ["OMP_NUM_THREADS"] = "36"

from alaska2.alaska_pytorch.config import EXPERIMENT_HYPER_PARAMETERS


@click.command()
@click.argument("experiment_number", type=int)
@click.argument("mode", type=str)
def main(experiment_number, mode):
    initialise_torch(experiment_number)

    # Add the imports here since we must set the CUDA_VISIBLE_DEVICES
    # environment variable before importing torch.
    from alaska2.alaska_pytorch.scripts.inference import perform_inference
    from alaska2.alaska_pytorch.train import run_stegoanalysis_experiment

    if mode == "train":
        print("Training model with PyTorch.")
        run_stegoanalysis_experiment(experiment_number)
    else:
        print("Performing inference with PyTorch.")
        perform_inference(experiment_number=experiment_number,)


def initialise_torch(experiment_number):
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    # Set which GPU will be visible (note: the code currently only supports
    # training with one GPU).
    if hyper_parameters["devices"]:
        gpus = ",".join([str(_) for _ in hyper_parameters["devices"]])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    if hyper_parameters["seed"]:
        # Set a seed for reproducibility. Note: torch will not be seeded as making
        # training deterministic will slow down training.
        random.seed(hyper_parameters["seed"])
        np.random.seed(hyper_parameters["seed"])

    from torch.backends import cudnn

    # Run a benchmark at the beginning of training that determines which
    # algorithm will be the best for our hardware. Only use this if the input
    # size is fixed, or the benchmark will be run every time the input size
    # changes.
    cudnn.benchmark = True


if __name__ == "__main__":
    main()
