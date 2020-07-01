import os
import click

# This should be set to the number of physical cores the system has.
os.environ["OMP_NUM_THREADS"] = "36"

from imagenet.config import EXPERIMENT_HYPER_PARAMETERS
from run_pytorch import initialise_torch


@click.command()
@click.argument("experiment_number", type=int)
def main(experiment_number):
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    initialise_torch(hyper_parameters)

    # Add the imports here since we must set the CUDA_VISIBLE_DEVICES
    # environment variable before importing torch.
    from imagenet.train import run_imagenet_experiment

    print("Training ImageNet.")
    run_imagenet_experiment(experiment_number)


if __name__ == "__main__":
    main()
