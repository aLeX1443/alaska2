import os
import click

# This should be set to the number of physical cores the system has.
os.environ["OMP_NUM_THREADS"] = "36"

from alaska_pytorch.scripts.inference import perform_inference
from alaska_pytorch.train import run_stegoanalysis_experiment


@click.command()
@click.argument("experiment_number", type=int)
@click.argument("mode", type=str)
def main(experiment_number, mode):
    if mode == "train":
        print("Training model with PyTorch.")
        run_stegoanalysis_experiment(experiment_number)
    else:
        print("Performing inference with PyTorch.")
        perform_inference(
            "alaska_pytorch/checkpoints/StegoEfficientNetB0_1590801582_best_val_checkpoint.pth"
        )
        # perform_inference(
        #     "alaska_pytorch/checkpoints/StegoEfficientNetB0_1590800801_best_val_checkpoint.pth"
        # )


if __name__ == "__main__":
    main()
