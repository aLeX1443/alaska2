"""
Athena Docker container ID: a9dabee28391
"""

import os
import click

# This should be set to the number of physical cores the system has.
os.environ["OMP_NUM_THREADS"] = "36"

from alaska2.alaska_tensorflow.train import run_training_experiment
from alaska2.alaska_tensorflow.config import EXPERIMENT_HYPER_PARAMETERS


@click.command()
@click.argument("experiment_number", type=int)
def main(experiment_number):
    initialise_tf(experiment_number)
    print(f"\nRunning experiment: {experiment_number}")
    run_training_experiment(experiment_number)


def initialise_tf(experiment_number):
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    """
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
    """

    # Set which GPUs will be visible.
    gpus = ",".join([str(_) for _ in hyper_parameters["visible_gpus"]])
    # TODO needs to be moved to above tf import
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # The imports must be performed after setting the environment variables
    # for them to take effect.
    import tensorflow as tf
    from tensorflow.keras.mixed_precision import (
        experimental as mixed_precision,
    )
    import tensorflow.keras.backend as K

    if hyper_parameters["enable_xla"]:
        tf.config.optimizer.set_jit(True)
        print("Enabled XLA (Accelerated Linear Algebra)")

    if hyper_parameters["enable_amp"]:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_policy(policy)
        print("Enabled AMP (Automatic Mixed Precision)")

    # Set the Keras image data format.
    K.set_image_data_format(hyper_parameters["image_data_format"])


if __name__ == "__main__":
    main()
