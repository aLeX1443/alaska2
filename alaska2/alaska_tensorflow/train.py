import time
import click
import os
import multiprocessing as mp

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from alaska2.alaska_tensorflow.config import EXPERIMENT_HYPER_PARAMETERS
from alaska2.alaska_tensorflow.lib.data_loaders import (
    create_train_and_validation_loaders,
)
from alaska2.alaska_tensorflow.lib.utils import load_trained_tf_model

print(f"Using TensorFlow version {tf.__version__}")
assert tf.__version__.startswith("2.")
K.set_image_data_format("channels_last")


@click.command()
@click.argument("experiment_number", type=int)
def main(experiment_number):
    initialise_tf(experiment_number)
    print(f"\nRunning experiment: {experiment_number}")
    run_training_experiment(experiment_number)


def run_training_experiment(experiment_number):
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    # Define what type of strategy will be used.
    if len(hyper_parameters["visible_gpus"]) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Number of devices: {strategy.num_replicas_in_sync}\n")

    # The batch size will be divided by all the devices within the scope.
    batch_size = hyper_parameters["batch_size"] * strategy.num_replicas_in_sync
    hyper_parameters["batch_size"] = batch_size

    model_name = f"{hyper_parameters['model_name']}_{round(time.time())}"
    hyper_parameters["model_name"] = model_name

    with strategy.scope():
        # Get the built and compiled model.
        model = get_compiled_model(hyper_parameters)

        # Get the training and validation data generators
        (
            training_data_generator,
            validation_data_generator,
        ) = get_train_and_validation_generators(hyper_parameters)
        hyper_parameters["num_train_batches"] = len(training_data_generator)
        hyper_parameters["num_validation_batches"] = len(
            validation_data_generator
        )

        # Get the training callbacks.
        callbacks = get_keras_callbacks(hyper_parameters)

        print(f"Model name: {model_name}")

        model.fit(
            x=training_data_generator,
            validation_data=validation_data_generator,
            epochs=hyper_parameters["n_epochs"],
            initial_epoch=hyper_parameters["initial_epoch"],
            callbacks=callbacks,
            # class_weight=train_class_weights,
            use_multiprocessing=False,
            max_queue_size=hyper_parameters["training_max_queue_size"],
            workers=hyper_parameters["training_workers"],
            shuffle=False,
        )

    print(f"Completed experiment: {hyper_parameters['model_name']}")


def get_compiled_model(hyper_parameters):
    model_save_dir = hyper_parameters["model_save_dir"]
    model_name = hyper_parameters["model_name"]

    metrics = get_metrics()

    if hyper_parameters["saved_model_path"] is not None:
        print("Continuing training model from last checkpoint")
        model = load_trained_tf_model(
            model_save_dir=hyper_parameters["model_save_dir"],
            model_name=hyper_parameters["saved_model_path"],
            use_best_val_loss=True,
        )
    else:
        # Build the model.
        # TODO try using this for training very large models:
        #  https://keras.io/getting-started/faq/#device-parallelism
        model = hyper_parameters["model"](
            input_shape=hyper_parameters["input_shape"],
            print_model_summary=hyper_parameters["print_model_summary"],
        )

    # Define the optimiser.
    optimiser = hyper_parameters["optimiser"](
        lr=hyper_parameters["learning_rate"]
    )

    # Enable mixed precision using dynamic loss scaling.
    if hyper_parameters["enable_amp"]:
        optimiser = mixed_precision.LossScaleOptimizer(
            optimiser, loss_scale="dynamic"
        )

    # Compile the model.
    model.compile(
        optimizer=optimiser, loss="categorical_crossentropy", metrics=metrics,
    )

    # Serialize the model to JSON and save.
    model_json = model.to_json()
    with open(f"{model_save_dir}{model_name}.json", "w") as json_file:
        json_file.write(model_json)

    return model


def get_train_and_validation_generators(hyper_parameters):
    (
        training_data_generator,
        validation_data_generator,
    ) = create_train_and_validation_loaders(
        batch_size=hyper_parameters["batch_size"],
    )
    return (
        training_data_generator,
        validation_data_generator,
    )


def get_metrics(use_xla=True):
    # TODO add weighted AUC from the competition
    if use_xla:
        return [
            tf.keras.metrics.Accuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    else:
        return [
            tf.keras.metrics.Accuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ]


def get_keras_callbacks(hyper_parameters):
    # TODO add learning rate drop after n epochs
    model_save_dir = hyper_parameters["model_save_dir"]
    model_name = hyper_parameters["model_name"]

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"tensorboard_logs/{model_name}",
        update_freq=int(
            hyper_parameters["num_train_batches"]
            / hyper_parameters["log_tensorboard_n_times_per_epoch"]
        ),
        profile_batch=0,  # '10, 100',
        write_graph=False,
    )

    val_loss_checkpointer = tf.keras.callbacks.ModelCheckpoint(
        f"{model_save_dir}{model_name}_best_val_loss.h5",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=True,
    )

    loss_checkpointer = tf.keras.callbacks.ModelCheckpoint(
        f"{model_save_dir}{model_name}.h5",
        monitor="loss",
        mode="min",
        save_best_only=False,
        save_weights_only=True,
        verbose=True,
    )

    return [tensorboard, val_loss_checkpointer, loss_checkpointer]


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
    gpus = "".join([str(_) for _ in hyper_parameters["visible_gpus"]])
    # TODO needs to be moved to above tf import
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Set this to the number of physical CPU cores in the system.
    # This will greatly speed up ICP and asynchronous data loading for PyTorch.
    os.environ["OMP_NUM_THREADS"] = str(int(mp.cpu_count() / 2))

    if hyper_parameters["enable_xla"]:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"

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
