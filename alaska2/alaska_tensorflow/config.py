import tensorflow as tf
import alaska2.alaska_tensorflow.models.efficientnet as efn


MODELS_SAVE_DIR = "alaska2/alaska_tensorflow/checkpoints/"
TENSORBOARD_LOGS_DIR = "alaska2/alaska_tensorflow/tensorboard_logs/"
SEED = 2020


EXPERIMENT_HYPER_PARAMETERS = {
    0: {
        "model_name": "efficientnet_b0",
        "model": efn.build_efficientnet_b0,
        "initial_epoch": 0,
        "model_save_dir": MODELS_SAVE_DIR,
        "visible_gpus": [1],  # [0, 1],
        "enable_xla": False,  # True
        "enable_amp": False,  # True
        "saved_model_path": None,  # "efficientnet_b0_1591273683",
        "pooling": "avg",
        "print_model_summary": True,
        "input_shape": (3, 512, 512),
        "optimiser": tf.keras.optimizers.Adam,
        "learning_rate": 0.0001,
        "n_epochs": 1000,
        "batch_size": 8,
        "validation_split": 0.2,
        "image_data_format": "channels_first",
        "log_tensorboard_n_times_per_epoch": 100,
        "training_max_queue_size": 10,
        "training_workers": 10,
        "num_train_batches": None,
        "num_validation_batches": None,
    }
}
