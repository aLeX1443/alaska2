import tensorflow as tf
import alaska2.alaska_tensorflow.models.efficientnet as efn
import alaska2.alaska_tensorflow.models.dct_lstm as dct_lstm


MODELS_SAVE_DIR = "alaska2/alaska_tensorflow/checkpoints/"
TENSORBOARD_LOGS_DIR = "alaska2/alaska_tensorflow/tensorboard_logs/"
SEED = 2020


EXPERIMENT_HYPER_PARAMETERS = {
    0: {
        "model_name": "efficientnet_b3",
        "model": efn.build_base_efficientnet_model,
        "input_data_type": "RGB",
        "initial_epoch": 0,
        "model_save_dir": MODELS_SAVE_DIR,
        "visible_gpus": [0],  # [0, 1],
        "enable_xla": True,
        "enable_amp": True,
        "saved_model_path": None,
        "pooling": "avg",
        "print_model_summary": True,
        "input_shape": (3, 512, 512),
        "optimiser": tf.keras.optimizers.Adam,
        "learning_rate": 0.0001,
        "n_epochs": 1000,
        "batch_size": 32,
        "validation_split": 0.2,
        "image_data_format": "channels_first",
        "log_tensorboard_n_times_per_epoch": 100,
        "training_max_queue_size": 10,
        "training_workers": 10,
        "num_train_batches": None,
        "num_validation_batches": None,
    },
    1: {
        "model_name": "dct_lstm",
        "model": dct_lstm.build_dct_lstm,
        "input_data_type": "DCT",
        "initial_epoch": 0,
        "model_save_dir": MODELS_SAVE_DIR,
        "visible_gpus": [0],  # [0, 1],
        "enable_xla": True,
        "enable_amp": True,
        "saved_model_path": None,
        "pooling": "avg",
        "print_model_summary": True,
        "input_shape": (12288, 64),
        "optimiser": tf.keras.optimizers.Adam,
        "learning_rate": 0.001,
        "n_epochs": 1000,
        "batch_size": 64,
        "validation_split": 0.2,
        "image_data_format": "channels_last",
        "log_tensorboard_n_times_per_epoch": 100,
        "training_max_queue_size": 25,
        "training_workers": 25,
        "num_train_batches": None,
        "num_validation_batches": None,
    },
}
