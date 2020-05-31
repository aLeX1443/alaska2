from alaska_pytorch.models import *


MODELS_SAVE_DIR = "alaska_pytorch/checkpoints/"
TENSORBOARD_LOGS_DIR = "alaska_pytorch/tensorboard_logs/"
SEED = 2020


EXPERIMENT_HYPER_PARAMETERS = {
    0: {
        "model_name": "resnext50_32x4d",
        "model": build_resnext50_32x4d,
        "device": 0,
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 36,  # cuDNN errors if batch_size is too big
        "n_epochs": 100,
        "learning_rate": 0.001,
        "lr_scheduler_exp_gamma": 0.98,
        "shuffle": True,
        "training_workers": 10,
        "val_epoch_freq": 1,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "continue_training_model_weights_path": None,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
        "validate_model_n_times_per_epoch": 5,
    },
    1: {
        "model_name": "resnext101_32x8d",
        "model": build_resnext101_32x8d,
        "device": 0,
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 16,
        "n_epochs": 100,
        "learning_rate": 0.001,
        "lr_scheduler_exp_gamma": 0.98,
        "shuffle": True,
        "training_workers": 10,
        "val_epoch_freq": 1,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "continue_training_model_weights_path": None,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
        "validate_model_n_times_per_epoch": 5,
    },
    2: {
        "model_name": "resnext50_32x4d",
        "model": build_resnext50_32x4d,
        "device": "cpu",
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 16,
        "n_epochs": 100,
        "learning_rate": 0.001,
        "lr_scheduler_exp_gamma": 0.98,
        "shuffle": True,
        "training_workers": 10,
        "val_epoch_freq": 1,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "continue_training_model_weights_path": None,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
        "validate_model_n_times_per_epoch": 5,
    },
}
