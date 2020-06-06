from alaska2.alaska_pytorch.models import *


MODELS_SAVE_DIR = "alaska2/alaska_pytorch/checkpoints/"
TENSORBOARD_LOGS_DIR = "alaska2/alaska_pytorch/tensorboard_logs/"
SEED = 2020

"""
Note on batch_size:
- Can get cuDNN errors if batch_size is too big.
- Can get NaN model output if batch_size is too small or learning rate is too 
  large when training with AMP.
"""


EXPERIMENT_HYPER_PARAMETERS = {
    0: {
        "model_name": "efficientnet_b0",
        "model": build_efficientnet_b0,
        "use_amp": True,
        "device": 0,
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 96,
        "n_epochs": 1000,
        "learning_rate": 0.001,
        "lr_scheduler_exp_gamma": 0.98,
        "training_workers": 5,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "continue_training_model_weights_path": None,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    1: {
        "model_name": "efficientnet_b3",
        "model": build_efficientnet_b3,
        "use_amp": True,
        "devices": [0, 1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": "alaska2/alaska_pytorch/checkpoints/efficientnet_b3_1591462030_checkpoint.pth",
        # Training loop:
        "batch_size": 48,
        "n_epochs": 1000,
        "learning_rate": 0.0001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "continue_training_model_weights_path": None,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    2: {
        "model_name": "efficientnet_b3",
        "model": build_efficientnet_b3,
        "use_amp": True,
        "devices": [0],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": "alaska2/alaska_pytorch/checkpoints/efficientnet_b3_1591462030_checkpoint.pth",
        # Training loop:
        "batch_size": 16,
        "n_epochs": 1000,
        "learning_rate": 0.0001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 3,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "continue_training_model_weights_path": None,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
}
