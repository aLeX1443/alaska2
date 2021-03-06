from alaska2.alaska_pytorch.models import *

MODELS_SAVE_DIR = "cifar10/checkpoints/"
TENSORBOARD_LOGS_DIR = "cifar10/tensorboard_logs/"
SEED = 2020


EXPERIMENT_HYPER_PARAMETERS = {
    0: {
        "model_name": "dct_efficientnet_b7_no_weight_sharing",
        "model": build_dct_efficientnet_b7_no_weight_sharing,
        "input_data_type": "DCT",
        "n_classes": 10,
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 64,
        "n_epochs": 1000,
        "learning_rate": 0.0002,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 20,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 25,
    },
}
