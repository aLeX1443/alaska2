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
        "model_name": "rgb_efficientnet_b0",
        "model": build_efficientnet_b0,
        "input_data_type": "RGB",
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0, 1],
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
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    1: {
        "model_name": "rgb_efficientnet_b3",
        "model": build_efficientnet_b3,
        "input_data_type": "RGB",
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0, 1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": "alaska2/alaska_pytorch/checkpoints/rgb_efficientnet_b3_1592168121_best_auc_checkpoint.pth",
        # Training loop:
        "batch_size": 48,
        "n_epochs": 1000,
        "learning_rate": 0.00005,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    2: {
        "model_name": "rgb_efficientnet_b5",
        "model": build_efficientnet_b5,
        "input_data_type": "RGB",
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0, 1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": "alaska2/alaska_pytorch/checkpoints/rgb_efficientnet_b5_1591917676_checkpoint.pth",
        # Training loop:
        "batch_size": 24,
        "n_epochs": 1000,
        "learning_rate": 0.0001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    3: {
        "model_name": "dct_efficientnet_b0",
        "model": build_dct_efficientnet_b0,
        "input_data_type": "DCT",
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 1024,
        "n_epochs": 1000,
        "learning_rate": 0.0001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    4: {
        "model_name": "dct_efficientnet_b0",
        "model": build_dct_efficientnet_b0,
        "input_data_type": "DCT",
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 1024,
        "n_epochs": 1000,
        "learning_rate": 0.001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    5: {
        "model_name": "dct_resnet_b0",
        "model": build_dct_resnet_50,
        "input_data_type": "DCT",
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 256,
        "n_epochs": 1000,
        "learning_rate": 0.0001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    6: {
        "model_name": "dct_multiple_input_efficientnet_b0",
        "model": build_dct_multiple_input_efficientnet_b0,
        "input_data_type": "DCT",
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0, 1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 256,
        "n_epochs": 1000,
        "learning_rate": 0.001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    7: {
        "model_name": "combined_dct_efficientnet_b0",
        "model": build_efficientnet_b0,
        "input_data_type": "Combined_DCT",
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0, 1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 48,
        "n_epochs": 1000,
        "learning_rate": 0.0001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 1,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    8: {
        "model_name": "ycbcr_qf_input_efficientnet_b3",
        "model": build_quality_factor_efficientnet_b3,
        "input_data_type": "YCbCr",
        "use_quality_factor": True,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0, 1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": "alaska2/alaska_pytorch/checkpoints/ycbcr_qf_input_efficientnet_b3_1592504869_checkpoint.pth",
        # Training loop:
        "batch_size": 48,
        "n_epochs": 1000,
        "learning_rate": 0.00001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    9: {
        "model_name": "rgb_qf_input_efficientnet_b3",
        "model": build_quality_factor_efficientnet_b3,
        "input_data_type": "RGB",
        "use_quality_factor": True,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0, 1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": "alaska2/alaska_pytorch/checkpoints/rgb_qf_input_efficientnet_b3_1592261292_checkpoint.pth",
        # Training loop:
        "batch_size": 48,
        "n_epochs": 1000,
        "learning_rate": 0.00005,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    10: {
        "model_name": "rgb_qf_input_efficientnet_b5",
        "model": build_quality_factor_efficientnet_b5,
        "input_data_type": "RGB",
        "use_quality_factor": True,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0, 1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": "alaska2/alaska_pytorch/checkpoints/rgb_qf_input_efficientnet_b5_1592316231_checkpoint.pth",
        # Training loop:
        "batch_size": 24,
        "n_epochs": 1000,
        "learning_rate": 0.000075,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
    11: {
        "model_name": "ycbcr_yedroudj_net",
        "model": build_yedroudj_net,
        "input_data_type": "YCbCr",
        "use_quality_factor": False,
        "separate_classes_by_quality_factor": False,
        "use_amp": True,
        "devices": [0, 1],
        "seed": 2020,
        "validation_split": 0.2,
        # Model parameters:
        "trained_model_path": None,
        # Training loop:
        "batch_size": 32,
        "n_epochs": 1000,
        "learning_rate": 0.001,
        "lr_scheduler_exp_gamma": 0.95,
        "training_workers": 10,
        # Other:
        "model_checkpoint_dir": MODELS_SAVE_DIR,
        "tensorboard_log_dir": TENSORBOARD_LOGS_DIR,
        "log_tensorboard_n_times_per_epoch": 100,
    },
}
