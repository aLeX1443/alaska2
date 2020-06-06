from typing import Dict
import gc
import os
import numpy as np
import time
from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.cuda
import torch.distributed
from torch import nn

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    print("Could not load torch.cuda.amp")

from tqdm import tqdm

from alaska2.alaska_pytorch.lib.utils import (
    make_dir_if_not_exists,
    LabelSmoothing,
    EmptyScaler,
)
from alaska2.alaska_pytorch.lib.data_loaders import (
    make_train_and_validation_data_loaders,
)
from alaska2.alaska_pytorch.lib.metrics import (
    MovingAverageMetric,
    WeightedAUCMeter,
)


class Trainer:
    def __init__(self, hyper_parameters: Dict) -> None:
        if not torch.cuda.is_available():
            raise ValueError("GPU not available.")

        self.main_device = "cuda:0"
        self.devices = hyper_parameters["devices"]

        # Build the model.
        self.model = hyper_parameters["model"](n_classes=4)

        # Distribute over all GPUs.
        self.model = nn.DataParallel(self.model, device_ids=self.devices).to(
            self.main_device
        )

        hyper_parameters["batch_size"] = (
            int(len(self.devices)) * hyper_parameters["batch_size"]
        )

        # Define a reusable loss criterion.
        self.criterion = LabelSmoothing().to(self.main_device)

        self.optimiser = torch.optim.AdamW(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=hyper_parameters["learning_rate"],
            amsgrad=False,
        )

        # Create a gradient scaler for Automatic Mixed Precision (AMP).
        # Note: the default value for init_scale is 2.0**16. This should be set
        # to the largest value that does not cause NaNs in the model's output
        # due to FP16 overflow.
        if hyper_parameters["use_amp"]:
            self.grad_scaler = GradScaler(
                init_scale=2.0 ** 12,
                growth_interval=np.iinfo(np.int64).max,
                growth_factor=1.000001,
                backoff_factor=0.999999,
            )
        else:
            self.grad_scaler = EmptyScaler()

        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimiser, hyper_parameters["lr_scheduler_exp_gamma"]
        # )

        # Define objects to keep track of smooth metrics.
        self.train_loss_metric = MovingAverageMetric()
        self.train_auc_metric = MovingAverageMetric()
        self.validation_loss_metric = MovingAverageMetric()
        self.validation_auc_metric = MovingAverageMetric()

        self.model_checkpoint_dir = hyper_parameters["model_checkpoint_dir"]
        make_dir_if_not_exists(self.model_checkpoint_dir)

        self.hyper_parameters = hyper_parameters
        self.seed = hyper_parameters["seed"]
        self.epoch = 0
        self.model_name = hyper_parameters["model_name"]
        self.training_run_name = f"{self.model_name}_{round(time.time())}"
        self.n_epochs = hyper_parameters["n_epochs"]
        self.best_val_epoch = 0
        self.best_val_loss = np.inf
        self.start_epoch = 1

        # Create the train and validation data loaders.
        (
            self.train_data_loader,
            self.val_data_loader,
        ) = make_train_and_validation_data_loaders(hyper_parameters)

        self.batch_size = int(
            hyper_parameters["batch_size"] * len(self.devices)
        )
        self.total_train_batches = len(self.train_data_loader)
        self.total_val_batches = len(self.val_data_loader)

        self.stat_freq = int(
            self.total_train_batches
            / hyper_parameters["log_tensorboard_n_times_per_epoch"]
        )

        # Check if we should continue training the model
        if hyper_parameters["trained_model_path"]:
            checkpoint = torch.load(hyper_parameters["trained_model_path"])
            self.model.module.load_state_dict(checkpoint["state_dict"])
            # self.scheduler.load_state_dict(checkpoint["scheduler"])
            # self.optimiser.load_state_dict(checkpoint["optimiser"])
            self.start_epoch = checkpoint["epoch"]
            self.best_val_epoch = checkpoint["best_val_epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
            print(f"Loaded weights and continuing to train model.")

        self.writer = SummaryWriter(
            logdir=f"{hyper_parameters['tensorboard_log_dir']}/{self.training_run_name}/"
        )

        print(f"Training run name: {self.training_run_name}")

    def train(self) -> None:
        """
        Full training logic
        """
        # # Baseline random feature performance.
        # with torch.no_grad():
        #     self._valid_epoch(epoch=self.start_epoch)

        for epoch in range(self.start_epoch, self.n_epochs + 1):
            self.epoch = epoch
            # lr = self.scheduler.get_last_lr()
            lr = self.hyper_parameters["learning_rate"]
            print(f" Epoch: {epoch}, LR: {lr}")

            self._train_epoch(epoch=epoch)

            self._save_checkpoint(
                epoch=epoch, filename=self.training_run_name + "_checkpoint"
            )
            # self.scheduler.step()

            with torch.no_grad():
                validation_loss = self._valid_epoch(epoch)

            if validation_loss <= self.best_val_loss:
                print(
                    f"Saving the best val model with loss: {validation_loss}"
                )
                self.best_val_loss = validation_loss
                self.best_val_epoch = epoch
                self._save_checkpoint(
                    epoch=epoch,
                    filename=self.training_run_name + "_best_val_checkpoint",
                )
            else:
                print(
                    f"Current best val model with loss: {self.best_val_loss} at epoch {self.best_val_epoch}"
                )

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        auc_meter = WeightedAUCMeter()
        # loss_criterion = nn.CrossEntropyLoss()

        for batch_idx, (images, targets) in tqdm(
            enumerate(self.train_data_loader),
            desc="Training model",
            total=self.total_train_batches,
        ):
            images = images.to(self.main_device).float()
            targets = targets.to(self.main_device).float()

            self.optimiser.zero_grad()

            if self.hyper_parameters["use_amp"]:
                with autocast():
                    outputs = self.model(images)
                    # Make sure there are no NaN values introduced by loss
                    # scaling.
                    assert not np.isnan(outputs.cpu().detach().numpy()).any()
                    # _, labels = targets.max(dim=1)
                    loss = self.criterion(outputs, targets)
                # Backpropagate and optimise using AMP.
                # print(self.grad_scaler.get_scale())
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimiser)
                self.grad_scaler.update()
            else:
                outputs = self.model(images)
                # _, labels = targets.max(dim=0)
                loss = self.criterion(outputs, targets)
                # Backpropagate.
                loss.backward()

            # Update the metrics.
            auc_meter.update(targets, outputs)
            self.train_loss_metric.update(loss.detach().item())
            self.train_auc_metric.update(auc_meter.avg)

            self.optimiser.step()

            gc.collect()

            if batch_idx % self.stat_freq == 0:
                start_iter = epoch * self.total_train_batches * len(self.devices)
                self.writer.add_scalar(
                    tag="train/loss",
                    scalar_value=self.train_loss_metric.moving_average,
                    global_step=start_iter + batch_idx,
                )
                self.writer.add_scalar(
                    tag="train/weighted_auc",
                    scalar_value=self.train_auc_metric.moving_average,
                    global_step=start_iter + batch_idx,
                )
        print(
            f"Train AUC: {self.train_auc_metric.moving_average}, "
            f"Train loss: {self.train_loss_metric.moving_average}\n"
        )

    def _valid_epoch(self, epoch: int) -> float:
        print("\nPerforming validation...")
        self.model.eval()
        auc_meter = WeightedAUCMeter()

        with torch.no_grad():
            for batch_idx, (images, targets) in tqdm(
                enumerate(self.val_data_loader),
                desc="Validating model",
                total=self.total_val_batches,
            ):
                images = images.to(self.main_device).float()
                targets = targets.to(self.main_device).float()

                if self.hyper_parameters["use_amp"]:
                    with autocast():
                        outputs = self.model(images)
                        assert not np.isnan(
                            outputs.cpu().detach().numpy()
                        ).any()
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                # Update the metrics.
                auc_meter.update(targets, outputs)
                self.validation_loss_metric.update(loss.detach().item())
                self.validation_auc_metric.update(auc_meter.avg)

                gc.collect()

        print(
            f"Validation AUC: {self.validation_auc_metric.moving_average}, "
            f"Validation loss: {self.validation_loss_metric.moving_average}\n"
        )
        self.writer.add_scalar(
            tag="validation/loss",
            scalar_value=self.validation_loss_metric.moving_average,
            global_step=epoch,
        )
        self.writer.add_scalar(
            tag="validation/weighted_auc",
            scalar_value=self.validation_auc_metric.moving_average,
            global_step=epoch,
        )

        return float(self.validation_loss_metric.moving_average)

    def _save_checkpoint(self, epoch: float, filename: str) -> None:
        state = {
            "epoch": epoch,
            "state_dict": self.model.module.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            # "scheduler": self.scheduler.state_dict(),
            "hyper_parameters": self.hyper_parameters,
            "best_val_loss": self.best_val_loss,
            "best_val_epoch": self.best_val_epoch,
        }
        filename = os.path.join(self.model_checkpoint_dir, f"{filename}.pth")
        print("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)
