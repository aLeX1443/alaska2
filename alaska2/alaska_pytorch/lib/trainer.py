from typing import Dict, Tuple
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
    from apex import amp
except ImportError:
    print("Could not import NVIDIA Apex")

from tqdm import tqdm

from alaska2.alaska_pytorch.lib.utils import (
    make_dir_if_not_exists,
    LabelSmoothing,
)
from alaska2.alaska_pytorch.lib.metrics import (
    MovingAverageMetric,
    WeightedAUCMeter,
    BatchAverage,
)


class Trainer:
    def __init__(
        self, hyper_parameters: Dict, train_data_loader, val_data_loader
    ) -> None:
        if not torch.cuda.is_available():
            raise ValueError("GPU not available.")
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.devices = hyper_parameters["devices"]
        self.main_device = "cuda:0"

        # Build the model.
        self.model = hyper_parameters["model"](
            n_classes=hyper_parameters["n_classes"]
        ).to(self.main_device)

        # self.model = apex.parallel.convert_syncbn_model(self.model)

        # Define a reusable loss criterion.
        self.criterion = LabelSmoothing().to(self.main_device)

        self.optimiser = torch.optim.RMSprop(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=hyper_parameters["learning_rate"],
        )

        # Enable Automatic Mixed Precision (AMP).
        if hyper_parameters["use_amp"]:
            self.model, self.optimiser = amp.initialize(
                self.model, self.optimiser, opt_level="O1", loss_scale=2 ** 10
            )

        # Synchronise Batch Normalisation across devices.
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Distribute over all GPUs.
        self.model = nn.DataParallel(self.model, device_ids=self.devices).to(
            self.main_device
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimiser, hyper_parameters["lr_scheduler_exp_gamma"]
        )

        # Define objects to keep track of smooth metrics.
        self.train_auc_metric = MovingAverageMetric(window_size=1)
        self.train_loss_metric = MovingAverageMetric(window_size=500)

        self.model_checkpoint_dir = hyper_parameters["model_checkpoint_dir"]
        make_dir_if_not_exists(self.model_checkpoint_dir)

        self.hyper_parameters = hyper_parameters
        self.input_data_type = hyper_parameters["input_data_type"]
        self.use_quality_factor = hyper_parameters["use_quality_factor"]
        self.separate_classes_by_quality_factor = hyper_parameters[
            "separate_classes_by_quality_factor"
        ]
        self.seed = hyper_parameters["seed"]
        self.epoch = 0
        self.model_name = hyper_parameters["model_name"]
        self.training_run_name = f"{self.model_name}_{round(time.time())}"
        self.n_epochs = hyper_parameters["n_epochs"]
        self.best_auc_epoch = 0
        self.best_auc_score = -np.inf
        self.best_val_epoch = 0
        self.best_val_loss = np.inf
        self.start_epoch = 1

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
            pre_trained_state_dict = checkpoint["state_dict"]
            if (
                "cifar10" in hyper_parameters["trained_model_path"]
                or "imagenet"
                in hyper_parameters["trained_model_path"]  # or True
            ):
                missing_keys = [
                    "_fc.weight",
                    "_fc.bias",
                ]
                for key in missing_keys:
                    pre_trained_state_dict.pop(key)
            else:
                try:
                    self.optimiser.load_state_dict(checkpoint["optimiser"])
                except ValueError as e:
                    print("Could not load optimiser state dict")
                    print(e)
            try:
                self.model.module.load_state_dict(
                    pre_trained_state_dict, strict=False
                )
            except Exception as e:
                print(e)
                self.model.load_state_dict(
                    pre_trained_state_dict, strict=False
                )
            # self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = checkpoint["epoch"] + 1
            try:
                self.best_auc_epoch = checkpoint["best_auc_epoch"]
                self.best_auc_score = checkpoint["best_auc_score"]
            except KeyError:
                pass
            self.best_val_epoch = checkpoint["best_val_epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
            print(f"Loaded weights and continuing to train model.")

        make_dir_if_not_exists(hyper_parameters["tensorboard_log_dir"])
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
            lr = self.scheduler.get_last_lr()
            # lr = self.hyper_parameters["learning_rate"]
            print(f" Epoch: {epoch}, LR: {lr}")

            self._train_epoch(epoch=epoch)

            self._save_checkpoint(
                epoch=epoch, filename=self.training_run_name + "_checkpoint"
            )
            self.scheduler.step()

            with torch.no_grad():
                validation_auc, validation_loss = self._valid_epoch(epoch)

            if validation_auc >= self.best_auc_score:
                print(
                    f"Saving the best AUC model with score: {validation_auc}"
                )
                self.best_auc_score = validation_auc
                self.best_auc_epoch = epoch
                self._save_checkpoint(
                    epoch=epoch,
                    filename=self.training_run_name + "_best_auc_checkpoint",
                )
            else:
                print(
                    f"Current best AUC model with score: {self.best_auc_score} at epoch {self.best_auc_epoch}"
                )

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        auc_meter = WeightedAUCMeter()
        loss_meter = BatchAverage()

        for batch_idx, input_batch in tqdm(
            enumerate(self.train_data_loader),
            desc="Training model",
            total=self.total_train_batches,
        ):
            model_input, targets = self.load_batch(input_batch)

            self.optimiser.zero_grad()

            outputs = self.model(*model_input)

            # Make sure there are no NaN values introduced by loss
            # scaling.
            if np.isnan(outputs.cpu().detach().numpy()).any():
                print("Encountered NaN in output, skipping batch")
                continue

            loss = self.criterion(outputs, targets)
            # Backpropagate.
            loss.backward()

            self.optimiser.step()

            # Update the batch metrics.
            auc_meter.update(targets, outputs)
            loss_meter.update(loss.detach().item())

            # Update the moving average metrics that will be logged in
            # TensorBoard.
            self.train_auc_metric.update(auc_meter.avg)
            self.train_loss_metric.update(loss.detach().item())

            gc.collect()

            if batch_idx % self.stat_freq == 0:
                start_iter = (
                    epoch * self.total_train_batches * len(self.devices)
                )
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
            f"Train AUC: {auc_meter.avg}, " f"Train loss: {loss_meter.avg}\n"
        )

    def _valid_epoch(self, epoch: int) -> Tuple[float, float]:
        print("\nPerforming validation...")
        self.model.eval()
        auc_meter = WeightedAUCMeter()
        loss_meter = BatchAverage()

        with torch.no_grad():
            for batch_idx, input_batch in tqdm(
                enumerate(self.val_data_loader),
                desc="Validating model",
                total=self.total_val_batches,
            ):
                model_input, targets = self.load_batch(input_batch)

                outputs = self.model(*model_input)

                # Make sure there are no NaN values introduced by loss
                # scaling.
                if np.isnan(outputs.cpu().detach().numpy()).any():
                    print("Encountered NaN in output, skipping batch")
                    continue

                loss = self.criterion(outputs, targets)

                # Update the metrics.
                auc_meter.update(targets, outputs)

                loss_meter.update(loss.detach().item())
                gc.collect()

        print(
            f"Validation AUC: {auc_meter.avg}, "
            f"Validation loss: {loss_meter.avg}\n"
        )
        self.writer.add_scalar(
            tag="validation/loss",
            scalar_value=loss_meter.avg,
            global_step=epoch,
        )
        self.writer.add_scalar(
            tag="validation/weighted_auc",
            scalar_value=auc_meter.avg,
            global_step=epoch,
        )

        return (
            float(auc_meter.avg),
            float(loss_meter.avg),
        )

    def load_batch(self, input_batch):
        if self.input_data_type in ["RGB", "YCbCr"]:
            if self.use_quality_factor:
                images = input_batch[0].to(self.main_device).float()
                quality_factor = input_batch[1].to(self.main_device).float()
                targets = input_batch[2].to(self.main_device).long()
                model_input = (images, quality_factor)
            else:
                images = input_batch[0].to(self.main_device).float()
                targets = input_batch[1].to(self.main_device).long()
                model_input = (images,)
        else:
            # Load the DCT data.
            images = (
                input_batch[0].to(self.main_device).float(),
                input_batch[1].to(self.main_device).float(),
                input_batch[2].to(self.main_device).float(),
            )
            targets = input_batch[3].to(self.main_device).float()
            model_input = images
        return model_input, targets

    def _save_checkpoint(self, epoch: float, filename: str) -> None:
        state = {
            "epoch": epoch,
            "state_dict": self.model.module.state_dict()
            if len(self.devices) > 1
            else self.model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            # "scheduler": self.scheduler.state_dict(),
            "hyper_parameters": self.hyper_parameters,
            "best_auc_score": self.best_auc_score,
            "best_auc_epoch": self.best_auc_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_epoch": self.best_val_epoch,
            # "amp": amp.state_dict(),
        }
        filename = os.path.join(self.model_checkpoint_dir, f"{filename}.pth")
        print("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)
