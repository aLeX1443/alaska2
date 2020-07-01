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
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    print("Could not load torch.cuda.amp")

from tqdm import tqdm

from alaska2.alaska_pytorch.lib.utils import (
    make_dir_if_not_exists,
    LabelSmoothing,
    EmptyScaler,
)
from alaska2.alaska_pytorch.lib.metrics import (
    MovingAverageMetric,
    WeightedAUCMeter,
    BatchAverage,
    top_k_accuracy,
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
        self.main_device = self.devices[0]

        # Build the model.
        self.model = hyper_parameters["model"](n_classes=1000)

        # Distribute over all GPUs.
        self.model = nn.DataParallel(self.model, device_ids=self.devices).to(
            self.main_device
        )

        hyper_parameters["batch_size"] = (
            int(len(self.devices)) * hyper_parameters["batch_size"]
        )

        # Define a reusable loss criterion.
        self.criterion = LabelSmoothing().to(self.main_device)

        self.optimiser = torch.optim.RMSprop(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=hyper_parameters["learning_rate"],
        )

        # Create a gradient scaler for Automatic Mixed Precision (AMP).
        # Note: the default value for init_scale is 2.0**16. This should be set
        # to the largest value that does not cause NaNs in the model's output
        # due to FP16 overflow.
        if hyper_parameters["use_amp"]:
            self.grad_scaler = GradScaler(
                init_scale=2.0 ** 4,
                growth_interval=np.iinfo(np.int64).max,
                growth_factor=1.000001,
                backoff_factor=0.999999,
            )
            # self.grad_scaler = GradScaler()
        else:
            self.grad_scaler = EmptyScaler()

        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimiser, hyper_parameters["lr_scheduler_exp_gamma"]
        # )

        # Define objects to keep track of smooth metrics.
        # TODO add top-1 and top-5 metrics
        self.train_auc_metric = MovingAverageMetric(window_size=200)
        self.train_top_1_acc_metric = MovingAverageMetric(window_size=200)
        self.train_top_5_acc_metric = MovingAverageMetric(window_size=200)
        self.train_loss_metric = MovingAverageMetric(window_size=200)

        self.model_checkpoint_dir = hyper_parameters["model_checkpoint_dir"]
        make_dir_if_not_exists(self.model_checkpoint_dir)

        self.hyper_parameters = hyper_parameters
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
            self.model.module.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            # self.scheduler.load_state_dict(checkpoint["scheduler"])
            try:
                self.optimiser.load_state_dict(checkpoint["optimiser"])
            except ValueError:
                pass
            self.start_epoch = checkpoint["epoch"] + 1
            try:
                self.best_auc_epoch = checkpoint["best_auc_epoch"]
                self.best_auc_score = checkpoint["best_auc_score"]
            except KeyError:
                pass
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
        top_1_acc_meter = BatchAverage()
        top_5_acc_meter = BatchAverage()

        for batch_idx, input_batch in tqdm(
            enumerate(self.train_data_loader),
            desc="Training model",
            total=self.total_train_batches,
        ):
            model_input, targets = self.load_batch(input_batch)

            self.optimiser.zero_grad()

            if self.hyper_parameters["use_amp"]:
                with autocast():
                    outputs = self.model(*model_input)
                    loss = self.criterion(outputs, targets)
                # Backpropagate and optimise using AMP.
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimiser)
                self.grad_scaler.update()
            else:
                with autocast(enabled=False):
                    outputs = self.model(*model_input)
                    loss = self.criterion(outputs, targets)
                    # Backpropagate.
                    loss.backward()

            self.optimiser.step()

            # Update the batch metrics.
            # top_1_acc = top_k_accuracy(targets, outputs, top_k=1)
            # top_5_acc = top_k_accuracy(targets, outputs, top_k=5)
            # Make sure there are no NaN values introduced by loss
            # scaling.
            if not np.isnan(outputs.cpu().detach().numpy()).any():
                auc_meter.update(targets, outputs)
            else:
                print("Encountered NaN in output. Not updating AUC metric.\n")
            # top_1_acc_meter.update(top_1_acc)
            # top_5_acc_meter.update(top_5_acc)
            loss_meter.update(loss.detach().item())

            # targets, outputs = get_argmax_from_prediction_and_target(
            #     targets, outputs
            # )

            # Update the moving average metrics that will be logged in
            # TensorBoard.
            self.train_auc_metric.update(auc_meter.avg)
            # self.train_top_1_acc_metric.update(top_1_acc)
            # self.train_top_5_acc_metric.update(top_5_acc)
            self.train_loss_metric.update(loss.detach().item())

            # print(
            #     f"Train AUC: {auc_meter.avg}, "
            #     f"Train Top1 Accuracy: {top_1_acc}"
            #     f"Train Top5 Accuracy: {top_5_acc}"
            #     f"Train loss: {loss.detach().item()}\n"
            # )

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
                # self.writer.add_scalar(
                #     tag="train/top_1_acc",
                #     scalar_value=self.train_top_1_acc_metric.moving_average,
                #     global_step=start_iter + batch_idx,
                # )
                # self.writer.add_scalar(
                #     tag="train/top_5_acc",
                #     scalar_value=self.train_top_1_acc_metric.moving_average,
                #     global_step=start_iter + batch_idx,
                # )
        print(
            f"Train metrics: "
            f"AUC: {auc_meter.avg}, "
            f"Loss: {loss_meter.avg}\n"
        )

    def _valid_epoch(self, epoch: int) -> Tuple[float, float]:
        print("\nPerforming validation...")
        self.model.eval()
        auc_meter = WeightedAUCMeter()
        loss_meter = BatchAverage()
        top_1_acc_meter = BatchAverage()
        top_5_acc_meter = BatchAverage()

        with torch.no_grad():
            for batch_idx, input_batch in tqdm(
                enumerate(self.val_data_loader),
                desc="Validating model",
                total=self.total_val_batches,
            ):
                model_input, targets = self.load_batch(input_batch)

                if self.hyper_parameters["use_amp"]:
                    with autocast():
                        outputs = self.model(*model_input)
                        loss = self.criterion(outputs, targets)
                else:
                    with autocast(enabled=False):
                        outputs = self.model(*model_input)
                        loss = self.criterion(outputs, targets)

                # Update the metrics.
                if not np.isnan(outputs.cpu().detach().numpy()).any():
                    # Make sure there are no NaN values introduced by loss
                    # scaling.
                    auc_meter.update(targets, outputs)
                else:
                    print("Encountered NaN in output. Not updating AUC metric.\n")
                loss_meter.update(loss.detach().item())
                # top_1_acc_meter.update(
                #     top_k_accuracy(targets, outputs, top_k=1)
                # )
                # top_5_acc_meter.update(
                #     top_k_accuracy(targets, outputs, top_k=5)
                # )

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
        # self.writer.add_scalar(
        #     tag="validation/top_1_acc",
        #     scalar_value=top_1_acc_meter.avg,
        #     global_step=epoch,
        # )
        # self.writer.add_scalar(
        #     tag="validation/top_5_acc",
        #     scalar_value=top_5_acc_meter.avg,
        #     global_step=epoch,
        # )

        return (
            float(auc_meter.avg),
            float(loss_meter.avg),
        )

    def load_batch(self, input_batch):
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
            "state_dict": self.model.module.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            # "scheduler": self.scheduler.state_dict(),
            "hyper_parameters": self.hyper_parameters,
            "best_auc_score": self.best_auc_score,
            "best_auc_epoch": self.best_auc_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_epoch": self.best_val_epoch,
        }
        filename = os.path.join(self.model_checkpoint_dir, f"{filename}.pth")
        print("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)
