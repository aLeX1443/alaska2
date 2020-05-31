import gc
import os
import numpy as np
import time
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.optim
import torch.cuda

# from torch.cuda.amp import GradScaler, autocast
import torch.nn.parallel as parallel

from tqdm import tqdm

from alaska_pytorch.lib.utils import make_dir_if_not_exists
from alaska_pytorch.lib.data_loaders import (
    make_train_and_validation_data_loaders,
)
from alaska_pytorch.lib.metrics import (
    MovingAverageMetric,
    binary_accuracy_from_model_output_and_target,
)


class Trainer:
    def __init__(self, hyper_parameters):
        if not torch.cuda.is_available():
            raise ValueError("GPU not available.")

        self.device = hyper_parameters["device"]

        # Copy the network to a GPU.
        self.model = hyper_parameters["model"](n_classes=1).to(self.device)

        self.optimiser = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=hyper_parameters["learning_rate"],
        )

        # # Create a gradient scaler for Automatic Mixed Precision (AMP).
        # self.grad_scaler = GradScaler()

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimiser, hyper_parameters["lr_scheduler_exp_gamma"]
        )

        self.model_checkpoint_dir = hyper_parameters["model_checkpoint_dir"]
        make_dir_if_not_exists(self.model_checkpoint_dir)

        self.hyper_parameters = hyper_parameters
        self.seed = hyper_parameters["seed"]
        self.epoch = 0
        self.model_name = hyper_parameters["model_name"]
        self.training_run_name = f"{self.model_name}_{round(time.time())}"
        self.n_epochs = hyper_parameters["n_epochs"]
        self.val_epoch_freq = hyper_parameters["val_epoch_freq"]
        self.best_val_epoch = 0
        self.best_val_loss = np.inf
        self.start_epoch = 1

        # Create the train and validation data loaders.
        (
            self.train_data_loader,
            self.val_data_loader,
            self.train_sample_weights,
            self.validation_sample_weights,
        ) = make_train_and_validation_data_loaders(self.hyper_parameters)

        self.batch_size = self.train_data_loader.batch_size
        self.total_train_batches = len(self.train_data_loader)
        self.total_val_batches = len(self.val_data_loader)

        self.stat_freq = int(
            self.total_train_batches
            / hyper_parameters["log_tensorboard_n_times_per_epoch"]
        )

        # Check if we should continue training the model
        if hyper_parameters["trained_model_path"]:
            checkpoint = torch.load(hyper_parameters["trained_model_path"])
            self.model.load_state_dict(checkpoint["state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])
            self.start_epoch = checkpoint["epoch"]
            self.best_val_epoch = checkpoint["best_val_epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
            print(f"Loaded weights and continuing to train model.")

        self.writer = SummaryWriter(
            logdir=f"{hyper_parameters['tensorboard_log_dir']}/{self.training_run_name}/"
        )

        # print(self.model)

        print(f"Training run name: {self.training_run_name}")

    def train(self):
        """
        Full training logic
        """
        # Baseline random feature performance.
        with torch.no_grad():
            self._valid_epoch(epoch=0)

        for epoch in range(self.start_epoch, self.n_epochs + 1):
            self.epoch = epoch
            lr = self.scheduler.get_last_lr()
            print(f" Epoch: {epoch}, LR: {lr}")

            self._train_epoch(epoch=epoch)

            self._save_checkpoint(
                epoch=epoch, filename=self.training_run_name + "_checkpoint"
            )
            self.scheduler.step()

            if epoch % self.val_epoch_freq == 0:
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
                        filename=self.training_run_name
                        + "_best_val_checkpoint",
                    )
                else:
                    print(
                        f"Current best val model with loss: {self.best_val_loss} at epoch {self.best_val_epoch}"
                    )

    def _train_epoch(self, epoch):
        self.model.train()
        loss_criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.train_sample_weights.to(self.device)
        )

        loss_metric = MovingAverageMetric()
        accuracy_metric = MovingAverageMetric()

        for batch_idx, train_dict in tqdm(
            enumerate(self.train_data_loader),
            desc="Training model",
            total=self.total_train_batches,
        ):
            self.optimiser.zero_grad()

            input_batch = train_dict[0]["image"].to(self.device)
            target_batch = train_dict[1].unsqueeze(1).float().to(self.device)

            # with autocast():
            output = self.model(input_batch)
            loss = loss_criterion(output, target_batch)

            # # TODO for multi-GPU use: https://pytorch.org/docs/master/notes/amp_examples.html#working-with-multiple-gpus
            # # Backpropagate and optimise using AMP.
            # self.grad_scaler.scale(loss).backward()
            # self.grad_scaler.step(self.optimiser)
            # self.grad_scaler.update()

            # Backpropagate and optimise.
            loss.backward()
            self.optimiser.step()

            # Add to the metrics.
            loss_metric.add_value(loss.item())
            accuracy_metric.add_value(
                binary_accuracy_from_model_output_and_target(
                    output, target_batch
                )
            )

            gc.collect()

            # TODO needs to be changed for multi-GPU settings
            if batch_idx % self.stat_freq == 0:
                start_iter = (epoch - 1) * self.total_train_batches

                current_loss = loss_metric.get_moving_average_point()
                current_accuracy = accuracy_metric.get_moving_average_point()

                print(f"Accuracy: {current_accuracy}, loss: {current_loss}")

                self.writer.add_scalar(
                    tag="train/loss",
                    scalar_value=current_loss,
                    global_step=start_iter + batch_idx,
                )
                self.writer.add_scalar(
                    tag="train/accuracy",
                    scalar_value=current_accuracy,
                    global_step=start_iter + batch_idx,
                )

    def _valid_epoch(self, epoch):
        print("\nPerforming validation...")
        self.model.eval()
        loss_criterion = torch.nn.BCEWithLogitsLoss()

        total_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, val_dict in tqdm(
                enumerate(self.val_data_loader),
                desc="Validating model",
                total=self.total_val_batches,
            ):
                input_batch = val_dict[0]["image"].to(self.device)
                target_batch = val_dict[1].unsqueeze(1).float().to(self.device)

                # with autocast():
                output = self.model(input_batch)
                loss = loss_criterion(output, target_batch)

                pred = torch.sigmoid(output) >= 0.5
                batch_correct = (
                    pred.eq(target_batch.view_as(pred)).sum().item()
                )

                correct += batch_correct
                total_loss += loss.item()

                gc.collect()

        total_loss /= self.total_val_batches
        accuracy = correct / (self.total_val_batches * self.batch_size)

        print(
            f"Validation loss: {total_loss}, "
            f"Validation accuracy: {round(accuracy * 100, 2)}%\n"
        )
        self.writer.add_scalar(
            tag="validation/loss", scalar_value=total_loss, global_step=epoch
        )
        self.writer.add_scalar(
            tag="validation/accuracy", scalar_value=accuracy, global_step=epoch
        )

        return total_loss

    def _save_checkpoint(self, epoch, filename):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "hyper_parameters": self.hyper_parameters,
            "best_val_loss": self.best_val_loss,
            "best_val_epoch": self.best_val_epoch,
        }
        filename = os.path.join(self.model_checkpoint_dir, f"{filename}.pth")
        print("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)


def data_parallel(
    model, input_data, target, loss_criterions, device_ids, output_device=None
):
    if output_device is None:
        output_device = device_ids[0]

    replicas = parallel.replicate(model, device_ids)

    outputs = parallel.parallel_apply(
        modules=replicas, inputs=input_data, devices=device_ids
    )

    losses = parallel.parallel_apply(
        modules=loss_criterions,
        inputs=tuple(zip(outputs, target)),
        devices=device_ids,
    )

    loss = parallel.gather(losses, output_device).mean()
    output = parallel.gather(outputs, output_device)
    target = parallel.gather(target, output_device)

    return output, target, loss
