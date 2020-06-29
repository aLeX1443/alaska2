from cifar10.config import EXPERIMENT_HYPER_PARAMETERS
from cifar10.lib.data_loaders import make_train_and_validation_data_loaders
from cifar10.lib.trainer import Trainer


def run_cifar10_experiment(experiment_number):
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    # Create the train and validation data loaders.
    (
        train_data_loader,
        val_data_loader,
    ) = make_train_and_validation_data_loaders(hyper_parameters)

    trainer = Trainer(
        hyper_parameters=hyper_parameters,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
    )
    trainer.train()
