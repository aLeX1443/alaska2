from alaska2.alaska_pytorch.config import EXPERIMENT_HYPER_PARAMETERS
from alaska2.alaska_pytorch.lib.trainer import Trainer


def run_stegoanalysis_experiment(experiment_number):
    hyper_parameters = EXPERIMENT_HYPER_PARAMETERS[experiment_number]

    trainer = Trainer(hyper_parameters=hyper_parameters)
    trainer.train()
