"""Validation-only Optuna search using NexuML's public session lifecycle."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from lightning import seed_everything
from nexuml.core.types import ScenarioSpec
from nexuml.training.lightning import NexuSession

from library.config.defaults import default_callbacks, default_logging
from library.config.scenario.mnist_resnet import mnist_resnet


def scenario(
    lr: float = 1e-3,
    batch_size: int = 32,
    encoder_width: int = 32,
    max_epochs: int = 1,
    name: str = "mnist_validation",
    include_test: bool = False,
) -> ScenarioSpec:
    spec = mnist_resnet(
        lr=lr,
        batch_size=batch_size,
        encoder_width=encoder_width,
        max_epochs=max_epochs,
    )
    spec.name = name
    spec.data.train_split, spec.data.val_split, spec.data.test_split = 0.8, 0.2, 0.0
    if not include_test:
        spec.data.datasets = [spec.data.datasets[0]]
    spec.data.loader.num_workers = 0
    spec.evaluation.algorithms = []
    spec.exports = []
    spec.tuning = None
    spec.logging = default_logging(name)
    spec.callbacks = default_callbacks(name)
    return spec


def objective(trial, max_epochs: int, study_name: str) -> float:
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "encoder_width": trial.suggest_categorical("encoder_width", [16, 32, 64]),
    }
    seed_everything(42, workers=True)
    spec = scenario(
        **params, max_epochs=max_epochs, name=f"{study_name}_trial_{trial.number}"
    )
    spec.logging.mlflow.experiment_name = study_name
    session = NexuSession.from_scenario(spec).setup()
    for logger in session.trainer_loggers:
        logger.log_hyperparams(params)
    results = session.fit().validate()
    if not results or "val/loss" not in results[0]:
        raise ValueError(
            "Expected val/loss from NexuSession.validate(); no test fallback."
        )
    loss = float(results[0]["val/loss"])
    if not math.isfinite(loss):
        raise ValueError(f"Validation objective is not finite: {loss}")
    for logger in session.trainer_loggers:
        logger.finalize("success")
    return loss


def tune(n_trials: int, max_epochs: int, study_name: str) -> None:
    import optuna

    if n_trials < 1 or max_epochs < 1:
        raise ValueError("n_trials and max_epochs must be positive")
    Path("logs/optuna").mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///logs/optuna/mnist.db",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective(trial, max_epochs, study_name), n_trials=n_trials
    )
    print(
        f"Best val/loss (minimize): {study.best_value}; parameters: {study.best_params}"
    )

    # Only the chosen configuration sees the official held-out test source.
    seed_everything(42, workers=True)
    selected = scenario(
        **study.best_params,
        max_epochs=max_epochs,
        name=f"{study_name}_selected",
        include_test=True,
    )
    selected.logging.mlflow.experiment_name = study_name
    session = NexuSession.from_scenario(selected).setup().fit()
    print("Selected model test results:", session.test())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--study-name", default="mnist-validation")
    args = parser.parse_args()
    tune(args.n_trials, args.max_epochs, args.study_name)


if __name__ == "__main__":
    main()
