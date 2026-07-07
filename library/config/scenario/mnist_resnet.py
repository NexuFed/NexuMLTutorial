"""MNIST and FashionMNIST ResNet classification scenarios."""

from __future__ import annotations

from nexuml.core.discovery import scenario
from nexuml.core.types import (
    EvaluationSpec,
    LoggingSpec,
    MLflowSpec,
    OptimizerSpec,
    ScenarioSpec,
    SchedulerSpec,
    TensorBoardSpec,
    TrainingSpec,
)

from library.config.data.mnist import mnist_data
from library.config.model import resnet_classifier


@scenario("mnist-resnet")
def mnist_resnet(
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
) -> ScenarioSpec:
    """MNIST image classification with ResNet backbone.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    name = "mnist_resnet"
    return ScenarioSpec(
        name=name,
        pipeline=resnet_classifier(),
        training=TrainingSpec(
            optimizer=OptimizerSpec(type="torch.optim.Adam", params={"lr": lr}),
            scheduler=SchedulerSpec(
                type="torch.optim.lr_scheduler.ConstantLR",
                params={"factor": 1.0, "total_iters": 0},
            ),
            loss_keys={"classification_loss": 1.0},
            metric_keys=[],
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
        ),
        data=mnist_data(download=True, root="data/mnist"),
        evaluation=EvaluationSpec(metrics=[], algorithms=[], test_result_metrics="all"),
        logging=LoggingSpec(
            tensorboard=TensorBoardSpec(log_dir="logs/tensorboard"),
            mlflow=MLflowSpec(
                tracking_uri="sqlite:///./logs/mlflow/mlflow.db",
                experiment_name=name,
                log_model=False,
            ),
            dvclive=None,
            experiment_name=name,
            run_name=None,
            log_system_metrics=False,
        ),
        callbacks=[],
        tuning=None,
        checkpoint=None,
        exports=[],
    )
