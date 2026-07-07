"""MNIST and FashionMNIST ResNet classification scenarios."""

from __future__ import annotations

from nexuml.core.discovery import scenario
from nexuml.core.types import (
    CallbackSpec,
    CheckpointLoadSpec,
    DiagramSpec,
    EvalAlgorithmSpec,
    EvaluationSpec,
    ExportSpec,
    LoggingSpec,
    MLflowSpec,
    OptimizerSpec,
    ScenarioSpec,
    SchedulerSpec,
    TensorBoardSpec,
    TrainingSpec,
    TuningSpec,
)

from ..data.mnist import mnist_data
from ..model import resnet_classifier


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
        data=mnist_data(download=True, root="data"),
        evaluation=EvaluationSpec(
            metrics=["accuracy", "f1"],
            algorithms=[
                EvalAlgorithmSpec(
                    type="tsne_visualizer",
                    params={
                        "feature_key": "pooled_embeddings",
                        "label_key": "class",
                        "max_samples": 1_000,
                    },
                ),
            ],
            test_result_metrics="all",
        ),
        logging=LoggingSpec(
            tensorboard=TensorBoardSpec(log_dir="logs/tensorboard"),
            mlflow=MLflowSpec(
                tracking_uri="sqlite:///./logs/mlflow/mlflow.db",
                experiment_name=name,
                log_model=False,
            ),
            dvclive=None,
            diagram=DiagramSpec(depth=2, output_dir="logs/diagrams"),
        ),
        callbacks=[
            CallbackSpec(
                type="early_stopping",
                params={"monitor": "val/loss", "patience": 5},
            ),
            CallbackSpec(
                type="lr_monitor",
            ),
            CallbackSpec(
                type="checkpoint",
                params={
                    "dirpath": "logs/checkpoints/cifar-resnet",
                    "monitor": "val/loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "filename": "{epoch:02d}-{val_loss:.4f}",
                    "save_last": True,
                },
            ),
            CallbackSpec(
                type="rich_progress",
            ),
            CallbackSpec(
                type="device_stats",
            ),
        ],
        tuning=TuningSpec(
            n_trials=2,
            directions=["minimize"],
            metric_key="val/loss",
            storage="logs/optuna/optuna.log",
            prune=False,
        ),
        checkpoint=CheckpointLoadSpec(
            source=None,
            allow_missing=True,
            allow_shape_mismatch=True,
            freeze_loaded=False,
        ),
        exports=[ExportSpec(kind="train_package", output="logs/models/mnist_resnet")],
    )
