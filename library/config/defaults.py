LOG_FOLDER = "logs"

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
    SchedulerSpec,
    TensorBoardSpec,
    TrainingSpec,
    TuningSpec,
)


def default_training(
    max_epochs: int = 10, batch_size: int = 64, lr=1e-3
) -> TrainingSpec:
    return TrainingSpec(
        optimizer=OptimizerSpec(type="torch.optim.Adam", params={"lr": lr}),
        scheduler=SchedulerSpec(
            type="torch.optim.lr_scheduler.ConstantLR",
            params={"factor": 1.0, "total_iters": 0},
        ),
        loss_keys={"classification_loss": 1.0},
        metric_keys=["accuracy", "f1"],
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
    )


def default_evaluation(
    feature_key: str = "pooled_embeddings", label_key: str = "class"
) -> EvaluationSpec:
    return EvaluationSpec(
        metrics=["accuracy", "f1"],
        algorithms=[
            EvalAlgorithmSpec(
                type="tsne_visualizer",
                params={
                    "feature_key": feature_key,
                    "label_key": label_key,
                    "max_samples": 1_000,
                },
            ),
        ],
        test_result_metrics="all",
    )


def default_logging(name: str) -> LoggingSpec:
    return LoggingSpec(
        tensorboard=TensorBoardSpec(log_dir=f"{LOG_FOLDER}/tensorboard"),
        mlflow=MLflowSpec(
            tracking_uri=f"sqlite:///./{LOG_FOLDER}/mlflow/mlflow.db",
            experiment_name=name,
            log_model=False,
        ),
        dvclive=None,
        diagram=DiagramSpec(depth=2, output_dir=f"{LOG_FOLDER}/diagrams"),
    )


def default_callbacks() -> list[CallbackSpec]:
    return [
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
                "dirpath": f"{LOG_FOLDER}/checkpoints/cifar-resnet",
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
    ]


def default_tuning(
    metric_key: str = "val/loss", direction: str = "minimize"
) -> TuningSpec:
    return TuningSpec(
        n_trials=2,
        directions=["minimize"],
        metric_key=metric_key,
        storage=f"{LOG_FOLDER}/optuna/optuna.log",
        prune=False,
    )


def default_checkpoint() -> CheckpointLoadSpec:
    return CheckpointLoadSpec(
        source=None,
        allow_missing=True,
        allow_shape_mismatch=True,
        freeze_loaded=False,
    )


def default_exports() -> list[ExportSpec]:
    return [ExportSpec(kind="train_package", output="logs/models/mnist_resnet")]
