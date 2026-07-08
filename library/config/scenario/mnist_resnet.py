"""MNIST and FashionMNIST ResNet classification scenarios."""

from __future__ import annotations

from nexuml.core.discovery import scenario
from nexuml.core.types import (
    ScenarioSpec,
)

from ..data.mnist import mnist_data
from ..defaults import (
    default_callbacks,
    default_checkpoint,
    default_evaluation,
    default_exports,
    default_logging,
    default_training,
    default_tuning,
)
from ..model import resnet_classifier


@scenario("mnist-resnet")
def mnist_resnet(
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
    encoder_width: int = 32,
    encoder_depth: int = 2,
    pooling_type: str = "GlobalAveragePooling",
    head_dropout: float = 0.0,
) -> ScenarioSpec:
    """MNIST image classification with ResNet backbone.

    Returns:
        ScenarioSpec: Assembled scenario with pipeline, training, data and evaluation.
    """
    name = "mnist_resnet"
    return ScenarioSpec(
        name=name,
        pipeline=resnet_classifier(
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            pooling_type=pooling_type,
            head_dropout=head_dropout,
        ),
        training=default_training(max_epochs=max_epochs, batch_size=batch_size, lr=lr),
        data=mnist_data(download=True, root="data"),
        evaluation=default_evaluation(
            feature_key="pooled_embeddings", label_key="class"
        ),
        logging=default_logging(name=name),
        callbacks=default_callbacks(),
        tuning=default_tuning(),
        checkpoint=default_checkpoint(),
        exports=default_exports(),
    )
