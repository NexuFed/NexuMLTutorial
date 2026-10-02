"""A new FashionMNIST task initialized from the MNIST encoder only."""

from nexuml.core.discovery import scenario
from nexuml.core.types import CheckpointLoadSpec, ScenarioSpec

from ..data.mnist import mnist_data
from ..defaults import default_callbacks, default_exports, default_logging
from .mnist_resnet import mnist_resnet


@scenario("tutorial-fashion-mnist-transfer")
def fashion_mnist_transfer(
    source: str | None = None,
    freeze_loaded: bool = True,
    lr: float = 1e-3,
    max_epochs: int = 1,
    batch_size: int = 64,
    encoder_width: int = 32,
    encoder_depth: int = 2,
) -> ScenarioSpec:
    spec = mnist_resnet(
        lr=lr,
        max_epochs=max_epochs,
        batch_size=batch_size,
        encoder_width=encoder_width,
        encoder_depth=encoder_depth,
    )
    spec.name = "fashion_mnist_transfer"
    spec.data = mnist_data(root="data/fashion_mnist", num_workers=0, fashion=True)
    spec.checkpoint = CheckpointLoadSpec(
        source=source,
        include=["stages.Encoder.*"],
        allow_missing=True,
        allow_shape_mismatch=False,
        freeze_loaded=freeze_loaded,
    )
    spec.evaluation.algorithms = []
    spec.logging = default_logging(spec.name)
    spec.callbacks = default_callbacks(spec.name)
    spec.exports = default_exports(spec.name)
    return spec
