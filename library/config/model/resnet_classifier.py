"""ResNet classification model scenario fragments."""

from __future__ import annotations

from nexuml.core.components import LayerDefinition
from nexuml.core.types import LayerSpec, PipelineSpec

from ...layers.head.classification import ClassificationHead
from ...layers.loss.cross_entropy import CrossEntropyLoss
from ...layers.metrics.classification_metrics import ClassificationMetrics
from ...layers.model.resnet import ResNetEncoder
from ...layers.utils.pooling import GlobalAveragePooling


def resnet_classifier(
    label_key: str = "class",
    encoder_width: int = 32,
    encoder_depth: int = 2,
    pooling: LayerDefinition | None = None,
    head_dropout: float = 0.0,
) -> PipelineSpec:
    """Create a PipelineSpec for a staged ResNet image classifier."""
    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    component=ResNetEncoder(width=encoder_width, depth=encoder_depth),
                    keys_in=["features"],
                    keys_out=["embeddings"],
                ),
            ],
            "Pooling": [
                LayerSpec(
                    component=pooling or GlobalAveragePooling(),
                    keys_in=["embeddings"],
                    keys_out=["pooled_embeddings"],
                ),
            ],
            "Head": [
                LayerSpec(
                    component=ClassificationHead(dropout=head_dropout),
                    keys_in=["pooled_embeddings"],
                    keys_out=["class_logits"],
                ),
            ],
            "Loss": [
                LayerSpec(
                    component=CrossEntropyLoss(),
                    keys_in=["class_logits"],
                    keys_out=["classification_loss"],
                    label_key=label_key,
                )
            ],
            "Metrics": [
                LayerSpec(
                    component=ClassificationMetrics(),
                    keys_in=["class_logits"],
                    keys_out=["accuracy", "f1"],
                    label_key=label_key,
                )
            ],
        }
    )
