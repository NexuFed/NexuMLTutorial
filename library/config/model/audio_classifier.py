"""Shared audio classification pipeline."""

from __future__ import annotations

from nexuml.core.components import LayerDefinition
from nexuml.core.types import LayerSpec, PipelineSpec

from ...layers.head.classification import ClassificationHead
from ...layers.loss.cross_entropy import CrossEntropyLoss
from ...layers.metrics.classification_metrics import ClassificationMetrics


def audio_classifier(
    encoder: LayerDefinition,
    label_key: str = "class",
    head_dropout: float = 0.0,
) -> PipelineSpec:
    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    component=encoder,
                    keys_in=["waveform"],
                    keys_out=["embeddings"],
                )
            ],
            "Head": [
                LayerSpec(
                    component=ClassificationHead(dropout=head_dropout),
                    keys_in=["embeddings"],
                    keys_out=["class_logits"],
                )
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
