"""Shared audio classification pipeline."""

from __future__ import annotations

from typing import Any

from nexuml.core.types import LayerSpec, PipelineSpec


def audio_classifier(
    encoder_type: str,
    num_classes: int = 8,
    label_key: str = "class",
    head_dropout: float = 0.0,
    encoder_params: dict[str, Any] | None = None,
) -> PipelineSpec:
    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    type_key=encoder_type,
                    keys_in=["waveform"],
                    keys_out=["embeddings"],
                    params=encoder_params or {},
                )
            ],
            "Head": [
                LayerSpec(
                    type_key="ClassificationHead",
                    keys_in=["embeddings"],
                    keys_out=["class_logits"],
                    params={"num_classes": num_classes, "dropout": head_dropout},
                )
            ],
            "Loss": [
                LayerSpec(
                    type_key="CrossEntropyLoss",
                    keys_in=["class_logits"],
                    keys_out=["classification_loss"],
                    params={"label_key": label_key},
                )
            ],
            "Metrics": [
                LayerSpec(
                    type_key="ClassificationMetrics",
                    keys_in=["class_logits"],
                    keys_out=["accuracy", "f1"],
                    params={"num_classes": num_classes, "label_key": label_key},
                )
            ],
        }
    )
