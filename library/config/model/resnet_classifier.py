"""ResNet classification model scenario fragments."""

from __future__ import annotations

from nexuml.core.types import LayerSpec, PipelineSpec


def resnet_classifier(
    num_classes: int = 10,
    label_key: str = "class",
    encoder_width: int = 32,
    encoder_depth: int = 2,
    pooling_type: str = "GlobalAveragePooling",
    head_dropout: float = 0.0,
) -> PipelineSpec:
    """Create a PipelineSpec for a staged ResNet image classifier."""
    return PipelineSpec(
        stages={
            "Encoder": [
                LayerSpec(
                    type_key="ResNetEncoder",
                    keys_in=["features"],
                    keys_out=["embeddings"],
                    params={
                        "width": encoder_width,
                        "depth": encoder_depth,
                    },
                ),
            ],
            "Pooling": [
                LayerSpec(
                    type_key=pooling_type,
                    keys_in=["embeddings"],
                    keys_out=["pooled_embeddings"],
                    params={},
                ),
            ],
            "Head": [
                LayerSpec(
                    type_key="LatentClassificationHead",
                    keys_in=["pooled_embeddings"],
                    keys_out=["class_logits"],
                    params={
                        "num_classes": num_classes,
                        "softmax": True,
                        "dropout": head_dropout,
                    },
                ),
            ],
            "Loss": [
                LayerSpec(
                    type_key="BCELoss",
                    keys_in=["class_logits"],
                    keys_out=["classification_loss"],
                    params={
                        "label_key": label_key,
                    },
                )
            ],
            "Metrics": [
                LayerSpec(
                    type_key="ClassificationMetrics",
                    keys_in=["class_logits"],
                    keys_out=["accuracy", "f1"],
                    params={
                        "num_classes": num_classes,
                        "label_key": label_key,
                    },
                )
            ],
        }
    )
