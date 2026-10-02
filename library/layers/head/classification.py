"""Generic single-label classification head."""

from __future__ import annotations

from typing import Any

import torch
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("ClassificationHead")
class ClassificationHead(LayerDefinition):
    dropout: float = 0.0

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _ClassificationHeadRuntime(
            dropout=self.dropout,
            **context.runtime_kwargs(num_classes=context.num_classes or 10),
        )


class _ClassificationHeadRuntime(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        num_classes: int = 10,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            num_classes=num_classes,
            **kwargs,
        )
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.classifier = torch.nn.Linear(input_sizes[keys_in[0]][0], num_classes)

    def forward_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classifier(self.dropout(x))
