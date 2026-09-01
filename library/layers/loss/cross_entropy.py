"""Single-label multiclass classification loss."""

from __future__ import annotations

from typing import Any

import torch
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import layer


@layer("CrossEntropyLoss")
class CrossEntropyLoss(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        label_key: str = "class",
        **kwargs: Any,
    ):
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            label_key=label_key,
            **kwargs,
        )
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def forward_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is None:
            return x.sum(dim=-1) * 0
        return self.loss(x, y.long().reshape(-1))
