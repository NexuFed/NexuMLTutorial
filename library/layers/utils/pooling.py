"""Pooling helpers for simple model heads."""

from __future__ import annotations

import torch
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import layer


@layer("GlobalAveragePooling")
class GlobalAveragePooling(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        **kwargs,
    ):
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward_tensor(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.pool(x).flatten(1)
