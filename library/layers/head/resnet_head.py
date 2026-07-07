"""ResNet classification head layer."""

from __future__ import annotations

from typing import Any

import torch
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import layer


@layer("LatentClassificationHead")
class LatentClassificationHead(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        num_classes: int = 10,
        softmax: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            num_classes=num_classes,
            **kwargs,
        )
        self.classifier = torch.nn.Linear(input_sizes[keys_in[0]][0], num_classes)
        self.softmax = softmax

    def forward_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.classifier(x)
        if self.softmax:
            return torch.nn.functional.softmax(x, dim=-1)
        return x
