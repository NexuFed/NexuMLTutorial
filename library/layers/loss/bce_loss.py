"""Classification loss layer."""

from __future__ import annotations

from typing import Any

import torch
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import layer


@layer("BCELoss")
class BCELoss(PipelineLayer):
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
        self.loss = torch.nn.BCELoss()

    def forward_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is None:
            # No labels available (e.g. shape-propagation dummy pass) - output zero loss.
            return torch.zeros(x.shape[0], device=x.device, requires_grad=True)
        if y.ndim == 1:
            y = torch.nn.functional.one_hot(y.long(), num_classes=x.shape[-1])
        return self.loss(x, y.float()).expand(x.shape[0])
