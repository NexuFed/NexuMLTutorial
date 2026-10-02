"""Pooling helpers for simple model heads."""

from __future__ import annotations

import torch
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("GlobalAveragePooling")
class GlobalAveragePooling(LayerDefinition):
    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _GlobalAveragePoolingRuntime(**context.runtime_kwargs())


class _GlobalAveragePoolingRuntime(PipelineLayer):
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


@layer("GlobalMaxPooling")
class GlobalMaxPooling(LayerDefinition):
    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _GlobalMaxPoolingRuntime(**context.runtime_kwargs())


class _GlobalMaxPoolingRuntime(PipelineLayer):
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
        self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))

    def forward_tensor(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.pool(x).flatten(1)
