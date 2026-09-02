"""Small raw-waveform audio encoders."""

from __future__ import annotations

from typing import Any

import torch
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("AudioCNNEncoder")
class AudioCNNEncoder(LayerDefinition):
    embedding_dim: int = 64

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _AudioCNNEncoderRuntime(**context.runtime_kwargs(), **self.model_dump())


class _AudioCNNEncoderRuntime(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        embedding_dim: int = 64,
        **kwargs: Any,
    ):
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            torch.nn.BatchNorm1d(32),
            torch.nn.GELU(),
            torch.nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU(),
            torch.nn.Conv1d(64, embedding_dim, kernel_size=9, stride=4, padding=4),
            torch.nn.BatchNorm1d(embedding_dim),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )

    def forward_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(x.unsqueeze(1)).flatten(1)


@layer("TinyAudioTransformerEncoder")
class TinyAudioTransformerEncoder(LayerDefinition):
    d_model: int = 64
    patch_size: int = 160
    num_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _TinyAudioTransformerEncoderRuntime(
            **context.runtime_kwargs(), **self.model_dump()
        )


class _TinyAudioTransformerEncoderRuntime(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        d_model: int = 64,
        patch_size: int = 160,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(
            input_sizes=input_sizes, keys_in=keys_in, keys_out=keys_out, **kwargs
        )
        input_samples = input_sizes[keys_in[0]][0]
        if patch_size <= 0 or input_samples < patch_size:
            raise ValueError(
                "patch_size must be positive and no larger than the waveform"
            )
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")

        token_count = input_samples // patch_size
        self.patch_embedding = torch.nn.Conv1d(
            1, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.position_embedding = torch.nn.Parameter(
            torch.zeros(1, token_count, d_model)
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.patch_embedding(x.unsqueeze(1)).transpose(1, 2)
        tokens = self.encoder(tokens + self.position_embedding)
        return self.norm(tokens).mean(dim=1)
