"""Super simple ResNet-style feature backbone."""

from __future__ import annotations

from typing import Any

import torch
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.discovery import layer


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
        )
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


@layer("ResNetEncoder")
class ResNetEncoder(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        num_classes: int = 10,
        width: int = 32,
        depth: int = 2,
        **kwargs: Any,
    ):
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            num_classes=num_classes,
            **kwargs,
        )

        first_key = keys_in[0]
        in_channels = input_sizes[first_key][0] if first_key in input_sizes else 1

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(width),
            torch.nn.ReLU(inplace=True),
        )
        self.blocks = torch.nn.ModuleList(ResidualBlock(width) for _ in range(depth))
        self.out_channels = width

    def forward_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x
