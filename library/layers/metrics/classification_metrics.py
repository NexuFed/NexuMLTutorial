"""Multiclass classification metrics layer."""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
import torch.nn as nn
from nexuml.core.base_layer import LightningMode, PipelineLayer
from nexuml.core.discovery import layer
from tensordict import TensorDict
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

_METRIC_BUILDERS = {
    "accuracy": lambda num_classes, average: MulticlassAccuracy(num_classes=num_classes),
    "f1": lambda num_classes, average: MulticlassF1Score(num_classes=num_classes, average=average),
}


@layer("ClassificationMetrics")
class ClassificationMetrics(PipelineLayer):
    """Accumulate multiclass accuracy/F1 metrics with torchmetrics.

    ``keys_out`` names which metrics to compute (e.g. ``["accuracy", "f1"]``)
    and doubles as the output TensorDict keys. Metrics are only tracked
    during validation/test (see ``lightning_mode``); training steps pass
    through unchanged.
    """

    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        num_classes: int,
        label_key: str = "class",
        average: str = "macro",
        **kwargs: Any,
    ):
        super().__init__(
            input_sizes=input_sizes,
            keys_in=keys_in,
            keys_out=keys_out,
            label_key=label_key,
            num_classes=num_classes,
            **kwargs,
        )
        self.average = cast(Literal["micro", "macro", "weighted", "none"], average)
        self.val_metrics = self._build_metrics()
        self.test_metrics = self._build_metrics()

    def _build_metrics(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                name: _METRIC_BUILDERS[name](self.num_classes, self.average)
                for name in self.keys_out
            }
        )

    def _active_metrics(self) -> nn.ModuleDict | None:
        if self.lightning_mode == LightningMode.VALIDATING:
            return self.val_metrics
        if self.lightning_mode == LightningMode.TESTING:
            return self.test_metrics
        return None

    def forward(
        self,
        x: TensorDict | torch.Tensor,
        y: TensorDict | None = None,
    ) -> tuple[TensorDict | torch.Tensor, TensorDict | None]:
        if not self.check_update():
            return x, y
        assert isinstance(x, TensorDict)

        if y is None or self.label_key not in y.keys():  # ty: ignore[unsupported-operator]
            zero = torch.tensor(0.0, device=x.device)
            for key in self.keys_out:
                x[key] = zero.expand(x.batch_size)
            return x, y

        metrics = self._active_metrics()
        if metrics is None:
            return x, y

        logits = cast(torch.Tensor, x[cast(list[str], self.keys_in)[0]])
        labels = self.get_label(x, y)
        preds = logits.reshape(-1, logits.shape[-1])
        labels = cast(torch.Tensor, labels).long().reshape(-1).to(preds.device)

        for key, metric in metrics.items():
            metric.to(preds.device)
            metric.update(preds, labels)  # ty: ignore[call-non-callable]
            x[key] = metric.compute().detach().expand(x.batch_size)  # ty: ignore[call-non-callable]
        return x, y

    def get_epoch_metrics(self, stage: str) -> dict[str, torch.Tensor]:
        metrics = (
            self.val_metrics if stage == "val" else self.test_metrics if stage == "test" else None
        )
        if metrics is None:
            return {}
        return {key: metric.compute().detach() for key, metric in metrics.items()}  # ty: ignore[call-non-callable]

    def on_validation_start(self):
        super().on_validation_start()
        for metric in self.val_metrics.values():
            metric.reset()  # ty: ignore[call-non-callable]

    def on_test_start(self):
        super().on_test_start()
        for metric in self.test_metrics.values():
            metric.reset()  # ty: ignore[call-non-callable]

    def forward_tensor(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
