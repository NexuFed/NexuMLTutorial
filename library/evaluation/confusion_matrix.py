"""Streaming confusion counts through NexuML's evaluator extension contract."""

from typing import Any

from nexuml.core.components import EvalAlgorithmDefinition, EvalBuildContext
from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.algorithm import (
    EvalAlgorithm,
    get_declared_axis,
    get_declared_tensor,
)
from pydantic import Field
from tensordict import TensorDict
from torchmetrics.classification import MulticlassConfusionMatrix

from ._artifact import log_figure


@eval_algorithm("TutorialConfusionMatrix")
class ConfusionMatrix(EvalAlgorithmDefinition):
    class_names: list[str] = Field(min_length=2)

    def build(self, context: EvalBuildContext) -> EvalAlgorithm:
        return _ConfusionMatrixRuntime(
            self.class_names,
            context.feature_key or "class_logits",
            context.label_key or "class",
        )


class _ConfusionMatrixRuntime(EvalAlgorithm):
    def __init__(self, class_names: list[str], feature_key: str, label_key: str):
        self.class_names = class_names
        self.feature_key, self.label_key = feature_key, label_key
        self.counts = MulticlassConfusionMatrix(num_classes=len(class_names))
        self.finished = False

    def fit_end(self) -> None:
        self.counts.reset()
        self.finished = False

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        if self.finished:
            self.fit_end()
        logits = get_declared_tensor(
            x, self.feature_key, algorithm="TutorialConfusionMatrix"
        )
        labels = get_declared_axis(
            x, y, self.label_key, algorithm="TutorialConfusionMatrix"
        )
        if logits.ndim != 2 or logits.shape[1] != len(self.class_names):
            raise ValueError("Expected logits [batch, number of class_names]")
        labels = labels.detach().cpu().reshape(-1)
        if not labels.eq(labels.long()).all():
            raise ValueError("Expected integer class labels")
        self.counts.update(logits.detach().cpu(), labels.long())

    def eval_end(self) -> None:
        self.results()  # Reject empty evaluation rather than publish NaN.
        self.finished = True

    def results(self) -> dict[str, float]:
        matrix = self.counts.compute()
        samples = matrix.sum().item()
        if samples == 0:
            raise ValueError("Confusion-matrix evaluation received no samples")
        return {
            "micro_accuracy": matrix.trace().item() / samples,
            "samples": float(samples),
        }

    def visualize(self, logger: Any) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matrix = self.counts.compute().numpy()
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, label="Samples")
        ax.set(
            xlabel="Predicted class",
            ylabel="True class",
            title="Confusion matrix",
            xticks=range(len(self.class_names)),
            yticks=range(len(self.class_names)),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        try:
            log_figure(logger, "eval/confusion_matrix", fig)
        finally:
            plt.close(fig)
