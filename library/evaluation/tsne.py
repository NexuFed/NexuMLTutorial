"""Latent space visualizer using t-SNE or UMAP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.algorithm import EvalAlgorithm
from nexuml.evaluation.storage import ReservoirTensorDictBuffer
from tensordict import TensorDict

from ._artifact import log_figure

logger = logging.getLogger(__name__)


@eval_algorithm("tsne_visualizer")
class tSNEVisualizer(EvalAlgorithm):
    """Visualizes latent space using t-SNE or UMAP projection.

    Uses reservoir sampling to handle large datasets efficiently.
    Train and test samples are projected together and colored by label/split.
    """

    type_key = "tsne_visualizer"

    def __init__(
        self,
        feature_key: str = "latent",
        label_key: str = "class",
        max_samples: int = 2_000,
        perplexity: int = 30,
        storage_backend: str = "memory",
        storage_path: str | None = None,
    ) -> None:
        self.feature_key = feature_key
        self.label_key = label_key
        self.perplexity = perplexity
        base_path = Path(storage_path) if storage_path is not None else None
        self._train_storage = ReservoirTensorDictBuffer(
            max_samples=max_samples,
            storage_backend=storage_backend,
            storage_path=base_path / "train" if base_path is not None else None,
        )
        self._test_storage = ReservoirTensorDictBuffer(
            max_samples=max_samples,
            storage_backend=storage_backend,
            storage_path=base_path / "test" if base_path is not None else None,
        )
        self._projection: Any = None
        self._split_labels: list[str] = []
        self._combined_labels: list[float | None] = []

    def fit_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        if self.feature_key not in x.keys():
            return
        features = x[self.feature_key].float()
        if features.dim() > 2:
            features = features.flatten(1)
        labels = (
            y[self.label_key].float()
            if y is not None and self.label_key in y.keys()
            else torch.full((features.shape[0],), float("nan"))
        )
        self._train_storage.add_batch(
            TensorDict(
                {"feature": features.cpu(), "label": labels.cpu().reshape(-1)},
                batch_size=[features.shape[0]],
            )
        )

    def fit_end(self) -> None:
        pass

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        if self.feature_key not in x.keys():
            return
        features = x[self.feature_key].float()
        if features.dim() > 2:
            features = features.flatten(1)
        labels = (
            y[self.label_key].float()
            if y is not None and self.label_key in y.keys()
            else torch.full((features.shape[0],), float("nan"))
        )
        self._test_storage.add_batch(
            TensorDict(
                {"feature": features.cpu(), "label": labels.cpu().reshape(-1)},
                batch_size=[features.shape[0]],
            )
        )

    def eval_end(self) -> None:
        train_td = self._train_storage.get()
        test_td = self._test_storage.get()
        train_data = train_td["feature"] if train_td is not None else None
        test_data = test_td["feature"] if test_td is not None else None
        train_labels = self._restore_labels(train_td)
        test_labels = self._restore_labels(test_td)
        if train_data is None and test_data is None:
            logger.warning("tSNEVisualizer: no features collected, skipping projection")
            return

        parts: list[torch.Tensor] = []
        split_labels = []
        if train_data is not None:
            parts.append(cast(torch.Tensor, train_data))
            split_labels.extend(["train"] * len(train_data))
        if test_data is not None:
            parts.append(cast(torch.Tensor, test_data))
            split_labels.extend(["test"] * len(test_data))

        combined = torch.cat(parts).numpy()
        n_samples = combined.shape[0]
        if n_samples < 3:
            logger.warning(
                "tSNEVisualizer: not enough samples (%d) for projection", n_samples
            )
            return

        try:
            from sklearn.manifold import TSNE

            perplexity = min(self.perplexity, max(2, (n_samples - 1) // 3))
            self._projection = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                init="pca",
            ).fit_transform(combined)
        except ImportError:
            logger.error("sklearn not available for t-SNE projection")

        self._split_labels = split_labels
        self._combined_labels = train_labels + test_labels

    def visualize(self, logger_obj: Any) -> None:  # ty: ignore[invalid-method-override]
        if self._projection is None:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        proj = self._projection
        labels = np.array(self._combined_labels) if self._combined_labels else None

        split_markers = {"train": "o", "test": "x"}
        if labels is not None and len(labels) == len(proj):
            unique_labels = sorted(
                {label for label in labels.tolist() if label is not None}
            )
            cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
            for split in ("train", "test"):
                split_mask = np.array(self._split_labels) == split
                for i, label in enumerate(unique_labels):
                    label_mask = labels == label
                    mask = split_mask & label_mask
                    if not np.any(mask):
                        continue
                    ax.scatter(
                        proj[mask, 0],
                        proj[mask, 1],
                        c=[cmap(i)],
                        label=f"{split}:{str(label)}",
                        alpha=0.72,
                        s=18,
                        marker=split_markers.get(split, "o"),
                        linewidths=0.6,
                    )
            ax.legend(markerscale=1.4, fontsize=7, frameon=False, ncol=2)
        else:
            for split in ("train", "test"):
                split_mask = np.array(self._split_labels) == split
                if not np.any(split_mask):
                    continue
                ax.scatter(
                    proj[split_mask, 0],
                    proj[split_mask, 1],
                    alpha=0.7,
                    s=18,
                    marker=split_markers.get(split, "o"),
                    label=split,
                )
            ax.legend(frameon=False)

        ax.set_title(f"t-SNE Space by {self.label_key}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.2, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        log_figure(logger_obj, f"eval/tSNE/{self.label_key}", fig)
        plt.close(fig)

    def results(self) -> dict[str, float]:
        return (
            {"n_projected_samples": float(len(self._split_labels))}
            if self._split_labels
            else {}
        )

    def _restore_labels(self, data: TensorDict | None) -> list[float | None]:
        if data is None or "label" not in data.keys():
            return []
        labels = data["label"].float().cpu().tolist()
        restored: list[float | None] = []
        for label in labels:
            restored.append(None if not np.isfinite(label) else float(label))
        return restored
