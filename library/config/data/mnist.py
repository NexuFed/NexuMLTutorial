"""CIFAR data scenario fragments."""

from __future__ import annotations

from nexuml.core.types import DatasetSpec, DataSpec, LoaderSpec
from nexuml.data.loaders.definitions import TorchLoader

from ...data.mnist import MNISTDataset


def mnist_data(
    download: bool = True, root: str = "data", num_workers: int = 4
) -> DataSpec:
    """Create a DataSpec for MNIST image classification.

    Returns:
        DataSpec: MNIST dataset specification with fit and test splits.
    """
    return DataSpec(
        datasets=[
            DatasetSpec(
                source=MNISTDataset(root=str(root), train=True, download=download),
                modality="image",
                split_type="fit",
            ),
            DatasetSpec(
                source=MNISTDataset(root=str(root), train=False, download=download),
                modality="image",
                split_type="test",
            ),
        ],
        input_shapes={"features": [1, 28, 28]},
        num_classes=10,
        feature_key="features",
        loader=LoaderSpec(backend=TorchLoader(), num_workers=num_workers),
    )
