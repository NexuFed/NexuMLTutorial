"""CIFAR data scenario fragments."""

from __future__ import annotations

from nexuml.core.types import DatasetSpec, DataSpec


def mnist_data(download: bool = True, root: str = "data/mnist") -> DataSpec:
    """Create a DataSpec for MNIST image classification.

    Returns:
        DataSpec: MNIST dataset specification with fit and test splits.
    """
    return DataSpec(
        source_type="mnist",
        datasets=[
            DatasetSpec(
                type_key="MNISTDataset",
                params={
                    "root": str(root),
                    "train": True,
                    "download": download,
                },
                modality="image",
                split_type="fit",
            ),
            DatasetSpec(
                type_key="MNISTDataset",
                params={
                    "root": str(root),
                    "train": False,
                    "download": download,
                },
                modality="image",
                split_type="test",
            ),
        ],
        # input_shapes={"features": [1, 28, 28]},
        # num_classes=10,
        # feature_key="features",
    )
