"""Prepared log-mel features with their original split assignments."""

from nexuml.core.types import DataSpec, DatasetSpec, LoaderSpec
from nexuml.data.loaders.definitions import TorchLoader

from ...data.exported import ExportedDataset


def exported_log_mel_data(root: str, num_workers: int = 0) -> DataSpec:
    return DataSpec(
        datasets=[
            DatasetSpec(
                source=ExportedDataset(root=root),
                split_type="keep",
                modality="audio",
            )
        ],
        input_shapes={"features": [1, 64, 101]},
        feature_key="features",
        num_classes=8,
        skip_pipeline_stages=["Preprocessing"],
        loader=LoaderSpec(backend=TorchLoader(), num_workers=num_workers),
    )
