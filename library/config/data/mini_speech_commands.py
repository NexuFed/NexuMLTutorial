"""Mini Speech Commands data configuration."""

from __future__ import annotations

from nexuml.core.types import DatasetSpec, DataSpec, LoaderSpec


def mini_speech_commands_data(
    root: str = "data/mini_speech_commands",
    download: bool = True,
    num_workers: int = 4,
) -> DataSpec:
    return DataSpec(
        source_type="MiniSpeechCommandsDataset",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        datasets=[
            DatasetSpec(
                type_key="MiniSpeechCommandsDataset",
                params={"root": root, "download": download},
                modality="audio",
                split_type="keep",
            )
        ],
        input_shapes={"waveform": [16000]},
        feature_key="waveform",
        num_classes=8,
        loader=LoaderSpec(backend="dali", num_workers=num_workers),
    )
