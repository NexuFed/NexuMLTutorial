import wave
from pathlib import Path

import pytest

from library.data.mini_speech_commands import MiniSpeechCommandsDataset


def _write_wav(path: Path, samples: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * samples)


def test_metadata_is_deterministic_and_groups_speakers(tmp_path: Path) -> None:
    root = tmp_path / "mini_speech_commands"
    _write_wav(root / "yes" / "shared_nohash_1.wav")
    _write_wav(root / "down" / "shared_nohash_0.wav")
    _write_wav(root / "go" / "other_nohash_0.wav")

    dataset = MiniSpeechCommandsDataset(root=str(root), download=False).build()

    assert dataset.meta is not None
    assert dataset.meta["class"].tolist() == [0, 1, 7]
    assert (
        dataset.meta["file"]
        .str.endswith(
            (
                "down/shared_nohash_0.wav",
                "go/other_nohash_0.wav",
                "yes/shared_nohash_1.wav",
            )
        )
        .all()
    )
    shared = dataset.meta[dataset.meta["file"].str.contains("shared_nohash")]
    assert shared["split"].nunique() == 1
    assert set(dataset.meta["split"]) <= {"train", "val", "test"}


def test_direct_item_and_native_audio_contract(tmp_path: Path) -> None:
    root = tmp_path / "mini_speech_commands"
    _write_wav(root / "right" / "speaker_nohash_0.wav")

    dataset = MiniSpeechCommandsDataset(root=str(root), download=False).build()
    x, y = dataset[0]

    assert x["waveform"].shape == (16_000,)
    assert y is not None and y["class"].item() == 4
    assert dataset.data is None
    assert dataset.sample_rate == 16_000
    assert dataset.dali_x_keys == ["waveform"]
    assert dataset.dali_layout == "T"
    assert dataset.dali_sequence_length == 16_000


def test_missing_dataset_fails_without_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="set download=True"):
        MiniSpeechCommandsDataset(
            root=str(tmp_path / "missing"), download=False
        ).build()
