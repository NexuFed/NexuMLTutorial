"""File-backed Mini Speech Commands dataset."""

from __future__ import annotations

import hashlib
import urllib.request
import wave
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.dataset import NexuDataset
from tensordict import TensorDict

DOWNLOAD_URL = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
SAMPLE_RATE = 16_000
CLIP_SAMPLES = 16_000
CLASS_NAMES = ("down", "go", "left", "no", "right", "stop", "up", "yes")


def _has_wav_files(root: Path) -> bool:
    return any(next((root / name).glob("*.wav"), None) for name in CLASS_NAMES)


def _download_dataset(root: Path) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    archive_path = root.parent / "mini_speech_commands.zip"
    urllib.request.urlretrieve(DOWNLOAD_URL, archive_path)

    target = root.parent.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not (target / member.filename).resolve().is_relative_to(target):
                raise ValueError(f"Unsafe archive entry: {member.filename}")
        archive.extractall(target)

    if not _has_wav_files(root):
        raise FileNotFoundError(
            f"Downloaded archive did not create WAV files under {root}"
        )
    archive_path.unlink()


def _speaker_split(path: Path) -> str:
    speaker = path.stem.split("_nohash_", 1)[0]
    bucket = (
        int.from_bytes(hashlib.sha1(speaker.encode("utf-8")).digest()[:4], "big") % 100
    )
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


@data_source("MiniSpeechCommandsDataset")
class MiniSpeechCommandsDataset(DataSourceDefinition):
    """Eight-command WAV dataset with an approximate 80/10/10 speaker split."""

    root: str = "data/mini_speech_commands"
    download: bool = True

    def build(self) -> NexuDataset:
        return _MiniSpeechCommandsDatasetRuntime(**self.model_dump())


class _MiniSpeechCommandsDatasetRuntime(NexuDataset):
    LABEL_NAMES = ["class"]
    MODALITY = "audio"

    def __init__(
        self,
        root: str = "data/mini_speech_commands",
        download: bool = True,
    ):
        self.root = Path(root)
        if not _has_wav_files(self.root):
            if not download:
                raise FileNotFoundError(
                    f"Mini Speech Commands not found at {self.root}; "
                    "set download=True to fetch it"
                )
            _download_dataset(self.root)

        rows = [
            {"file": str(path), "class": class_id, "split": _speaker_split(path)}
            for class_id, name in enumerate(CLASS_NAMES)
            for path in sorted((self.root / name).glob("*.wav"))
        ]
        if not rows:
            raise FileNotFoundError(f"No WAV files found under {self.root}")

        super().__init__(
            data=None,
            meta=pd.DataFrame(rows),
            label_names=self.LABEL_NAMES,
            modality=self.MODALITY,
        )
        self.sample_rate = SAMPLE_RATE
        self.dali_x_keys = ["waveform"]
        self.dali_layout = "T"
        self.dali_sequence_length = CLIP_SAMPLES

    def load_item(self, idx: int, row: pd.Series) -> TensorDict:
        path = Path(str(row["file"]))
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getcomptype() != "NONE":
                raise ValueError(f"Expected mono PCM WAV: {path}")
            if wav.getframerate() != SAMPLE_RATE or wav.getsampwidth() != 2:
                raise ValueError(f"Expected 16-bit {SAMPLE_RATE} Hz WAV: {path}")
            waveform = np.frombuffer(
                wav.readframes(wav.getnframes()), dtype="<i2"
            ).astype(np.float32)

        waveform /= 32768.0
        waveform = np.pad(
            waveform[:CLIP_SAMPLES], (0, max(0, CLIP_SAMPLES - len(waveform)))
        )
        return TensorDict({"waveform": torch.from_numpy(waveform)}, batch_size=[])
