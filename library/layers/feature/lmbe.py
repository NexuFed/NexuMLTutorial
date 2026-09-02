"""Log Mel Band Energy (LMBE) feature extractor layer."""

from __future__ import annotations

import torch
import torchaudio
from nexuml.core.base_layer import PipelineLayer
from nexuml.core.components import LayerBuildContext, LayerDefinition
from nexuml.core.discovery import layer


@layer("LMBE")
class LMBE(LayerDefinition):
    """Compute Log Mel Band Energies from a raw waveform.

    Uses torchaudio's MelSpectrogram + AmplitudeToDB pipeline.
    """

    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int | None = None
    power: int = 2
    fmin: int = 0
    fmax: int = 8000
    sr: int | None = None
    sample_rate: int | None = None
    sampling_rate: int | None = None
    mel_scale: str = "slaney"
    pad_mode: str = "constant"
    to_db: bool = True

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _LMBERuntime(**context.runtime_kwargs(), **self.model_dump())


class _LMBERuntime(PipelineLayer):
    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int | None = None,
        power: int = 2,
        fmin: int = 0,
        fmax: int = 8000,
        sr: int | None = None,
        sample_rate: int | None = None,
        sampling_rate: int | None = None,
        mel_scale: str = "slaney",
        pad_mode: str = "constant",
        to_db: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        resolved_sr = (
            sample_rate
            if sample_rate is not None
            else (
                sr
                if sr is not None
                else (sampling_rate if sampling_rate is not None else 16000)
            )
        )
        self.sr = resolved_sr
        self.power = power
        self.mel_scale = mel_scale
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_length = win_length if win_length is not None else n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.to_db = to_db

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=float(power),
            normalized=False,
            mel_scale=mel_scale,
            norm=mel_scale,
            pad_mode=pad_mode,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power" if power == 2 else "magnitude",
        )

    def forward_tensor(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:

        mel = self.mel_spectrogram(x)

        if self.to_db:
            mel = self.amplitude_to_db(mel)

        # PatchEmbedding expects (B, C, H, W), so expose spectrograms with an
        # explicit channel dimension.
        if mel.ndim == 3:
            mel = mel.unsqueeze(1)
        elif mel.ndim == 2:
            mel = mel.unsqueeze(0).unsqueeze(0)

        return mel
