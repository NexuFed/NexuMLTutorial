import pytest
import torch
from nexuml.core.base_layer import PipelineLayer

from library.layers.model.audio import AudioCNNEncoder, TinyAudioTransformerEncoder


@pytest.mark.parametrize("encoder_type", [AudioCNNEncoder, TinyAudioTransformerEncoder])
def test_audio_encoder_contract(encoder_type: type[PipelineLayer]) -> None:
    encoder = encoder_type(
        input_sizes={"waveform": (16_000,)},
        keys_in=["waveform"],
        keys_out=["embeddings"],
    )

    embeddings = encoder.forward_tensor(torch.randn(2, 16_000))

    assert embeddings.shape == (2, 64)
    assert torch.isfinite(embeddings).all()
