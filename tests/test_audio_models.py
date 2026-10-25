import pytest
import torch
from nexuml.core.components import LayerBuildContext, LayerDefinition

from library.layers.model.audio import AudioCNNEncoder, TinyAudioTransformerEncoder


@pytest.mark.parametrize(
    "definition_type", [AudioCNNEncoder, TinyAudioTransformerEncoder]
)
def test_audio_encoder_contract(definition_type: type[LayerDefinition]) -> None:
    encoder = definition_type().build(
        LayerBuildContext(
            input_sizes={"waveform": (16_000,)},
            keys_in=["waveform"],
            keys_out=["embeddings"],
        )
    )

    embeddings = encoder.forward_tensor(torch.randn(2, 16_000))

    assert embeddings.shape == (2, 64)
    assert torch.isfinite(embeddings).all()
