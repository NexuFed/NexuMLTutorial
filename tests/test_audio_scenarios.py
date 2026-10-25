from nexuml.data.loaders.definitions import DaliLoader

from library.config.model import resnet_classifier
from library.config.scenario import speech_commands_cnn, speech_commands_transformer
from library.layers.head.classification import ClassificationHead
from library.layers.loss.cross_entropy import CrossEntropyLoss
from library.layers.model.audio import AudioCNNEncoder, TinyAudioTransformerEncoder


def test_audio_scenarios_only_swap_encoder() -> None:
    cnn = speech_commands_cnn(download=False)
    transformer = speech_commands_transformer(download=False)

    assert isinstance(cnn.pipeline.stages["Encoder"][0].component, AudioCNNEncoder)
    assert isinstance(
        transformer.pipeline.stages["Encoder"][0].component,
        TinyAudioTransformerEncoder,
    )
    for stage in ("Head", "Loss", "Metrics"):
        assert cnn.pipeline.stages[stage] == transformer.pipeline.stages[stage]
    assert cnn.data == transformer.data

    for spec in (cnn, transformer):
        assert isinstance(spec.data.loader.backend, DaliLoader)
        assert spec.data.datasets[0].split_type == "keep"
        assert isinstance(spec.pipeline.stages["Head"][0].component, ClassificationHead)
        assert isinstance(spec.pipeline.stages["Loss"][0].component, CrossEntropyLoss)


def test_mnist_uses_generic_classification_layers() -> None:
    pipeline = resnet_classifier()

    assert isinstance(pipeline.stages["Head"][0].component, ClassificationHead)
    assert isinstance(pipeline.stages["Loss"][0].component, CrossEntropyLoss)
