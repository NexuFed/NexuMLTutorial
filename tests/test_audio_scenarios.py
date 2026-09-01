from library.config.model import resnet_classifier
from library.config.scenario import speech_commands_cnn, speech_commands_transformer


def test_audio_scenarios_only_swap_encoder() -> None:
    cnn = speech_commands_cnn(download=False)
    transformer = speech_commands_transformer(download=False)

    assert cnn.pipeline.stages["Encoder"][0].type_key == "AudioCNNEncoder"
    assert (
        transformer.pipeline.stages["Encoder"][0].type_key
        == "TinyAudioTransformerEncoder"
    )
    for stage in ("Head", "Loss", "Metrics"):
        assert cnn.pipeline.stages[stage] == transformer.pipeline.stages[stage]
    assert cnn.data == transformer.data

    for spec in (cnn, transformer):
        assert spec.data.loader.backend == "dali"
        assert spec.data.datasets[0].split_type == "keep"
        assert spec.pipeline.stages["Head"][0].type_key == "ClassificationHead"
        assert spec.pipeline.stages["Loss"][0].type_key == "CrossEntropyLoss"


def test_mnist_uses_generic_classification_layers() -> None:
    pipeline = resnet_classifier()

    assert pipeline.stages["Head"][0].type_key == "ClassificationHead"
    assert pipeline.stages["Loss"][0].type_key == "CrossEntropyLoss"
