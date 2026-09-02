"""Mini Speech Commands classification scenarios."""

from __future__ import annotations

from nexuml.core.components import LayerDefinition
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec

from ...layers.model.audio import AudioCNNEncoder, TinyAudioTransformerEncoder
from ..data import mini_speech_commands_data
from ..defaults import (
    default_callbacks,
    default_checkpoint,
    default_evaluation,
    default_exports,
    default_logging,
    default_training,
    default_tuning,
)
from ..model import audio_classifier


def _speech_commands_scenario(
    name: str,
    encoder: LayerDefinition,
    root: str,
    download: bool,
    lr: float,
    batch_size: int,
    max_epochs: int,
    num_workers: int,
) -> ScenarioSpec:
    return ScenarioSpec(
        name=name,
        pipeline=audio_classifier(encoder=encoder),
        training=default_training(max_epochs=max_epochs, batch_size=batch_size, lr=lr),
        data=mini_speech_commands_data(
            root=root, download=download, num_workers=num_workers
        ),
        evaluation=default_evaluation(feature_key="embeddings", label_key="class"),
        logging=default_logging(name=name),
        callbacks=default_callbacks(name=name),
        tuning=default_tuning(),
        checkpoint=default_checkpoint(),
        exports=default_exports(name=name),
    )


@scenario("speech-commands-cnn")
def speech_commands_cnn(
    root: str = "data/mini_speech_commands",
    download: bool = True,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
    num_workers: int = 4,
) -> ScenarioSpec:
    return _speech_commands_scenario(
        name="speech_commands_cnn",
        encoder=AudioCNNEncoder(),
        root=root,
        download=download,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
    )


@scenario("speech-commands-transformer")
def speech_commands_transformer(
    root: str = "data/mini_speech_commands",
    download: bool = True,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
    num_workers: int = 4,
) -> ScenarioSpec:
    return _speech_commands_scenario(
        name="speech_commands_transformer",
        encoder=TinyAudioTransformerEncoder(),
        root=root,
        download=download,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        num_workers=num_workers,
    )
