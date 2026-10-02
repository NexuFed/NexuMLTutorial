"""Scenario configuration fragments."""

from .mnist_resnet import mnist_resnet
from .transfer import fashion_mnist_transfer
from .speech_commands import (
    speech_commands_cnn,
    speech_commands_transformer,
    speech_commands_log_mel,
    speech_commands_log_mel_prepared,
)
