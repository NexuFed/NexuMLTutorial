import pytest
from nexuml.core.config import ResolvedConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR

from library.config.scenario import (
    mnist_resnet,
    speech_commands_cnn,
    speech_commands_transformer,
)
from library.config.tune.mnist_resnet import scenario as tune_scenario


@pytest.mark.parametrize(
    "make_scenario",
    [mnist_resnet, speech_commands_cnn, speech_commands_transformer, tune_scenario],
)
def test_scenario_uses_current_factory_api(make_scenario) -> None:
    scenario = make_scenario()
    loaded = ResolvedConfig.from_yaml(ResolvedConfig.from_scenario(scenario).to_yaml())

    assert loaded.name == scenario.name
    assert loaded.training.optimizer.resolve() is Adam
    assert loaded.training.scheduler.resolve() is ConstantLR
    assert loaded.callbacks
    assert all(callback.resolve() for callback in loaded.callbacks)
