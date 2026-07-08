"""Optuna tuning file.

Exposes:
  scenario()     - base ScenarioSpec (default mnist_resnet architecture)
  build(**arch)  - factory rebuilding scenario from architecture params
  SEARCH_SPACE   - Optuna hyperparameter architecture search space
  TUNING_SPEC    - tuning configuration

"""

from __future__ import annotations

from nexuml.core.types import ScenarioSpec, TuningSpec

from library.config.defaults import LOG_FOLDER
from library.config.scenario.mnist_resnet import mnist_resnet

SEARCH_SPACE = {
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    "training.lr": {
        "type": "float",
        "low": 1e-4,
        "high": 1e-2,
        "log": True,
    },
    # Architecture params below are grouped under dotted prefixes ("encoder.*",
    # "pooling.*", "head.*") so the tuner nests them into dicts and forwards
    # them to build(**arch_params) instead of setting them as scalar overrides.
    "encoder.width": {"type": "categorical", "choices": [16, 32, 64]},
    "encoder.depth": {"type": "categorical", "choices": [1, 2, 3, 4]},
    "pooling.type": {
        "type": "categorical",
        "choices": ["GlobalAveragePooling", "GlobalMaxPooling"],
    },
    "head.use_dropout": {
        "type": "categorical",
        "choices": [True, False],
        "when": {
            True: {
                "head.dropout": {"type": "float", "low": 0.1, "high": 0.5},
            },
        },
    },
}

TUNING_SPEC = TuningSpec(
    n_trials=20,
    directions=["maximize"],
    # NOTE: "test/f1" is used because nexuml's train() runs fit() then test()
    # sequentially, and Trainer.logged_metrics only retains the last loop's
    # keys - "val/f1" doesn't survive. This means the sweep selects
    # architectures by test-set score, which is test-set leakage. Fine for
    # this demo; a real project should tune on a validation-only metric.
    metric_key="test/f1",
    storage=f"{LOG_FOLDER}/optuna/optuna.log",
    prune=False,
)


def scenario() -> ScenarioSpec:
    """Return the default base scenario for tuning."""
    return build(
        batch_size=32,
    )


def build(
    batch_size: int,
    encoder: dict | None = None,
    pooling: dict | None = None,
    head: dict | None = None,
) -> ScenarioSpec:
    """Build a scenario from sampled hyperparameters and architecture params.

    Returns:
        ScenarioSpec: Assembled scenario with tuning spec attached.
    """
    encoder = encoder or {}
    pooling = pooling or {}
    head = head or {}
    scenario_spec = mnist_resnet(
        batch_size=batch_size,
        max_epochs=10,
        encoder_width=encoder.get("width", 32),
        encoder_depth=encoder.get("depth", 2),
        pooling_type=pooling.get("type", "GlobalAveragePooling"),
        head_dropout=head.get("dropout", 0.0),
    )
    scenario_spec.tuning = TUNING_SPEC
    # The tuner disables the trainer's progress bar; drop the rich_progress
    # callback so it doesn't conflict with that setting.
    scenario_spec.callbacks = [c for c in scenario_spec.callbacks if c.type != "rich_progress"]
    return scenario_spec
