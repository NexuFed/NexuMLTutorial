# Tuning and Experiment Tracking

Reuse the [MNIST classifier](01_mnist_from_scratch.md). Optuna selects learning rate, batch size, and encoder width; NexuML still builds the data/model and owns training. There is no custom training loop.

## Setup

Run from the repository root in the README's activated environment:

```bash
uv pip install -e '.[advanced]'
python -m library.config.tune.mnist_resnet --help
```

The extra installs NexuML's Optuna and MLflow integrations. TensorBoard is already a NexuML dependency. The first run downloads torchvision MNIST under `data/` if necessary.

## Keep Test Data Out of Search

`library/config/tune/mnist_resnet.py` creates each trial from the existing scenario factory, then:

- keeps **only the official training source**;
- splits that source 80/20 into training/validation;
- resets seed 42 before building every trial;
- disables post-training evaluators and package exports;
- calls `NexuSession.setup().fit().validate()` and minimizes finite `val/loss`.

The session lifecycle matters:

```python
session = NexuSession.from_scenario(spec).setup()
results = session.fit().validate()
validation_loss = results[0]["val/loss"]
```

`validate()` returns metric dictionaries, not the session. Do not chain `.test()` onto its return value. `NexuSession.run()` includes testing; it is deliberately **not** used inside a trial. NexuML 0.2.1's built-in CLI tuner also runs the full test lifecycle, so this chapter replaces the old `nexuml tune` recipe rather than changing `test/f1` to another key and hoping that solves leakage.

## Two-Trial Smoke Run

```bash
python -m library.config.tune.mnist_resnet --n-trials 2 --max-epochs 1
```

Expect two finite validation objectives, the minimum objective and its sampled parameters, followed by **one final test evaluation**. After selection, the script trains the selected configuration afresh with the same seed and train/validation split, then evaluates its last trained weights on the official 10,000-image test source. This is not a best-epoch reload or evaluation of every trial on test.

Artifacts:

| Location | Contents |
| --- | --- |
| `logs/optuna/mnist.db` | SQLite study, parameters, validation objectives |
| `logs/tensorboard/` | Trial/selected-run metrics and configuration artifacts |
| `logs/mlflow/mlflow.db` | Local tracking store; trials share the study's experiment |
| `logs/mlflow/mlflow_artifacts/` | MLflow artifacts |
| `logs/checkpoints/<study>_trial_<n>/` | Trial checkpoints, separate from the selected run |

The default study is `mnist-validation`. Existing study names are rejected rather than silently mixing another search into old results. For another run:

```bash
python -m library.config.tune.mnist_resnet --study-name mnist-validation-second --n-trials 2 --max-epochs 1
```

## Compare Runs

Start either viewer locally, in a separate terminal:

```bash
tensorboard --logdir logs/tensorboard --host 127.0.0.1
mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db --host 127.0.0.1
```

Trial identities end in `_trial_0`, `_trial_1`, etc.; the final run ends in `_selected`. Sampled parameters are logged explicitly, and NexuML records the resolved configuration. Compare validation results, not test results, to select hyperparameters. No tracking server or credentials are required for the training command.

## Offline Verification

```bash
python -m pytest tests/test_advanced_tutorials.py -k tuning -q
```

These checks reject test access inside trial sessions and missing/non-finite objectives. They use synthetic session results and do not download MNIST; the real smoke command separately exercises Optuna, tracking, and NexuML training.

Next: [audio preprocessing and dataset export](04_audio_preprocessing_and_export.md).
