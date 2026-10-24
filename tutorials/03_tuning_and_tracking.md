# Tuning and Experiment Tracking

Run from the repository root after the [README setup](../README.md#setup):

```bash
uv pip install -e '.[advanced]'
python -m library.config.tune.mnist_resnet --n-trials 2 --max-epochs 1
```

Optuna searches learning rate, batch size, and encoder width using finite `val/loss`. Trials use a seeded 80/20 split of the official training source; only the selected configuration is retrained and tested on the official test source. No custom training loop or test-based selection is used.

Study names must be new. To repeat:

```bash
python -m library.config.tune.mnist_resnet --study-name mnist-validation-second --n-trials 2 --max-epochs 1
```

## Compare Runs

Start either viewer in a separate terminal:

```bash
tensorboard --logdir logs/tensorboard --host 127.0.0.1
mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db --host 127.0.0.1
```

Compare `_trial_0`, `_trial_1`, and `_selected` runs. Parameters, metrics, and resolved configurations are logged locally; the Optuna study is `logs/optuna/mnist.db`. No tracking server credentials are required.

Next: [preprocessing and dataset export](04_audio_preprocessing_and_export.md).
