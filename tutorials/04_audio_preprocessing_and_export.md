# Audio Preprocessing, Dataset Export, and WebDataset

Requires the [README setup](../README.md#setup), `.[advanced]`, and native DALI. `TutorialLMBE` converts 16 kHz waveforms into float32 log-mel features `[B, 1, 64, 101]`, then reuses the image ResNet classifier.

## Train Online

```bash
nexuml backend list data-loader
nexuml resolve speech-commands-log-mel
nexuml train speech-commands-log-mel --max-epochs 1
```

## Prepare Features on GPU

Choose new output directories; existing destinations are rejected.

```bash
python -m examples.prepare_speech_commands --backend numpy --device cuda --output data/prepared/speech_commands_numpy
python -m examples.prepare_speech_commands --backend webdataset --device cuda --output data/prepared/speech_commands
```

Both exports preserve labels, identities, and speaker-owned splits. The boundary stops at `features`, before the encoder; trained weights are unnecessary. `--device cpu` moves the transform, not the raw DALI loader. Inspect each export's `config.yaml` and metadata file.

## Train Prepared Features

The default uses WebDataset with the Torch loader and skips `Preprocessing`:

```bash
nexuml resolve speech-commands-log-mel-prepared
nexuml train speech-commands-log-mel-prepared --max-epochs 1
```

For the NumPy export:

```bash
python - <<'PY'
from library.config.scenario import speech_commands_log_mel_prepared
from nexuml.training.lightning import NexuSession

spec = speech_commands_log_mel_prepared(root="data/prepared/speech_commands_numpy", max_epochs=1)
NexuSession.from_scenario(spec).run()
PY
```

For native DALI loading of prepared WebDataset:

```bash
python - <<'PY'
from library.config.scenario import speech_commands_log_mel_prepared
from nexuml.data.loaders.definitions import DaliLoader
from nexuml.training.lightning import NexuSession

spec = speech_commands_log_mel_prepared(max_epochs=1)
spec.data.loader.backend = DaliLoader()
NexuSession.from_scenario(spec).run()
PY
```

## Automatic Materialization Alternative

Use a separate destination. This route defaults to NumPy and does not explicitly select CUDA for the transform:

```bash
python - <<'PY'
from library.config.scenario import speech_commands_log_mel
from nexuml.core.types import PreprocessingSpec
from nexuml.training.lightning import NexuSession

spec = speech_commands_log_mel(max_epochs=1)
spec.data.preprocessing = PreprocessingSpec(
    enabled=True, path="data/prepared/speech_commands_auto",
    until_x_keys=["features"], x_keys=["features"], y_keys=["class"],
)
spec.data.skip_pipeline_stages = ["Preprocessing"]
NexuSession.from_scenario(spec).run()
PY
```

Next: [custom evaluation](05_custom_evaluation.md).
