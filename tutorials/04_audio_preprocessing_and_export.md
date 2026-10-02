# Audio Preprocessing, Dataset Export, and WebDataset

Start with the [native-DALI Speech Commands dataset](02_audio_native_dali.md). Add the existing torchaudio `TutorialLMBE` stage, then reuse the MNIST ResNet, pooling, head, loss, and metrics. No new encoder or DSP implementation is needed.

## Feature Contract

```text
waveform [B, 16000], float32, 16 kHz
  -> Preprocessing: TutorialLMBE
features [B, 1, 64, 101], float32, log-mel power in dB
  -> Encoder -> Pooling -> Head
class_logits [B, 8]
```

The typed transform settings are visible in `library/config/scenario/speech_commands.py`: 400-point FFT/window, 160-sample hop, 64 Slaney mel bands, 0–8 kHz, power 2, constant padding, centered STFT, and dB conversion. Centering gives 101 frames for 16,000 samples. `MelSpectrogram` and `AmplitudeToDB` are registered Torch submodules; `.to("cuda")` moves their buffers, and computation stays on the waveform's device. Silent input remains finite. There is no librosa or NumPy DSP detour.

Class order is unchanged: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes`. Speaker-owned splits remain intact.

## Train Online

Use the README setup plus the compatible DALI/CUDA prerequisites from the previous chapter. Installing `.[advanced]` enables the default MLflow logger.

```bash
nexuml backend list data-loader
nexuml resolve speech-commands-log-mel
nexuml train speech-commands-log-mel --max-epochs 1
```

This runs log-mel extraction on every batch. Training controls model device movement; the raw source still uses NexuML's native WAV reader/decoder.

## Export the Prepared Boundary on GPU

From the repository root:

```bash
python -m examples.prepare_speech_commands --backend numpy --device cuda --output data/prepared/speech_commands_numpy
python -m examples.prepare_speech_commands --backend webdataset --device cuda --output data/prepared/speech_commands
```

Each command exports all existing train/validation/test views. Destinations must not already exist; use a new path rather than overwrite an artifact. `--root` changes the raw dataset path; `--batch-size` changes export batching. The first command can download the approximately 182 MB archive if absent. Both commands require the raw DALI backend; `--device cpu` changes transform placement, not the raw loader implementation.

The script calls public APIs directly:

```python
transform = LightningFeatureExtractor(session.lightning_module, x_keys=["features"])
export_data_module(
    session.data_module, output,
    backend="webdataset", transform=transform,
    x_keys=["features"], y_keys=["class"], device="cuda",
)
```

The boundary stops after `features`, **before the encoder**. Model weights are not required to prepare log-mel features. NexuML writes float32 feature tensors, labels, metadata, key/shape configuration, and format-specific files. NumPy stores per-sample arrays; WebDataset stores shards plus its indexes. NumPy conversion for serialization happens after feature computation; it does not move DSP to CPU.

Inspect `config.yaml` and `metadata.parquet` (or core's CSV fallback) under the output directory. Compare identities and split assignments, not shuffled batch positions.

## Reload Without Preprocessing Twice

`TutorialExportedDataset` is a small typed definition delegating to core `nexuml.data.exported.ExportedDataset`. There is no tutorial tar reader, writer, or dependency on `nexuml_library`.

The prepared scenario retains the same model stages but sets:

```python
feature_key = "features"
input_shapes = {"features": [1, 64, 101]}
skip_pipeline_stages = ["Preprocessing"]
```

Its source uses `split_type="keep"`. The default prepared loader is core `TorchLoader`, supporting both local export formats:

```bash
nexuml resolve speech-commands-log-mel-prepared
nexuml train speech-commands-log-mel-prepared --max-epochs 1
```

The default root matches the WebDataset export above. To use the NumPy destination, compose the public scenario in Python:

```python
from library.config.scenario import speech_commands_log_mel_prepared
from nexuml.training.lightning import NexuSession

spec = speech_commands_log_mel_prepared(root="data/prepared/speech_commands_numpy", max_epochs=1)
NexuSession.from_scenario(spec).run()
```

For the native prepared-WebDataset route, change **only** the loader:

```python
from nexuml.data.loaders.definitions import DaliLoader

spec = speech_commands_log_mel_prepared(max_epochs=1)
spec.data.loader.backend = DaliLoader()
NexuSession.from_scenario(spec).run()
```

This requires native DALI to be installed and usable. Core consumes its shard/index contract; the tutorial does not implement a fallback while claiming native execution.

## Automatic Materialization Alternative

`DataSpec.preprocessing` provides the same boundary for core-managed materialization:

```python
from nexuml.core.types import PreprocessingSpec
from library.config.scenario import speech_commands_log_mel

spec = speech_commands_log_mel(max_epochs=1)
spec.data.preprocessing = PreprocessingSpec(
    enabled=True, path="data/prepared/speech_commands_auto",
    until_x_keys=["features"], x_keys=["features"], y_keys=["class"],
)
spec.data.skip_pipeline_stages = ["Preprocessing"]
```

The default writer is NumPy. In NexuML 0.2.1, this automatic route and the CLI `export-dataset` route do not select CUDA for the transform. Use the explicit export script above when demonstrating GPU preprocessing. Choose one materialization route, not both on the same destination.

## Verify

```bash
python -m pytest tests/test_advanced_tutorials.py -k 'log_mel or prepared' -q
```

Synthetic CPU/CUDA checks cover shape, finiteness, singleton/batch parity, exported labels/identities/splits, skipped preprocessing, and matching downstream logits. CUDA skips explicitly if unavailable; this is not proof that a DALI real-data run worked. Float32 round-trip checks use `torch.testing.assert_close`'s default tolerance.

Next: [custom evaluation](05_custom_evaluation.md).
