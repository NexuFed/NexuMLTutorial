# Proposal

## Why

The learning path stops after basic MNIST and raw-waveform audio classification, leaving five advanced topics without runnable tutorials. Complete those topics as small, connected examples that teach correct NexuML usage, including GPU-capable audio preprocessing and validation-only model selection.

## What Changes

- Add five runnable tutorials and their documentation:
  - tuning and experiment tracking;
  - log-mel preprocessing, dataset export, and WebDataset reuse;
  - custom evaluation;
  - checkpoints and transfer learning;
  - model export and inference.
- Reuse the existing MNIST/audio datasets, torchaudio `LMBE`, ResNet encoder, pooling, classifier, loss, metrics, and logging/artifact helpers. Add only components needed to demonstrate a new contract.
- Replace the unsafe tuning example's `test/f1` objective with a short Optuna objective using NexuML's public fit/validate session lifecycle. Keep the test split out of all trials.
- Demonstrate online log-mel extraction and exported-feature training with the same downstream pipeline. Delegate serialization, metadata, and loading to NexuML core.
- Add a TorchMetrics-backed registered confusion-matrix evaluator, separate from training metrics.
- Demonstrate trainer resume separately from selective encoder weight loading. Use torchvision FashionMNIST as a small transfer target sharing the MNIST adapter and classification pipeline.
- Demonstrate trained-package reload, selective SafeTensors reuse, and ONNX inference using NexuML's supported Python APIs. Do not imply that every `ExportSpec` kind is automatically executed by the CLI.
- Add focused offline checks, documented real-data smoke commands, and a CUDA preprocessing check when CUDA is available.
- Update the README learning path and link all implemented stages. Distributed execution stays planned.

### Goals and learning outcome

Readers can compose typed scenarios, tune without test-set leakage, reuse prepared dataset views, register an evaluator, resume or transfer weights intentionally, and reload a trained model for inference. Each chapter explains the configuration seam and provides commands, prerequisites, expected outputs, and a small verification step.

### Non-goals

- Distributed execution, cloud/S3 deployment, benchmarks, or accuracy targets.
- A tutorial-owned training loop, tuner framework, DALI pipeline, dataset writer/reader, export wrapper framework, or service.
- Custom DSP, librosa, CPU-only audio feature extraction, or unnecessary dependencies.
- A dependency on `nexuml_library`, copied base-library implementations, compatibility aliases, or unrelated cleanup.
- Changes to the sibling NexuML repository. Unsupported upstream behavior is reported rather than patched around or implemented locally.

## Capabilities

### New Capabilities

- `tutorial-tuning-and-tracking`: validation-only Optuna model selection with NexuML sessions and explicit tracking configuration.
- `tutorial-audio-preprocessing-and-export`: GPU-capable torchaudio features and NexuML-owned dataset export/reload contracts.
- `tutorial-custom-evaluation`: registered, bounded-memory confusion-matrix evaluation and artifact logging.
- `tutorial-checkpoints-and-transfer`: resume versus selective weight reuse, freezing, and fine-tuning on FashionMNIST.
- `tutorial-model-export-and-inference`: trained package, SafeTensors, and ONNX examples with reload/inference verification.
- `advanced-tutorial-documentation`: a linked, progressive learning path with reproducible commands and honest prerequisites.

### Modified Capabilities

None. `openspec list --specs` reports no main specs; the earlier audio change remains complete but unarchived and is not modified by this change.

## Impact

- Documentation: `README.md`, existing tutorial cross-links where needed, and five new files under `tutorials/`.
- Configuration: `library/config/tune/mnist_resnet.py`, minimal shared data/model fragments, new log-mel and transfer scenarios, and their discovery imports.
- Components: the existing torchvision dataset adapter, a thin local typed definition delegating to `nexuml.data.exported.ExportedDataset`, and one registered confusion-matrix evaluator. Reuse `library/layers/feature/lmbe.py`; change it only if focused device/shape tests identify a relevant defect.
- Execution: small checkpoint and export/inference scripts under `examples/`, using public NexuML APIs rather than a shared runner abstraction.
- Packaging: optional extras for tuning/tracking and ONNX only where required; verify the minimum NexuML version against the APIs actually used.
- Tests: extend existing scenario checks and add one focused advanced-tutorial test module using synthetic inputs and local temporary artifacts.

### Upstream contracts

The examples rely on typed `ScenarioSpec`/`LayerSpec`, `NexuSession` fit/validate/test, `PreprocessingSpec`, `LightningFeatureExtractor`, `export_data_module`, core `ExportedDataset`, `skip_pipeline_stages`, evaluator registration/lifecycle, `CheckpointLoadSpec`, trainer checkpoint restore, and package/SafeTensors/ONNX export APIs. The current built-in tuner runs test before reading logged metrics, so the new validation-only example uses the public session lifecycle instead of copying the tuner or silently optimizing a test metric.
