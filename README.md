# NexuML Tutorial

Runnable examples implemented as a standalone NexuML library. See the [NexuML documentation](https://nexufed.github.io/NexuML/) for framework concepts and API reference.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

export NEXUML_DATA_ROOT=$(pwd)/data
export NEXUML_LOGS_ROOT=$(pwd)
```

Audio examples require native DALI (`nexuml[dali]`) and a compatible NVIDIA driver/CUDA runtime. Check loader registration:

```bash
nexuml backend list data-loader
```

Scenario log paths already include `logs/`; keep `NEXUML_LOGS_ROOT` at the repository root.

## Local Library

The editable install registers this library via its package entry point. Inspect its components:

```bash
nexuml library list
nexuml registry list data
nexuml registry list layers
nexuml registry list eval
nexuml registry list scenarios
```

Do not add `library/` itself as a local library root: that imports the same components a second time under different module names. If you previously added it, run `nexuml library delete "$(pwd)/library"`; the editable install still discovers it.

## Learning Path

| Stage | Topic | Status |
| --- | --- | --- |
| 1 | [MNIST: custom library basics](tutorials/01_mnist_from_scratch.md) | Available |
| 2 | [Speech Commands CNN: file-backed native DALI](tutorials/02_audio_native_dali.md) | Available |
| 3 | [Speech Commands Transformer: pipeline composition](tutorials/02_audio_native_dali.md#swap-only-the-encoder) | Available |
| 4 | [Tuning and experiment tracking](tutorials/03_tuning_and_tracking.md) | Available |
| 5 | [Preprocessing, dataset export, and WebDataset](tutorials/04_audio_preprocessing_and_export.md) | Available |
| 6 | [Custom evaluation](tutorials/05_custom_evaluation.md) | Available |
| 7 | [Checkpoints and transfer learning](tutorials/06_checkpoints_and_transfer_learning.md) | Available |
| 8 | [Model export and inference](tutorials/07_model_export_and_inference.md) | Available |
| 9 | Distributed execution after NexuML exposes a stable backend | Planned |

## Coming Up

- [ ] Bring your own dataset: CSV metadata and files, label mapping, validation, splits, and registration.
- [ ] Augmentation and normalization: training-only transforms, deterministic validation/test preprocessing, and consistent inference preprocessing.
- [ ] Understand model failures: misclassified samples, per-class precision/recall, class imbalance, and confidence.
- [ ] Reproducible experiments: seeds, configuration capture, dataset identity, dependency versions, and GPU nondeterminism.
- [ ] Troubleshooting and a CPU-first smoke test: tensor keys/shapes, missing backends, CUDA/DALI setup, and checkpoint compatibility.
- [ ] Regression or multilabel classification beyond single-label classification.
- [ ] Distributed training once a stable NexuML backend is available.
- [ ] Inference service deployment once a supported serving workflow is established.

## Quick Commands

```bash
nexuml resolve tutorial-mnist-resnet
nexuml build configs/tutorial-mnist-resnet.yaml
nexuml train tutorial-mnist-resnet --max-epochs 1

nexuml resolve speech-commands-cnn
nexuml build configs/speech-commands-cnn.yaml
nexuml train speech-commands-cnn --max-epochs 10

nexuml resolve speech-commands-transformer
nexuml build configs/speech-commands-transformer.yaml
nexuml train speech-commands-transformer --max-epochs 10
```

Advanced examples reuse these components. Install optional search/tracking and export runtimes:

```bash
uv pip install -e '.[advanced,onnx]'
python -m library.config.tune.mnist_resnet --n-trials 2 --max-epochs 1
python -m examples.prepare_speech_commands --device cuda --output data/prepared/speech_commands
python -m examples.export_and_infer --source logs/models/mnist_resnet --output logs/export/mnist-first --onnx
```

Tuning uses validation-only sessions: no test dataset is included in trials. Export scripts require new destinations and explicit trained sources. Run the scripts as modules from the repository root.

Focused offline checks (no dataset downloads):

```bash
uv pip install pytest
python -m pytest tests -q
```
