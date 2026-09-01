# NexuML Tutorial

This repository is a standalone external NexuML library. Every dataset, layer, evaluation, and scenario shown by the registry commands below is implemented in this repository.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

export NEXUML_DATA_ROOT=$(pwd)/data
export NEXUML_LOGS_ROOT=$(pwd)/logs
```

The `nexuml[dali]` dependency installs NexuML's supported DALI extra. Native DALI is platform-specific; before running the audio tutorial, verify that this environment reports the loader:

```bash
nexuml backend list
```

On GPU systems, the installed DALI build also needs a compatible NVIDIA driver/CUDA runtime.

## Local Library

Register this checkout and inspect its components:

```bash
nexuml library add $(pwd)/library
nexuml library list

nexuml registry --help
nexuml registry list data
nexuml registry list layers
nexuml registry list eval
nexuml registry list scenarios
```

## Learning Path

| Stage | Topic | Status |
| --- | --- | --- |
| 1 | [MNIST: custom library basics](tutorials/01_mnist_from_scratch.md) | Available |
| 2 | [Speech Commands CNN: file-backed native DALI](tutorials/02_audio_native_dali.md) | Available |
| 3 | [Speech Commands Transformer: pipeline composition](tutorials/02_audio_native_dali.md#swap-only-the-encoder) | Available |
| 4 | Tuning and experiment tracking | Planned |
| 5 | Preprocessing, dataset export, and WebDataset | Planned |
| 6 | Custom evaluation | Planned |
| 7 | Checkpoints and transfer learning | Planned |
| 8 | Model export and inference | Planned |
| 9 | Distributed execution after NexuML exposes a stable backend | Planned |

## Quick Commands

```bash
nexuml resolve mnist-resnet
nexuml build configs/mnist-resnet.yaml
nexuml train mnist-resnet --max-epochs 1

nexuml resolve speech-commands-cnn
nexuml build configs/speech-commands-cnn.yaml
nexuml train speech-commands-cnn --max-epochs 10

nexuml resolve speech-commands-transformer
nexuml build configs/speech-commands-transformer.yaml
nexuml train speech-commands-transformer --max-epochs 10
```

The existing tuning example remains available:

```bash
nexuml tune --scenario-file library/config/tune/mnist_resnet.py --n-trials 10
```
