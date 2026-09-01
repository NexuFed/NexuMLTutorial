# MNIST: Build a Local NexuML Library

MNIST is the smallest example in this repository. It introduces NexuML registration and pipeline composition without requiring a file-backed loader.

## What Is Local

The package entry point exposes the importable `library` package. Its tutorial-owned pieces are:

- `library/data/mnist.py`: the registered `MNISTDataset` source;
- `library/layers/model/resnet.py`: the registered `ResNetEncoder`;
- `library/layers/head/classification.py`: the shared logits head;
- `library/layers/loss/cross_entropy.py`: the shared multiclass loss;
- `library/config/scenario/mnist_resnet.py`: the registered `mnist-resnet` scenario.

Register the checkout and inspect those registries:

```bash
nexuml library add $(pwd)/library
nexuml library list
nexuml registry list data
nexuml registry list layers
nexuml registry list scenarios
```

## Resolve, Build, Train

`resolve` expands the registered scenario into a concrete YAML configuration. `build` validates and constructs its dataset/model contracts. `train` runs the configured lifecycle.

```bash
nexuml resolve mnist-resnet
nexuml build configs/mnist-resnet.yaml
nexuml train mnist-resnet --max-epochs 1
```

The scenario composes `ScenarioSpec` sections for data, pipeline, training, evaluation, logging, callbacks, and export. TensorDict keys connect its stages:

```text
features -> embeddings -> pooled_embeddings -> class_logits
                                      class -> classification_loss, accuracy, f1
```

MNIST is held by torchvision in `dataset.data`, so NexuML treats it as an in-memory dataset. That is useful for a first example, but it intentionally does not demonstrate native DALI file loading. Continue with [the audio tutorial](02_audio_native_dali.md) for that path.
