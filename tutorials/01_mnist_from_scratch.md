# MNIST: Build a Local NexuML Library

Run from the repository root after the [README setup](../README.md#setup).

```bash
nexuml library list
nexuml registry list data
nexuml registry list layers
nexuml registry list scenarios

nexuml resolve tutorial-mnist-resnet
nexuml build configs/tutorial-mnist-resnet.yaml
nexuml train tutorial-mnist-resnet --max-epochs 1
```

Local components live in `library/data/mnist.py`, `library/layers/`, and `library/config/scenario/mnist_resnet.py`. The scenario uses typed definitions and connects stages through TensorDict keys:

```text
features -> embeddings -> pooled_embeddings -> class_logits
                                      class -> classification_loss, accuracy, f1
```

MNIST uses the Torch loader with an in-memory torchvision dataset. The run records accuracy/F1, embedding visualizations, checkpoints, and a trained package.

Next: [native-DALI audio](02_audio_native_dali.md).
