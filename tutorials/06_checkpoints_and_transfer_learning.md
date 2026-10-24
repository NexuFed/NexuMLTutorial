# Checkpoints and Transfer Learning

Use the [README setup](../README.md#setup) with `.[advanced]`. Load only trusted checkpoints/packages: they can contain Python objects or code.

## Produce a Source

Skip this command if you retained the trained MNIST artifacts from chapter 1:

```bash
nexuml train tutorial-mnist-resnet --max-epochs 1
```

Use the concrete artifact paths from your run. Defaults are `logs/checkpoints/mnist_resnet/last.ckpt` and `logs/models/mnist_resnet/`; repeated runs may use alternate checkpoint names. Choose new callback/export paths for independent experiments.

## Resume the Trainer

```bash
python -m examples.checkpoints_and_transfer resume --checkpoint logs/checkpoints/mnist_resnet/last.ckpt --max-epochs 2
```

Resume restores model, optimizer/scheduler, epoch/step, and compatible callbacks. Two means two total epochs, not two additional epochs. Keep the original data and compatible callback paths.

Use the example script on the tested NexuML checkout: checkpoint-only CLI resume ignores the `--max-epochs` override.

## Transfer to FashionMNIST

```bash
python -m examples.checkpoints_and_transfer transfer --source logs/models/mnist_resnet --output logs/transfer/fashion-first --max-epochs 1
```

The destination must be new. The source may also be a compatible SafeTensors file or trainer checkpoint.

The example loads only `stages.Encoder.*` into a matching 32/2 encoder, rejects shape mismatches, and requires nonempty matches. It trains a fresh clothing head with frozen encoder parameters (`head/`), then fine-tunes all target weights with a new optimizer at `1e-4` (`fine/`). FashionMNIST downloads if absent.

Freezing parameters does not freeze BatchNorm running statistics. No claim is made that digit pretraining improves clothing accuracy.

Next: [model export and inference](07_model_export_and_inference.md).
