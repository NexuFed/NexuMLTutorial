# Checkpoints and Transfer Learning

These are different operations:

| Operation | Restores | Training state |
| --- | --- | --- |
| Trainer resume | Original model, optimizer/scheduler, epoch/step, compatible callbacks | Continues the same run |
| Selective transfer | Explicitly matched weights | New task, head, optimizer, and run |

Use the README environment and `uv pip install -e '.[advanced]'` for default tracking. Run commands from the repository root. Only load trusted checkpoints/packages: these formats can contain Python objects/code.

## Produce an Explicit Source

```bash
nexuml train tutorial-mnist-resnet --max-epochs 1
```

The configured callback saves `logs/checkpoints/mnist_resnet/last.ckpt` and a validation-monitored checkpoint. The CLI also exports the **live trained pipeline** to `logs/models/mnist_resnet/`. Retain the printed concrete artifact paths; repeated runs can create alternate checkpoint names. Do not guess a latest directory.

The first run downloads MNIST if absent. For a new independent source experiment, choose new callback/export paths in the scenario instead of overwriting earlier work.

## Resume the Trainer

```bash
python -m examples.checkpoints_and_transfer resume --checkpoint logs/checkpoints/mnist_resnet/last.ckpt --max-epochs 2
```

The script uses `NexuSession.from_trainer_checkpoint`, sets the session's total target `max_epochs`, then calls `fit()`. Two means **two total epochs**, not two more epochs. It prints the resumed epoch/global step. The equivalent NexuML CLI is:

```bash
nexuml train --trainer-checkpoint logs/checkpoints/mnist_resnet/last.ckpt --max-epochs 2
```

CLI training additionally performs validation/test/export. Selective `CheckpointLoadSpec` is not a replacement for this resume path. Keep original data and compatible callbacks when resuming; changing checkpoint directories can prevent restoration of checkpoint-selection state.

## Transfer MNIST's Encoder to FashionMNIST

`TutorialFashionMNISTDataset` delegates to torchvision FashionMNIST through the existing adapter. The image shape and ten-class head contract stay `[1, 28, 28]` and `[B, 10]`; labels now refer to clothing categories, not digits.

`tutorial-fashion-mnist-transfer` reuses the MNIST model but declares:

```python
CheckpointLoadSpec(
    source=source,
    include=["stages.Encoder.*"],
    allow_missing=True,       # head is deliberately new
    allow_shape_mismatch=False,
    freeze_loaded=True,
)
```

The actual compiled prefix is `stages.Encoder.`. The script prints the load report and requires a nonempty match before training; selected shape mismatches fail. The source must have the same encoder width/depth as the target (the documented workflow uses the default 32/2 encoder). Matching head shape is **not** permission to copy the source digit classifier.

```bash
python -m examples.checkpoints_and_transfer transfer --source logs/models/mnist_resnet --output logs/transfer/fashion-first --max-epochs 1
```

`--source` accepts a package directory, SafeTensors file, or trainer checkpoint supported by core's weight loader. The output directory must be new. This run may download FashionMNIST under `data/fashion_mnist/`.

The example performs two phases:

1. **Head-only:** load and freeze encoder parameters, initialize a fresh clothing head, train for one epoch, and export `head/`.
2. **Fine-tune:** load all target weights from `head/` into a new session with trainable parameters and learning rate `1e-4`, then train/test/export `fine/`.

The second phase intentionally starts a **new optimizer**, not a resumed source optimizer. Checkpoints live under `head_checkpoints/` and `fine_checkpoints/` within the chosen output.

`freeze_loaded` freezes parameters, **not BatchNorm running statistics**. Those buffers still update during head-only training. Fixed statistics would require an explicitly eval-mode backbone; this example does not add a freezing manager. It makes no claim that digit pretraining improves clothing accuracy.

The target's ten labels, in torchvision order, are T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot.

## Offline Verification

```bash
python -m pytest tests/test_advanced_tutorials.py -k 'transfer or resume' -q
```

Synthetic checks cover source selection, encoder-only matches, fresh head state, parameter freezing/head updates, trainable fine-tuning, incompatible shapes, and trainer model/optimizer/scheduler/epoch-step recovery. No datasets are downloaded by these tests.

Next: [model export and inference](07_model_export_and_inference.md).
