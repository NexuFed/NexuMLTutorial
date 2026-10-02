# Model Export and Label-Free Inference

Reuse the trained MNIST classifier from the [checkpoint chapter](06_checkpoints_and_transfer_learning.md). A package, a weights file, and an ONNX graph solve different problems; none should silently substitute random weights for a missing trained source.

## Obtain a Trained Package

```bash
uv pip install -e '.[advanced,onnx]'
nexuml train tutorial-mnist-resnet --max-epochs 1
```

The configured `train_package` export writes the live trained model under `logs/models/mnist_resnet/`. When using sessions directly, export explicitly after `fit()`:

```python
from pathlib import Path
from nexuml.core.export import export_package

export_package(session.pipeline, Path("logs/models/my_run"), trainer=session.trainer)
```

NexuML CLI training automatically handles `train_package` in 0.2.1, not every `ExportSpec` kind. Standalone `nexuml export` without `--checkpoint` does **not** discover the latest trained checkpoint. Prefer an explicit source path.

## Reload Two Ways

```python
import torch
from pathlib import Path
from tensordict import TensorDict
from nexuml.core.export import infer, load_package, load_inference_package

pipeline, config, metadata = load_package(Path("logs/models/mnist_resnet"))
packaged, _, _ = load_inference_package(Path("logs/models/mnist_resnet"))
x = TensorDict({"features": torch.rand(2, 1, 28, 28)}, batch_size=[2])
logits = infer(pipeline, x)["class_logits"]
print(logits.shape, logits.argmax(-1))
```

`load_package` rebuilds from the resolved configuration using current installed tutorial code. `load_inference_package` loads the packaged pipeline/runtime source. Both still require compatible Python, Torch, NexuML, TensorDict, and external dependencies; packaging is not a standalone executable or an ABI guarantee. Load only trusted packages.

Input is float image tensors `[B, 1, 28, 28]` in `[0, 1]`, under `features`. Output is unnormalized `class_logits [B, 10]`. `infer` selects eval/no-grad behavior. Labels are unnecessary; loss/metric stages do not block label-free inference. Use `argmax` for class IDs or `softmax` when probabilities are needed.

## SafeTensors and ONNX

```bash
python -m examples.export_and_infer --source logs/models/mnist_resnet --output logs/export/mnist-first
python -m examples.export_and_infer --source logs/models/mnist_resnet --output logs/export/mnist-onnx-first --onnx
```

Destinations must be new. The example first checks package reload parity, then exports `weights.safetensors` plus a JSON manifest and reloads weights into the **same configured architecture**, rejecting missing/mismatched state. SafeTensors is a weights artifact: model configuration/code are still required.

With `--onnx`, the public core `export_onnx` consumes `features` and produces `class_logits`. ONNX Runtime verifies the graph on CPU for batch sizes **1 and 2**, comparing logits and predicted classes. The image classifier avoids unnecessary STFT/complex-op export problems; the [audio chapter](04_audio_preprocessing_and_export.md) already demonstrates GPU-capable DSP separately.

Both reload and ONNX comparisons use `rtol=1e-4`, `atol=1e-5`. The script uses fixed seeded synthetic image inputs for portability checks, not accuracy measurement. Exporting the model is separate from deploying an inference service; no service framework is added.

## Verify

```bash
python -m pytest tests/test_advanced_tutorials.py -k 'package or safetensors or onnx' -q
```

The synthetic test trains a tiny classifier before comparing its live, rebuilt, packaged, SafeTensors, and ONNX outputs. Optional ONNX dependencies skip explicitly when absent; a skipped ONNX check is not a passing graph verification. CUDA/DALI is not needed for image export or CPU ONNX Runtime.

Distributed execution remains a deferred roadmap topic.
