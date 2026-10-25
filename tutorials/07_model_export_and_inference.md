# Model Export and Label-Free Inference

Reuse the trained MNIST package from [chapter 1](01_mnist_from_scratch.md). Run from the repository root; export destinations must be new:

```bash
uv pip install -e '.[advanced,onnx]'
python -m examples.export_and_infer --source logs/models/mnist_resnet --output logs/export/mnist-first
python -m examples.export_and_infer --source logs/models/mnist_resnet --output logs/export/mnist-onnx-first --onnx
```

The script verifies rebuilt/package parity, exports and reloads SafeTensors into the same architecture, and optionally checks CPU ONNX Runtime for batch sizes 1 and 2. Comparisons use `rtol=1e-4`, `atol=1e-5`; seeded inputs check portability, not accuracy.

## Infer Without Labels

```bash
python - <<'PY'
from pathlib import Path
import torch
from tensordict import TensorDict
from nexuml.core.export import infer, load_package

pipeline, config, metadata = load_package(Path("logs/models/mnist_resnet"))
inputs = TensorDict({"features": torch.rand(2, 1, 28, 28)}, batch_size=[2])
logits = infer(pipeline, inputs)["class_logits"]
print(logits.shape, logits.argmax(-1))
PY
```

Input: float images in `[0, 1]`. Output: unnormalized logits `[B, 10]`; labels are unnecessary. `infer` uses evaluation/no-grad behavior.

`load_package` rebuilds using installed code; `load_inference_package` loads packaged runtime code. Both require compatible dependencies. Load only trusted packages; SafeTensors alone still needs model configuration/code. Always supply an explicit trained source rather than assuming standalone export discovers the latest checkpoint.

Inference service deployment and distributed execution remain roadmap topics.
