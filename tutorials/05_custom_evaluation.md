# Custom Evaluation: A Streaming Confusion Matrix

Reuse the [MNIST scenario](01_mnist_from_scratch.md). Add one consumer-only evaluator; leave the training loop and classification pipeline unchanged.

## Definition Versus Runtime

`library/evaluation/confusion_matrix.py` registers the typed `TutorialConfusionMatrix` definition. Its `build(EvalBuildContext)` constructs mutable runtime state using declared `feature_key` and `label_key` routing.

The runtime delegates counting to TorchMetrics `MulticlassConfusionMatrix`. It stores only a `C × C` matrix, not all predictions. Training-fit batches do not contribute. Test batches supply `class_logits [B, C]` and integer-valued `class [B]`; missing declared tensors, wrong logits shape, nonintegral labels, and empty evaluations fail clearly.

Rows are **true** classes; columns are **predicted** classes. Scalar results are:

- `micro_accuracy = trace(matrix) / matrix.sum()`;
- `samples = matrix.sum()`.

This explicitly defined micro accuracy need not equal a macro-averaged training metric.

## Run Through the Public Extension Contract

Install the README environment (`.[advanced]` enables MLflow), then run from the repository root:

```bash
nexuml registry list eval
python - <<'PY'
from library.config.scenario import mnist_resnet
from library.evaluation import ConfusionMatrix
from nexuml.core.types import EvalAlgorithmSpec
from nexuml.training.lightning import NexuSession

spec = mnist_resnet(max_epochs=1)
spec.evaluation.algorithms = [EvalAlgorithmSpec(
    algorithm=ConfusionMatrix(class_names=[str(i) for i in range(10)]),
    feature_key="class_logits", label_key="class",
)]
session = NexuSession.from_scenario(spec)
session.fit()
session.test()
print(session.lightning_module.evaluation_results)
PY
```

The first run may download MNIST. Expected scalar keys:

```text
TutorialConfusionMatrix/micro_accuracy
TutorialConfusionMatrix/samples
```

The sample count is 10,000 for the official MNIST test split. NexuML calls `eval_batch`, `eval_end`, visualization, and scalar-result collection. The runtime clears prior counts on the first batch after a completed evaluation, so repeating `session.test()` does not double the count. Its optional `fit_end()` reset also supports direct use of the full evaluator lifecycle.

## View the Artifact

The evaluator reuses `library/evaluation/_artifact.py`, rather than implementing another logger adapter. TensorBoard receives the labeled figure at `eval/confusion_matrix`; MLflow receives `eval/confusion_matrix.png` under its run's artifacts.

```bash
tensorboard --logdir logs/tensorboard --host 127.0.0.1
mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db --host 127.0.0.1
```

For scalar results without tracking, construct the session with `enable_loggers=False`. The matrix remains evaluator state, not another pipeline output key; score-producing transformations belong in the model pipeline instead.

## Offline Verification

```bash
python -m pytest tests/test_advanced_tutorials.py -k confusion -q
```

Checks use known counts including an absent class, reject missing keys/empty evaluation, verify repeated lifecycle reset, inspect a TensorBoard image event, and execute a synthetic NexuML fit/test session without network downloads.

Next: [checkpoints and transfer learning](06_checkpoints_and_transfer_learning.md).
