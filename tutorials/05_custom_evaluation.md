# Custom Evaluation: A Streaming Confusion Matrix

After the [README setup](../README.md#setup), run from the repository root:

```bash
uv pip install -e '.[advanced]'
nexuml registry list eval
python - <<'PY'
from library.config.scenario import mnist_resnet
from library.evaluation import ConfusionMatrix
from nexuml.core.types import EvalAlgorithmSpec
from nexuml.training.lightning import NexuSession

spec = mnist_resnet(max_epochs=1)
spec.data.loader.num_workers = 0  # stdin scripts cannot be reopened by spawned workers.
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

Expected results: `TutorialConfusionMatrix/micro_accuracy` and `TutorialConfusionMatrix/samples` (10,000 for MNIST test). Rows are true classes; columns are predicted classes. Only test batches contribute; repeated evaluation resets counts.

The evaluator in `library/evaluation/confusion_matrix.py` retains only a class-count matrix. TensorBoard receives `eval/confusion_matrix`; MLflow receives `eval/confusion_matrix.png`. Use the [tracking viewers](03_tuning_and_tracking.md#compare-runs) to inspect them.

Next: [checkpoints and transfer learning](06_checkpoints_and_transfer_learning.md).
