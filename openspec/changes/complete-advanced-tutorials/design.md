# Design

## Context

See `proposal.md` for motivation and the six capability specs for acceptance contracts. This repository already supplies typed MNIST and audio sources, a channel-inferred 2D ResNet, pooling/head/loss/metrics, a torchaudio log-mel component, logger defaults, and a visualization helper. The completed, unarchived audio change is context only; do not rewrite it.

The inspected NexuML source exposes the required public session, preprocessing, dataset-reader, checkpoint, and export APIs. Two current behaviors matter: the built-in tuner runs test before retrieving its objective, and automatic CLI post-training exports handle `train_package` only. Documentation must not hide either limitation.

## Goals / Non-Goals

**Goals:**

- Demonstrate each new contract with a small extension of an existing experiment.
- Keep TensorDict keys, dataset-owned splits, devices, and trained-weight provenance explicit.
- Make offline checks independent of network downloads and optional CUDA/ONNX/tracking installations.

**Non-Goals:**

- No generic example runner, backend abstraction, custom batch/training loop, or copied framework implementation.
- No parallel CPU/GPU preprocessing implementations or new spectrogram encoder.
- No new base-library dependency, upstream repository edits, legacy shims, or broad cleanup.

## Decisions

### 1. Extend existing fragments; add only purposeful entrypoints

Keep configuration in `library/config/`, components in their existing component directories, and three small executable API demonstrations under `examples/`. They use `argparse` and normal functions; they are not a framework. Registry/discovery imports follow the existing package pattern.

Intended implementation structure (`*` means a new file):

```text
README.md
pyproject.toml
library/
  data/
    mnist.py                         # shared torchvision adapter; FashionMNIST sibling definition
    exported.py *                    # typed wrapper around the NexuML core exported dataset
    __init__.py
  config/
    data/
      mnist.py                       # retain MNIST defaults; allow the FashionMNIST source
      exported.py *                  # prepared-feature data contract
      __init__.py
    model/
      resnet_classifier.py           # existing downstream image classifier, unchanged if possible
    scenario/
      speech_commands.py             # raw/prepared log-mel scenario composition
      transfer.py *                  # FashionMNIST with explicit selective load configuration
      __init__.py
    tune/
      mnist_resnet.py                # validation-only Optuna example and executable entrypoint
  evaluation/
    confusion_matrix.py *
    _artifact.py                     # reuse; do not introduce another logger adapter
    __init__.py
  layers/feature/lmbe.py              # reuse; edit only for a proven relevant defect
examples/
  prepare_speech_commands.py *        # explicit export device and preprocessing boundary
  checkpoints_and_transfer.py *       # public session lifecycle and checkpoint/load reports
  export_and_infer.py *               # package, SafeTensors, and ONNX demonstrations
tutorials/
  03_tuning_and_tracking.md *
  04_audio_preprocessing_and_export.md *
  05_custom_evaluation.md *
  06_checkpoints_and_transfer_learning.md *
  07_model_export_and_inference.md *
tests/
  test_scenario_api.py                # extend existing typed-config coverage
  test_advanced_tutorials.py *         # focused offline checks, not a backend test matrix
```

Make small updates to existing introductory chapters only to link later work or remove claims that log-mel/export are unavailable. Generated resolved YAML is an output of documented commands, not a second manually maintained source of truth.

Alternative rejected: independent implementations of each chapter. They obscure composition and multiply drift. Extract a helper only when two real call sites need the same behavior.

### 2. Use public fit/validate sessions for tuning

Replace the unsafe example in `library/config/tune/mnist_resnet.py` with a short standard Optuna objective. Sample only learning rate, batch size, and encoder width, call the existing MNIST scenario factory, then use `NexuSession.from_scenario(...).setup().fit().validate()`. Return an explicitly checked finite validation loss. Never call `run()`, `train()`, or `test()` inside a trial.

Use a fixed seed and reproducible split configuration for every trial. Trial data contains only the official MNIST training source, split 80/20 into training/validation with no test allocation; the official test source is omitted entirely. Restore the official test source only for the final selected experiment while preserving the training/validation split. Disable post-training evaluators and per-trial package exports because model selection does not need them. Keep tracking/run names distinct, use a documented local study store, and record sampled parameters. Evaluate the chosen model on held-out test data only after selection, using an explicitly identified trained/best model.

The executable command is `python -m library.config.tune.mnist_resnet --n-trials 2 --max-epochs 1`; keep this entrypoint specific to the tutorial. Update the README's existing unsafe `nexuml tune` recommendation. Explain why this example uses the session API rather than presenting a broken validation CLI recipe.

Alternative rejected: replacing `test/f1` with `val/loss` in the existing file without proving the CLI exposes it. Reading the current tuner shows that would not solve test access and might fail metric retrieval. No tuner monkeypatch or copied search engine is needed: Optuna owns search and NexuML owns training.

### 3. Log-mel is one pipeline stage; ResNet is already sufficient

Compose the existing image-classification pipeline with a preceding `Preprocessing` stage:

```text
waveform [B, 16000], float32, 16 kHz
  -> torchaudio log-mel
features [B, 1, 64, 101]
  -> existing ResNet -> pooling -> existing head
class_logits [B, 8]
```

Use explicit LMBE settings: sample rate 16,000; `n_fft=400`; `win_length=400`; `hop_length=160`; `n_mels=64`; power 2; frequency range 0–8,000 Hz; constant padding; Slaney scale; dB conversion enabled. Confirm the 101-frame contract against the installed transform's centering behavior. Keep the transform as registered Torch submodules so normal model device movement moves its buffers too. Test silence and batched inputs; make no CPU/NumPy detour for DSP.

Reuse `resnet_classifier()` as the entire downstream fragment: its `features` input and channel-inferred stem already accept spectrograms. Registered scenario names are `speech-commands-log-mel` and `speech-commands-log-mel-prepared`. Both preserve the eight-class mapping and speaker-owned splits.

Alternative rejected: a separate spectrogram CNN, handwritten STFT/mel filters, or librosa. Existing code plus torchaudio already covers the requested lesson.

### 4. Delegate prepared-view storage and loading to core

`examples/prepare_speech_commands.py` builds the raw log-mel scenario, creates its runtime, and calls public `export_data_module` with `LightningFeatureExtractor(x_keys=["features"])`, `x_keys=["features"]`, `y_keys=["class"]`, and an explicit device. The only format selection is NumPy or WebDataset, because both are requested. Default to float32 storage; do not add compression/precision knobs without a demonstrated need.

Passing `device="cuda"` delegates module/batch device movement to the core exporter. Document that the existing CLI `export-dataset` route does not select CUDA automatically; it is not evidence of GPU preprocessing. Also explain the equivalent `DataSpec.preprocessing` boundary without adding a second custom materialization path.

Add a small local `DataSourceDefinition` that constructs `nexuml.data.exported.ExportedDataset`. Do not import/copy the wrapper from `nexuml_library`, parse its schema manually, or write a tar reader. The prepared scenario sets `split_type="keep"`, `feature_key="features"`, the known feature shape, and `skip_pipeline_stages=["Preprocessing"]`. It retains the same pipeline/stage names so the skip is visible rather than replacing the model.

Use native `DaliLoader` for the existing raw WAV path and the documented WebDataset loading path where supported; keep a core `TorchLoader` prepared-data path available for NumPy/offline verification. Verify backend requirements rather than silently downgrading a claimed native DALI run.

Compare exported features by stable sample identity, not shuffled iteration position. Validate labels, splits, counts, float32 feature tolerance, and downstream logits. Refuse accidental overwrite in the example rather than silently replacing an existing export; let users choose a new destination.

Alternative rejected: tutorial-owned dataset export/reader utilities or adding the entire base library for a tiny typed adapter.

### 5. The evaluator accumulates counts, not predictions

Register one typed `TutorialConfusionMatrix` evaluator backed by the already-installed TorchMetrics `MulticlassConfusionMatrix`. Read `class_logits` and `class` through the evaluation contract. Training-fit batches do not contribute to test counts; reset at the start of each full fit/evaluate lifecycle using the supported lifecycle seam, and verify repeated runs.

Retain only the class-count matrix, compute micro accuracy as `trace(matrix) / matrix.sum()`, and expose it plus the evaluated sample count through scalar results. Raise a clear error for missing declared tensors or an empty evaluation rather than emit NaN. The matrix remains evaluator state, not a new pipeline score key. Plot a labeled true/predicted matrix with Matplotlib and reuse `_artifact.log_figure`.

Configure the evaluator directly in a scenario/example, using the existing MNIST factory and `EvalAlgorithmSpec`; do not create a duplicate classifier just to register an evaluation lesson. Document class order and the scalar definition so it is not confused with macro-averaged training metrics.

Alternative rejected: buffering all logits and computing sklearn reports after training. Streaming counts are smaller and teach the extension lifecycle cleanly.

### 6. Resume and transfer are separate operations

Resume uses the saved callback path and the public trainer-checkpoint mechanism; do not feed a resume checkpoint to a selective load and call it equivalent. Verify recovered model and optimizer state plus epoch/global-step continuation on a tiny synthetic run.

For actual transfer, add a typed FashionMNIST sibling source using torchvision and the existing MNIST runtime adapter/data fragment. This adds a real target task without another dataset loader or classification stack. The `tutorial-fashion-mnist-transfer` scenario preserves encoder structure, constructs a fresh head, and sets `CheckpointLoadSpec` with an explicit encoder allow-list, `allow_missing=True` for intentionally new head state, and `allow_shape_mismatch=False` for selected encoder tensors. Use actual compiled state-dict keys; assert the load report matched an encoder tensor.

The script first uses `freeze_loaded=True` to train the target head, then starts a new fine-tuning session from the target package with trainable weights and a fresh optimizer. This is deliberate phase initialization, not same-run resume. Do not claim MNIST features improve FashionMNIST accuracy.

`freeze_loaded` freezes parameters, not BatchNorm running statistics. Document that buffer updates continue in train mode; do not add a callback or override normal session mode behavior to hide it. A relevant `ponytail:` comment in the example names this limitation and points to an explicitly frozen/eval backbone if fixed statistics become necessary.

Alternative rejected: artificial transfer between identical heads on the same dataset, pretrained-model frameworks, or a generic freezing manager.

### 7. Export the existing image classifier, not DSP-heavy ONNX

`examples/export_and_infer.py` uses an explicitly trained MNIST checkpoint or package; it never falls back to a random model. Prefer the existing automatic `train_package` export to provide the source artifact. For an explicit checkpoint export, use NexuML's package API and retain source metadata.

Demonstrate `load_package` and `load_inference_package`, then label-free `infer` with `features` inputs and `class_logits` outputs. Export SafeTensors through `export_safetensors`, load them into the same configured model with the public weight loader, and compare predictions. Export ONNX through `export_onnx(input_key="features", output_key="class_logits")`, then compare with ONNX Runtime on fixed image inputs and the batch sizes actually supported.

The image classifier avoids unnecessary STFT/complex-op portability problems; GPU-capable audio preprocessing is already taught separately. Do not write a tutorial ONNX wrapper or imply that `ExportSpec(kind="onnx")` is automatically handled after CLI training.

### 8. Dependencies, proof, and documentation stay scoped

Use project optional groups `advanced` for NexuML tuning/tracking extras and `onnx` for actual exporter/runtime requirements. Keep the base install free of avoidable integrations and keep registry imports usable without those optional packages. Torchaudio, TorchMetrics, torchvision, and Matplotlib are already present; no new DSP or model library is needed. Inspect installed/package metadata before declaring minimum versions; do not invent a version bump.

Extend the existing scenario/config test and add one `test_advanced_tutorials.py` module. Use synthetic waveforms/images and temporary local artifacts. Cover validation-only tuning, finite device/shape contracts, export/reload split parity, confusion counts/reset, selective-load/freeze/resume behavior, and inference round trips. CUDA and ONNX checks have explicit skips when prerequisites are unavailable; default checks must not download datasets.

Each new tutorial states what changed from the previous chapter, which components were reused, setup commands, minimal execution commands, expected keys/artifacts, and verification. README stages 4–8 become available only after their implementations and checks exist; stage 9 stays planned. No placeholder code is added just to make the table look finished.

## Risks / Trade-offs

- [Installed NexuML may differ from the inspected sibling source] → Verify actual import targets and public contracts first; fail with a clear minimum-version prerequisite. No compatibility layer or sibling-repo edits.
- [DALI/CUDA may be unavailable] → Separate offline CPU proof from documented native real-data runs; explicitly report GPU/DALI checks that could not run.
- [Core export schema or prepared WebDataset loading may reject a contract] → Add a focused round-trip proof before writing final commands; report an upstream blocker instead of duplicating the backend.
- [Batch-dependent dB scaling or transform shape assumptions] → Verify batch/singleton parity and silence against the fixed transform configuration before claiming prepared/online equivalence.
- [Normalization buffers still update while parameters are frozen] → State the precise freeze semantics and test parameters; do not claim immutable encoder statistics.
- [ONNX exporter compatibility varies across Torch versions] → Use the existing single-input image model and public exporter; record tested dependencies/tolerances, and do not label an untested export successful.
- [Study/tracking/checkpoint artifacts can collide across runs] → Use explicit destinations/run identities and refuse destructive example overwrite; avoid guessed latest paths.

## Migration Plan

This change is additive except for the deliberately corrected tuning example and related commands. Existing MNIST and raw-waveform scenarios retain their registry names and default behavior. No user dataset or saved artifact is rewritten. Replace the old unsafe tuning recommendation rather than retain a legacy demo alias; document the new module entrypoint.

Implementation is scoped to this repository. Rollback removes the added examples/components/documents and restores only changes made for this change; generated model/data/study artifacts remain user-owned and are never deleted as part of rollback.
