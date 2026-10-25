# Tasks

## 1. Verify public contracts and scope dependencies

- [x] 1.1 Inspect the installed NexuML import location, version, and required session/data/export APIs; verify they match the design with a read-only import/signature check and record any unsupported upstream contract before implementation. Do not edit the sibling NexuML repository.
- [x] 1.2 Run the existing baseline with `.venv/bin/python -m pytest tests/test_scenario_api.py tests/test_audio_models.py tests/test_audio_scenarios.py tests/test_mini_speech_commands.py -q`; report pre-existing failures separately and preserve existing scenario names/defaults.
- [x] 1.3 Add only required optional `advanced` and `onnx` dependency groups, keeping new optional imports out of registry discovery; verify editable base/extra installations in the supported environment and `.venv/bin/nexuml registry list scenarios` without discovery errors. Do not add `nexuml_library` or a DSP/model convenience framework.

## 2. Validation-only tuning and tracking

- [x] 2.1 Replace the existing unsafe tuning example with a small Optuna objective calling public NexuML fit/validate sessions; omit the official test source from trials, use a fixed 80/20 training/validation split and seed, check finite `val/loss`, and suppress unnecessary evaluators/exports. Verify a synthetic two-trial check fails if any trial calls test or includes a test source.
- [ ] 2.2 Add the module's minimal trial/epoch arguments, explicit local study storage, distinguishable tracking runs/parameters, and a final selected-model evaluation after restoring the held-out source; verify best parameters/objective are reported, missing/non-finite objectives fail clearly, and test is accessed only after selection.
- [ ] 2.3 Write `tutorials/03_tuning_and_tracking.md` with optional setup, the session-versus-current-CLI distinction, artifact/viewing commands, and the corrected entrypoint; verify `python -m library.config.tune.mnist_resnet --n-trials 2 --max-epochs 1` on available real data and the offline tuning check with `.venv/bin/python -m pytest tests/test_advanced_tutorials.py -k tuning -q`.

## 3. Log-mel preprocessing and prepared dataset reuse

- [x] 3.1 Compose `speech-commands-log-mel` from the existing LMBE and ResNet classification fragments with the documented 16 kHz/64-mel configuration; verify synthetic random and silent inputs yield finite `[B, 1, 64, 101]` features and `[B, 8]` logits using `.venv/bin/python -m pytest tests/test_advanced_tutorials.py -k log_mel -q`. Run the same device check on CUDA when available and report a skip otherwise.
- [x] 3.2 Add the minimal typed exported-data definition/data fragment and `speech-commands-log-mel-prepared` scenario, delegating to core `ExportedDataset` and skipping only `Preprocessing`; verify YAML round-trip/discovery and an offline check that prepared features reach the reused classifier without another log-mel transform.
- [x] 3.3 Add `examples/prepare_speech_commands.py` as a direct public-export API example with explicit device, feature/label boundary, NumPy/WebDataset selection, and non-destructive destination handling; verify small synthetic export/reload parity for sample identities, labels, splits, counts, and features with `.venv/bin/python -m pytest tests/test_advanced_tutorials.py -k prepared -q`.
- [x] 3.4 Write `tutorials/04_audio_preprocessing_and_export.md` with raw/prepared pipeline contracts, explicit GPU export, core materialization configuration, and native WebDataset prerequisites; verify `.venv/bin/nexuml resolve speech-commands-log-mel`, `.venv/bin/nexuml resolve speech-commands-log-mel-prepared`, and `.venv/bin/nexuml backend list data-loader`, then execute the documented export/reload/train smoke path when DALI/CUDA and real data are available. Report unexecuted native checks explicitly.

## 4. Custom evaluation

- [x] 4.1 Implement and register the minimal TorchMetrics-backed confusion-matrix evaluator with declared key/class validation, lifecycle reset, bounded counts, micro-accuracy/sample-count results, and no training-fit contamination; verify exact counts including an absent class, a repeated lifecycle, and missing-key/empty-evaluation errors with `.venv/bin/python -m pytest tests/test_advanced_tutorials.py -k confusion -q`.
- [x] 4.2 Reuse the existing Matplotlib artifact helper and configure the evaluator through the existing MNIST scenario's evaluation extension surface; verify scalar results without logging, a labeled saved/logged figure with a supported local logger, and `.venv/bin/nexuml registry list eval` with the new evaluator present.
- [x] 4.3 Write `tutorials/05_custom_evaluation.md` with a complete runnable configuration/session snippet, extension lifecycle, class order, scalar semantics, and artifact path; execute that snippet on available data or the synthetic equivalent and verify its stated outputs match the evaluator check.

## 5. Checkpoints and real transfer learning

- [x] 5.1 Extend the existing torchvision MNIST adapter/data fragment minimally for a typed FashionMNIST sibling source without changing MNIST defaults; verify dataset selection, label shape/class count, and official train/test separation offline using a small mocked torchvision payload plus the existing scenario tests.
- [x] 5.2 Add `tutorial-fashion-mnist-transfer` with an explicit encoder-only checkpoint allow-list and strict selected-shape handling; verify actual matched encoder values, a fresh head, nonempty match reporting, and clear missing/no-match/shape-error failures with `.venv/bin/python -m pytest tests/test_advanced_tutorials.py -k transfer -q`.
- [x] 5.3 Add the checkpoint/transfer example using public resume and selective-load APIs; demonstrate frozen-encoder, head-only training followed by a fresh fine-tuning phase. Verify frozen encoder parameters stay unchanged, head parameters can update, and fine-tuning enables encoder gradients; document BatchNorm buffer behavior with a targeted `ponytail:` comment naming the fixed-statistics upgrade path.
- [x] 5.4 Add a focused synthetic resume check proving model/optimizer state recovery and epoch/global-step continuation from an explicit callback checkpoint; verify `.venv/bin/python -m pytest tests/test_advanced_tutorials.py -k resume -q` and reject any use of selective loading as a substitute for trainer resume.
- [x] 5.5 Write `tutorials/06_checkpoints_and_transfer_learning.md` with source training, explicit saved paths, resume, FashionMNIST head training, and fine-tuning commands; verify the example's help and minimal real-data workflow, or report missing downloads/runtime prerequisites without claiming end-to-end execution.

## 6. Model export and inference

- [x] 6.1 Add `examples/export_and_infer.py` using an explicit trained source and public package reload/inference APIs; verify both current-code and packaged-runtime reloads return matching label-free `class_logits`, and missing trained sources fail, with `.venv/bin/python -m pytest tests/test_advanced_tutorials.py -k package -q`.
- [x] 6.2 Demonstrate SafeTensors export and reload into the matching classifier configuration using core APIs; verify source/reloaded logits and predicted classes agree within a stated tolerance with `.venv/bin/python -m pytest tests/test_advanced_tutorials.py -k safetensors -q`.
- [x] 6.3 Demonstrate the public ONNX exporter on the existing image classifier and compare ONNX Runtime logits/predictions for the supported tested batch sizes; verify `.venv/bin/python -m pytest tests/test_advanced_tutorials.py -k onnx -q` with the optional dependencies installed, and report an explicit missing-prerequisite skip otherwise. Do not add a custom exporter to conceal an upstream failure.
- [x] 6.4 Write `tutorials/07_model_export_and_inference.md` with trained-source provenance, package/weights/ONNX distinctions, dependency setup, key/shape/tolerance contracts, and runnable commands; execute the chapter's supported-format smoke workflow and confirm it never describes an omitted checkpoint as automatic latest-model selection.

## 7. Learning-path integration and final proof

- [x] 7.1 Link the five completed chapters in README stages 4–8, replace the old unsafe tuning command, and update only relevant cross-links/future-work statements in the introductory chapters; verify every link resolves, each documented scenario/component/script exists, and distributed execution remains planned.
- [ ] 7.2 Run `.venv/bin/python -m pytest tests -q` and the available documented chapter smoke commands; record actual pass/skip/blocker results separately for offline CPU, CUDA/DALI, tracking, and ONNX rather than equating skipped checks with success.
- [ ] 7.3 Perform the final simplicity/dependency audit with `git diff --check` and a review of changed files: verify every new abstraction/dependency serves a requested chapter, no framework/backend/base-library implementation was copied, no user artifact is overwritten, and only real deliberate limitations receive `ponytail:` comments.

## Verification so far — 2026-10-01

- Baseline: 11 passed after installing the missing project/test dependencies. Installed NexuML 0.2.1 public contracts verified; sibling repository unchanged.
- Full offline suite: 28 passed, including CUDA log-mel and optional ONNX Runtime checks. No skipped checks in that run.
- CLI verification uses `HOME=/tmp/opencode/nexuml-tutorial-home`: the user's saved `/workspaces/NexuMLTutorial/library` root otherwise duplicates the editable entry point. Saved settings remain untouched.
- Real CUDA exports: NumPy and WebDataset each contain 8,000 samples; splits 6,290 train / 953 val / 757 test. Speaker groups remain disjoint. Float32 features are identical; label values agree although NumPy/core readback uses float32 labels and WebDataset uses int64.
- Native prepared WebDataset training: one epoch / 99 optimizer steps; finite validation loss 2.180734 and test loss 2.182805; no compiled Preprocessing stage. Backend iterator-reset warning did not prevent completion.
- Real resume: epoch 1 / step 797 continued to epoch 2 / step 1,594. Real FashionMNIST head/fine-tune workflow matched 30 encoder tensors and completed (test accuracy 0.7709).
- Real trained-package, SafeTensors, and ONNX Runtime comparisons passed, including ONNX batches 1 and 2.
- Full-MNIST tuning smoke hit its 600-second execution limit: trial 0 completed with val/loss 0.41821911931037903; trial 1 was interrupted, and final selected-model testing did not execute. Existing partial study `advanced-smoke-20261001` was preserved, not rewritten. Tasks 2.2, 2.3, and final verification remain pending.
- README/tutorial relative links resolve; example help commands, `git diff --check`, and strict OpenSpec validation passed. Final audit remains pending.
