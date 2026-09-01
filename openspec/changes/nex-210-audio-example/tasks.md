# nex-210-audio-example Tasks

## 1. Make the tutorial package truly standalone

- [ ] 1.1 Update `pyproject.toml` to remove the `nexuml-library>=0.1.0` dependency.
- [ ] 1.2 Prefer `nexuml[dali]>=0.1.0` as the NexuML dependency so the tutorial consumes NexuML's own DALI dependency policy; if the current installation workflow cannot consume that extra cleanly, keep `nexuml>=0.1.0` and document the DALI extra install command instead of pinning DALI separately.
- [ ] 1.3 Fix `[project.entry-points."nexuml.libraries"]` so `nexuml-tutorial = "library"`, matching the actual importable package built by Hatch.
- [ ] 1.4 Remove the README workflow that installs and then uninstalls `nexuml_library`.
- [ ] 1.5 Search the repository for `nexuml_library` and ensure there are no imports, dependencies, docs instructions, or hidden runtime assumptions left.
- [ ] 1.6 Do not add compatibility shims for old tutorial-only layer names during this cleanup.

## 2. Add Mini Speech Commands as a local file-backed dataset

- [ ] 2.1 Create `library/data/mini_speech_commands.py` and register `MiniSpeechCommandsDataset` with `@data_source("MiniSpeechCommandsDataset")`.
- [ ] 2.2 Define the stable constants in that module:
  - download URL `http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip`;
  - sample rate `16_000`;
  - clip samples `16_000`;
  - class order `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes`.
- [ ] 2.3 Implement a small standard-library download/extract helper using `urllib`, `zipfile`, and `pathlib`; do not import TensorFlow, torchaudio, Hugging Face datasets, or another dataset manager.
- [ ] 2.4 Keep `root` semantics simple: the configured root is the extracted directory containing the eight class subdirectories.
- [ ] 2.5 If the dataset is missing and `download=False`, raise an actionable `FileNotFoundError` rather than returning an empty dataset.
- [ ] 2.6 If downloading, extract into `root.parent`, validate that expected WAV files exist, and remove the temporary archive after success.
- [ ] 2.7 Add a short safe-extraction check so archive entries cannot escape the intended directory; do not create a generic archive abstraction.
- [ ] 2.8 Build metadata deterministically by iterating the explicit class list and sorted WAV paths.
- [ ] 2.9 Store exactly the needed training metadata: `file`, `class`, and `split` (plus only genuinely useful standard metadata such as `basename` if it improves readability).
- [ ] 2.10 Keep `self.data is None` so NexuML's DALI backend recognizes the dataset as file-backed.

## 3. Add deterministic speaker-grouped train/val/test splitting

- [ ] 3.1 Extract the speaker key from the filename prefix before `_nohash_`; fall back to the full stem if the marker is absent.
- [ ] 3.2 Use a stable `hashlib.sha1` digest bucket rather than Python's randomized `hash()`.
- [ ] 3.3 Assign buckets `<80` to `train`, `<90` to `val`, and the rest to `test`.
- [ ] 3.4 Ensure every recording for one speaker receives the same split regardless of class or file order.
- [ ] 3.5 Document that the split is approximately 80/10/10 by speaker bucket, not an exact sample-count split.

## 4. Declare the audio/NexuML data contract

- [ ] 4.1 In `MiniSpeechCommandsDataset`, set:
  - `sample_rate = 16_000`;
  - `dali_x_keys = ["waveform"]`;
  - `dali_layout = "T"`;
  - `dali_sequence_length = 16_000`;
  - `label_names = ["class"]` through the base dataset constructor;
  - modality `audio`.
- [ ] 4.2 Implement the smallest direct `load_item(...)` needed for a coherent `NexuDataset` inspection/fallback contract using Python `wave` + NumPy/Torch.
- [ ] 4.3 Limit the direct loader to the official tutorial data assumptions: mono PCM, 16 kHz, normalized float32, pad/truncate to 16,000 samples; fail clearly for unsupported input rather than growing an audio framework.
- [ ] 4.4 Ensure direct `dataset[i]` returns `x["waveform"]` with shape `[16000]` and `y["class"]`.
- [ ] 4.5 Create `library/config/data/mini_speech_commands.py` with `mini_speech_commands_data(...)`.
- [ ] 4.6 Configure the dataset through `DatasetSpec(type_key="MiniSpeechCommandsDataset", modality="audio", split_type="keep")`.
- [ ] 4.7 Configure `DataSpec` with `input_shapes={"waveform": [16000]}`, `feature_key="waveform"`, `num_classes=8`, and 0.8/0.1/0.1 split ratios.
- [ ] 4.8 Configure `LoaderSpec(backend="dali", num_workers=num_workers)` explicitly.
- [ ] 4.9 Do not import `nvidia.dali` or call any DALI operator in tutorial code.
- [ ] 4.10 Update `library/config/data/__init__.py` and any package exports needed for discovery/readability.

## 5. Generalize the tutorial classification head and loss

- [ ] 5.1 Add `library/layers/head/classification.py` with `@layer("ClassificationHead")`.
- [ ] 5.2 Keep the head minimal: optional dropout + one `torch.nn.Linear`, returning raw class logits with no softmax option.
- [ ] 5.3 Add `library/layers/loss/cross_entropy.py` with `@layer("CrossEntropyLoss")`.
- [ ] 5.4 Use `torch.nn.CrossEntropyLoss(reduction="none")` and scalar class labels cast to `long`.
- [ ] 5.5 Preserve a small `y is None` path required for NexuML dummy/shape propagation without introducing fake training behavior.
- [ ] 5.6 Update `library/config/model/resnet_classifier.py` to use `ClassificationHead` + `CrossEntropyLoss` and raw logits.
- [ ] 5.7 Update existing MNIST callers/imports to the generic classification components.
- [ ] 5.8 Update `library/layers/head/__init__.py` and `library/layers/loss/__init__.py`.
- [ ] 5.9 Delete `library/layers/head/resnet_head.py` after all callers are migrated.
- [ ] 5.10 Delete `library/layers/loss/bce_loss.py` after all callers are migrated.
- [ ] 5.11 Do not preserve `LatentClassificationHead` or `BCELoss` aliases.
- [ ] 5.12 Keep `library/layers/metrics/classification_metrics.py` unless a concrete incompatibility with raw logits is found; do not rewrite working metrics just for stylistic consistency.

## 6. Add the two raw-waveform encoders

- [ ] 6.1 Create `library/layers/model/audio.py` and keep both tutorial encoders in this single module.
- [ ] 6.2 Implement `@layer("AudioCNNEncoder")` with default embedding size 64.
- [ ] 6.3 Accept `[B, 16000]`, add a channel dimension internally, run three small Conv1d/normalization/activation blocks, global temporal average pooling, and return `[B, 64]`.
- [ ] 6.4 Keep the CNN free of residual hierarchies, spectrogram frontends, custom blocks spread over additional files, or external model libraries.
- [ ] 6.5 Implement `@layer("TinyAudioTransformerEncoder")` with defaults:
  - `d_model=64`;
  - `patch_size=160`;
  - `num_layers=2`;
  - `num_heads=4`;
  - `dim_feedforward=128`;
  - `dropout=0.1`.
- [ ] 6.6 Use Conv1d patch embedding, learned positional embeddings, `torch.nn.TransformerEncoder`, final LayerNorm, and mean token pooling.
- [ ] 6.7 Derive the positional-token count from `input_sizes` and `patch_size`; validate impossible configurations clearly.
- [ ] 6.8 Do not add a CLS token, custom attention, pretrained weights, or a `transformers` dependency.
- [ ] 6.9 Ensure both encoders return exactly the same semantic output key/shape contract: `embeddings` with dimension 64 by default.
- [ ] 6.10 Update `library/layers/model/__init__.py`.

## 7. Assemble one reusable audio classifier and two scenarios

- [ ] 7.1 Create `library/config/model/audio_classifier.py`.
- [ ] 7.2 Add one `audio_classifier(...) -> PipelineSpec` function whose only architecture-specific choice is the encoder type/params.
- [ ] 7.3 Use the common stage contract:
  - `waveform -> embeddings`;
  - `embeddings -> class_logits` via `ClassificationHead`;
  - `class_logits -> classification_loss` via `CrossEntropyLoss`;
  - `class_logits -> accuracy/f1` via `ClassificationMetrics`.
- [ ] 7.4 Create `library/config/scenario/speech_commands.py`.
- [ ] 7.5 Add one private/shared `_speech_commands_scenario(...)` assembly helper to avoid duplicated data/training/evaluation/logging setup.
- [ ] 7.6 Register `@scenario("speech-commands-cnn")` using `AudioCNNEncoder`.
- [ ] 7.7 Register `@scenario("speech-commands-transformer")` using `TinyAudioTransformerEncoder`.
- [ ] 7.8 Expose useful public scenario parameters (`root`, `download`, `lr`, `batch_size`, `max_epochs`, `num_workers`) without turning the scenario into an arbitrary untyped dict interface.
- [ ] 7.9 Add only a few model-specific scalar parameters if they materially improve the tutorial; prefer defaults over a large option surface.
- [ ] 7.10 Update `library/config/model/__init__.py`, `library/config/scenario/__init__.py`, and `library/config/__init__.py`.

## 8. Remove stale scenario-name hard-coding from shared defaults

- [ ] 8.1 Update `default_callbacks(...)` in `library/config/defaults.py` to accept `name` and write checkpoints to `logs/checkpoints/{name}` instead of the stale `cifar-resnet` path.
- [ ] 8.2 Update `default_exports(...)` to accept `name` and write to `logs/models/{name}` instead of hard-coded `mnist_resnet`.
- [ ] 8.3 Update `mnist_resnet.py` and both audio scenarios to pass their own scenario name.
- [ ] 8.4 Do not redesign logging, checkpointing, tuning, evaluation, or callback APIs in this change.

## 9. Add focused tests only

### Dataset tests

- [ ] 9.1 Create `tests/test_mini_speech_commands.py` using `tmp_path` and Python `wave` to generate a few small 16 kHz mono PCM WAV fixtures; no network access.
- [ ] 9.2 Verify explicit class mapping and deterministic metadata discovery.
- [ ] 9.3 Verify files from the same speaker prefix always share a split.
- [ ] 9.4 Verify direct item loading returns waveform shape `[16000]` and the class label.
- [ ] 9.5 Verify the native DALI declaration (`data is None`, sample rate/layout/key/sequence length).
- [ ] 9.6 Combine related assertions so this file remains roughly 2-3 tests rather than a large matrix.

### Model tests

- [ ] 9.7 Create `tests/test_audio_models.py` with one parametrized or two small forward tests for the CNN and Transformer.
- [ ] 9.8 Assert both accept the same waveform input contract and return finite `[B, 64]` embeddings.
- [ ] 9.9 Do not test every hyperparameter combination or PyTorch Transformer internals.

### Scenario tests

- [ ] 9.10 Create `tests/test_audio_scenarios.py` with structural spec assertions rather than long training jobs.
- [ ] 9.11 Verify CNN vs Transformer differs at the encoder while head/loss/metrics/data contracts remain identical.
- [ ] 9.12 Verify both specs select `loader.backend == "dali"`, preserve `split_type == "keep"`, use `ClassificationHead`, and use `CrossEntropyLoss`.
- [ ] 9.13 Verify the existing MNIST pipeline resolves through the new generic classification head/loss after migration.
- [ ] 9.14 Keep the total new test surface small; do not add a mandatory live dataset download or DALI/GPU CI matrix.

## 10. Write the progressive tutorial documentation

- [ ] 10.1 Rewrite the root `README.md` setup so it installs only this tutorial package + NexuML and no longer discusses uninstalling the public library.
- [ ] 10.2 Keep the current registry commands, but explain that every registered tutorial component comes from this repository.
- [ ] 10.3 Add `tutorials/01_mnist_from_scratch.md` for the existing first-library workflow:
  - library registration;
  - local dataset/layers/config;
  - `resolve`;
  - `build`;
  - `train`.
- [ ] 10.4 Add `tutorials/02_audio_native_dali.md` explaining why the file-backed example uses native DALI while in-memory MNIST does not.
- [ ] 10.5 Explain Mini Speech Commands layout, `file/class/split` metadata, the native waveform contract, and `LoaderSpec(backend="dali")`.
- [ ] 10.6 Include runnable CNN commands:

```bash
nexuml resolve speech-commands-cnn
nexuml build configs/speech-commands-cnn.yaml
nexuml train speech-commands-cnn --max-epochs 10
```

- [ ] 10.7 Include runnable Transformer commands:

```bash
nexuml resolve speech-commands-transformer
nexuml build configs/speech-commands-transformer.yaml
nexuml train speech-commands-transformer --max-epochs 10
```

- [ ] 10.8 Show a short code/config comparison demonstrating that only the encoder changes while dataset/head/loss/metrics/training remain shared.
- [ ] 10.9 State DALI platform/runtime prerequisites clearly and show `nexuml backend list` as the verification command.
- [ ] 10.10 Add a root README learning-path table covering:
  1. MNIST/custom library;
  2. audio CNN/native DALI;
  3. audio Transformer/composition;
  4. tuning/tracking;
  5. preprocessing/data export/WebDataset;
  6. custom evaluation;
  7. checkpoints/transfer;
  8. export/inference;
  9. distributed execution when stable.
- [ ] 10.11 Link only the implemented tutorial files; mark future items as planned rather than creating placeholder files/code.

## 11. Validate native DALI behavior on a supported environment

- [ ] 11.1 Verify `nexuml backend list` shows `data-loader dali`.
- [ ] 11.2 With no local dataset present, run `nexuml train speech-commands-cnn --max-epochs 1` and verify the dataset downloads/extracts successfully.
- [ ] 11.3 Verify training builds from a metadata-backed dataset with `self.data is None` and WAV `file` paths, satisfying the NexuML native-file route rather than the known in-memory Torch fallback condition.
- [ ] 11.4 Verify one CNN epoch completes and logs normal training/validation metrics.
- [ ] 11.5 Run `nexuml build` for the Transformer and, if practical on the validation machine, a short training smoke run; do not require two full training jobs merely for acceptance.
- [ ] 11.6 Do not add tutorial-specific loader instrumentation if existing NexuML logs are sufficient; backend observability belongs in NexuML core.

## 12. Final validation and simplicity audit

- [ ] 12.1 Run the repository test suite (`pytest -q` or the repository-defined equivalent).
- [ ] 12.2 Run registry checks:

```bash
nexuml library list
nexuml registry list data
nexuml registry list layers
nexuml registry list scenarios
```

- [ ] 12.3 Resolve/build MNIST to prove the generic classification cleanup did not break the first tutorial.
- [ ] 12.4 Resolve/build both Speech Commands scenarios.
- [ ] 12.5 Search the final repository and prove there is no `nexuml_library` dependency/import.
- [ ] 12.6 Search `library/` and `tutorials/` and prove there is no direct `nvidia.dali` import.
- [ ] 12.7 Prove there is no TensorFlow dependency.
- [ ] 12.8 Confirm there is one audio dataset, one shared audio classifier assembly, one generic classification head, and one multiclass loss implementation.
- [ ] 12.9 Confirm CNN and Transformer do not duplicate scenario/head/loss/metrics/training code.
- [ ] 12.10 Confirm deleted `resnet_head.py` and `bce_loss.py` have no compatibility aliases left behind.
- [ ] 12.11 Run any existing lint/type commands configured by the repository; do not introduce a new heavyweight tool just for this change.
- [ ] 12.12 Run strict OpenSpec validation for `nex-210-audio-example`.
- [ ] 12.13 Review the final diff for tutorial readability: a new user should be able to understand the dataset and both models without reading NexuML's DALI implementation.
