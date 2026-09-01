# nex-210-audio-example Design

## Context

`NexuMLTutorial` is intended to show how a user builds a NexuML library without relying on the built-in `nexuml_library`. At the start of NEX-210 the repository contains a small MNIST example with local dataset/layer/config code, but two details obscure that goal:

1. `pyproject.toml` still depends on `nexuml-library>=0.1.0` even though the README tells the user to uninstall `nexuml_library`.
2. the only dataset is in-memory (`torchvision.datasets.MNIST` stored as `self.data`), so it cannot demonstrate NexuML's native file-backed DALI loading path.

The current NexuML core behavior that NEX-210 intentionally targets is:

- `LoaderSpec.backend` selects the loader backend; its default is `dali` in current NexuML core.
- `DaliLoaderBackend` falls back to Torch for datasets whose root dataset has `data is not None`.
- for metadata-backed datasets with `meta["file"]`, DALI infers the native source from modality/file suffix;
- for audio it calls NexuML's native file loader with `kind="audio"`;
- the native audio pipeline uses DALI `readers.file` plus `decoders.audio`, resamples to the dataset sample rate, pads/truncates to the declared sequence length, and returns keyed TensorDict batches;
- explicit dataset attributes `dali_x_keys`, `dali_layout`, and `dali_sequence_length` let a file-backed dataset define the input contract without forcing Python item loading to infer shapes;
- `DatasetSpec(split_type="keep")` preserves a dataset-provided `split` column;
- `NexuDataCreator.split_meta(...)` only converts `fit`/`all` values, so pre-existing `train`/`val`/`test` values remain stable.

Relevant NexuML core files at planning time:

- `src/nexuml/core/types.py`
  - `DatasetSpec`
  - `LoaderSpec`
  - `DataSpec`
- `src/nexuml/data/dataset.py`
  - `NexuDataset`
  - metadata-backed `load_labels`
  - `split_meta`
- `src/nexuml/data/creator.py`
  - dataset instantiation
  - `split_type="keep"`
  - loader backend selection
- `src/nexuml/data/loaders/dali_backend.py`
  - native-file routing
  - in-memory Torch fallback
- `src/nexuml/data/loaders/dali_multimodal.py`
  - `audio_file_pipeline`
  - native `readers.file` / `decoders.audio` path

External references checked for this design:

- TensorFlow Mini Speech Commands tutorial:
  - `https://www.tensorflow.org/tutorials/audio/simple_audio`
  - archive: `http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip`
  - 8,000 files, 8 classes, approximately 182 MB, <=1 s, 16 kHz WAV.
- NVIDIA DALI file reader:
  - `https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/operations/nvidia.dali.fn.readers.file.html`
  - supports file lists and WAV/FLAC/OGG file filtering.
- NVIDIA DALI audio decoder:
  - `https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/operations/nvidia.dali.fn.decoders.audio.html`
  - decodes audio, supports downmixing, and supports target sample-rate resampling.

NEX-210 uses those capabilities only through NexuML public configuration. Tutorial code does not directly call DALI.

## Goals & Non-Goals

### Goals

- Make the tutorial package genuinely independent from `nexuml_library`.
- Add a small real-world-ish file-backed audio classification problem that automatically downloads.
- Exercise NexuML's native DALI audio path, not DALI `external_source` and not a tutorial-owned loader.
- Keep the raw input contract extremely simple: one 16,000-sample waveform and one scalar class label.
- Provide two compact models with an identical `embeddings` contract so users can see pipeline composition rather than duplicated experiments.
- Reuse one generic classification head, loss, and metrics path for MNIST and audio.
- Keep implementation code small enough to read completely during a tutorial.
- Keep tests focused on tutorial-owned contracts rather than retesting NexuML or DALI internals.
- Document a clear progression from first model to advanced NexuML features.

### Non-Goals

- No changes to NexuML core.
- No `nexuml_library` dependency or import.
- No tutorial-owned DALI backend/pipeline/iterator.
- No TensorFlow runtime dependency.
- No torchaudio/librosa/Hugging Face dependency just to download or read this dataset.
- No spectrogram, mel filter bank, log-mel, MFCC, SpecAugment, or advanced audio preprocessing in NEX-210.
- No pretrained audio model.
- No custom attention implementation.
- No implementation of the later tuning/export/distributed tutorial stages.
- No backwards-compatibility aliases for tutorial-only renamed layer keys.
- No broad refactor of the existing evaluation code.

## Core Decisions

### D1 — Mini Speech Commands is the canonical tutorial audio dataset

The dataset root is:

```text
data/mini_speech_commands/
```

Expected extracted shape:

```text
data/mini_speech_commands/
├── README.md
├── down/*.wav
├── go/*.wav
├── left/*.wav
├── no/*.wav
├── right/*.wav
├── stop/*.wav
├── up/*.wav
└── yes/*.wav
```

The class mapping is explicit and never derived from filesystem iteration order:

```python
CLASS_NAMES = (
    "down",
    "go",
    "left",
    "no",
    "right",
    "stop",
    "up",
    "yes",
)
```

Rationale:

- explicit ordering makes checkpoints/configs reproducible;
- 8 classes are enough to be non-trivial without turning onboarding into a dataset-management exercise;
- WAV + 16 kHz + <=1 second maps exactly to the native DALI contract already supported by NexuML;
- the official mini archive avoids a multi-gigabyte first-run download.

### D2 — Download uses only the standard library

`library/data/mini_speech_commands.py` owns a tiny download helper using:

- `urllib.request.urlretrieve` (or `urlopen` + copy if implementation preference requires progress handling);
- `zipfile.ZipFile`;
- `pathlib.Path`.

Algorithm:

1. Resolve `root = Path(root)`.
2. If `root` already contains WAV class directories, do not download.
3. If the dataset is missing and `download=False`, raise a clear `FileNotFoundError` describing the expected root and how to enable download.
4. If the dataset is missing and `download=True`:
   - create `root.parent`;
   - download the archive to `root.parent / "mini_speech_commands.zip"`;
   - extract into `root.parent` so the archive-created `mini_speech_commands/` directory becomes `root`;
   - delete the archive after successful extraction.
5. Validate that at least one expected class contains WAV files; raise an actionable error if extraction produced no dataset.

The implementation shall not depend on TensorFlow's `tf.keras.utils.get_file`.

The extraction helper SHOULD reject archive members that would escape the target directory. This can remain a short local safety check and must not grow into a generic archive framework.

### D3 — Metadata owns labels and split; files stay on disk

`MiniSpeechCommandsDataset` is registered locally:

```python
@data_source("MiniSpeechCommandsDataset")
class MiniSpeechCommandsDataset(NexuDataset):
    LABEL_NAMES = ["class"]
```

The main state is metadata rather than `self.data`:

```text
file                                             class   split
.../down/004ae714_nohash_0.wav                  0       train
.../yes/01bb6a2a_nohash_0.wav                   7       val
...
```

Important invariants:

- `self.data` remains `None`.
- `meta["file"]` is a string path to a WAV file.
- `meta["class"]` is an integer in `[0, 7]`.
- `meta["split"]` is exactly `train`, `val`, or `test`.
- rows are built in deterministic class/path order.
- `label_names=["class"]` is supplied explicitly.
- modality is `audio`.

Rationale: leaving `self.data` unset is what allows NexuML's DALI backend to use the native file reader rather than deliberately falling back to Torch for an in-memory object.

### D4 — Split by speaker, deterministically

Speech Commands filenames encode the speaker before `_nohash_`, for example:

```text
01bb6a2a_nohash_0.wav
```

The speaker key is:

```python
speaker_id = Path(file).stem.split("_nohash_", 1)[0]
```

If a filename unexpectedly lacks `_nohash_`, use the full stem as the grouping key rather than failing.

Do not use Python's built-in `hash()` because its randomized seed makes assignments process-dependent.

Use a deterministic digest bucket:

```python
digest = hashlib.sha1(speaker_id.encode("utf-8")).digest()
bucket = int.from_bytes(digest[:4], "big") % 100

if bucket < 80:
    split = "train"
elif bucket < 90:
    split = "val"
else:
    split = "test"
```

All recordings from one speaker therefore receive one split.

Rationale:

- avoids speaker leakage;
- needs no manifest file or extra dependency;
- stable across machines and Python versions;
- visually understandable in tutorial code.

NEX-210 does not promise exact 80/10/10 sample counts; it promises deterministic speaker-group assignment using 80/10/10 hash buckets.

### D5 — The dataset declares the native DALI contract explicitly

The dataset constructor sets:

```python
self.sample_rate = 16_000
self.dali_x_keys = ["waveform"]
self.dali_layout = "T"
self.dali_sequence_length = 16_000
```

The scenario declares:

```python
input_shapes={"waveform": [16000]}
feature_key="waveform"
num_classes=8
```

These names are stable across both models and all stages in this change.

The expected dataloader batch contract is:

```text
x["waveform"]    float tensor [B, 16000]
x["sample_index"] long tensor [B]
y["class"]       class tensor [B]
```

The tutorial must not rely on DALI's reader-generated label as the semantic class. NexuML uses the native reader's file index to look up metadata labels, so `meta["class"]` remains the source of truth.

### D6 — Provide a minimal Python WAV item loader only as the dataset fallback/inspection path

Although training is explicitly configured for DALI, `MiniSpeechCommandsDataset` should remain a valid `NexuDataset` when inspected directly. Implement `load_item(...)` with Python `wave` + NumPy/Torch only:

- expect mono PCM WAV;
- read frames;
- normalize integer samples to float32;
- verify/respect 16 kHz;
- pad or truncate to 16,000 samples;
- return `TensorDict({"waveform": waveform}, batch_size=[])`.

This function is NOT the training loader in the tutorial's audio scenarios. Its purpose is:

- direct dataset inspection;
- small unit tests without DALI;
- a coherent `NexuDataset.__getitem__` contract.

Do not build resampling, stereo handling, codecs, augmentation, multiprocessing, or caching here. If a WAV violates the deliberately narrow tutorial assumptions, raise a clear error. DALI remains responsible for the configured training path.

### D7 — Audio `DataSpec` explicitly preserves native DALI and existing splits

Create `library/config/data/mini_speech_commands.py`:

```python
def mini_speech_commands_data(
    root: str = "data/mini_speech_commands",
    download: bool = True,
    num_workers: int = 4,
) -> DataSpec:
    return DataSpec(
        source_type="MiniSpeechCommandsDataset",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        datasets=[
            DatasetSpec(
                type_key="MiniSpeechCommandsDataset",
                params={"root": root, "download": download},
                modality="audio",
                split_type="keep",
            )
        ],
        input_shapes={"waveform": [16000]},
        num_classes=8,
        feature_key="waveform",
        loader=LoaderSpec(
            backend="dali",
            num_workers=num_workers,
        ),
    )
```

`source_type` is descriptive here; the real dataset instantiation is driven by `datasets`.

Do not add DALI-specific pipeline arguments to `LoaderSpec.params` unless an implementation blocker in current NexuML requires one. The tutorial should demonstrate the default native-file contract, not advanced backend tuning.

### D8 — CNN operates directly on waveform and returns embeddings

`library/layers/model/audio.py` contains both audio encoders to keep the example compact.

`AudioCNNEncoder` public defaults:

```python
@layer("AudioCNNEncoder")
class AudioCNNEncoder(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        embedding_dim: int = 64,
        **kwargs,
    ): ...
```

Recommended internal shape:

```text
[B, 16000]
 -> unsqueeze -> [B, 1, 16000]
 -> Conv1d(1, 32, kernel_size=9, stride=4, padding=4)
 -> BatchNorm1d(32)
 -> GELU
 -> Conv1d(32, 64, kernel_size=9, stride=4, padding=4)
 -> BatchNorm1d(64)
 -> GELU
 -> Conv1d(64, embedding_dim, kernel_size=9, stride=4, padding=4)
 -> BatchNorm1d(embedding_dim)
 -> GELU
 -> AdaptiveAvgPool1d(1)
 -> flatten
 -> [B, embedding_dim]
```

Minor kernel/stride adjustments are acceptable during implementation if shape propagation or training stability gives a concrete reason. Do not add residual stacks, configuration hierarchies, squeeze-excitation, frontend transforms, or model factories inside the layer.

### D9 — Tiny Transformer uses Conv1d patching and standard PyTorch TransformerEncoder

`TinyAudioTransformerEncoder` lives in the same `audio.py` module:

```python
@layer("TinyAudioTransformerEncoder")
class TinyAudioTransformerEncoder(PipelineLayer):
    def __init__(
        self,
        input_sizes: dict[str, tuple],
        keys_in: list[str],
        keys_out: list[str],
        d_model: int = 64,
        patch_size: int = 160,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ): ...
```

Internal contract for the default 16,000-sample input:

```text
[B, 16000]
 -> unsqueeze -> [B, 1, 16000]
 -> Conv1d(1, 64, kernel_size=160, stride=160)
 -> [B, 64, 100]
 -> transpose -> [B, 100, 64]
 -> add learned positional embedding [1, 100, 64]
 -> 2 x nn.TransformerEncoderLayer(
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1,
        batch_first=True,
        norm_first=True,
    )
 -> LayerNorm(64)
 -> mean(dim=1)
 -> [B, 64]
```

Implementation constraints:

- use `torch.nn.TransformerEncoder` / `TransformerEncoderLayer`;
- no CLS token;
- no custom attention;
- no pretrained weights;
- no `transformers` dependency;
- derive the positional-embedding token count from `input_sizes` and `patch_size`;
- fail clearly if an impossible/zero token configuration is requested.

### D10 — Both encoders deliberately share `embeddings`

The architectural teaching point is that everything after the encoder remains unchanged.

Stable pipeline contract:

```text
waveform
   |
   +-- AudioCNNEncoder -------------------+
   |                                      |
   +-- TinyAudioTransformerEncoder -------+--> embeddings [64]
                                              |
                                              v
                                      ClassificationHead
                                              |
                                         class_logits [8]
                                          /          \
                                         v            v
                              CrossEntropyLoss   ClassificationMetrics
```

The scenario/model config changes the encoder type key, not the rest of the training program.

### D11 — Replace tutorial softmax+BCE multiclass path with logits+cross entropy

The current tutorial head/loss are unnecessarily tied to the first ResNet example:

- `resnet_head.py` defines a reusable linear classifier but names it `LatentClassificationHead` and optionally applies softmax;
- `bce_loss.py` one-hot encodes scalar multiclass labels and applies `BCELoss` to probabilities.

For ordinary single-label multiclass classification, use a simpler generic contract.

Replace:

```text
library/layers/head/resnet_head.py
library/layers/loss/bce_loss.py
```

with:

```text
library/layers/head/classification.py
library/layers/loss/cross_entropy.py
```

`ClassificationHead`:

```python
@layer("ClassificationHead")
class ClassificationHead(PipelineLayer):
    # Linear embedding -> raw class logits.
```

- optional dropout is allowed because the current head already supports it;
- no softmax parameter;
- output is raw logits.

`CrossEntropyLoss`:

```python
@layer("CrossEntropyLoss")
class CrossEntropyLoss(PipelineLayer):
    # nn.CrossEntropyLoss(reduction="none")
```

- default `label_key="class"`;
- cast/reshape class labels to `long`;
- return a per-sample loss vector compatible with NexuML's loss-key aggregation;
- preserve the existing dummy/shape-propagation behavior when `y is None` with the smallest safe zero-loss implementation.

Update `resnet_classifier.py` and the MNIST scenario to use the new keys. Do not preserve `LatentClassificationHead` or `BCELoss` aliases in the tutorial package.

`ClassificationMetrics` remains in place because it already consumes class logits and scalar labels through torchmetrics.

### D12 — One audio pipeline factory, two scenario wrappers

Create `library/config/model/audio_classifier.py` with a small function that assembles the common pipeline:

```python
def audio_classifier(
    encoder_type: str,
    num_classes: int = 8,
    label_key: str = "class",
    head_dropout: float = 0.0,
    encoder_params: dict | None = None,
) -> PipelineSpec:
    ...
```

Stages:

```text
Encoder
  type_key=<encoder_type>
  keys_in=["waveform"]
  keys_out=["embeddings"]

Head
  type_key="ClassificationHead"
  keys_in=["embeddings"]
  keys_out=["class_logits"]

Loss
  type_key="CrossEntropyLoss"
  keys_in=["class_logits"]
  keys_out=["classification_loss"]
  label_key="class"

Metrics
  type_key="ClassificationMetrics"
  keys_in=["class_logits"]
  keys_out=["accuracy", "f1"]
  num_classes=8
  label_key="class"
```

Create `library/config/scenario/speech_commands.py` with one private/shared assembly helper and two decorated public scenario functions:

```python
@scenario("speech-commands-cnn")
def speech_commands_cnn(...):
    return _speech_commands_scenario(
        name="speech_commands_cnn",
        encoder_type="AudioCNNEncoder",
        ...,
    )


@scenario("speech-commands-transformer")
def speech_commands_transformer(...):
    return _speech_commands_scenario(
        name="speech_commands_transformer",
        encoder_type="TinyAudioTransformerEncoder",
        ...,
    )
```

Both public functions SHOULD expose at least:

- `root`;
- `download`;
- `lr`;
- `batch_size`;
- `max_epochs`;
- `num_workers`.

Model-specific parameters may be exposed only where useful for the tutorial; do not turn the scenario into a generic arbitrary-dict command surface.

### D13 — Make existing defaults scenario-name-safe, not audio-specific

`library/config/defaults.py` currently hard-codes multiple example names in paths. Fix only this leakage:

- `default_callbacks(name: str)` writes checkpoints under `logs/checkpoints/{name}`;
- `default_exports(name: str)` writes models under `logs/models/{name}`;
- existing `default_logging(name)` continues to use the scenario name;
- update `mnist_resnet.py` and new audio scenarios to pass `name`.

Do not redesign the defaults module, callback system, logging stack, or checkpoint behavior in NEX-210.

### D14 — Package metadata must reflect a real external library

Update `pyproject.toml`:

- remove `nexuml-library>=0.1.0`;
- keep NexuML as the framework dependency;
- use the NexuML DALI extra so an audio-capable Linux install pulls the supported DALI package via NexuML rather than duplicating DALI version policy in this repository;
- fix the library entry point value to the importable package name:

```toml
[project.entry-points."nexuml.libraries"]
nexuml-tutorial = "library"
```

- keep Hatch building the `library` package.

Preferred dependency declaration:

```toml
dependencies = [
    "nexuml[dali]>=0.1.0",
]
```

If package-resolution constraints make the extra form invalid for the current unpublished/dev installation workflow, keep `nexuml>=0.1.0` and put the DALI installation command in the README. Do NOT directly pin a separate DALI package version in the tutorial unless NexuML's own extra cannot be consumed.

README setup shall no longer tell users to uninstall `nexuml_library`; it should never be installed by this repository in the first place.

### D15 — Documentation is progressive, but NEX-210 creates only real tutorials

Add:

```text
tutorials/01_mnist_from_scratch.md
tutorials/02_audio_native_dali.md
```

Do not add empty placeholder tutorial files for future stages.

`01_mnist_from_scratch.md` documents the existing local MNIST library as the first conceptual step:

- local package/registry;
- dataset;
- layer;
- scenario;
- `resolve`, `build`, `train`.

`02_audio_native_dali.md` documents:

1. why MNIST does not demonstrate native DALI (it is in-memory);
2. why file-backed audio does;
3. Mini Speech Commands download/layout;
4. the metadata contract;
5. `dali_x_keys`/layout/sequence length at a conceptual level;
6. `LoaderSpec(backend="dali")` as the only backend selection needed in tutorial config;
7. commands:

```bash
nexuml resolve speech-commands-cnn
nexuml build configs/speech-commands-cnn.yaml
nexuml train speech-commands-cnn --max-epochs 10

nexuml resolve speech-commands-transformer
nexuml build configs/speech-commands-transformer.yaml
nexuml train speech-commands-transformer --max-epochs 10
```

8. a short comparison showing the encoder type key is the meaningful architecture difference.

The root README contains a compact learning-path table with statuses such as `available` / `next`, linking only to actual files for available stages.

### D16 — Future learning path is documented as architecture, not implemented scope

README shall describe these future steps without creating code for them:

1. MNIST / custom library basics.
2. Audio CNN / file-backed DALI.
3. Audio Transformer / composition.
4. Tuning + tracking.
5. Preprocessing + dataset export + WebDataset.
6. Custom evaluation.
7. Checkpoint loading / transfer learning.
8. Model export / inference.
9. Distributed execution when a stable NexuML execution backend exists.

This avoids a common tutorial failure mode where later concepts exist as half-maintained placeholders.

## Exact File Plan

### Add

```text
library/data/mini_speech_commands.py
library/config/data/mini_speech_commands.py
library/layers/model/audio.py
library/config/model/audio_classifier.py
library/config/scenario/speech_commands.py
library/layers/head/classification.py
library/layers/loss/cross_entropy.py
tutorials/01_mnist_from_scratch.md
tutorials/02_audio_native_dali.md
tests/test_mini_speech_commands.py
tests/test_audio_models.py
tests/test_audio_scenarios.py
```

### Modify

```text
README.md
pyproject.toml
library/config/__init__.py
library/config/defaults.py
library/config/data/__init__.py
library/config/model/__init__.py
library/config/scenario/__init__.py
library/config/model/resnet_classifier.py
library/config/scenario/mnist_resnet.py
library/layers/head/__init__.py
library/layers/loss/__init__.py
library/layers/model/__init__.py
```

### Delete after callers are migrated

```text
library/layers/head/resnet_head.py
library/layers/loss/bce_loss.py
```

No file should be added merely to preserve the deleted names.

## Testing Strategy

Keep the suite intentionally small. The tutorial should not retest PyTorch Lightning, NexuML registry internals, or DALI operators.

### `tests/test_mini_speech_commands.py`

Use `tmp_path` and Python `wave` to create a few tiny valid 16 kHz mono PCM WAV fixtures. No network access.

Focused assertions:

1. metadata/class contract:
   - paths discovered from class directories;
   - explicit class IDs are correct;
   - splits are only train/val/test.
2. speaker grouping:
   - two files sharing the same `<speaker>_nohash_...` prefix always have the same split.
3. direct item contract:
   - `x["waveform"].shape == (16000,)`;
   - `y["class"]` exists.
4. native DALI declaration:
   - `data is None`;
   - `sample_rate == 16000`;
   - `dali_x_keys == ["waveform"]`;
   - `dali_layout == "T"`;
   - `dali_sequence_length == 16000`.

These can be combined into two or three tests rather than one test per assertion.

The automatic-download code should be structured so it is easy to reason about, but do not add a live network test. If a download-specific unit test is useful, monkeypatch the tiny download helper; otherwise validate it in the manual smoke run.

### `tests/test_audio_models.py`

Two focused tests or one parametrized test:

- instantiate `AudioCNNEncoder` with `input_sizes={"waveform": (16000,)}`;
- instantiate `TinyAudioTransformerEncoder` with the same input contract;
- forward a small random batch;
- assert both return shape `[B, 64]`;
- assert outputs are finite.

Do not test every model hyperparameter combination.

### `tests/test_audio_scenarios.py`

Inspect the specs rather than launching a long training job:

- `speech_commands_cnn(..., download=False)` uses `AudioCNNEncoder`;
- `speech_commands_transformer(..., download=False)` uses `TinyAudioTransformerEncoder`;
- both use `LoaderSpec.backend == "dali"`;
- both use `split_type == "keep"`;
- both expose `waveform -> embeddings -> class_logits` and the same loss/metric keys;
- both use `CrossEntropyLoss` and `ClassificationHead`;
- the MNIST pipeline also resolves to the new generic head/loss.

Prefer one or two structural tests, not a large snapshot.

### DALI smoke validation

The actual native-DALI execution belongs in validation rather than a large mandatory unit test because DALI availability is platform-specific.

On a supported Linux environment with DALI installed:

```bash
nexuml backend list
nexuml resolve speech-commands-cnn
nexuml build configs/speech-commands-cnn.yaml
nexuml train speech-commands-cnn --max-epochs 1
```

Acceptance evidence should show:

- `data-loader dali` is registered;
- dataset downloads automatically when absent;
- one epoch runs;
- the runtime uses the DALI backend without falling back because `self.data` is `None` and metadata contains WAV `file` paths.

If logging does not currently make the exact native route obvious, do not add tutorial-side instrumentation. Inspect or rely on NexuML's existing loader log/contract; backend observability is a NexuML core concern.

## Validation Commands

From the tutorial repository:

```bash
uv sync
source .venv/bin/activate

nexuml library list
nexuml registry list data
nexuml registry list layers
nexuml registry list scenarios
nexuml backend list

pytest -q

nexuml resolve mnist-resnet
nexuml build configs/mnist-resnet.yaml

nexuml resolve speech-commands-cnn
nexuml build configs/speech-commands-cnn.yaml
nexuml train speech-commands-cnn --max-epochs 1

nexuml resolve speech-commands-transformer
nexuml build configs/speech-commands-transformer.yaml
```

If the repository has lint/type tooling configured by implementation time, run the repository-defined commands as well. Do not introduce a heavyweight new quality tool solely for NEX-210.

Run OpenSpec validation:

```bash
openspec validate nex-210-audio-example --strict
```

or the exact equivalent supported by the installed OpenSpec CLI.

## Acceptance Architecture

A successful implementation has a simple dependency direction:

```text
NexuMLTutorial local code
    |
    +--> nexuml.core public specs/decorators/base classes
    |
    +--> torch / tensordict / pandas / numpy as ordinary runtime libraries
    |
    X--> nexuml_library                 (forbidden)
    X--> nvidia.dali direct import      (forbidden in tutorial code)
    X--> tensorflow                     (forbidden dependency)
```

And a simple data path:

```text
Mini Speech Commands WAV files
       |
       v
MiniSpeechCommandsDataset metadata
(file, class, split; data=None)
       |
       v
LoaderSpec(backend="dali")
       |
       v
NexuML DaliLoaderBackend
       |
       v
NexuML native audio_file_pipeline
readers.file -> decoders.audio -> pad/truncate
       |
       v
TensorDict waveform [B,16000]
       |
       +-----------------------------+
       v                             v
AudioCNNEncoder             TinyAudioTransformerEncoder
       |                             |
       +-------- embeddings [B,64] --+
                     |
                     v
            ClassificationHead
                     |
               class_logits
                /        \
               v          v
      CrossEntropyLoss   metrics
```

That architecture is the central learning outcome of NEX-210.

## Risks & Mitigations

### DALI is platform-specific

Risk: a user on an unsupported platform cannot run the audio training path.

Mitigation:

- make DALI availability explicit in the audio tutorial;
- keep dataset/model unit tests independent of DALI;
- do not silently present Torch loading as equivalent to the tutorial objective;
- rely on NexuML's DALI extra/version policy rather than pinning another copy here.

### Raw waveform models may train less accurately than spectrogram models

Risk: users may expect state-of-the-art keyword spotting accuracy.

Mitigation:

- state that the goal is demonstrating NexuML composition and native file loading;
- keep the models intentionally small;
- place log-mel preprocessing in the documented next stage where it naturally teaches NexuML preprocessing/data-export concepts.

### Speaker-hash split may not be exactly balanced

Risk: hash buckets create approximate rather than exact sample percentages.

Mitigation:

- split by speaker to avoid leakage;
- document bucket semantics rather than promising exact counts;
- do not add complex stratified group-splitting dependencies for a tutorial.

### Generic classification cleanup changes existing tutorial layer keys

Risk: old local generated configs referencing `LatentClassificationHead`/`BCELoss` stop resolving.

Mitigation:

- this tutorial repo is a learning example, not a stable compatibility package;
- update all checked-in callers/docs together;
- do not add compatibility aliases that make the example harder to understand.

## Implementation Quality Bar

Before declaring NEX-210 complete, review the resulting diff for the following:

- no `nexuml_library` dependency/import remains;
- no direct `nvidia.dali` import exists in `library/` or `tutorials/`;
- no TensorFlow dependency exists;
- only one audio dataset implementation exists;
- only one shared audio pipeline assembly exists;
- only one generic classification head and one single-label multiclass loss exist;
- CNN and Transformer do not duplicate head/loss/metrics/scenario logic;
- no future tutorial placeholder code was added;
- tests are small and contract-focused;
- README commands match real registered scenario names;
- all paths use scenario names rather than stale `cifar-resnet`/`mnist_resnet` hard-coding;
- the final code remains readable without knowing the implementation of NexuML's DALI backend.
