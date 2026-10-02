# nex-210-audio-example

## Why

`NexuMLTutorial` currently demonstrates NexuML with an MNIST/ResNet example, but it does not demonstrate a file-backed modality or NexuML's native NVIDIA DALI loading path. The current MNIST source stores a torchvision dataset in `self.data`; NexuML's DALI loader intentionally falls back to the Torch loader for in-memory datasets. That makes the existing tutorial unsuitable for teaching how NexuML handles real file-backed data with a native DALI reader.

The tutorial repository also currently declares `nexuml-library>=0.1.0` in `pyproject.toml` even though the README explicitly installs NexuML and then uninstalls `nexuml_library`. That contradicts the tutorial's intended purpose: users should be able to build a clean external library from scratch using only NexuML core contracts.

NEX-210 shall add one small, real audio-classification example that exercises NexuML's native WAV/DALI path while remaining easy to download, inspect, train, and modify. It shall also make the tutorial repository genuinely standalone and document a progressive path from the existing MNIST example to more advanced NexuML features.

## What Changes

- Make `NexuMLTutorial` independent from `nexuml_library`:
  - remove the `nexuml-library` dependency;
  - fix the package entry point so it points at the importable local `library` package;
  - keep all tutorial dataset/model/loss/metric/scenario implementations local to this repository;
  - do not import or copy implementations from `nexuml_library`.
- Add Google's Mini Speech Commands dataset as the audio example:
  - official small Speech Commands subset;
  - approximately 182 MB download;
  - 8,000 WAV files;
  - eight classes: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes`;
  - clips are 16 kHz and at most one second;
  - download automatically with Python standard-library tools only;
  - build deterministic speaker-grouped train/validation/test metadata so recordings from one speaker never cross splits.
- Add a file-backed `MiniSpeechCommandsDataset` using NexuML's dataset contract:
  - `meta["file"]` contains the WAV path;
  - `meta["class"]` contains the integer class;
  - `meta["split"]` preserves the deterministic split;
  - modality is `audio`;
  - native DALI contract is `waveform` with layout `T`, 16 kHz sample rate, and 16,000 samples;
  - no tutorial-owned DALI pipeline is implemented.
- Configure the scenario with `LoaderSpec(backend=DaliLoader())` so NexuML core selects its native audio path (`readers.file` + `decoders.audio`) rather than using an external-source or tutorial-defined loader.
- Add two deliberately small raw-waveform encoders sharing the same output contract:
  - `AudioCNNEncoder`: a small 1D CNN and temporal global pooling;
  - `TinyAudioTransformerEncoder`: Conv1d patch embedding plus a two-layer Transformer encoder;
  - both output a fixed 64-dimensional `embeddings` tensor so the same classifier, loss, metrics, training setup, and dataset can be reused.
- Generalize the tutorial's single-label classification pieces while touching the minimum code needed:
  - replace the ResNet-specific `LatentClassificationHead` naming with a generic `ClassificationHead` that returns logits;
  - replace softmax + BCE for ordinary multiclass classification with raw logits + `CrossEntropyLoss`;
  - update the existing MNIST scenario to use the same generic classification components;
  - do not keep legacy aliases for the old tutorial-only layer names.
- Add two scenarios:
  - `speech-commands-cnn` as the default/simple audio tutorial;
  - `speech-commands-transformer` as the architecture-swap example;
  - both are assembled through one small shared audio-classifier scenario factory rather than duplicating the scenario.
- Expand tutorial documentation:
  - document how and why the audio example uses native DALI;
  - show `resolve -> build -> train` for both audio scenarios;
  - explicitly explain that changing from CNN to Transformer changes the encoder while leaving the rest of the pipeline intact;
  - document the recommended learning sequence for subsequent NexuML features without implementing all of them in NEX-210.

## Dataset Choice

Mini Speech Commands is selected because it is small enough for an onboarding tutorial but still represents real file-backed audio:

- official TensorFlow tutorial source: `http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip`;
- 8,000 WAV files / 8 commands;
- 16 kHz, <= 1 second clips;
- directory-per-class layout;
- no TensorFlow runtime dependency is required by this tutorial;
- the archive is downloaded and extracted using the Python standard library.

The tutorial shall use the explicit class ordering:

```text
0 down
1 go
2 left
3 no
4 right
5 stop
6 up
7 yes
```

## Native DALI Requirement

The audio example exists specifically to exercise the NexuML core DALI backend. The tutorial MUST NOT import `nvidia.dali` directly or define its own DALI pipeline.

The intended public configuration is:

```python
DataSpec(
    datasets=[
        DatasetSpec(
            source=MiniSpeechCommandsDataset(
                root="data/mini_speech_commands",
                download=True,
            ),
            modality="audio",
            split_type="keep",
        )
    ],
    input_shapes={"waveform": [16000]},
    num_classes=8,
    feature_key="waveform",
    loader=LoaderSpec(backend=DaliLoader(), num_workers=4),
)
```

The dataset shall expose the native-file contract expected by NexuML core:

```python
self.sample_rate = 16_000
self.dali_x_keys = ["waveform"]
self.dali_layout = "T"
self.dali_sequence_length = 16_000
```

`split_type="keep"` is required because the dataset owns a deterministic speaker-grouped `split` column that must not be overwritten by `DataCreator`.

## Model Choice

### `AudioCNNEncoder`

The default model shall be intentionally small and understandable:

```text
waveform [B, 16000]
  -> add channel dimension
  -> Conv1d block
  -> Conv1d block
  -> Conv1d block
  -> AdaptiveAvgPool1d(1)
  -> embeddings [B, 64]
```

The exact implementation may tune kernel/stride values during implementation, but it shall remain a compact raw-waveform CNN with no spectrogram dependency and no third-party model package.

### `TinyAudioTransformerEncoder`

The second model shall demonstrate that only the encoder needs to change:

```text
waveform [B, 16000]
  -> add channel dimension
  -> Conv1d patch embedding, patch/stride 160
  -> ~100 tokens x 64 dimensions
  -> positional embedding
  -> 2 x TransformerEncoderLayer
       d_model=64
       nhead=4
       dim_feedforward=128
       batch_first=True
  -> LayerNorm
  -> mean token pooling
  -> embeddings [B, 64]
```

No CLS token, custom attention implementation, pretrained model, or audio-specific transformer package shall be introduced.

## Progressive Learning Path

NEX-210 shall document the following recommended progression. Only stages 1 and 2/3 are implemented by this change; later stages are roadmap items for future changes.

1. **MNIST — first custom library**
   - register a local dataset, layers, and scenario;
   - inspect registries;
   - use `resolve`, `build`, and `train`;
   - understand `ScenarioSpec` and TensorDict keys.
2. **Speech Commands CNN — real file-backed data + native DALI**
   - learn metadata-backed file datasets;
   - understand modality, `file`, `split`, and `LoaderSpec`;
   - see native DALI selected through configuration only.
3. **Speech Commands Transformer — pipeline composition**
   - swap only the encoder;
   - reuse the same dataset, head, loss, metrics, and training setup.
4. **Tuning and tracking**
   - Optuna over learning rate/model width/depth;
   - compare TensorBoard/MLflow runs.
5. **Preprocessing and data-export backends**
   - add log-mel preprocessing;
   - materialize prepared features;
   - compare NumPy/WebDataset and DALI loading without changing the training loop.
6. **Custom evaluation**
   - add a small confusion-matrix or embedding evaluation algorithm;
   - demonstrate `@eval_algorithm` separately from the training loop.
7. **Checkpoints and transfer learning**
   - resume, selectively load, freeze, and fine-tune an encoder.
8. **Export and inference**
   - package/SafeTensors/ONNX export and reload.
9. **Distributed execution**
   - add only after the corresponding NexuML execution backend is stable; do not document WIP backend APIs in this change.

## Capabilities

### New

- `standalone-tutorial-library`
- `native-dali-audio-example`
- `progressive-tutorial-path`

### Modified

None. This repository has no existing OpenSpec capability baseline yet; NEX-210 introduces the initial tutorial-specific specifications.

## Impact

### Expected repository structure after implementation

```text
NexuMLTutorial/
├── README.md
├── pyproject.toml
├── openspec/
│   ├── config.yaml
│   └── changes/
│       └── nex-210-audio-example/
│           ├── .openspec.yaml
│           ├── proposal.md
│           ├── design.md
│           ├── tasks.md
│           └── specs/
│               ├── standalone-tutorial-library/spec.md
│               ├── native-dali-audio-example/spec.md
│               └── progressive-tutorial-path/spec.md
├── library/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   └── mini_speech_commands.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── defaults.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── mnist.py
│   │   │   └── mini_speech_commands.py
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   ├── resnet_classifier.py
│   │   │   └── audio_classifier.py
│   │   └── scenario/
│   │       ├── __init__.py
│   │       ├── mnist_resnet.py
│   │       └── speech_commands.py
│   ├── evaluation/
│   │   └── ... existing evaluation code ...
│   └── layers/
│       ├── model/
│       │   ├── __init__.py
│       │   ├── resnet.py
│       │   └── audio.py
│       ├── head/
│       │   ├── __init__.py
│       │   └── classification.py
│       ├── loss/
│       │   ├── __init__.py
│       │   └── cross_entropy.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   └── classification_metrics.py
│       └── utils/
│           └── pooling.py
├── tutorials/
│   ├── 01_mnist_from_scratch.md
│   └── 02_audio_native_dali.md
└── tests/
    ├── test_mini_speech_commands.py
    ├── test_audio_models.py
    └── test_audio_scenarios.py
```

### Existing files intentionally replaced/renamed

- `library/layers/head/resnet_head.py` -> `library/layers/head/classification.py`
- `library/layers/loss/bce_loss.py` -> `library/layers/loss/cross_entropy.py`

The old tutorial-only files/component identities shall be removed rather than kept as compatibility aliases.

## Non-Goals

- Do not modify the NexuML repository or NexuML's DALI implementation in this change.
- Do not import from or depend on `nexuml_library`.
- Do not copy CIFAR, AudioSet, DCASE, or other implementations from `nexuml_library` into the tutorial.
- Do not implement a tutorial-specific DALI loader, DALI iterator, DALI augmentation graph, or DALI preprocessing backend.
- Do not add TensorFlow as a dependency; it is only the host/documentation source for Mini Speech Commands.
- Do not add Hugging Face datasets/models, torchaudio datasets, librosa, timm, transformers, or other convenience frameworks for this example.
- Do not add spectrogram/log-mel preprocessing yet; keep NEX-210 focused on raw waveform classification and native file loading.
- Do not implement all future tutorials listed in the learning path.
- Do not add a large test matrix, network-dependent CI test, or performance benchmark suite.
- Do not add legacy aliases for renamed tutorial components.
