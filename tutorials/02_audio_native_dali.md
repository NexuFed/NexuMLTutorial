# Speech Commands: Native DALI Audio

Mini Speech Commands adds a real file-backed modality while keeping the pipeline readable. The official archive is about 182 MB and contains 8,000 one-second-or-shorter, 16 kHz WAV clips in eight command directories:

```text
data/mini_speech_commands/
├── down/*.wav
├── go/*.wav
├── left/*.wav
├── no/*.wav
├── right/*.wav
├── stop/*.wav
├── up/*.wav
└── yes/*.wav
```

The dataset downloads and extracts this archive with Python's standard library. TensorFlow is not installed or used.

## Why This Uses Native DALI

The MNIST source keeps a torchvision dataset in `self.data`, so NexuML's DALI backend uses its in-memory fallback. `MiniSpeechCommandsDataset` instead keeps `self.data is None` and supplies metadata rows:

| Column | Meaning |
| --- | --- |
| `file` | WAV path consumed by NexuML's native file route |
| `class` | Stable integer ID for `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes` |
| `split` | Deterministic `train`, `val`, or `test` assignment |

The split hashes the speaker prefix before `_nohash_`. Buckets 0-79 train, 80-89 validation, and 90-99 test, so one speaker never crosses splits. This is approximately 80/10/10 by speaker bucket, not an exact sample-count split.

The dataset declares the native waveform contract:

```python
sample_rate = 16_000
dali_x_keys = ["waveform"]
dali_layout = "T"
dali_sequence_length = 16_000
```

The data configuration selects the backend and preserves dataset-owned splits:

```python
DatasetSpec(
    type_key="MiniSpeechCommandsDataset",
    modality="audio",
    split_type="keep",
)
LoaderSpec(backend="dali", num_workers=4)
```

No tutorial module imports DALI or defines a DALI pipeline. NexuML core maps the WAV metadata to its native reader/decoder and returns `x["waveform"]` with shape `[B, 16000]`.

## Runtime Prerequisites

NVIDIA DALI availability depends on the host platform. Install this project through its `nexuml[dali]` dependency, use a compatible NVIDIA driver/CUDA runtime for GPU execution, and confirm registration before training:

```bash
nexuml backend list
```

The output must include `data-loader dali`. Dataset and model unit tests do not require a DALI-capable GPU, but this tutorial's training path does require the DALI backend to be available.

## Train the CNN

```bash
nexuml resolve speech-commands-cnn
nexuml build configs/speech-commands-cnn.yaml
nexuml train speech-commands-cnn --max-epochs 10
```

The first run downloads and extracts Mini Speech Commands under `data/mini_speech_commands`.

## Swap Only the Encoder

The Transformer scenario changes only the encoder type:

```python
audio_classifier(encoder_type="AudioCNNEncoder")
audio_classifier(encoder_type="TinyAudioTransformerEncoder")
```

Both scenarios reuse the same dataset, `ClassificationHead`, `CrossEntropyLoss`, `ClassificationMetrics`, training, evaluation, callbacks, and logging setup. Both encoders emit `embeddings` with dimension 64.

```bash
nexuml resolve speech-commands-transformer
nexuml build configs/speech-commands-transformer.yaml
nexuml train speech-commands-transformer --max-epochs 10
```

These deliberately small raw-waveform models teach NexuML composition and native file loading, not state-of-the-art keyword spotting. Log-mel preprocessing and exported/WebDataset features are later learning-path stages.
