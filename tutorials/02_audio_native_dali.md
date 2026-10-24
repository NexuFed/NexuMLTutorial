# Speech Commands: Native DALI Audio

Mini Speech Commands contains 8,000 WAV clips across eight classes: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes`. The first run downloads approximately 182 MB; TensorFlow is not required.

## Train the CNN

After the [README setup](../README.md#setup), confirm that `dali` is listed:

```bash
nexuml backend list data-loader
nexuml resolve speech-commands-cnn
nexuml build configs/speech-commands-cnn.yaml
nexuml train speech-commands-cnn --max-epochs 10
```

The dataset provides `file`, `class`, and `split` metadata to NexuML's native WAV reader. Speaker hashing keeps speakers in one split, approximately 80/10/10. Input is `waveform [B, 16000]` at 16 kHz.

## Swap Only the Encoder

```bash
nexuml resolve speech-commands-transformer
nexuml build configs/speech-commands-transformer.yaml
nexuml train speech-commands-transformer --max-epochs 10
```

`AudioCNNEncoder` and `TinyAudioTransformerEncoder` both emit 64-dimensional embeddings. Dataset, head, loss, metrics, and training configuration are shared; see `library/config/scenario/speech_commands.py`.

These small raw-waveform models demonstrate composition, not state-of-the-art keyword spotting. DALI registration alone does not prove native GPU execution works.

Next: [tuning and tracking](03_tuning_and_tracking.md).
