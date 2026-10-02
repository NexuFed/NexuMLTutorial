# Spec Delta

## Purpose

Teach device-aware audio feature extraction and reuse of exported dataset views without changing the downstream classification experiment.

## ADDED Requirements

### Requirement: Log-mel preprocessing is batch- and device-aware
The audio example SHALL transform a batch of 16 kHz waveforms into finite channel-first log-mel features on the input tensor's device. Its configuration SHALL explicitly state the sample rate and feature parameters, and the implementation SHALL NOT require CPU-only feature computation or conversion through NumPy.

#### Scenario: CUDA feature extraction
- **WHEN** the preprocessing component and waveform batch are placed on CUDA in an available CUDA environment
- **THEN** the output remains on CUDA with shape `[B, 1, n_mels, frames]`
- **AND** the values are finite, including for silent input.

#### Scenario: CPU shape verification
- **WHEN** the same preprocessing configuration receives a synthetic CPU batch
- **THEN** it produces the same documented shape contract and finite features
- **AND** CPU verification does not imply that a CUDA/DALI execution was tested.

### Requirement: Prepared dataset views preserve the experiment contract
The tutorial SHALL export prepared features in both NumPy and WebDataset formats through NexuML and SHALL preserve feature keys, class labels, sample identities, and dataset-owned train/validation/test splits. Reloaded prepared features SHALL use the same encoder, pooling, head, loss, and metric contract as online preprocessing.

#### Scenario: Exported features are reloaded
- **WHEN** a small prepared dataset is exported and reloaded in either documented format
- **THEN** sample counts, labels, identities, and split assignments match the source view
- **AND** reloaded features match the online transform within the documented storage tolerance
- **AND** no speaker is moved between splits.

### Requirement: Prepared features bypass preprocessing exactly once
The prepared-data example SHALL skip only the preprocessing stage and SHALL retain normal downstream classification behavior.

#### Scenario: A prepared batch is classified
- **WHEN** the prepared-data scenario receives exported log-mel features
- **THEN** preprocessing is not called again
- **AND** the downstream pipeline produces class logits with eight output classes.
