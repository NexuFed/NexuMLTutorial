# Spec Delta

## Purpose

Teach the difference between continuing a training run and initializing a new task from selected pretrained weights.

## ADDED Requirements

### Requirement: Resume restores the original training lifecycle
The checkpoint tutorial SHALL save a checkpoint through the configured training callback and SHALL demonstrate continuation with model, optimizer/scheduler, epoch/global-step, and compatible callback state restored. It SHALL use an explicit checkpoint path rather than guessing the latest run directory.

#### Scenario: An interrupted run continues
- **WHEN** a compatible run resumes from its saved trainer checkpoint
- **THEN** its restored state matches the checkpoint and subsequent training advances the saved epoch/global-step
- **AND** the operation is documented as resume, not transfer learning.

### Requirement: Transfer initializes a new target task selectively
The transfer tutorial SHALL reuse the MNIST encoder on FashionMNIST while initializing a fresh target head and training lifecycle. The example SHALL report the actually matched weights, reject incompatible selected encoder shapes, and demonstrate both frozen-encoder training and subsequent fine-tuning.

#### Scenario: Encoder-only initialization
- **WHEN** the target scenario loads selected source encoder weights
- **THEN** at least one encoder parameter is matched and equals its source value
- **AND** target head parameters are not loaded from the source
- **AND** the target uses FashionMNIST labels and a new optimizer/training state.

#### Scenario: Frozen training transitions to fine-tuning
- **WHEN** the target first trains with the loaded encoder frozen and then starts the documented fine-tuning phase
- **THEN** encoder parameters remain unchanged during the frozen phase while the head can update
- **AND** encoder parameters become trainable during fine-tuning
- **AND** the documentation explicitly explains the treatment of normalization buffers in the frozen phase.

#### Scenario: A selective load is invalid
- **WHEN** the source is missing, no encoder weights match, or a selected encoder tensor has an incompatible shape
- **THEN** the example fails clearly rather than silently training an uninitialized or partially incompatible encoder.
