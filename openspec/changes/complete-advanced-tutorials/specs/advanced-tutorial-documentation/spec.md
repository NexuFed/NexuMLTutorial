# Spec Delta

## Purpose

Provide an honest, reproducible learning progression covering every requested advanced tutorial without presenting deferred work as complete.

## ADDED Requirements

### Requirement: The learning path links every implemented advanced topic
The README SHALL link real runnable tutorial documents for tuning/tracking, preprocessing/dataset export, custom evaluation, checkpoints/transfer, and model export/inference. Distributed execution SHALL remain explicitly planned, and the existing introductory examples SHALL remain accessible.

#### Scenario: A reader follows the roadmap
- **WHEN** a reader opens the README learning path after implementation
- **THEN** each of the five advanced topics links an existing tutorial
- **AND** distributed execution is not marked available
- **AND** the introductory MNIST and raw-waveform audio links still resolve.

### Requirement: Chapters explain a minimal reproducible workflow
Each advanced chapter SHALL explain the newly introduced NexuML contract, reused components, prerequisites, copyable commands, artifact locations, expected results, and a focused verification step. It SHALL distinguish network downloads, optional dependencies, and CUDA/DALI requirements from offline CPU checks.

#### Scenario: A reader runs a smoke workflow
- **WHEN** a reader follows a chapter's setup and minimal smoke commands from the repository root
- **THEN** all referenced scenarios, components, scripts, and paths exist
- **AND** the workflow produces the stated artifacts or reports an actionable missing prerequisite
- **AND** the documentation does not imply that an unexecuted GPU or optional-format check passed.

### Requirement: Examples remain a standalone external library
The completed tutorials SHALL rely on NexuML public contracts and directly appropriate existing libraries, without importing or copying the NexuML base library or implementing tutorial-owned training, loader, writer, or export infrastructure.

#### Scenario: A reader inspects the example code
- **WHEN** a reader traces any documented advanced workflow
- **THEN** configuration and small tutorial components remain visible in this repository
- **AND** training orchestration, dataset serialization/loading, and model artifact handling remain delegated to their public library owners.
