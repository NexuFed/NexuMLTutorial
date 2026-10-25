## ADDED Requirements

### Requirement: Tutorial library is independent from nexuml_library
The tutorial repository SHALL run as its own external NexuML library without installing, importing, or copying implementations from `nexuml_library`.

#### Scenario: Project dependencies contain NexuML but not nexuml_library
- **WHEN** a user inspects `pyproject.toml`
- **THEN** the project SHALL depend on NexuML core/framework packages needed by the tutorial
- **AND** it SHALL NOT declare `nexuml-library` as a dependency.

#### Scenario: Tutorial code contains no nexuml_library imports
- **WHEN** the repository's Python tutorial code is searched for `nexuml_library`
- **THEN** no runtime import from `nexuml_library` SHALL exist
- **AND** the audio example SHALL use only tutorial-owned implementations plus NexuML core/public contracts and ordinary runtime libraries.

#### Scenario: Library entry point targets the local importable package
- **WHEN** the tutorial package is installed
- **THEN** the `nexuml.libraries` entry point SHALL target the importable `library` package
- **AND** NexuML SHALL be able to discover decorated tutorial components from that package.

### Requirement: Single-label classification uses a generic logits contract
The tutorial SHALL expose one generic classification head and one cross-entropy loss for the MNIST and Speech Commands single-label multiclass examples.

#### Scenario: Classification head returns logits
- **WHEN** `ClassificationHead` processes an embedding tensor
- **THEN** it SHALL return raw `class_logits`
- **AND** it SHALL NOT apply softmax internally.

#### Scenario: Multiclass loss consumes scalar class labels
- **WHEN** `CrossEntropyLoss` receives `class_logits` and the configured scalar class label
- **THEN** it SHALL compute per-sample multiclass cross-entropy loss
- **AND** the caller SHALL NOT need to one-hot encode labels.

#### Scenario: Existing MNIST tutorial uses generic classification components
- **WHEN** the `mnist-resnet` pipeline is resolved after NEX-210
- **THEN** it SHALL use `ClassificationHead`
- **AND** it SHALL use `CrossEntropyLoss`
- **AND** it SHALL continue to expose the existing classification metrics.

### Requirement: Obsolete tutorial-only classification names are removed
The tutorial SHALL prefer the clean current API over compatibility aliases for tutorial-only code.

#### Scenario: Old head and loss are absent
- **WHEN** NEX-210 is complete
- **THEN** `library/layers/head/resnet_head.py` SHALL be removed
- **AND** `library/layers/loss/bce_loss.py` SHALL be removed
- **AND** no `LatentClassificationHead` compatibility registration SHALL remain
- **AND** no `BCELoss` compatibility registration SHALL remain.

### Requirement: Shared defaults do not hard-code another scenario name
Shared tutorial configuration helpers SHALL derive artifact paths from the active scenario name.

#### Scenario: Scenario-specific output paths are generated
- **WHEN** MNIST or either Speech Commands scenario creates default callbacks/exports
- **THEN** checkpoint paths SHALL use that scenario's name
- **AND** model export paths SHALL use that scenario's name
- **AND** no stale `cifar-resnet` or unrelated `mnist_resnet` path SHALL be injected into another scenario.
