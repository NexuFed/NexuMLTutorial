## ADDED Requirements

### Requirement: Tutorial documentation progresses from simple to advanced concepts
The repository SHALL present a recommended learning order that introduces one major NexuML concept at a time and reuses previously introduced concepts rather than replacing them with unrelated examples.

#### Scenario: First tutorial teaches an external library with MNIST
- **WHEN** a new user follows the first tutorial
- **THEN** the tutorial SHALL explain local library registration/discovery
- **AND** a local dataset, layers, and scenario
- **AND** the `resolve`, `build`, and `train` lifecycle
- **AND** it SHALL use the existing MNIST example as the simplest starting point.

#### Scenario: Second tutorial teaches file-backed audio and native DALI
- **WHEN** a user follows the audio tutorial
- **THEN** it SHALL explain why the in-memory MNIST example does not exercise native DALI file loading
- **AND** explain the Mini Speech Commands `file`/`class`/`split` metadata contract
- **AND** explain that `LoaderSpec(backend="dali")` selects NexuML's backend without requiring a tutorial-owned DALI implementation
- **AND** provide runnable `resolve`, `build`, and `train` commands for the CNN scenario.

#### Scenario: Transformer demonstrates composition rather than a second training stack
- **WHEN** a user advances from the CNN to the Transformer example
- **THEN** the documentation SHALL identify the encoder as the meaningful architecture change
- **AND** show that the dataset, head, loss, metrics, and training assembly are reused
- **AND** provide runnable `resolve`, `build`, and `train` commands for the Transformer scenario.

### Requirement: README exposes the complete recommended learning roadmap
The root README SHALL describe how users can progress through the important NexuML features after the initial examples.

#### Scenario: Learning path lists current and future stages
- **WHEN** a user reads the root README
- **THEN** it SHALL present the following conceptual progression:
  1. MNIST / custom library basics;
  2. Speech Commands CNN / file-backed native DALI;
  3. Speech Commands Transformer / pipeline composition;
  4. tuning and experiment tracking;
  5. preprocessing and dataset export, including a future WebDataset exercise;
  6. custom evaluation;
  7. checkpoints and transfer learning;
  8. model export and inference;
  9. distributed execution once a stable NexuML execution backend exists.

#### Scenario: Implemented tutorials are distinguishable from roadmap items
- **WHEN** the learning path is displayed
- **THEN** implemented tutorials SHALL link to real tutorial files
- **AND** future stages SHALL be clearly marked as planned/next
- **AND** the repository SHALL NOT contain empty placeholder implementations solely to make future stages appear complete.

### Requirement: Tutorial code and documentation stay aligned with real registered names
The documentation SHALL use scenario, layer, and backend names that actually resolve in the repository after NEX-210.

#### Scenario: Audio commands match registered scenarios
- **WHEN** a user copies the documented audio commands
- **THEN** `speech-commands-cnn` SHALL be a registered scenario
- **AND** `speech-commands-transformer` SHALL be a registered scenario
- **AND** documented `resolve`, `build`, and `train` commands SHALL use those exact names.

#### Scenario: DALI availability is made explicit
- **WHEN** a user prepares to run the audio tutorial
- **THEN** the tutorial SHALL instruct the user to verify loader availability with `nexuml backend list`
- **AND** it SHALL distinguish platform/runtime DALI availability from the dataset/model unit-test path.

### Requirement: Advanced topics are not prematurely coupled to NEX-210
The NEX-210 implementation SHALL stay focused on the standalone audio example while preserving a clear place for later NexuML feature tutorials.

#### Scenario: Spectrogram and data-export work remains future scope
- **WHEN** NEX-210 is complete
- **THEN** the audio training examples SHALL still operate on raw waveforms
- **AND** log-mel/spectrogram preprocessing SHALL remain a documented next step
- **AND** WebDataset/data-export exercises SHALL remain a documented next step rather than hidden inside the initial audio dataset.

#### Scenario: Distributed tutorial waits for stable backend contract
- **WHEN** the learning roadmap mentions distributed execution
- **THEN** it SHALL not document a work-in-progress execution API as if it were stable
- **AND** implementation SHALL be deferred until NexuML exposes the relevant stable backend contract.
