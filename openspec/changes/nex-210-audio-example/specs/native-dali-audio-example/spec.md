## ADDED Requirements

### Requirement: Mini Speech Commands downloads automatically without TensorFlow
The tutorial SHALL provide an automatically downloadable Mini Speech Commands dataset without adding TensorFlow or another dataset framework as a runtime dependency.

#### Scenario: Missing dataset downloads from the official mini archive
- **WHEN** a `MiniSpeechCommandsDataset(download=True)` definition is built
- **AND** the configured root does not contain the dataset
- **THEN** the tutorial SHALL download the official `mini_speech_commands.zip` archive
- **AND** extract it to the configured data location
- **AND** discover WAV files from the eight supported command directories.

#### Scenario: Download disabled fails clearly
- **WHEN** the configured dataset root is missing
- **AND** `download=False`
- **THEN** building the dataset SHALL raise an actionable file-not-found error
- **AND** it SHALL NOT silently create an empty training dataset.

#### Scenario: TensorFlow is not required
- **WHEN** the audio example is installed and executed
- **THEN** TensorFlow SHALL NOT be required for download, metadata creation, loading, model definition, or training.

### Requirement: Audio metadata uses a stable classification and split contract
The dataset SHALL expose deterministic file-backed metadata with stable class IDs and speaker-grouped train/validation/test assignments.

#### Scenario: Class IDs are stable
- **WHEN** dataset metadata is built
- **THEN** the class mapping SHALL be exactly `down=0`, `go=1`, `left=2`, `no=3`, `right=4`, `stop=5`, `up=6`, `yes=7`
- **AND** the mapping SHALL NOT depend on filesystem enumeration order.

#### Scenario: Speakers do not cross dataset splits
- **WHEN** two WAV files share the same speaker prefix before `_nohash_`
- **THEN** both files SHALL receive the same `train`, `val`, or `test` value
- **AND** the assignment SHALL be deterministic across Python processes and machines.

#### Scenario: Metadata remains file-backed
- **WHEN** the dataset definition is built
- **THEN** `meta["file"]` SHALL contain WAV file paths
- **AND** `meta["class"]` SHALL contain scalar class IDs
- **AND** `meta["split"]` SHALL contain `train`, `val`, or `test`
- **AND** the dataset's `data` attribute SHALL remain `None`.

### Requirement: Audio scenario selects NexuML native DALI loading through public configuration
The Speech Commands scenarios SHALL select the NexuML DALI loader via `LoaderSpec` and SHALL NOT implement or call a tutorial-owned DALI pipeline.

#### Scenario: DataSpec selects DALI and preserves dataset splits
- **WHEN** either Speech Commands scenario is resolved
- **THEN** `data.loader.backend` SHALL be a `DaliLoader` definition
- **AND** the dataset specification SHALL use `split_type="keep"`
- **AND** the feature key SHALL be `waveform`
- **AND** the declared input shape SHALL be `[16000]`
- **AND** the declared number of classes SHALL be 8.

#### Scenario: Dataset declares the native audio contract
- **WHEN** a `MiniSpeechCommandsDataset` definition is built
- **THEN** its sample rate SHALL be `16000`
- **AND** `dali_x_keys` SHALL equal `["waveform"]`
- **AND** `dali_layout` SHALL equal `T`
- **AND** `dali_sequence_length` SHALL equal `16000`.

#### Scenario: Tutorial code does not import DALI directly
- **WHEN** Python files under the tutorial's `library/` are inspected
- **THEN** they SHALL NOT import `nvidia.dali`
- **AND** no tutorial-defined DALI iterator, file reader, audio decoder, or external-source pipeline SHALL exist.

#### Scenario: Supported runtime executes the native file path
- **WHEN** DALI is installed on a supported environment
- **AND** `speech-commands-cnn` is trained
- **THEN** NexuML SHALL receive a metadata-backed dataset with WAV file paths and `data is None`
- **AND** the configured DALI backend SHALL be eligible for NexuML's native audio file-loader route rather than its in-memory Torch fallback condition.

### Requirement: Direct dataset inspection has the same waveform contract
The dataset SHALL remain inspectable through `NexuDataset.__getitem__` without making that Python item path the configured training loader.

#### Scenario: Direct item returns padded waveform and label
- **WHEN** a valid official-format WAV sample is loaded directly from the dataset
- **THEN** `x["waveform"]` SHALL be a float waveform with shape `[16000]`
- **AND** `y["class"]` SHALL contain the sample's scalar class label.

### Requirement: CNN and Transformer share one downstream classifier contract
The tutorial SHALL provide one small CNN encoder and one tiny Transformer encoder whose outputs are interchangeable for the rest of the NexuML pipeline.

#### Scenario: CNN emits fixed-size embeddings
- **WHEN** `AudioCNNEncoder` receives a batch of `[16000]` waveforms with default settings
- **THEN** it SHALL output finite embeddings with dimension 64.

#### Scenario: Tiny Transformer emits the same embedding shape
- **WHEN** `TinyAudioTransformerEncoder` receives the same waveform contract with default settings
- **THEN** it SHALL use standard PyTorch Transformer encoder layers
- **AND** output finite embeddings with dimension 64.

#### Scenario: Architecture swap does not duplicate downstream stages
- **WHEN** `speech-commands-cnn` and `speech-commands-transformer` are compared
- **THEN** the CNN scenario SHALL select `AudioCNNEncoder`
- **AND** the Transformer scenario SHALL select `TinyAudioTransformerEncoder`
- **AND** both SHALL reuse the same data contract, `ClassificationHead`, `CrossEntropyLoss`, classification metrics, and shared scenario/training assembly.
