# Spec Delta

## Purpose

Teach how to export actual trained model artifacts and verify predictions after loading them through the appropriate runtime.

## ADDED Requirements

### Requirement: Package export uses known trained weights
The export tutorial SHALL package a live trained model or an explicitly selected trained checkpoint, SHALL state the required external runtime dependencies, and SHALL demonstrate reload and label-free inference with explicit input/output keys.

#### Scenario: A trained package is reloaded
- **WHEN** the documented example reloads a trained MNIST package and predicts on the same fixed input as the source model
- **THEN** the class-logit shape and predicted classes match
- **AND** the logits agree within a documented floating-point tolerance
- **AND** no training labels are required for inference.

#### Scenario: The requested trained source does not exist
- **WHEN** the export example receives a missing explicit checkpoint or package
- **THEN** it reports that missing source rather than falling back to randomly initialized weights.

### Requirement: Alternative formats have explicit reload semantics
The tutorial SHALL demonstrate SafeTensors weight export/reload and ONNX export/inference for the existing image classifier. It SHALL distinguish a weights artifact from a full package and SHALL name any optional exporter/runtime installation requirements.

#### Scenario: SafeTensors weights are reused
- **WHEN** the exported weights are loaded into the matching configured classifier
- **THEN** its inference agrees with the original model within the documented tolerance
- **AND** the tutorial explains that matching model configuration is still required.

#### Scenario: ONNX inference is verified
- **WHEN** the optional ONNX dependencies are installed and the ONNX example is run
- **THEN** the exported graph consumes image tensors and returns classification logits
- **AND** runtime predictions agree with the source model within the documented tolerance for the tested batch sizes.

### Requirement: Export documentation reflects supported APIs
The tutorial SHALL distinguish automatic trained-package export from Python-only alternative format export and SHALL NOT claim that the standalone export command discovers a latest trained checkpoint automatically.

#### Scenario: A reader chooses an export route
- **WHEN** a reader follows the package, SafeTensors, or ONNX instructions
- **THEN** each route calls an API supported by the declared NexuML version and explains where its trained weights originate.
