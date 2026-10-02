# Spec Delta

## Purpose

Teach reproducible hyperparameter selection and experiment comparison without exposing held-out test data to model selection.

## ADDED Requirements

### Requirement: Hyperparameter trials use validation data only
The tuning example SHALL select hyperparameters using an explicitly named validation metric and SHALL NOT execute test evaluation during any trial. The held-out test split SHALL be evaluated only after hyperparameter selection for the final chosen model.

#### Scenario: A small tuning run selects a model
- **WHEN** a reader runs the documented two-trial smoke command
- **THEN** each trial records its sampled parameters and finite validation objective
- **AND** the run reports the best parameters and objective direction
- **AND** no trial reads the test split or selects on a test metric.

#### Scenario: The objective is unavailable
- **WHEN** the configured validation metric is absent or non-finite
- **THEN** the example reports an actionable error rather than substituting a training or test metric.

### Requirement: Tracking supports experiment comparison
The example SHALL expose explicit local tracking configuration and distinguish trial runs sufficiently for a reader to compare parameters and validation results. Optional tracking integrations SHALL have documented installation and viewing instructions.

#### Scenario: A reader compares trials
- **WHEN** a tuning run completes with a documented tracking backend enabled
- **THEN** the tracking output distinguishes trials and shows their parameters and validation results
- **AND** the documentation identifies the local artifact location and command used to inspect it.
