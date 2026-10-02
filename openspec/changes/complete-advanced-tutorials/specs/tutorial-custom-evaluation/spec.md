# Spec Delta

## Purpose

Teach how an external NexuML library registers a post-training evaluator and publishes useful results independently of training metrics.

## ADDED Requirements

### Requirement: Confusion-matrix evaluation is a discoverable extension
The tutorial SHALL provide a registered evaluator accepting classification logits and integer class labels. It SHALL aggregate a true-class-by-predicted-class matrix without retaining all predictions, and SHALL report an explicitly defined scalar summary and a labeled visualization.

#### Scenario: A known classification batch is evaluated
- **WHEN** a synthetic batch with known labels and predictions is evaluated
- **THEN** the matrix matches the expected counts, including classes absent from the batch
- **AND** the scalar summary agrees with those counts
- **AND** the matrix dimensions match the declared class count.

#### Scenario: A second evaluation run begins
- **WHEN** the evaluator is reused for a new evaluation lifecycle
- **THEN** it resets accumulated state instead of carrying counts from the previous run.

### Requirement: Evaluation remains separate from training metrics
The example SHALL configure the evaluator through the evaluation extension contract rather than adding custom work to the training loop. Scalar results SHALL remain available without an external logger, and supported configured loggers SHALL receive the visualization.

#### Scenario: Evaluation runs with and without tracking
- **WHEN** the documented evaluation scenario runs
- **THEN** its scalar summary is exposed in NexuML evaluation results
- **AND** when a supported logger is enabled, its artifact output includes a confusion matrix with class labels and clearly named axes.
