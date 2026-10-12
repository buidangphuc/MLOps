## Purpose

Fine-tune a small causal LM on a labelled text-classification dataset posed as generation, and evaluate and serve it with deterministic label scoring.

## ADDED Requirements

### Requirement: Configuration loads from params.yaml
The configuration layer SHALL load every pipeline section from `config/params.yaml` without requiring GLiNER sections, and SHALL reject an unknown fine-tune method or a metric outside the supported set.

#### Scenario: Shipped params.yaml loads
- **WHEN** the configuration is built from the repository's `params.yaml`
- **THEN** ingestion, transformation, trainer, evaluation, gate and registry configs are returned without error

#### Scenario: Unknown fine-tune method rejected
- **WHEN** `finetune.method` is set to a value outside `full | lora | qlora | qat`
- **THEN** configuration loading fails with an error naming the allowed methods

### Requirement: Training examples mask the prompt
Each training example SHALL be the chat-formatted prompt followed by the label name as the completion, and the loss labels SHALL ignore every prompt and padding token so only completion tokens are learned.

#### Scenario: Prompt tokens are ignored by the loss
- **WHEN** a row with text T and label L is turned into a training example
- **THEN** label ids equal the ignore index on every prompt position and equal the input ids on the completion positions of L

### Requirement: Label names come from the training split
The set of candidate labels used for training, evaluation and prediction SHALL be derived from the training split only.

#### Scenario: Label absent from train
- **WHEN** a label appears in val or test but not in train
- **THEN** it is not a candidate label and its rows are excluded from evaluation

### Requirement: Fine-tune methods
The trainer SHALL support `full`, `lora`, `qlora` and `qat` methods selected by `finetune.method`, and the saved model directory SHALL always be loadable as a plain causal-LM directory (adapters merged).

#### Scenario: LoRA trains only adapters
- **WHEN** the method is `lora`
- **THEN** only adapter parameters are trainable during training and the saved model directory loads without the adapter library

### Requirement: Evaluation scores labels by log-likelihood
Evaluation and prediction SHALL score each candidate label by the model's log-likelihood of the label completion given the prompt, predict the highest-scoring label, and expose the per-label scores as logits to the existing metrics, calibration and bootstrap code. The metric names the gate reads SHALL be unchanged.

#### Scenario: Highest-likelihood label wins
- **WHEN** a model assigns label B the highest completion log-likelihood for a text
- **THEN** the predicted label is B and the per-label score vector has one entry per candidate label

#### Scenario: Metrics file keeps gate keys
- **WHEN** evaluation finishes
- **THEN** the metrics file contains every metric in `metrics.compute` at top level, including `metrics.objective`

### Requirement: Tuned perf config applies only when opted in
When `perf.apply_tuned` is true, the trainer SHALL apply the training-throughput config tuned for the current box's train role; when false (the default) it SHALL not read perf results. A result tuned on different hardware SHALL fail rather than be applied.

#### Scenario: Opt-in off
- **WHEN** `perf.apply_tuned` is false
- **THEN** training arguments contain no values from stored perf results

#### Scenario: Result from another GPU
- **WHEN** `perf.apply_tuned` is true and the stored train result was tuned on a different hw_id
- **THEN** training stops with a hardware-mismatch error
