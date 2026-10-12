# qat Specification

## Purpose
Quantization-aware fine-tuning that trains a model for a specific quantization scheme the serving GPU can run, and ships it as that quantized model.

## Requirements

### Requirement: QAT scheme must run on the infer target
When `finetune.method` is `qat`, training SHALL refuse to start unless `finetune.qat.scheme` is a QAT-capable scheme (`int8-dyn` or `int4-wo`) that the infer target supports.

#### Scenario: Scheme unsupported by the serving GPU
- **WHEN** the method is `qat`, the scheme is `int4-wo` and the infer target is a T4
- **THEN** training fails before any step with an error naming the scheme and the infer target

#### Scenario: Scheme not QAT-capable
- **WHEN** the method is `qat` and the scheme is `nf4`
- **THEN** configuration or training fails with an error listing the QAT-capable schemes

### Requirement: QAT model is evaluated and served quantized
A QAT-trained model directory SHALL record its quantization scheme, and evaluation and prediction SHALL load it with that scheme applied unless an explicit scheme is requested.

#### Scenario: Scheme recorded and reused
- **WHEN** a QAT run with scheme `int8-dyn` saves its model
- **THEN** the model directory records `int8-dyn` and evaluation loads the model with `int8-dyn` applied
