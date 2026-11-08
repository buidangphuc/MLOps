## MODIFIED Requirements

### Requirement: Configuration loads from params.yaml
The configuration layer SHALL load the common sections of `config/text_classification/params.yaml` through the platform and the example pack's sections through `packs/text_classification/config.py`, and SHALL reject an unknown fine-tune method or a metric outside the pack's `METRICS`.

#### Scenario: Shipped params.yaml loads
- **WHEN** the configuration is built from the repository's `config/text_classification/params.yaml`
- **THEN** the pack's ingestion, transformation, trainer and evaluation configs and the platform's gate and registry configs are returned without error

#### Scenario: Unknown fine-tune method rejected
- **WHEN** `finetune.method` is set to a value outside `full | lora | qlora | qat`
- **THEN** configuration loading fails with an error naming the allowed methods
