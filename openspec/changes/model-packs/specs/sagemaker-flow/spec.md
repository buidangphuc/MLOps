## ADDED Requirements

### Requirement: Training on SageMaker without a laptop deploy
The platform SHALL run a model's stages as a SageMaker Training Job (`train-remote`) or as a SageMaker Pipeline (Train, Evaluate, Gate, Register), building the AWS requests itself. Account values (image, role, bucket, region) SHALL come from the environment, never from tracked files.

#### Scenario: Valid requests
- **WHEN** building the training-job request and the pipeline definition for the toy pack
- **THEN** the CreateTrainingJob and CreateProcessingJob arguments validate against botocore's SageMaker model

#### Scenario: Gate in the pipeline
- **WHEN** the pipeline definition is built
- **THEN** the Gate step has one condition per params.yaml gate rule on `evaluation.json`, and a Fail branch only when `gate.on_fail` is `error`

#### Scenario: Missing account values
- **WHEN** `train-remote` runs without `SAGEMAKER_IMAGE_URI`
- **THEN** it exits before any AWS call and names the missing variable

### Requirement: Steps inside a job
`sm-step train|evaluate|register` SHALL move the model, split and training run id between SageMaker job directories and the model's artifact tree, so that the three steps run in separate containers and still register the evaluated model.

#### Scenario: Round trip
- **WHEN** the toy pack runs train, evaluate and register with fresh artifact trees between steps
- **THEN** `evaluation.json` holds the toy metrics and the MLflow alias points at the registered version

### Requirement: Deploy by CI per environment
`deploy/<model>/<env>.yaml` SHALL hold the endpoint and the desired release version per target. The deploy job SHALL copy each changed target's tar from the build store into the endpoint's model prefix without overwriting an existing key, refuse tars that are not in the build's manifest or endpoints that are not multi-model, and smoke-invoke the new target model.

#### Scenario: Copy once
- **WHEN** the target key is absent, then present
- **THEN** the first sync copies and invokes it, the second invokes it without copying

#### Scenario: Plan
- **WHEN** an env file changes one target's version and adds a new environment
- **THEN** the plan lists exactly those targets
