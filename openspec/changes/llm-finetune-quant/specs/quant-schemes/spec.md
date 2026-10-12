## Purpose

Named post-training quantization schemes that inference can be swept over, each bound to the GPU capability it needs so a scheme is only ever chosen for a target that can run it.

## ADDED Requirements

### Requirement: Schemes declare their hardware requirement
Every PTQ scheme (`fp16, bf16, int8-wo, int8-dyn, int4-wo, fp8, nf4, bnb-int8`) SHALL declare the minimum compute capability it needs, and capability checks SHALL be decided from the role's TARGET in `hardware.yaml`, never from the current box.

#### Scenario: T4 infer target
- **WHEN** the infer target has compute capability 7.5
- **THEN** `fp16`, `int8-wo`, `int8-dyn`, `nf4` and `bnb-int8` are supported and `bf16`, `int4-wo` and `fp8` are not

#### Scenario: H100 infer target
- **WHEN** the infer target has compute capability 9.0
- **THEN** every scheme is supported

### Requirement: Infer sweep filters unsupported schemes before measuring
The perf infer sweep SHALL drop precisions the infer target cannot run before any trial is measured, and a scheme a backend cannot build SHALL be recorded as an infeasible trial rather than abort the sweep.

#### Scenario: Unsupported scheme never measured
- **WHEN** the grid contains `fp8` and the infer target is a T4
- **THEN** no trial with precision `fp8` is run or stored

#### Scenario: Backend cannot build a scheme
- **WHEN** a backend raises Unsupported for a scheme
- **THEN** the trial is stored as infeasible and the sweep continues

### Requirement: Quantized trials are quality-guarded
Each quantized trial SHALL report quality and quality_drop against the fp32 baseline using the task metric on a fixed dev slice, and a trial whose drop exceeds the configured constraint SHALL not be chosen.

#### Scenario: Large quality drop rejected
- **WHEN** a scheme is faster than all others but its quality_drop exceeds `infer.constraints.quality_drop.max`
- **THEN** the chosen config is a different, feasible trial

### Requirement: Real model entry
The package SHALL provide an infer entry and a train entry for the perf platform backed by the pipeline's own predictor and trainer, so `perf tune|profile` runs on the fine-tuned model without edits to the perf layer. When the target generates tokens, trials SHALL also report tokens per second.

#### Scenario: Entry resolves
- **WHEN** `--entry datascience.infer.perf_entry:target` is resolved
- **THEN** it provides `predictor`, `inputs` and `score` as the infer workload contract requires
