## MODIFIED Requirements

### Requirement: Real model entry
The example pack SHALL provide an infer entry and a train entry for the perf platform backed by its own predictor and trainer, so `perf tune|profile` runs on the fine-tuned model without edits to the perf layer. When the target generates tokens, trials SHALL also report tokens per second.

#### Scenario: Entry resolves
- **WHEN** `--entry packs.text_classification.perf_entry:target` is resolved
- **THEN** it provides `predictor`, `inputs` and `score` as the infer workload contract requires
