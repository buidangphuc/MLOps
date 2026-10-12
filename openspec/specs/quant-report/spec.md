# quant-report Specification

## Purpose
One table that compares quantization schemes and fine-tune methods on speed, memory and quality for the current hardware, so the trade-off is decided from data.

## Requirements

### Requirement: Report from stored results
`datascience perf report` SHALL build a table from the current box's stored infer sweep (one row per trial: scheme, batch, throughput or tokens/s, p99, peak memory, quality, quality_drop, feasible) and from tune runs when present (one row per variant: method, objective metric, train speed and peak memory when recorded), write it as markdown and CSV under the infer role's hw_id directory, and print it.

#### Scenario: Infer results only
- **WHEN** a stored infer sweep exists and there are no tune runs
- **THEN** the report has one row per stored trial and no tune section, and both files are written

#### Scenario: No stored results for this box
- **WHEN** no infer sweep is stored for the current hw_id
- **THEN** the command fails with a message to run `perf tune infer` on this box first

### Requirement: Report marks the chosen config
The report SHALL mark the trial chosen by the sweep and SHALL show infeasible trials with the reason they were rejected.

#### Scenario: Chosen and rejected rows
- **WHEN** the sweep chose one trial and rejected another for quality_drop
- **THEN** the chosen row is marked and the rejected row shows the violated constraint
