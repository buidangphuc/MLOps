---
name: compare-and-gate
description: Judge a model with the quality gate, tuning variants and MLflow. Use when deciding whether a model is good enough, comparing variants or backbones, choosing what to promote, or changing gate rules.
---

# Compare models and set the gate

`metrics.objective` is the one number the gate checks and `tune` ranks by; every name in
`metrics.compute` and in gate rules comes from the pack's `METRICS`.

## The gate

`params.yaml` `gate.rules: [{metric, min}]`, `gate.on_fail: error | warn`. A model failing a rule is
never registered; `error` also exits non-zero (for CI), `warn` only logs. A `min` comes from a
measured baseline: the currently served model scored on the same test set, or a labelled set the
team trusts. Until one exists the rule stays a placeholder (`min: 0.0`, `on_fail: warn`) and a human
reads the numbers.

## Compare

1. Write each option as a sparse override in `config/<id>/experiments.yaml` (a backbone, a
   fine-tune method, a data recipe).
2. `python main.py --model <id> tune`: every pack stage per variant, one MLflow run each, ranked in
   `artifacts/<id>/tune_runs/results.json`. A failed variant is recorded with its error and ranked
   last.
3. Read the ranking with its basis: variants that change data get their own split and
   `data_version`, so their test sets differ. Compare like with like: same data and seed, or score
   every candidate on one fixed test file.
4. `python main.py --model <id> promote <variant>`: the gate re-checks it and registers it with
   lineage (run id, config snapshot, data version), moving the MLflow alias.

Inspect runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db` (or the team server).

Done when the decision names the objective value, the comparison basis (same test set or not), and
the gate result for the chosen model.
