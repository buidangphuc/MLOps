"""Layer-2 tuning: a loop over manual variants (loop.py), manual promotion to the
registry with lineage (promote.py), and the R4 DAPT pre-stage (dapt.py).

Versioning is NOT re-invented here: each variant is logged as an MLflow run (params +
metrics) and, when promoted, versioned in the registry. This layer only adds the loop
+ the lineage that links a registry version back to its run and config snapshot.
"""
