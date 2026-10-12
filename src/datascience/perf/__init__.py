"""GPU performance platform: hardware-bound tuning (train/infer/kernel), profiling, store.

Task-agnostic: workloads take a model/trainer from an `--entry module:fn` the caller
provides; nothing here imports the pipeline's trainer or predictor.
"""
