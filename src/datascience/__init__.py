import os

# We are torch-only. Disable the TF/Flax backends in HF transformers so an
# installed TensorFlow (with Keras 3) can't crash imports. Must run before
# transformers is imported anywhere — this package is imported first.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .log import log

__all__ = ["log"]
