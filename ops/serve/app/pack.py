import importlib
import os

_mod = importlib.import_module(f"packs.{os.environ.get('DS_PACK', 'text_classification')}.serving_app")

Request = _mod.Request
WARMUP = _mod.WARMUP
load = _mod.load
respond = _mod.respond
