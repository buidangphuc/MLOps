"""PTQ schemes: name -> (hardware feature it needs, how it is built).

Kinds: `dtype` casts the module; `torchao` runs `torchao.quantization.quantize_` on a loaded
module; `load` hands a bitsandbytes config to `from_pretrained`, so it needs a model DIRECTORY.
torchao / bitsandbytes (the `quant` extra) are imported lazily; a missing library, a missing GPU
or an unknown name raises `Unsupported`, which the sweep records as an infeasible trial.

Task-agnostic: the pipeline calls `apply` (model.load_trained, perf infer builders).
"""

import importlib

from datascience.perf.hw import supports
from datascience.perf.runner import Unsupported

SCHEMES = {                       # name -> (min-capability feature in hw._MIN_CC, kind)
    "fp32": (None, "dtype"),
    "fp16": ("fp16", "dtype"),
    "bf16": ("bf16", "dtype"),
    "int8-wo": ("int8", "torchao"),
    "int8-dyn": ("int8", "torchao"),
    "int4-wo": ("int4", "torchao"),
    "fp8": ("fp8", "torchao"),
    "nf4": ("nf4", "load"),
    "bnb-int8": ("int8", "load"),
}
_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
_TORCHAO = {"int8-wo": "Int8WeightOnlyConfig", "int8-dyn": "Int8DynamicActivationInt8WeightConfig",
            "int4-wo": "Int4WeightOnlyConfig", "fp8": "Float8DynamicActivationFloat8WeightConfig"}
_CUDA_ONLY = {"int4-wo", "fp8"}   # the other torchao schemes also run on CPU; `load` needs CUDA too


def supported(role: str, targets: dict | None = None) -> list[str]:
    """Schemes the role's TARGET GPU can run (decided from hardware.yaml, not this box)."""
    return [n for n, (feature, _) in SCHEMES.items() if feature is None or supports(feature, role, targets)]


def _import(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise Unsupported(f"{module} is not installed (pip install -e .[quant])") from e


def _need_cuda(scheme: str):
    import torch

    if not torch.cuda.is_available():
        raise Unsupported(f"{scheme} needs a CUDA GPU")
    return torch


def apply(source, scheme: str, loader=None):
    """The model under `scheme`. `source` is a loaded module (dtype / torchao kinds) or a model
    directory (load kinds; `loader(path, **kwargs)` defaults to AutoModelForCausalLM.from_pretrained)."""
    if scheme not in SCHEMES:
        raise Unsupported(f"unknown scheme {scheme!r} (choices: {list(SCHEMES)})")
    kind = SCHEMES[scheme][1]

    if kind == "dtype":
        import torch

        return source.to(dtype=getattr(torch, _DTYPES[scheme]))

    if kind == "torchao":
        if scheme in _CUDA_ONLY:
            _need_cuda(scheme)
        ao = _import("torchao.quantization")
        try:
            config = getattr(ao, _TORCHAO[scheme])()
        except AttributeError as e:
            raise Unsupported(f"installed torchao has no {_TORCHAO[scheme]}") from e
        if scheme in _CUDA_ONLY:
            import torch

            source = source.to(torch.bfloat16)   # int4 / fp8 kernels are bf16-only
        ao.quantize_(source, config)
        return source

    torch = _need_cuda(scheme)                   # kind == "load": bitsandbytes, from a directory
    _import("bitsandbytes")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if scheme == "nf4":
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        config = BitsAndBytesConfig(load_in_8bit=True)
    loader = loader or AutoModelForCausalLM.from_pretrained
    return loader(str(source), quantization_config=config, device_map={"": torch.cuda.current_device()})
