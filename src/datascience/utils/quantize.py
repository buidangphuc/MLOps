"""Shared ONNX INT8 dynamic-quantization step.

One place for the optimum/onnxruntime quantize invocation so every
pipeline's optimize step behaves identically.
"""

from datascience import log


def quantize_dynamic_int8(model_dir, *, matmul_only: bool = False) -> None:
    """Apply dynamic INT8 quantization in place on ``model_dir``.

    ``matmul_only=True`` restricts quantization to MatMul ops — required for
    models with a conv patch-embed (e.g. LayoutLMv3): a quantized Conv
    becomes a ConvInteger op the CPU provider cannot execute.
    """
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    log.info(f"applying dynamic INT8 quantization{' (MatMul only)' if matmul_only else ''}")
    quantizer = ORTQuantizer.from_pretrained(model_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(
        is_static=False,
        per_channel=False,
        **({"operators_to_quantize": ["MatMul"]} if matmul_only else {}),
    )
    quantizer.quantize(save_dir=model_dir, quantization_config=qconfig)
