"""BENCH — measure inference latency/throughput to tune GPU efficiency.

local : sweep batch sizes on the local ONNX model (runs anywhere, incl Mac).
triton: emit the Triton `perf_analyzer` command + sample input (run on the GPU box;
        read GPU utilisation from nvidia-smi or Triton's :8002/metrics).
"""

import json
import time

import numpy as np

from datascience import log
from datascience.constants import paths

_SAMPLE = "Central bank raises interest rates amid persistent inflation concerns"


def bench_local(batch_sizes=(1, 8, 16, 32), iters_per_batch=50):
    from datascience.serve.local import _load_model, _runtime

    model = _load_model()
    log.info(f"local bench (runtime={_runtime()})")
    for bs in batch_sizes:
        texts = [_SAMPLE] * bs
        model.predict(texts)  # warmup
        lat = []
        for _ in range(iters_per_batch):
            t = time.perf_counter()
            model.predict(texts)
            lat.append((time.perf_counter() - t) * 1000)
        lat = np.array(lat)
        throughput = bs * 1000.0 / lat.mean()
        log.info(
            f"batch={bs:>3}  p50={np.percentile(lat,50):6.1f}ms  "
            f"p95={np.percentile(lat,95):6.1f}ms  throughput={throughput:7.0f} samples/s"
        )


def bench_triton(url="localhost:8000", model_name="text_classifier", concurrency="1:16"):
    # perf_analyzer needs an input file for the STRING input.
    input_file = paths.ARTIFACTS / "perf_input.json"
    input_file.write_text(json.dumps({"data": [{"TEXT": [_SAMPLE]}]}))
    cmd = (
        f"perf_analyzer -m {model_name} -u {url} "
        f"--concurrency-range {concurrency} --input-data {input_file} --percentile=95"
    )
    log.info("run on the GPU box (Triton must be serving):")
    log.info(f"  {cmd}")
    log.info("GPU utilisation: `nvidia-smi dmon` or Triton metrics at http://%s/metrics" % url.replace(":8000", ":8002"))
    return cmd
