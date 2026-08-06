"""CPU/local GLiNER endpoint: FastAPI wrapping the Predictor — NO Triton.

This is the cheap TEST path (serving.device: cpu): it exposes the SAME request/response
contract as the Triton MME ({text|texts} -> {text, entities}), so a client invokes either
identically. On the ONNX path it serves the ONNX+int8 model produced by `optimize`
(quality guarded by optimize.py's parity gate); otherwise the trained torch model.
"""

import os

import yaml

from src.datascience import log
from src.datascience.constants import paths


def _serving():
    return yaml.safe_load(paths.SERVING_FILE.read_text()) or {}


def _resolve_weights(sv):
    """Where the trained weights come from.

    model_source: registry  -> pull the version from the S3 model registry into a local cache
                               (the serving image ships NO weights — this is the pull step).
                               version = env MODEL_VERSION > serving.yaml model_version > latest.
    model_source: local (default) -> use the on-disk artifacts/model_trainer/model.

    env MODEL_SOURCE overrides serving.yaml model_source (so a Docker run can flip to the
    registry pull with -e MODEL_SOURCE=registry, no rebuild).

    Returns a model dir, or None to let Predictor fall back to its default MODEL_DIR.
    """
    source = os.environ.get("MODEL_SOURCE") or sv.get("model_source")
    if source != "registry":
        return None
    reg = sv.get("registry") or {}
    version = os.environ.get("MODEL_VERSION") or sv.get("model_version") or "latest"
    # Infra coords come from the environment first (pick a `.env.<env>` file: test/staging),
    # falling back to serving.yaml. Keeps real bucket names out of the committed config.
    bucket = os.environ.get("S3_BUCKET") or reg.get("s3_bucket")
    prefix = os.environ.get("S3_PREFIX") or reg.get("s3_prefix")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or reg.get("region")
    from src.datascience.registry.s3 import S3Registry

    r = S3Registry(bucket=bucket, prefix=prefix, region=region, name=reg.get("name") or "GlinerNER")
    log.info(f"pulling weights from S3 registry: {r.name} version={version}")
    return r.pull(version, paths.REGISTRY_CACHE)


def _load_predictor():
    """GLiNER Predictor on the configured device. runtime=onnx loads the optimized ONNX dir."""
    from src.datascience.infer.predict import Predictor

    sv = _serving()
    device = "cpu" if sv.get("device") == "cpu" else None   # None => Predictor auto-picks cuda>cpu
    weights = _resolve_weights(sv)                           # None => Predictor default MODEL_DIR
    if sv.get("runtime") == "onnx":
        onnx_dir = str(weights) if weights else str(paths.OPTIMIZED_DIR)
        if not os.path.exists(os.path.join(onnx_dir, "gliner_config.json")):
            raise SystemExit("runtime=onnx but no ONNX model — run `optimize` first (or set runtime: torch)")
        log.info(f"serving ONNX+int8 GLiNER from {onnx_dir} (device={device or 'auto'})")
        return Predictor(model_dir=onnx_dir, device=device, onnx=True)
    log.info(f"serving torch GLiNER from {weights or paths.MODEL_DIR} (device={device or 'auto'})")
    return Predictor(model_dir=weights, device=device)


def create_app():
    from fastapi import FastAPI
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
    from pydantic import BaseModel

    predictor = _load_predictor()
    latency = Histogram("predict_seconds", "predict latency")
    app = FastAPI(title="gliner-ner")

    class Req(BaseModel):
        text: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Path + shape match bds-genai-sse's GlinerFastAPIClient exactly, so the SSE service can
    # point its GLINER_ENDPOINT here and swap test<->prod with NO client code change:
    #   request  {"text": "..."}
    #   response {"output": {"intent": "...", "propertyType": "...", "location": "...", ...}}
    # (keys = label with first letter lowercased, one merged string per label — the same
    #  parsed["output"] the SageMaker/Triton clients normalise to).
    @app.post("/api/v1/predict")
    def predict(req: Req):
        with latency.time():
            parsed = predictor.predict_parser([req.text])[0]
        return {"output": {k[:1].lower() + k[1:]: v for k, v in parsed.items()}}

    return app


def serve(port=None):
    import uvicorn

    port = port or _serving().get("cpu_test", {}).get("port", 8080)
    # Build the app FIRST — create_app() pulls the model from the registry and loads it, the
    # slow part. Only after it returns do we bind the port, so the moment /health answers 200
    # the model is genuinely ready (no need to poke /predict to find out).
    log.info("loading model (registry pull + init) before accepting traffic…")
    app = create_app()
    log.info(f"READY — model loaded, serving GLiNER on :{port} (FastAPI, no Triton)")
    uvicorn.run(app, host="0.0.0.0", port=port)
