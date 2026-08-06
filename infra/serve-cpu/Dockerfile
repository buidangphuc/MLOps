# CPU serving image for the GLiNER FastAPI endpoint (test/staging self-host on EC2 or any box).
# No Triton, no SageMaker: just `serve/local.py` wrapping Predictor on CPU. Same request/response
# contract as the prod Triton MME (POST /api/v1/predict {"text"} -> {"output": {...}}), so the
# bds-genai-sse client swaps test<->prod with NO code change.
#
# NO WEIGHTS ARE BAKED IN. The image is code+deps only, so it's model-agnostic — one image
# serves any model version. Weights come in at RUNTIME one of two ways:
#
# Build from the REPO ROOT:
#   docker build -f infra/serve-cpu/Dockerfile -t gliner-serve-cpu .
#
# (A) Pull from the S3 model registry — the "build serving anywhere" path. Set serving.yaml
#     model_source: registry (+ registry.s3_bucket), then pick the version by env:
#   docker run --rm -p 8080:8080 \
#     -e MODEL_VERSION=v3 \
#     -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_REGION=ap-southeast-1 \
#     gliner-serve-cpu
#     -> startup pulls s3://.../registry/GlinerNER/v3/ into the cache, then serves it.
#
# (B) Mount local artifacts — dev on the training box (serving.yaml model_source: local):
#   docker run --rm -p 8080:8080 -v /path/to/MLOps/artifacts:/app/artifacts:ro gliner-serve-cpu
#
#   curl -s localhost:8080/api/v1/predict -H 'content-type: application/json' \
#     -d '{"text":"cho thuê căn hộ 2 phòng ngủ quận 7 giá dưới 15 triệu"}'
#   -> {"output": {"intent":"thuê", "propertyType":"căn hộ", "location":"quận 7", ...}}

FROM python:3.10-slim

WORKDIR /app

# CPU-only torch first, from the CPU wheel index (avoids the huge CUDA build).
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY infra/serve-cpu/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + config only. The model is mounted at runtime (keeps the image small + reusable).
COPY main.py ./main.py
COPY src ./src
COPY config ./config

# Empty CUDA_VISIBLE_DEVICES forces CPU regardless of serving.yaml device — Predictor's
# pick_device() falls to CPU when no GPU is visible.
ENV CUDA_VISIBLE_DEVICES="" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8080

# Healthy ONLY when the model is ready: uvicorn doesn't bind the port until create_app()
# finishes pulling+loading the model, so /health returning 200 == usable. `docker ps` shows
# "(health: starting)" during the S3 pull, then "(healthy)" — no need to poke /predict to know.
# start-period covers a cold ~1.1GB registry pull (uses python since slim has no curl).
HEALTHCHECK --start-period=600s --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health',timeout=2).status==200 else 1)"

# serve/local.py: FastAPI on :8080 (port from serving.yaml cpu_test.port). torch runtime = default.
CMD ["python", "main.py", "serve", "--backend", "local", "--port", "8080"]
