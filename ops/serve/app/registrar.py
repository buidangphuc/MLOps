"""App factory: FastAPI + request-id middleware + a catch-all exception handler + a lifespan that
loads the model in the background, exposing /health (liveness), /ready (readiness), /metrics.
Trimmed to the minimum.
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import get_settings
from app.predictor import is_ready, load_predictor
from app.request_id import RequestIdMiddleware, get_request_id
from app.router import router


def register_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load in the background so /health answers immediately and a slow S3 pull + CPU load does
        # not trip the liveness probe. /ready flips to 200 once the model is loaded + warmed up.
        logger.info(
            f"starting model-serving env={settings.ENVIRONMENT} "
            f"source={settings.MODEL_SOURCE} version={settings.MODEL_VERSION}"
        )
        threading.Thread(target=load_predictor, args=(settings,), daemon=True).start()
        yield

    app = FastAPI(title="model-serving", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware, header=settings.X_REQUEST_ID_HEADER)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception(f"unhandled error on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500, content={"error": "internal error", "request_id": get_request_id()}
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if is_ready():
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "loading"})

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router)
    return app
