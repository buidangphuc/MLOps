"""The one real endpoint: POST /api/v1/predict. The pack defines the request and response."""

from fastapi import APIRouter
from prometheus_client import Counter, Histogram

from app import pack
from app.predictor import get_predictor

router = APIRouter()

PREDICT_LATENCY = Histogram("predict_seconds", "predict latency (s)")
REQUESTS = Counter("predict_requests_total", "predict requests")
ERRORS = Counter("predict_errors_total", "predict errors")


@router.post("/api/v1/predict")
def predict(req: pack.Request):
    REQUESTS.inc()
    with PREDICT_LATENCY.time():
        try:
            return pack.respond(get_predictor(), req)
        except Exception:
            ERRORS.inc()
            raise  # -> global exception handler returns a clean 500 (worker stays alive)
