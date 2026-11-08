"""Per-request correlation id: a ContextVar + middleware that reads/echoes X-Request-ID.

One module: the ContextVar and the middleware.
"""

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return _request_id.get()


def set_request_id(value: str | None) -> str:
    rid = value or uuid.uuid4().hex
    _request_id.set(rid)
    return rid


def clear_request_id() -> None:
    _request_id.set("-")


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header: str = "X-Request-ID"):
        super().__init__(app)
        self.header = header

    async def dispatch(self, request, call_next):
        rid = set_request_id(request.headers.get(self.header))
        try:
            response = await call_next(request)
            response.headers[self.header] = rid
            return response
        finally:
            clear_request_id()
