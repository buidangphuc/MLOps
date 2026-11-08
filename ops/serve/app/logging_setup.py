"""Loguru → STDOUT with the request-id injected, and stdlib/uvicorn logs bridged into loguru.

STDOUT-only on purpose: the platform's log agent tails container stdout, so tracing/log shipping
is an infra concern — no tracer or file sinks here.
"""

import logging
import sys

from loguru import logger

from app.request_id import get_request_id

_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "req={extra[request_id]} | {name}:{function}:{line} - {message}"
)


class InterceptHandler(logging.Handler):
    """Route stdlib logging (uvicorn, transformers, ...) into loguru."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _inject_request_id(record):
    record["extra"].setdefault("request_id", get_request_id())


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.configure(extra={"request_id": "-"}, patcher=_inject_request_id)
    logger.add(sys.stdout, level=level, format=_FORMAT, backtrace=False, diagnose=False, enqueue=True)
    logging.getLogger().handlers = [InterceptHandler()]
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "transformers"):
        lg = logging.getLogger(name)
        lg.handlers = [InterceptHandler()]
        lg.propagate = False
