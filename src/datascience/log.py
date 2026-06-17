from __future__ import annotations

import logging
import os
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """Route stdlib logging (mlflow, urllib3, ...) through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _configure():
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()  # drop the default handler so we don't double-log
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
    logger.add(
        os.path.join(log_dir, "running.log"),
        level="DEBUG",
        rotation="10 MB",
        retention="15 days",
        compression="tar.gz",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    return logger


log = _configure()
