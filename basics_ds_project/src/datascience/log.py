from __future__ import annotations

import logging
import os

from functools import lru_cache

import loguru

from loguru import logger

from core.conf import settings


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class Logger:
    def __init__(self):
        self.log_path = settings.LOG_DIR

    @lru_cache
    def get_logger(self) -> loguru.Logger:
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        log_stdout_file = os.path.join(self.log_path, settings.LOG_STDOUT_FILENAME)
        log_stderr_file = os.path.join(self.log_path, settings.LOG_STDERR_FILENAME)

        log_config = {
            "rotation": "10MB",
            "retention": "15 days",
            "compression": "tar.gz",
            "enqueue": True,
        }

        logger.add(
            log_stdout_file,
            **log_config,
            level="INFO",
            filter=lambda record: record["level"].name == "INFO" or record["level"].no <= 25,
            backtrace=False,
            diagnose=False,
        )

        logger.add(
            log_stdout_file,
            **log_config,
            level="DEBUG",
            filter=lambda record: record["level"].name == "DEBUG" or record["level"].no <= 20,
            backtrace=True,
            diagnose=True,
        )

        logger.add(
            log_stderr_file,
            **log_config,
            level="WARNING",
            filter=lambda record: record["level"].name == "WARNING" or record["level"].no <= 30,
            backtrace=True,
            diagnose=True,
        )

        logger.add(
            log_stderr_file,
            **log_config,
            level="ERROR",
            filter=lambda record: record["level"].name == "ERROR" or record["level"].no >= 30,
            backtrace=True,
            diagnose=True,
        )
        logging.getLogger().handlers = [InterceptHandler()]
        for name in logging.root.manager.loggerDict.keys():
            logging.getLogger(name).handlers = []
            logging.getLogger(name).propagate = True
        return logger


log = Logger().get_logger()
