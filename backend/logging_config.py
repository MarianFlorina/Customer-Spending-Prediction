import os
import sys
import logging
import logging.config
from datetime import datetime

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
LOG_FILE = os.getenv("LOG_FILE", "")


def setup_logging():
    """Configure structured logging for the application."""
    formatters = {
        "standard": {
            "format": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }

    # Optional: JSON formatter if pythonjsonlogger is installed
    try:
        import pythonjsonlogger.jsonlogger  # noqa: F401
        formatters["json"] = {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    except ImportError:
        pass

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "stream": sys.stdout,
        },
    }

    console_handlers = ["console"]

    if LOG_FILE:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "filename": LOG_FILE,
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        }
        console_handlers.append("file")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            "": {"level": LOG_LEVEL, "handlers": console_handlers},
            "uvicorn": {"level": "WARNING"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"level": "WARNING"},
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
        },
    }

    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {LOG_LEVEL} level")
    if LOG_FILE:
        logger.info(f"Log file: {LOG_FILE}")
