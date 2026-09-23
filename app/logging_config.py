"""Structured logging.

Two formats:

* ``json``  - one JSON object per line, including extra fields such as
  ``camera_id``, ``det_ms``, ``emb_ms``, ``batch_size`` ... this is what you
  want in production to grep GPU throughput.
* ``text``  - human friendly ``time LEVEL logger message key=value`` lines.

Usage::

    logger = logging.getLogger("app.pipeline")
    logger.info("batch processed",
                extra={"camera_id": 3, "batch_size": 8, "total_ms": 12.3})
"""

from __future__ import annotations

import json
import logging
import logging.config
import sys
import time
from typing import Any

from .config import Settings

# Attributes every LogRecord already carries - everything else is "extra".
_STANDARD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "msg", "message", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            payload[key] = _jsonable(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{time.strftime('%H:%M:%S', time.gmtime(record.created))}."
            f"{int(record.msecs):03d} {record.levelname:<7} "
            f"{record.name}: {record.getMessage()}"
        )
        extras = []
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            extras.append(f"{key}={value}")
        if extras:
            base += "  " + " ".join(extras)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def configure(settings: Settings) -> None:
    """Install logging configuration for the whole process (idempotent)."""
    formatter: logging.Formatter
    if settings.log_format.lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = TextFormatter("%(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # Third-party noise control.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class Timer:
    """Tiny context manager measuring elapsed milliseconds.

    >>> with Timer() as t: ...
    >>> logger.info("done", extra={"total_ms": t.ms})
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
