"""
Structured logging for SHL Assessment Recommender.

Provides consistent JSON-style logging across all modules.
Every request logs: intent, constraints, retrieved docs, scores, latency, reason.

Improves: Observability, Debugging, Interview defensibility.
"""

import logging
import sys
import time
from typing import Any

from app.config import LOG_LEVEL


def setup_logger(name: str) -> logging.Logger:
    """Create a structured logger for a module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    return logger


class RequestLogger:
    """Context manager for logging per-request metrics.

    Usage:
        with RequestLogger(logger, request_id="abc") as rl:
            rl.log("intent", "search")
            rl.log("retrieved_count", 15)
            # ... processing ...
        # Automatically logs total latency on exit.
    """

    def __init__(self, logger: logging.Logger, **context: Any) -> None:
        self._logger = logger
        self._context = context
        self._data: dict[str, Any] = {}
        self._start: float = 0.0

    def __enter__(self) -> "RequestLogger":
        self._start = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed_ms = (time.time() - self._start) * 1000
        self._data["latency_ms"] = round(elapsed_ms, 1)
        self._logger.info(
            "request_complete | %s | %s",
            " ".join(f"{k}={v}" for k, v in self._context.items()),
            " | ".join(f"{k}={v}" for k, v in self._data.items()),
        )

    def log(self, key: str, value: Any) -> None:
        """Log a key-value pair for this request."""
        self._data[key] = value
        self._logger.debug("%s = %s", key, value)
