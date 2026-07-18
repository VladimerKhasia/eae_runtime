"""
Structured logging: every runtime event should be observable. This wires
the EventBus to Python's logging module using a JSON-ish structured
formatter, so events can be piped to any log aggregator.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from .events import Event, EventBus

_LOGGER_NAME = "eae_runtime"


def get_logger(level: str = "WARNING") -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))
    return logger


def attach_structured_logging(event_bus: EventBus, level: str = "WARNING", logger: Optional[logging.Logger] = None) -> None:
    """Subscribe a structured JSON logger to every event on `event_bus`.

    `level` sets the logger's threshold (e.g. "WARNING" hides events by
    default). Individual event records are always emitted at DEBUG
    severity, so setting `level="DEBUG"` is what makes them visible.
    """
    logger = logger or get_logger(level)

    def _log(event: Event) -> None:
        record = {"event": event.type, "timestamp": event.timestamp, **event.payload}
        logger.debug(json.dumps(record, default=str))

    event_bus.subscribe("*", _log)
