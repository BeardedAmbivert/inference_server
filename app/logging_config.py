"""Structured JSON logging with per-request correlation IDs.

`configure_logging` installs a single stdout handler that emits one JSON object per log
record. The HTTP middleware (app/main.py) stores a request ID in `request_id_var`, so every
log emitted while handling a request carries the same `request_id`.
"""

import json
import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Standard LogRecord attributes; anything else on a record is caller-supplied `extra`.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JSONFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id
        # Include any extra={...} fields the caller attached to the record.
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter as the sole root-logger handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
