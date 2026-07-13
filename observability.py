"""Structured operational logging with safe contextual fields."""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Iterator


_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "theoria_log_context", default={}
)
_SENSITIVE_KEY = re.compile(
    r"(authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|"
    r"password|secret|credential|cookie)",
    re.IGNORECASE,
)
_SENSITIVE_TEXT = (
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)"
        r"(\s*[:=]\s*)['\"]?([^\s,'\"}]{8,})"
    ),
)


def redact_text(value: str) -> str:
    """Redact common credential forms from console-safe text previews."""
    value = _SENSITIVE_TEXT[0].sub(r"\1[REDACTED]", value)
    return _SENSITIVE_TEXT[1].sub(r"\1\2[REDACTED]", value)


def _redact(value: Any, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line for ingestion by standard log systems."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            **_context.get(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(_redact(payload), ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = {**_context.get(), **getattr(record, "fields", {})}
        suffix = " ".join(f"{k}={v}" for k, v in sorted(_redact(context).items()))
        base = f"{record.levelname.lower()} {record.name}: {record.getMessage()}"
        if suffix:
            base += f" [{suffix}]"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(
    *,
    level: str | None = None,
    log_format: str | None = None,
) -> None:
    """Configure the package root logger once, replacing stale handlers."""
    resolved_level = (level or os.getenv("THEORIA_LOG_LEVEL") or "INFO").upper()
    resolved_format = (
        log_format or os.getenv("THEORIA_LOG_FORMAT") or "text"
    ).lower()
    if resolved_format not in {"text", "json"}:
        raise ValueError("log format must be 'text' or 'json'")

    logger = logging.getLogger("theoria")
    logger.setLevel(resolved_level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if resolved_format == "json" else TextFormatter())
    logger.handlers[:] = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"theoria.{name}")


@contextlib.contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Bind correlation fields across nested async tasks."""
    merged = {**_context.get(), **{k: v for k, v in fields.items() if v is not None}}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


def event(
    logger: logging.Logger,
    level: int,
    event_name: str,
    message: str,
    **fields: Any,
) -> None:
    logger.log(level, message, extra={"event": event_name, "fields": fields})
