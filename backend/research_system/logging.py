"""Structured JSON logging.

One pipeline for two sources: our own structlog calls and the stdlib `logging`
records emitted by third-party libraries. Both come out as JSON carrying the
`run_id` bound for the current context, with key-shaped values redacted.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

REDACTED = "[redacted]"
# Deliberately not a bare "token": this system logs `tokens` and `token_count`
# on every trace entry, and those must survive.
_SECRET_HINTS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "bearer",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
)


def redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Replace values of key-shaped fields with a placeholder."""
    for key in event_dict:
        if any(hint in key.lower() for hint in _SECRET_HINTS):
            event_dict[key] = REDACTED
    return event_dict


def configure_logging(
    level: str,
    *,
    json_logs: bool,
    cache_loggers: bool = True,
) -> None:
    """Configure structlog and route stdlib logging through the same pipeline.

    `level` and `json_logs` have no defaults here: they come from
    `settings.logging`, which is where those values are defined.

    Safe to call twice; the second call replaces the first. Tests pass
    `cache_loggers=False` so reconfiguring actually takes effect.
    """
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact_secrets,
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=cache_loggers,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Runs only on records from stdlib loggers, not on our own. ExtraAdder
        # lifts `logger.info(..., extra={...})` fields into the event dict, so
        # they are rendered and, crucially, reach the redactor.
        foreign_pre_chain=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.ExtraAdder(),
            *shared,
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to `name`."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_run_id(run_id: str) -> None:
    """Bind `run_id` to every subsequent log line in this context."""
    structlog.contextvars.bind_contextvars(run_id=run_id)


def clear_context() -> None:
    """Drop all context-bound fields."""
    structlog.contextvars.clear_contextvars()
