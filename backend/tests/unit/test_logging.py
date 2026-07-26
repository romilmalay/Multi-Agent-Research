"""Logging tests.

`configure_logging` is called inside each test body, never from a fixture:
pytest rebinds `sys.stdout` between the setup and call phases, and a handler
built during setup would write into the wrong capture buffer.
"""

import json
import logging
from typing import Any

import pytest

from research_system.logging import (
    REDACTED,
    bind_run_id,
    clear_context,
    configure_logging,
    get_logger,
    redact_secrets,
)


def _configure(**kwargs: Any) -> None:
    """Configure with caching off, so repeated calls in one session take effect."""
    configure_logging(cache_loggers=False, **kwargs)


def _emit(capsys: pytest.CaptureFixture[str], **kwargs: Any) -> dict[str, Any]:
    get_logger("test").info("event", **kwargs)
    line: dict[str, Any] = json.loads(capsys.readouterr().out)
    return line


def test_output_is_json_with_level_and_timestamp(capsys: pytest.CaptureFixture[str]) -> None:
    _configure()
    line = _emit(capsys)
    assert line["event"] == "event"
    assert line["level"] == "info"
    assert line["timestamp"].endswith("Z")


def test_bound_run_id_appears_on_every_line(capsys: pytest.CaptureFixture[str]) -> None:
    _configure()
    bind_run_id("run-123")
    assert _emit(capsys)["run_id"] == "run-123"
    assert _emit(capsys)["run_id"] == "run-123"


def test_clear_context_drops_run_id(capsys: pytest.CaptureFixture[str]) -> None:
    _configure()
    bind_run_id("run-123")
    clear_context()
    assert "run_id" not in _emit(capsys)


def test_secrets_are_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    _configure()
    line = _emit(capsys, GOOGLE_API_KEY="abc", auth_token="xyz", query="safe")
    assert line["GOOGLE_API_KEY"] == REDACTED
    assert line["auth_token"] == REDACTED
    assert line["query"] == "safe"


def test_redact_secrets_leaves_plain_fields_untouched() -> None:
    assert redact_secrets(None, "info", {"user": "romil"}) == {"user": "romil"}


def test_token_accounting_fields_are_not_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    """`tokens` must survive: the trace table and cost analysis depend on it."""
    _configure()
    line = _emit(capsys, tokens=412, token_count=1024, input_tokens=88, auth_token="leak")
    assert line["tokens"] == 412
    assert line["token_count"] == 1024
    assert line["input_tokens"] == 88
    assert line["auth_token"] == REDACTED


def test_console_mode_is_not_json(capsys: pytest.CaptureFixture[str]) -> None:
    _configure(json_logs=False)
    get_logger("test").info("hello")
    assert "hello" in capsys.readouterr().out


def test_level_filters_lower_records(capsys: pytest.CaptureFixture[str]) -> None:
    _configure(level="WARNING")
    get_logger("test").info("dropped")
    assert capsys.readouterr().out == ""


def test_stdlib_library_logs_become_json_with_run_id(capsys: pytest.CaptureFixture[str]) -> None:
    """A third-party library logging via stdlib is shaped like our own lines."""
    _configure()
    bind_run_id("run-123")
    logging.getLogger("httpx").warning("Retrying request after 429")

    line = json.loads(capsys.readouterr().out)
    assert line["event"] == "Retrying request after 429"
    assert line["level"] == "warning"
    assert line["run_id"] == "run-123"
    assert line["logger"] == "httpx"


def test_stdlib_exception_includes_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    _configure()
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("sqlalchemy").exception("query failed")

    line = json.loads(capsys.readouterr().out)
    assert "ValueError: boom" in line["exception"]


def test_stdlib_secrets_are_redacted_too(capsys: pytest.CaptureFixture[str]) -> None:
    _configure()
    logging.getLogger("httpx").warning("calling api", extra={"api_key": "sk-real-key"})

    out = capsys.readouterr().out
    assert "sk-real-key" not in out
    assert REDACTED in out
