import time

import pytest

from research_system.domain.state import SearchResult
from research_system.errors import PermanentToolError, TransientToolError
from research_system.settings import RetrySettings, get_settings
from research_system.tools.base import Tool
from research_system.tools.retry import RetryingTool

RESULTS: list[SearchResult] = [{"title": "t", "snippet": "s", "url": "u", "date": ""}]


class FlakyTool:
    """Fails `failures` times, then succeeds. Counts every call."""

    name = "tavily"

    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self._remaining = failures
        self._error = error or TransientToolError("upstream is down", tool="tavily")
        self.calls = 0

    async def search(self, query: str) -> list[SearchResult]:
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise self._error
        return RESULTS


@pytest.fixture
def settings() -> RetrySettings:
    """Real policy, microsecond waits, so the tests measure behaviour not sleep."""
    return RetrySettings(
        max_attempts=3,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.004,
        budget_seconds=30.0,
    )


async def test_a_working_tool_is_called_once(settings: RetrySettings) -> None:
    tool = FlakyTool(failures=0)
    assert await RetryingTool(tool, settings).search("q") == RESULTS
    assert tool.calls == 1


async def test_a_transient_failure_is_retried_and_recovers(settings: RetrySettings) -> None:
    """A 503 on the first attempt must not cost the run its sources."""
    tool = FlakyTool(failures=2)
    assert await RetryingTool(tool, settings).search("q") == RESULTS
    assert tool.calls == 3


async def test_a_permanent_failure_is_never_retried(settings: RetrySettings) -> None:
    """A 404 will still be a 404 in one second; retrying only wastes time."""
    tool = FlakyTool(failures=1, error=PermanentToolError("gone", tool="tavily"))
    with pytest.raises(PermanentToolError):
        await RetryingTool(tool, settings).search("q")
    assert tool.calls == 1


async def test_attempts_are_capped(settings: RetrySettings) -> None:
    tool = FlakyTool(failures=99)
    with pytest.raises(TransientToolError):
        await RetryingTool(tool, settings).search("q")
    assert tool.calls == settings.max_attempts


async def test_the_tools_own_error_survives(settings: RetrySettings) -> None:
    """Not tenacity's RetryError: the researcher logs which tool failed."""
    with pytest.raises(TransientToolError) as caught:
        await RetryingTool(FlakyTool(failures=99), settings).search("q")
    assert caught.value.tool == "tavily"
    assert "upstream is down" in str(caught.value)


async def test_our_own_bugs_are_not_retried(settings: RetrySettings) -> None:
    """Running a KeyError three times just hides the real cause."""
    tool = FlakyTool(failures=1, error=KeyError("results"))
    with pytest.raises(KeyError):
        await RetryingTool(tool, settings).search("q")
    assert tool.calls == 1


async def test_the_budget_stops_a_dead_api(settings: RetrySettings) -> None:
    """One unreachable service must not hold the whole run open."""
    spent = RetrySettings(
        max_attempts=100,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.02,
        budget_seconds=0.05,
    )
    tool = FlakyTool(failures=99)
    started = time.monotonic()
    with pytest.raises(TransientToolError):
        await RetryingTool(tool, spent).search("q")
    assert time.monotonic() - started < 1.0
    assert tool.calls < spent.max_attempts


async def test_it_waits_between_attempts(settings: RetrySettings) -> None:
    """Retrying instantly would hammer a service that asked us to slow down."""
    slow = RetrySettings(
        max_attempts=3,
        initial_backoff_seconds=0.05,
        max_backoff_seconds=0.2,
        budget_seconds=30.0,
    )
    started = time.monotonic()
    with pytest.raises(TransientToolError):
        await RetryingTool(FlakyTool(failures=99), slow).search("q")
    assert time.monotonic() - started >= 0.05


async def test_a_wrapped_tool_is_still_a_tool(settings: RetrySettings) -> None:
    """The researcher calls .search() and never knows retrying is in the way."""
    wrapped: Tool = RetryingTool(FlakyTool(failures=0), settings)
    assert wrapped.name == "tavily"
    assert await wrapped.search("q") == RESULTS


def test_the_configured_policy_is_sane() -> None:
    retry = get_settings().search.retry
    assert retry.max_attempts >= 2
    assert retry.initial_backoff_seconds <= retry.max_backoff_seconds
    # A run must not be able to spend longer retrying one tool than researching.
    assert retry.budget_seconds <= 60


def test_backoff_bounds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="max_backoff_seconds"):
        RetrySettings(
            max_attempts=3,
            initial_backoff_seconds=10.0,
            max_backoff_seconds=1.0,
            budget_seconds=30.0,
        )
