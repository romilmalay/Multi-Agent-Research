"""Retrying, for the failures worth retrying.

Wraps any tool in the retry policy without the tool knowing: `RetryingTool` is a
`Tool` itself, so the researcher calls `.search()` and never branches on whether
a call is a first attempt or a fourth.

The whole decision is one line — `retry_if_exception_type(TransientToolError)`.
Nothing here reads a status code, because `tools.base` already turned every
status into the type that answers the question.
"""

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

from research_system.domain.state import SearchResult
from research_system.errors import TransientToolError
from research_system.settings import RetrySettings
from research_system.tools.base import Tool


class RetryingTool:
    """One tool, retried when it fails in a way that might clear."""

    def __init__(self, tool: Tool, settings: RetrySettings) -> None:
        self._tool = tool
        self._settings = settings
        self.name = tool.name

    async def search(self, query: str) -> list[SearchResult]:
        retrying: AsyncRetrying = AsyncRetrying(
            retry=retry_if_exception_type(TransientToolError),
            # Two limits: attempts, and total time. The second is the budget that
            # stops one unreachable API from holding up the whole run.
            stop=(
                stop_after_attempt(self._settings.max_attempts)
                | stop_after_delay(self._settings.budget_seconds)
            ),
            # Jitter is sized to the first wait, so three researchers that failed
            # together do not come back in step and repeat the overload.
            wait=wait_exponential_jitter(
                initial=self._settings.initial_backoff_seconds,
                max=self._settings.max_backoff_seconds,
                jitter=self._settings.initial_backoff_seconds,
            ),
            # Raise the tool's own error, not tenacity's wrapper: the researcher
            # records which tool failed and why.
            reraise=True,
        )
        results: list[SearchResult] = await retrying(self._tool.search, query)
        return results
