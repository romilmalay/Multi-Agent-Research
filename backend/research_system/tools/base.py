"""What every tool looks like, and how a failed HTTP call becomes a typed error.

Three tools sit on three unrelated upstreams — Tavily, the Wikipedia REST API, a
raw page scraper — and all three hand back the same four-key dict. Everything
downstream, from the quality gate to the citations in the report, therefore never
learns where a source came from.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

import httpx

from research_system.domain.state import SearchResult
from research_system.errors import PermanentToolError, ToolError, TransientToolError

# 408 and 429 clear on their own, and so does any 5xx. A 404 or a rejected
# request will not, however long we wait.
_TRANSIENT_STATUS = frozenset({408, 429})

# Wikimedia's policy is to reject anonymous agents with a 403, and most sites
# worth scraping do the same.
USER_AGENT = "research-system/0.1 (https://github.com/SevenTrippingChakras)"


class Tool(Protocol):
    """A named, async source of search results."""

    name: str

    async def search(self, query: str) -> list[SearchResult]: ...


def make_result(
    *,
    title: str,
    url: str,
    snippet: str,
    max_chars: int,
    date: str = "",
) -> SearchResult:
    """Build the uniform result, snippet cut to the configured length.

    Truncating here rather than in each tool is what keeps one upstream's
    thousand-word extract from crowding a Tavily snippet out of the analyst's
    context. `date` is empty when the upstream does not report one.
    """
    return SearchResult(
        title=title.strip(),
        snippet=snippet.strip()[:max_chars],
        url=url,
        date=date,
    )


def error_for_status(status_code: int, *, tool: str) -> ToolError:
    """The error a HTTP status calls for, transient or permanent."""
    message = f"{tool} returned HTTP {status_code}"
    if status_code in _TRANSIENT_STATUS or status_code >= 500:
        return TransientToolError(message, tool=tool)
    return PermanentToolError(message, tool=tool)


@contextmanager
def http_errors(tool: str) -> Iterator[None]:
    """Translate httpx failures into the errors the retry policy selects on."""
    try:
        yield
    except httpx.HTTPStatusError as exc:
        raise error_for_status(exc.response.status_code, tool=tool) from exc
    except httpx.TransportError as exc:
        # Timeouts and connection failures both land here, both worth a retry.
        raise TransientToolError(f"{tool} request failed: {exc}", tool=tool) from exc
