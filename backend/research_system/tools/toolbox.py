"""The assembled tool set: every tool wrapped, plus the two things researchers share.

Three researchers run at once, and two of their dependencies must be one object
rather than three. The rate limiter paces calls against an API's quota, so a
limiter per researcher would let three times the configured rate through. The
cache counts its own hits and misses, and three counters cannot be added up
afterwards into the number that matters.

Building it needs the API key, so it happens here and not in `selector.py`, which
stays a pure function of the query text.
"""

from dataclasses import dataclass
from functools import lru_cache

from research_system.guardrails.rate_limit import RateLimiter
from research_system.settings import Settings, get_settings
from research_system.tools.base import Tool
from research_system.tools.cache import SearchCache
from research_system.tools.retry import RetryingTool
from research_system.tools.scraper import ScraperTool
from research_system.tools.search import TavilyTool
from research_system.tools.selector import SCRAPER, TAVILY, WIKIPEDIA
from research_system.tools.wikipedia import WikipediaTool


@dataclass(frozen=True, slots=True)
class Toolbox:
    """Every tool a researcher may run, keyed by the names `selector` returns."""

    tools: dict[str, Tool]
    cache: SearchCache
    limiter: RateLimiter


def build_toolbox(settings: Settings) -> Toolbox:
    """Assemble the tools, each already wrapped in the retry policy."""
    if settings.tavily_api_key is None:
        raise ValueError("TAVILY_API_KEY is unset, so the primary search tool cannot be built")

    search = settings.search
    tools: dict[str, Tool] = {
        TAVILY: TavilyTool(search, settings.tavily_api_key),
        WIKIPEDIA: WikipediaTool(search),
        SCRAPER: ScraperTool(search),
    }
    return Toolbox(
        # Retrying is a wrapper, not a mode: the caller sees a plain `Tool`.
        tools={name: RetryingTool(tool, search.retry) for name, tool in tools.items()},
        cache=SearchCache(search.cache),
        limiter=RateLimiter(settings.guardrails.rate_limit),
    )


@lru_cache(maxsize=1)
def default_toolbox() -> Toolbox:
    """The process-wide toolbox, built on first use.

    Process-wide rather than per-run because that is what the two shared objects
    are actually scoped to: a rate limit belongs to an API key, and the cache is a
    file on disk. A worker running several research jobs at once should pace them
    against one bucket, not one bucket each.
    """
    return build_toolbox(get_settings())
