"""Researcher tests. No network: the toolbox is built from stub tools.

The cache and the rate limiter are the real ones — they are the two pieces of
behaviour under test, and both are local and fast.
"""

from pathlib import Path

import pytest

from research_system.agents.researcher import research
from research_system.domain.state import ResearchState, SearchResult
from research_system.errors import PermanentToolError
from research_system.guardrails.rate_limit import RateLimiter
from research_system.settings import CacheSettings, RateLimitSettings, Settings
from research_system.tools.cache import SearchCache
from research_system.tools.toolbox import Toolbox

PLAIN = "microplastic concentrations in the north pacific"
DEFINITIONAL = "what is a microplastic"
TRACE_KEYS = {
    "agent",
    "duration_ms",
    "tokens",
    "input_tokens",
    "output_tokens",
    "summary",
    "prompt_hash",
}


def source(url: str, title: str = "a source") -> SearchResult:
    return SearchResult(title=title, snippet="some text", url=url, date="")


class FakeTool:
    """Answers with canned results, or raises, and records every call."""

    def __init__(self, name: str, results: list[SearchResult] | Exception) -> None:
        self.name = name
        self.results = results
        self.queries: list[str] = []

    async def search(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        if isinstance(self.results, Exception):
            raise self.results
        return self.results


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def toolbox(tmp_path: Path) -> Toolbox:
    """A toolbox whose tools are stubs; the cache and limiter are real."""
    return Toolbox(
        tools={
            "tavily": FakeTool("tavily", [source("https://tavily.example/1")]),
            "wikipedia": FakeTool("wikipedia", [source("https://en.wikipedia.org/wiki/A")]),
            "scraper": FakeTool("scraper", [source("https://scraped.example/page")]),
        },
        cache=SearchCache(
            CacheSettings(enabled=True, path=tmp_path / "cache.sqlite", ttl_seconds=3600)
        ),
        # Generous enough that no test ever waits on it.
        limiter=RateLimiter(RateLimitSettings(requests_per_minute=6000, burst=100)),
    )


def tool(toolbox: Toolbox, name: str) -> FakeTool:
    found = toolbox.tools[name]
    assert isinstance(found, FakeTool)
    return found


def answers(toolbox: Toolbox, name: str, results: list[SearchResult] | Exception) -> None:
    tool(toolbox, name).results = results


def state(query: str, token_count: int = 0) -> ResearchState:
    """What a `Send` hands one researcher: its own sub-topic, and the spend so far."""
    return ResearchState(query=query, token_count=token_count)


async def test_the_sub_topic_is_searched_and_the_sources_returned(
    toolbox: Toolbox, settings: Settings
) -> None:
    update = await research(state(PLAIN), toolbox=toolbox, settings=settings)

    assert tool(toolbox, "tavily").queries == [PLAIN]
    assert update["sources"] == [source("https://tavily.example/1")]
    assert update["search_queries_used"] == [PLAIN]
    assert update["errors"] == []


async def test_only_the_selected_tools_run(toolbox: Toolbox, settings: Settings) -> None:
    """A plain question needs no encyclopaedia and names no URL."""
    await research(state(PLAIN), toolbox=toolbox, settings=settings)

    assert tool(toolbox, "wikipedia").queries == []
    assert tool(toolbox, "scraper").queries == []


async def test_a_definitional_question_also_asks_wikipedia(
    toolbox: Toolbox, settings: Settings
) -> None:
    update = await research(state(DEFINITIONAL), toolbox=toolbox, settings=settings)

    assert tool(toolbox, "wikipedia").queries == [DEFINITIONAL]
    assert len(update["sources"]) == 2


async def test_a_tool_disabled_in_config_is_never_selected(
    toolbox: Toolbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agents.researcher.tools` is what this deployment is allowed to run."""
    monkeypatch.setenv("AGENTS__RESEARCHER__TOOLS", '["tavily"]')
    update = await research(state(DEFINITIONAL), toolbox=toolbox, settings=Settings())

    assert tool(toolbox, "wikipedia").queries == []
    assert len(update["sources"]) == 1


async def test_the_same_url_from_two_tools_is_kept_once(
    toolbox: Toolbox, settings: Settings
) -> None:
    """Wikipedia is a top web result as often as not."""
    shared = "https://en.wikipedia.org/wiki/Microplastics"
    answers(toolbox, "tavily", [source(shared, title="from tavily"), source("https://other/1")])
    answers(toolbox, "wikipedia", [source(shared, title="from wikipedia")])

    update = await research(state(DEFINITIONAL), toolbox=toolbox, settings=settings)

    urls = [result["url"] for result in update["sources"]]
    assert urls == [shared, "https://other/1"]
    # First occurrence wins, and the tools run primary first.
    assert update["sources"][0]["title"] == "from tavily"


async def test_sources_are_capped_per_sub_topic(toolbox: Toolbox, settings: Settings) -> None:
    answers(toolbox, "tavily", [source(f"https://example.com/{i}") for i in range(20)])

    update = await research(state(PLAIN), toolbox=toolbox, settings=settings)

    assert len(update["sources"]) == settings.agents.researcher.max_sources_per_topic


async def test_an_exhausted_budget_makes_zero_tool_calls(
    toolbox: Toolbox, settings: Settings
) -> None:
    """The sources it would add are paid for by the analyst, not by this node."""
    update = await research(
        state(PLAIN, token_count=settings.budget.max_tokens_per_run),
        toolbox=toolbox,
        settings=settings,
    )

    assert tool(toolbox, "tavily").queries == []
    assert toolbox.limiter.acquired == 0
    assert update["sources"] == []
    assert update["search_queries_used"] == []
    assert "budget exhausted" in update["errors"][0]


async def test_a_degraded_budget_drops_the_supplementary_tools(
    toolbox: Toolbox, settings: Settings
) -> None:
    update = await research(
        state(DEFINITIONAL, token_count=settings.budget.degrade_at_tokens),
        toolbox=toolbox,
        settings=settings,
    )

    assert tool(toolbox, "tavily").queries == [DEFINITIONAL]
    assert tool(toolbox, "wikipedia").queries == []
    assert len(update["sources"]) == 1


async def test_one_failing_tool_does_not_lose_the_others_results(
    toolbox: Toolbox, settings: Settings
) -> None:
    answers(toolbox, "tavily", PermanentToolError("tavily returned HTTP 401", tool="tavily"))

    update = await research(state(DEFINITIONAL), toolbox=toolbox, settings=settings)

    assert update["sources"] == [source("https://en.wikipedia.org/wiki/A")]
    assert len(update["errors"]) == 1
    assert "tavily failed" in update["errors"][0]
    assert "HTTP 401" in update["errors"][0]


async def test_every_tool_failing_returns_no_sources_and_no_exception(
    toolbox: Toolbox, settings: Settings
) -> None:
    for name in ("tavily", "wikipedia"):
        answers(toolbox, name, PermanentToolError("down", tool=name))

    update = await research(state(DEFINITIONAL), toolbox=toolbox, settings=settings)

    assert update["sources"] == []
    assert len(update["errors"]) == 2


async def test_a_repeated_search_is_served_from_the_cache(
    toolbox: Toolbox, settings: Settings
) -> None:
    """The quality-gate retry re-runs sub-topics the first pass already searched."""
    first = await research(state(PLAIN), toolbox=toolbox, settings=settings)
    second = await research(state(PLAIN), toolbox=toolbox, settings=settings)

    assert tool(toolbox, "tavily").queries == [PLAIN]
    assert second["sources"] == first["sources"]
    assert toolbox.cache.stats().hits == 1


async def test_a_cached_answer_costs_no_rate_limit_permit(
    toolbox: Toolbox, settings: Settings
) -> None:
    """The permit paces calls to an upstream a cache hit never touches."""
    await research(state(PLAIN), toolbox=toolbox, settings=settings)
    assert toolbox.limiter.acquired == 1

    await research(state(PLAIN), toolbox=toolbox, settings=settings)
    assert toolbox.limiter.acquired == 1


async def test_every_network_call_takes_a_permit(toolbox: Toolbox, settings: Settings) -> None:
    await research(state(DEFINITIONAL), toolbox=toolbox, settings=settings)

    assert toolbox.limiter.acquired == 2


async def test_the_update_is_list_shaped_so_parallel_copies_merge(
    toolbox: Toolbox, settings: Settings
) -> None:
    """Everything this node returns has an `operator.add` reducer behind it."""
    update = await research(state(PLAIN), toolbox=toolbox, settings=settings)

    assert set(update) == {"sources", "search_queries_used", "errors", "pipeline_trace"}
    assert all(isinstance(value, list) for value in update.values())


async def test_the_trace_entry_has_the_agreed_shape(toolbox: Toolbox, settings: Settings) -> None:
    update = await research(state(DEFINITIONAL), toolbox=toolbox, settings=settings)

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "researcher"
    assert entry["tokens"] == 0
    assert entry["prompt_hash"] == ""
    assert entry["summary"] == "2 sources via tavily, wikipedia"
    assert entry["duration_ms"] >= 0.0


async def test_a_skipped_researcher_is_still_traced(toolbox: Toolbox, settings: Settings) -> None:
    update = await research(
        state(PLAIN, token_count=settings.budget.max_tokens_per_run),
        toolbox=toolbox,
        settings=settings,
    )

    (entry,) = update["pipeline_trace"]
    assert entry["summary"] == "skipped: budget exhausted"


def test_the_toolbox_holds_every_tool_the_selector_can_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research_system.tools.selector import select_tools
    from research_system.tools.toolbox import build_toolbox

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    built = build_toolbox(Settings())

    assert set(built.tools) == {"tavily", "wikipedia", "scraper"}
    assert set(select_tools("what is x, see https://a.example/b")) <= set(built.tools)


def test_every_tool_is_wrapped_in_the_retry_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_system.tools.retry import RetryingTool
    from research_system.tools.toolbox import build_toolbox

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    built = build_toolbox(Settings())

    assert all(isinstance(t, RetryingTool) for t in built.tools.values())


def test_a_missing_search_key_fails_at_build_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tavily is always primary: no key means no search at all, so say so early."""
    from research_system.tools.toolbox import build_toolbox

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        build_toolbox(Settings(_env_file=None))


def test_the_default_toolbox_is_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three researchers must share one limiter and one set of cache counters."""
    from research_system.tools import toolbox as toolbox_module

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    toolbox_module.default_toolbox.cache_clear()
    try:
        assert toolbox_module.default_toolbox() is toolbox_module.default_toolbox()
    finally:
        toolbox_module.default_toolbox.cache_clear()
