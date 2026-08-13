"""The researcher: one sub-topic in, a deduplicated list of sources out.

This node is the fan-out. The graph sends one copy of it per sub-topic, and each
copy is handed its own sub-topic as `query` and knows nothing about the others.
That isolation is what lets three of them run at once, and it is also why every
field this node returns is list-shaped: three copies finish at three different
moments, and `operator.add` merges what they return without any of them having to
coordinate.

It spends no LLM tokens and still checks the budget, because the sources it adds
are pasted into the analyst's prompt later. It is refusing to hand a bill to
someone else.

A tool that fails does not fail the node. Tavily being down is a reason to report
fewer sources, not a reason to lose the two that Wikipedia returned.
"""

import asyncio
import time
from typing import Any

from research_system.agents.trace import trace_entry
from research_system.domain.state import ResearchState, SearchResult
from research_system.guardrails.budget import check_budget
from research_system.logging import get_logger
from research_system.settings import Settings, get_settings
from research_system.tools.selector import select_tools
from research_system.tools.toolbox import Toolbox, default_toolbox

AGENT = "researcher"

logger = get_logger(__name__)


async def research(
    state: ResearchState,
    *,
    toolbox: Toolbox | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Search for one sub-topic. `state["query"]` is that sub-topic, not the run's."""
    settings = settings or get_settings()
    toolbox = toolbox or default_toolbox()
    sub_topic = state["query"]
    started = time.perf_counter()

    budget = check_budget(state["token_count"], settings.budget)
    if budget.exhausted:
        logger.warning("researcher_skipped", sub_topic=sub_topic, spent=budget.spent)
        return _update(
            started=started,
            sources=[],
            queries=[],
            errors=[f"budget exhausted, no sources gathered for {sub_topic!r}"],
            summary="skipped: budget exhausted",
        )

    # The selector says which tools suit this query; config says which this
    # deployment is allowed to use at all.
    enabled = settings.agents.researcher.tools
    tool_names = [name for name in select_tools(sub_topic) if name in enabled]
    if budget.degraded:
        # The primary alone. Every supplement costs a call now and analyst
        # tokens later, and `selector` orders the primary first for this reason.
        tool_names = tool_names[:1]

    outcomes = await asyncio.gather(
        *(_search(name, sub_topic, toolbox) for name in tool_names),
        return_exceptions=True,
    )

    found: list[SearchResult] = []
    errors: list[str] = []
    for name, outcome in zip(tool_names, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            logger.warning("tool_failed", tool=name, sub_topic=sub_topic, error=str(outcome))
            errors.append(f"{name} failed on {sub_topic!r}: {outcome}")
        else:
            found.extend(outcome)

    sources = _dedupe(found)[: settings.agents.researcher.max_sources_per_topic]
    return _update(
        started=started,
        sources=sources,
        queries=[sub_topic],
        errors=errors,
        summary=f"{len(sources)} sources via {', '.join(tool_names)}",
    )


async def _search(name: str, query: str, toolbox: Toolbox) -> list[SearchResult]:
    """One tool's results, from the cache when they are there.

    A cache hit costs no rate-limit permit, because the permit exists to pace
    calls to an upstream this one is not making. An empty list is a real answer —
    tools raise when they fail — so "found nothing" is cached like anything else.
    """
    cached = toolbox.cache.get(name, query)
    if cached is not None:
        return cached

    await toolbox.limiter.acquire()
    results = await toolbox.tools[name].search(query)
    toolbox.cache.set(name, query, results)
    return results


def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
    """First occurrence of each URL wins.

    Tools run in the order `selector` returned them, primary first, so the copy
    that survives is the one from the tool trusted most for this query — and the
    supplements are what the source cap trims.
    """
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for result in results:
        if result["url"] not in seen:
            seen.add(result["url"])
            unique.append(result)
    return unique


def _update(
    *,
    started: float,
    sources: list[SearchResult],
    queries: list[str],
    errors: list[str],
    summary: str,
) -> dict[str, Any]:
    """The state delta. Every field merges, so parallel copies never collide."""
    return {
        "sources": sources,
        "search_queries_used": queries,
        "errors": errors,
        "pipeline_trace": [
            trace_entry(AGENT, started=started, summary=summary),
        ],
    }
