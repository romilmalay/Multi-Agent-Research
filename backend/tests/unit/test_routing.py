"""Routing tests. Routers are pure functions of state, so nothing is mocked."""

import pytest
from langgraph.graph import START
from langgraph.runtime import Runtime

from research_system.domain.state import ResearchState, default_state
from research_system.graph.builder import build_graph
from research_system.graph.context import RunContext
from research_system.graph.routing import (
    RETRY_RESEARCHER,
    route_after_quality,
    route_to_researchers,
)
from research_system.settings import Settings

QUERY = "What are the effects of microplastics on marine life?"
SUB_TOPICS = [
    "microplastic ingestion by marine species",
    "bioaccumulation up the food chain",
    "population-level effects on fisheries",
]


def state(sub_topics: list[str], token_count: int = 0) -> ResearchState:
    planned = default_state(QUERY)
    planned["sub_topics"] = sub_topics
    planned["token_count"] = token_count
    return planned


def test_one_send_per_sub_topic() -> None:
    sends = route_to_researchers(state(SUB_TOPICS))
    assert len(sends) == len(SUB_TOPICS)
    assert {send.node for send in sends} == {"researcher"}


def test_each_send_carries_its_own_sub_topic_as_the_query() -> None:
    """The researcher reads `query`, so the sub-topic has to arrive under that name."""
    sends = route_to_researchers(state(SUB_TOPICS))
    assert [send.arg["query"] for send in sends] == SUB_TOPICS


def test_each_send_carries_the_tokens_spent_so_far() -> None:
    """Without it the researcher's budget check reads a state key that is not there."""
    sends = route_to_researchers(state(SUB_TOPICS, token_count=12_000))
    assert all(send.arg["token_count"] == 12_000 for send in sends)


def test_send_carries_nothing_else() -> None:
    """A researcher that cannot see the other sub-topics cannot depend on them."""
    sends = route_to_researchers(state(SUB_TOPICS))
    assert all(set(send.arg) == {"query", "token_count"} for send in sends)


def test_one_sub_topic_fans_out_to_one_researcher() -> None:
    sends = route_to_researchers(state(["microplastic ingestion"]))
    assert len(sends) == 1
    assert sends[0].arg["query"] == "microplastic ingestion"


def test_no_sub_topics_falls_back_to_the_query_itself() -> None:
    """Fanning out to zero researchers would mean a run with no sources at all."""
    sends = route_to_researchers(state([]))
    assert len(sends) == 1
    assert sends[0].arg["query"] == QUERY


def scored(*, passed: bool, retry_count: int) -> ResearchState:
    gated = default_state(QUERY)
    gated["quality_passed"] = passed
    gated["quality_score"] = 0.8 if passed else 0.3
    gated["retry_count"] = retry_count
    return gated


def runtime(settings: Settings | None = None) -> Runtime[RunContext]:
    return Runtime(context=RunContext(settings=settings))


def test_good_sources_go_straight_to_the_analyst() -> None:
    assert route_after_quality(scored(passed=True, retry_count=0), runtime()) == "analyst"


def test_weak_sources_buy_one_more_search_pass() -> None:
    assert route_after_quality(scored(passed=False, retry_count=0), runtime()) == RETRY_RESEARCHER


def test_weak_sources_past_the_retry_allowance_continue_anyway() -> None:
    """Sources this bad may be all that exist, so the run reports on them, not fails."""
    assert route_after_quality(scored(passed=False, retry_count=1), runtime()) == "analyst"


def test_a_pass_ends_the_retry_loop_even_with_retries_left() -> None:
    assert route_after_quality(scored(passed=True, retry_count=1), runtime()) == "analyst"


def test_the_retry_allowance_comes_from_the_run_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero retries configured means the gate never sends the run backwards."""
    monkeypatch.setenv("PIPELINE__MAX_QUALITY_RETRIES", "0")
    settings = Settings()

    assert route_after_quality(scored(passed=False, retry_count=0), runtime(settings)) == "analyst"


def test_graph_enters_at_the_planner() -> None:
    assert (START, "planner") in build_graph().edges


def test_planner_fans_out_through_the_router() -> None:
    branches = build_graph().branches["planner"]
    assert [branch.path.name for branch in branches.values()] == ["route_to_researchers"]
    assert [branch.ends for branch in branches.values()] == [{"researcher": "researcher"}]


def test_every_researcher_joins_at_the_quality_gate() -> None:
    assert ("researcher", "quality_gate") in build_graph().edges


def test_a_retry_goes_back_through_the_gate() -> None:
    """Re-scored, not trusted: the retry's sources face the same threshold."""
    assert (RETRY_RESEARCHER, "quality_gate") in build_graph().edges


def test_quality_gate_branches_to_the_analyst_or_a_retry() -> None:
    branches = build_graph().branches["quality_gate"]
    assert [branch.path.name for branch in branches.values()] == ["route_after_quality"]
    assert [set(branch.ends or {}) for branch in branches.values()] == [
        {"analyst", RETRY_RESEARCHER}
    ]


def test_the_analyst_hands_off_to_the_synthesizer() -> None:
    assert ("analyst", "synthesizer") in build_graph().edges


def test_the_synthesizer_hands_off_to_the_writer() -> None:
    assert ("synthesizer", "writer") in build_graph().edges
