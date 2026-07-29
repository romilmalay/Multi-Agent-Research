"""Graph node registration. Every agent is stubbed: no model, no tool, no network."""

from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from research_system.agents import analyst as analyst_module
from research_system.agents import planner as planner_module
from research_system.agents import quality_gate as quality_gate_module
from research_system.agents import researcher as researcher_module
from research_system.agents import reviewer as reviewer_module
from research_system.agents import synthesizer as synthesizer_module
from research_system.agents import writer as writer_module
from research_system.domain.state import ResearchState, default_state
from research_system.graph.builder import (
    analyst_node,
    build_graph,
    human_review_node,
    planner_node,
    quality_gate_node,
    researcher_node,
    retry_researcher_node,
    reviewer_node,
    synthesizer_node,
    writer_node,
)
from research_system.graph.context import RunContext
from research_system.graph.routing import HUMAN_REVIEW, RETRY_RESEARCHER, route_after_quality
from research_system.settings import Settings
from research_system.tools.toolbox import Toolbox

QUERY = "What are the effects of microplastics on marine life?"
EXPECTED_NODES = {
    "planner",
    "researcher",
    "quality_gate",
    RETRY_RESEARCHER,
    "analyst",
    "synthesizer",
    HUMAN_REVIEW,
    "writer",
    "reviewer",
}

AsyncNode = Callable[[ResearchState, Runtime[RunContext]], Awaitable[dict[str, Any]]]

# Every node whose agent takes settings and nothing else.
SETTINGS_ONLY_NODES = [
    (planner_node, planner_module, "plan"),
    (analyst_node, analyst_module, "analyse"),
    (synthesizer_node, synthesizer_module, "synthesise"),
    (writer_node, writer_module, "write"),
    (reviewer_node, reviewer_module, "review"),
]


@pytest.fixture
def settings() -> Settings:
    return Settings()


def runtime(
    settings: Settings | None = None, toolbox: Toolbox | None = None
) -> Runtime[RunContext]:
    return Runtime(context=RunContext(settings=settings, toolbox=toolbox))


def test_registers_exactly_the_nine_nodes() -> None:
    assert set(build_graph().nodes) == EXPECTED_NODES


def test_context_schema_is_the_run_context() -> None:
    assert build_graph().context_schema is RunContext


def test_run_context_defaults_to_no_overrides() -> None:
    """No context means each agent falls back to the process defaults."""
    context = RunContext()
    assert context.settings is None
    assert context.toolbox is None


@pytest.mark.parametrize(("node", "module", "function"), SETTINGS_ONLY_NODES)
async def test_node_hands_its_agent_the_run_settings(
    node: AsyncNode,
    module: Any,
    function: str,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
        seen["state"], seen["settings"] = state, settings
        return {"synthesis": "from the stub"}

    monkeypatch.setattr(module, function, fake)
    state = default_state(QUERY)

    update = await node(state, runtime(settings))

    assert seen["state"] is state
    assert seen["settings"] is settings
    assert update == {"synthesis": "from the stub"}


def test_quality_gate_node_hands_its_agent_the_run_settings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one synchronous node: it calls no model and no tool."""
    seen: dict[str, Any] = {}

    def fake(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
        seen["settings"] = settings
        return {"quality_passed": True}

    monkeypatch.setattr(quality_gate_module, "assess", fake)

    update = quality_gate_node(default_state(QUERY), runtime(settings))

    assert seen["settings"] is settings
    assert update == {"quality_passed": True}


async def test_researcher_node_hands_its_agent_the_toolbox(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    toolbox = cast(Toolbox, object())

    async def fake(
        state: ResearchState,
        *,
        toolbox: Toolbox | None = None,
        settings: Settings | None = None,
    ) -> dict[str, Any]:
        seen["toolbox"], seen["settings"] = toolbox, settings
        return {"sources": []}

    monkeypatch.setattr(researcher_module, "research", fake)

    await researcher_node(default_state(QUERY), runtime(settings, toolbox))

    assert seen["toolbox"] is toolbox
    assert seen["settings"] is settings


async def test_node_without_context_falls_back_to_the_process_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
        seen["settings"] = settings
        return {}

    monkeypatch.setattr(planner_module, "plan", fake)

    await planner_node(default_state(QUERY), runtime())

    assert seen["settings"] is None


async def test_node_survives_a_run_started_with_no_context_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that omits `context=` leaves `runtime.context` at None, not at defaults."""
    seen: dict[str, Any] = {}

    async def fake(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
        seen["settings"] = settings
        return {}

    monkeypatch.setattr(planner_module, "plan", fake)

    # The type says RunContext; LangGraph passes None anyway, which is the point.
    await planner_node(default_state(QUERY), Runtime(context=cast(RunContext, None)))

    assert seen["settings"] is None


async def test_retry_researcher_researches_the_original_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fan-out searched the sub-topics; the retry asks the question itself."""
    seen: dict[str, Any] = {}

    async def fake(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        seen["query"] = state["query"]
        return {"sources": [], "errors": []}

    monkeypatch.setattr(researcher_module, "research", fake)
    state = default_state(QUERY)
    state["sub_topics"] = ["microplastic ingestion", "bioaccumulation"]

    await retry_researcher_node(state, runtime())

    assert seen["query"] == QUERY


async def test_retry_researcher_increments_the_retry_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counter is what bounds the loop, so the node owns it, not the router."""

    async def fake(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"sources": [], "errors": ["tavily failed"]}

    monkeypatch.setattr(researcher_module, "research", fake)
    state = default_state(QUERY)
    state["retry_count"] = 1

    update = await retry_researcher_node(state, runtime())

    assert update["retry_count"] == 2
    assert update["errors"] == ["tavily failed"]


async def test_a_gate_that_never_passes_still_reaches_the_analyst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry loop, run for real: gate, retry, gate, on to the analyst.

    Only the three nodes of the loop are wired up, so what is under test is the
    loop itself — the router's bound and the counter the retry node increments.
    """
    gate_runs = 0

    def never_passes(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
        nonlocal gate_runs
        gate_runs += 1
        return {"quality_passed": False, "quality_score": 0.2, "source_ranking": []}

    async def fake_research(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"sources": [], "errors": []}

    async def analyst_stub(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
        return {"key_claims": [{"claim": "reached the analyst"}]}

    monkeypatch.setattr(quality_gate_module, "assess", never_passes)
    monkeypatch.setattr(researcher_module, "research", fake_research)

    builder: StateGraph[ResearchState, RunContext, ResearchState, ResearchState] = StateGraph(
        ResearchState, context_schema=RunContext
    )
    builder.add_node("quality_gate", quality_gate_node)
    builder.add_node(RETRY_RESEARCHER, retry_researcher_node)
    builder.add_node("analyst", analyst_stub)
    builder.add_edge(START, "quality_gate")
    builder.add_conditional_edges(
        "quality_gate", route_after_quality, ["analyst", RETRY_RESEARCHER]
    )
    builder.add_edge(RETRY_RESEARCHER, "quality_gate")
    builder.add_edge("analyst", END)

    final = await builder.compile().ainvoke(default_state(QUERY))

    assert final["retry_count"] == 1  # the configured allowance, spent once
    assert gate_runs == 2  # the retry is re-scored, not trusted
    assert final["key_claims"] == [{"claim": "reached the analyst"}]


async def test_human_review_interrupts_and_records_the_verdict() -> None:
    """The node pauses on the first pass and finishes on the resumed one."""
    builder: StateGraph[ResearchState, RunContext, ResearchState, ResearchState] = StateGraph(
        ResearchState, context_schema=RunContext
    )
    builder.add_node(HUMAN_REVIEW, human_review_node)
    builder.add_edge(START, HUMAN_REVIEW)
    builder.add_edge(HUMAN_REVIEW, END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "run-1"}}

    state = default_state(QUERY)
    state["conflicts"] = ["source 0 and source 2 disagree on the ingestion rate"]
    paused = await graph.ainvoke(state, config=config)

    assert paused["__interrupt__"][0].value["conflicts"] == state["conflicts"]
    assert paused["pipeline_trace"] == []

    resumed = await graph.ainvoke(Command(resume="approved, keep both claims"), config=config)

    entry = resumed["pipeline_trace"][-1]
    assert entry["agent"] == HUMAN_REVIEW
    assert "approved, keep both claims" in entry["summary"]
