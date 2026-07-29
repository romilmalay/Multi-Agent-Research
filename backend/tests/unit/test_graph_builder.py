"""Graph node registration. Every agent is stubbed: no model, no tool, no network."""

from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from research_system.agents import analyst as analyst_module
from research_system.agents import planner as planner_module
from research_system.agents import quality_gate as quality_gate_module
from research_system.agents import researcher as researcher_module
from research_system.agents import reviewer as reviewer_module
from research_system.agents import synthesizer as synthesizer_module
from research_system.agents import writer as writer_module
from research_system.domain.state import ResearchState, default_state
from research_system.graph.builder import (
    LLM_RETRY,
    analyst_node,
    build_graph,
    compile_graph,
    planner_node,
    quality_gate_node,
    researcher_node,
    retry_researcher_node,
    reviewer_node,
    synthesizer_node,
    writer_node,
)
from research_system.graph.context import RunContext
from research_system.graph.routing import (
    RETRY_RESEARCHER,
    route_after_quality,
    route_after_review,
)
from research_system.settings import Settings
from research_system.tools.toolbox import Toolbox

QUERY = "What are the effects of microplastics on marine life?"
SOURCE = {
    "title": "a study",
    "snippet": "measured 1,200 samples over 18 months",
    "url": "https://arxiv.org/abs/1",
    "date": "2026-01-01",
}
EXPECTED_NODES = {
    "planner",
    "researcher",
    "quality_gate",
    RETRY_RESEARCHER,
    "analyst",
    "synthesizer",
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


def test_registers_exactly_the_eight_nodes() -> None:
    assert set(build_graph().nodes) == EXPECTED_NODES


def test_context_schema_is_the_run_context() -> None:
    assert build_graph().context_schema is RunContext


def test_the_graph_compiles() -> None:
    """Compiling is the validation: every node reachable, every edge resolvable."""
    compiled = build_graph().compile()
    assert set(compiled.nodes) == EXPECTED_NODES | {START}


def test_only_the_model_nodes_retry() -> None:
    """The two that call no model have nothing transient to retry."""
    nodes = build_graph().nodes
    retrying = {name for name, node in nodes.items() if node.retry_policy}
    assert retrying == {"planner", "analyst", "synthesizer", "writer", "reviewer"}


def test_a_retry_is_spent_on_transient_failures_only() -> None:
    """A bad prompt fails the same way twice; retrying it just bills the tokens twice."""
    should_retry = LLM_RETRY.retry_on
    assert callable(should_retry)
    assert should_retry(ConnectionError("upstream dropped")) is True
    assert should_retry(ValueError("schema violation")) is False


def test_no_node_is_cached() -> None:
    """Deliberate: a cache hit replays the node's routing decision without re-deciding.

    Caching `quality_gate` — the only pure node — would serve its first "retry"
    verdict on every later visit, and the run would bounce between the gate and
    the retry until the step limit killed it.
    """
    nodes = build_graph().nodes
    assert not [name for name, node in nodes.items() if node.cache_policy]


def test_the_compiled_graph_carries_the_configured_step_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set here, not at the call site: it bounds the whole run, not one invocation."""
    monkeypatch.setenv("PIPELINE__RECURSION_LIMIT", "12")

    assert compile_graph(settings=Settings()).config == {"recursion_limit": 12}


def test_the_run_starts_at_the_planner() -> None:
    drawn = build_graph().compile().get_graph()
    assert [edge.target for edge in drawn.edges if edge.source == START] == ["planner"]


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


async def test_the_step_limit_stops_a_loop_the_counters_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backstop: a writer that forgets to count would otherwise revise forever."""
    monkeypatch.setenv("PIPELINE__RECURSION_LIMIT", "8")

    async def plan(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"sub_topics": ["one sub-topic"]}

    async def research(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"sources": [SOURCE]}

    def assess(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"quality_passed": True, "quality_score": 0.9}

    async def analyse(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"key_claims": [{"claim": "a claim", "confidence": 0.9}]}

    async def synthesise(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"synthesis": "the sources agree"}

    async def write_without_counting(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"current_draft": "a draft"}  # the bug: `revision_count` never moves

    async def reject(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"review": {"score": 3, "issues": [], "suggestions": [], "passed": False}}

    for module, name, stub in (
        (planner_module, "plan", plan),
        (researcher_module, "research", research),
        (quality_gate_module, "assess", assess),
        (analyst_module, "analyse", analyse),
        (synthesizer_module, "synthesise", synthesise),
        (writer_module, "write", write_without_counting),
        (reviewer_module, "review", reject),
    ):
        monkeypatch.setattr(module, name, stub)

    with pytest.raises(GraphRecursionError):
        await compile_graph(settings=Settings()).ainvoke(default_state(QUERY))


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


async def test_a_reviewer_that_never_passes_still_ends(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refinement loop, run for real: every draft rejected, the run still ends.

    Two nodes wired to the router, so what is under test is the bound — a writer
    that keeps being sent back until `revision_count` catches up with the config.
    """
    drafts = 0

    async def fake_write(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        nonlocal drafts
        drafts += 1
        return {
            "drafts": [*state["drafts"], f"draft {drafts}"],
            "current_draft": f"draft {drafts}",
            "revision_count": state["revision_count"] + 1,
        }

    async def fake_review(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"review": {"score": 4, "issues": ["thin"], "suggestions": [], "passed": False}}

    monkeypatch.setattr(writer_module, "write", fake_write)
    monkeypatch.setattr(reviewer_module, "review", fake_review)

    builder: StateGraph[ResearchState, RunContext, ResearchState, ResearchState] = StateGraph(
        ResearchState, context_schema=RunContext
    )
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_edge(START, "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_conditional_edges("reviewer", route_after_review, ["writer", END])

    final = await builder.compile().ainvoke(default_state(QUERY))

    assert drafts == settings.pipeline.max_revisions
    assert final["revision_count"] == settings.pipeline.max_revisions
    assert final["review"]["passed"] is False
