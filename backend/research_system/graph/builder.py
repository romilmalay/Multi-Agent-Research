"""The eight nodes of the research graph, registered on one `StateGraph`.

Every node here is a thin adapter, and that is the point. The agents in
`research_system.agents` take their dependencies as keyword arguments and know
nothing about LangGraph; these wrappers are the only place that knows a node is
called with `(state, runtime)`. Testing an agent therefore needs no graph, and
swapping a dependency for a run needs no change to an agent.

The dependencies those adapters pass on travel in `RunContext`, and the decisions
between them live in `routing`; this module is the nodes and the wiring.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, RetryPolicy

from research_system.agents import (
    analyst,
    planner,
    quality_gate,
    researcher,
    reviewer,
    synthesizer,
    writer,
)
from research_system.domain.state import ResearchState
from research_system.graph.context import RunContext, run_context
from research_system.graph.routing import (
    RETRY_RESEARCHER,
    route_after_quality,
    route_after_review,
    route_to_researchers,
)
from research_system.settings import Settings, get_settings

LLM_RETRY = RetryPolicy(max_attempts=2)
"""For nodes that call a model.

The client already retries a failed HTTP call (`llm.max_retries`), and every LLM
agent catches its own exceptions and degrades, so this covers the narrow band
left over: something raised outside the agent's own guard. `default_retry_on`
declines to retry `ValueError` and friends, which is what a bad prompt or a
schema violation raises — retrying those would just spend the tokens twice.
"""

# No node carries a `CachePolicy`, and that is a decision rather than an omission.
# The only candidate is `quality_gate` — the one node that is a pure function of
# its input — and a cache hit there replays the node's writes *including its
# routing decision*, without calling the router again. The gate's first verdict is
# "retry", so every later visit would be served that same answer from the cache and
# the run would bounce between the gate and the retry until it hit the step limit.
# The gate costs microseconds of arithmetic. There is nothing here worth caching.


def _announce(runtime: Runtime[RunContext], agent: str, detail: str) -> None:
    """Tell whoever is streaming that this node has started, and on what.

    A node's `updates` event arrives only when it finishes, which for a model call
    is tens of seconds of silence. This is the node saying it is alive in the
    meantime, and it belongs here rather than in the agent for the same reason the
    rest of this module does: the agents know nothing about LangGraph.

    `stream_writer` is a no-op when nobody is streaming, so this costs a dict.
    """
    runtime.stream_writer({"agent": agent, "detail": detail})


async def planner_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Split the query into sub-topics."""
    _announce(runtime, planner.AGENT, "splitting the query")
    return await planner.plan(state, settings=run_context(runtime).settings)


async def researcher_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Search for one sub-topic. Fanned out, one copy per sub-topic."""
    context = run_context(runtime)
    # `query` is this copy's sub-topic, which is the only way to tell three
    # simultaneous researchers apart while they are all still running.
    _announce(runtime, researcher.AGENT, f"searching: {state['query']}")
    return await researcher.research(state, toolbox=context.toolbox, settings=context.settings)


def quality_gate_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Score the gathered sources. No model, no tool, so no `await`.

    No announcement either: it is arithmetic, and its `updates` event arrives
    before anyone could have read a message saying it was about to start.
    """
    return quality_gate.assess(state, settings=run_context(runtime).settings)


async def retry_researcher_node(
    state: ResearchState, runtime: Runtime[RunContext]
) -> dict[str, Any]:
    """One more search pass when the gate rejected what the sub-topics found.

    It researches `query` — the original question, which the fan-out never
    searched — so the retry asks something the first pass did not, instead of
    replaying sub-topic queries the cache would answer identically.
    """
    context = run_context(runtime)
    _announce(runtime, researcher.AGENT, f"searching again: {state['query']}")
    update = await researcher.research(state, toolbox=context.toolbox, settings=context.settings)
    return {**update, "retry_count": state["retry_count"] + 1}


async def analyst_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Extract grounded claims from the ranked sources."""
    _announce(runtime, analyst.AGENT, f"extracting claims from {len(state['sources'])} sources")
    return await analyst.analyse(state, settings=run_context(runtime).settings)


async def synthesizer_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Cross-reference the claims into one narrative, and name the conflicts."""
    _announce(runtime, synthesizer.AGENT, f"cross-referencing {len(state['key_claims'])} claims")
    return await synthesizer.synthesise(state, settings=run_context(runtime).settings)


async def writer_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Write a draft, or revise the last one against the reviewer's issues."""
    _announce(runtime, writer.AGENT, f"writing draft {state['revision_count'] + 1}")
    return await writer.write(state, settings=run_context(runtime).settings)


async def reviewer_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Score the draft, and on acceptance publish the scrubbed final report."""
    _announce(runtime, reviewer.AGENT, "reviewing the draft")
    return await reviewer.review(state, settings=run_context(runtime).settings)


def build_graph() -> StateGraph[ResearchState, RunContext, ResearchState, ResearchState]:
    """The eight nodes, the edges between them, and the per-node policies."""
    graph: StateGraph[ResearchState, RunContext, ResearchState, ResearchState] = StateGraph(
        ResearchState, context_schema=RunContext
    )
    graph.add_node(planner.AGENT, planner_node, retry_policy=LLM_RETRY)
    graph.add_node(researcher.AGENT, researcher_node)
    graph.add_node(quality_gate.AGENT, quality_gate_node)
    graph.add_node(RETRY_RESEARCHER, retry_researcher_node)
    graph.add_node(analyst.AGENT, analyst_node, retry_policy=LLM_RETRY)
    graph.add_node(synthesizer.AGENT, synthesizer_node, retry_policy=LLM_RETRY)
    graph.add_node(writer.AGENT, writer_node, retry_policy=LLM_RETRY)
    graph.add_node(reviewer.AGENT, reviewer_node, retry_policy=LLM_RETRY)

    graph.add_edge(START, planner.AGENT)
    # The planner has one destination, reached once per sub-topic rather than once.
    graph.add_conditional_edges(planner.AGENT, route_to_researchers, [researcher.AGENT])
    # Every researcher, however many were sent, joins here before the gate scores them.
    graph.add_edge(researcher.AGENT, quality_gate.AGENT)
    graph.add_conditional_edges(
        quality_gate.AGENT, route_after_quality, [analyst.AGENT, RETRY_RESEARCHER]
    )
    # The retry is re-scored, not trusted: it re-enters the gate it was sent back by.
    # `retry_count`, which the node itself increments, is what stops that being a loop.
    graph.add_edge(RETRY_RESEARCHER, quality_gate.AGENT)

    graph.add_edge(analyst.AGENT, synthesizer.AGENT)
    graph.add_edge(synthesizer.AGENT, writer.AGENT)
    graph.add_edge(writer.AGENT, reviewer.AGENT)
    # The refinement loop, bounded by `revision_count`, which the writer increments
    # on every pass it makes — including the ones that produced nothing.
    graph.add_conditional_edges(reviewer.AGENT, route_after_review, [writer.AGENT, END])
    return graph


def compile_graph(
    *,
    checkpointer: Checkpointer = None,
    settings: Settings | None = None,
) -> CompiledStateGraph[ResearchState, RunContext, ResearchState, ResearchState]:
    """The runnable graph, with a bounded step count and somewhere to save its state.

    `recursion_limit` is the backstop under both loops: the retry counter and the
    revision counter are what should stop them, and this is what stops a run that
    escapes both from spinning until the budget is gone. It is a whole-run step
    count, so it is set here rather than left to whichever caller invokes.

    The checkpointer is passed in rather than opened here, because it owns a
    connection and the caller owns its lifetime. Without one the graph still runs
    — it just cannot be resumed, which is what tests and one-shot runs want.
    """
    settings = settings or get_settings()
    compiled = build_graph().compile(checkpointer=checkpointer)
    compiled.config = {"recursion_limit": settings.pipeline.recursion_limit}
    return compiled
