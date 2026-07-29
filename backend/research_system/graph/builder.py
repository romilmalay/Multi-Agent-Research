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

from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime

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
    route_to_researchers,
)


async def planner_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Split the query into sub-topics."""
    return await planner.plan(state, settings=run_context(runtime).settings)


async def researcher_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Search for one sub-topic. Fanned out, one copy per sub-topic."""
    context = run_context(runtime)
    return await researcher.research(state, toolbox=context.toolbox, settings=context.settings)


def quality_gate_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Score the gathered sources. No model, no tool, so no `await`."""
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
    update = await researcher.research(state, toolbox=context.toolbox, settings=context.settings)
    return {**update, "retry_count": state["retry_count"] + 1}


async def analyst_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Extract grounded claims from the ranked sources."""
    return await analyst.analyse(state, settings=run_context(runtime).settings)


async def synthesizer_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Cross-reference the claims into one narrative, and name the conflicts."""
    return await synthesizer.synthesise(state, settings=run_context(runtime).settings)


async def writer_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Write a draft, or revise the last one against the reviewer's issues."""
    return await writer.write(state, settings=run_context(runtime).settings)


async def reviewer_node(state: ResearchState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Score the draft, and on acceptance publish the scrubbed final report."""
    return await reviewer.review(state, settings=run_context(runtime).settings)


def build_graph() -> StateGraph[ResearchState, RunContext, ResearchState, ResearchState]:
    """The eight nodes and the edges between them. Later steps add the policies."""
    graph: StateGraph[ResearchState, RunContext, ResearchState, ResearchState] = StateGraph(
        ResearchState, context_schema=RunContext
    )
    graph.add_node(planner.AGENT, planner_node)
    graph.add_node(researcher.AGENT, researcher_node)
    graph.add_node(quality_gate.AGENT, quality_gate_node)
    graph.add_node(RETRY_RESEARCHER, retry_researcher_node)
    graph.add_node(analyst.AGENT, analyst_node)
    graph.add_node(synthesizer.AGENT, synthesizer_node)
    graph.add_node(writer.AGENT, writer_node)
    graph.add_node(reviewer.AGENT, reviewer_node)

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
    return graph
