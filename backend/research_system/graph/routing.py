"""Where the run goes next. Pure functions of state: they read it and pick an edge.

Routing lives apart from the nodes because it is the part of the graph that is
worth testing on its own. A router takes a state and returns a destination, so
every branch can be exercised by handing it a dict — no model, no tools, no
compiled graph.
"""

from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Send

from research_system.agents import analyst, researcher
from research_system.domain.state import ResearchState
from research_system.graph.context import RunContext, run_context
from research_system.settings import get_settings

# The one node with no agent module of its own is named here, where the router
# that reaches it lives. The other seven answer to their agent's `AGENT`.
RETRY_RESEARCHER = "retry_researcher"


def route_to_researchers(state: ResearchState) -> list[Send]:
    """Fan out: one researcher per sub-topic, each handed only its own.

    The payload is that researcher's entire view of the world — a `Send` replaces
    the state for the node it starts, so what is left out is simply not there. It
    gets `query` (its sub-topic, because the node is written not to know it is one
    of three) and `token_count`, which its budget check reads. The sources it
    returns merge back through the reducers on the main state.

    `sub_topics` is empty only if the planner returned nothing usable, and fanning
    out to zero researchers would mean a run with no sources at all, so the
    original query stands in as the plan of last resort.
    """
    sub_topics = state["sub_topics"] or [state["query"]]
    return [Send(researcher.AGENT, _payload(topic, state)) for topic in sub_topics]


def _payload(sub_topic: str, state: ResearchState) -> dict[str, Any]:
    """The slice of state one researcher runs on."""
    return {"query": sub_topic, "token_count": state["token_count"]}


def route_after_quality(state: ResearchState, runtime: Runtime[RunContext]) -> str:
    """Weak sources buy one more search pass; after that the run continues anyway.

    Refusing to proceed on a low score would be the wrong trade. The gate scores
    what search returned, and for some questions the good sources do not exist —
    so past the retry allowance the answer is a report that says so, carrying its
    own `quality_score`, rather than a run that fails or spins.
    """
    if state["quality_passed"]:
        return analyst.AGENT

    settings = run_context(runtime).settings or get_settings()
    if state["retry_count"] < settings.pipeline.max_quality_retries:
        return RETRY_RESEARCHER
    return analyst.AGENT
