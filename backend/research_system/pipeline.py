"""One question in, one finished run out: the entry point to the whole system.

Everything above this line — the CLI, the worker, the evaluation harness — wants
the same four things and should not each work out how to do them: build the
starting state, compile the graph, name the run so its checkpoints and its logs
file under one id, and await it. That is `run_pipeline`, and with its resuming
twin it is deliberately the only place that knows the shape of an invocation.

What it does not do is validate the query. A guardrail rejection has to happen at
the edge, before a run row exists to reject — the API refuses a bad query in the
handler, the CLI refuses it before it prints anything. By the time a query
reaches here it has already been accepted.
"""

import time
from typing import cast

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from research_system.domain.state import ResearchState, default_state
from research_system.errors import UnknownRunError
from research_system.graph.builder import compile_graph
from research_system.graph.checkpointer import thread
from research_system.graph.context import RunContext
from research_system.logging import bind_run_id, get_logger
from research_system.settings import Settings
from research_system.tools.toolbox import Toolbox

logger = get_logger(__name__)


async def run_pipeline(
    query: str,
    run_id: str | None = None,
    *,
    settings: Settings | None = None,
    toolbox: Toolbox | None = None,
    checkpointer: Checkpointer = None,
) -> ResearchState:
    """Research `query` from the planner to the final report, and return the state.

    `run_id` is generated when the caller has none, and is returned in the state
    either way — it is the checkpoint thread and the log field, so a caller that
    wants to resume or trace the run needs to be able to read it back.

    `settings` and `toolbox` are per-run overrides travelling in the context;
    omitting them leaves every agent on the process defaults. The checkpointer is
    the caller's to open and close, because it owns a connection; without one the
    run still completes, it just cannot be resumed.

    Failures inside a node are already handled by the node — every agent degrades
    rather than raises. What reaches here is the graph itself failing, which is
    not something this function can do anything useful about, so it propagates.
    """
    state = default_state(query, run_id)
    run_id = state["run_id"]
    bind_run_id(run_id)
    logger.info("run_started", query=query)

    graph = compile_graph(checkpointer=checkpointer, settings=settings)
    return await _finish(graph, state, run_id, settings=settings, toolbox=toolbox)


async def resume_pipeline(
    run_id: str,
    *,
    settings: Settings | None = None,
    toolbox: Toolbox | None = None,
    checkpointer: Checkpointer,
) -> ResearchState:
    """Carry on the run saved under `run_id`, from the step it stopped at.

    The input is `None`, and that is the whole mechanism: it tells LangGraph to
    load the thread's saved state instead of starting a new run from an initial
    one. Passing the state back in would re-apply every reducer — the token count
    would double, the sources would appear twice.

    The checkpointer is required rather than optional, because without one there
    is no saved state and so nothing to resume. A run id with nothing saved under
    it is the caller's mistake, and is reported as one.
    """
    bind_run_id(run_id)
    graph = compile_graph(checkpointer=checkpointer, settings=settings)

    saved = await graph.aget_state(thread(run_id))
    if not saved.values:
        raise UnknownRunError(f"no saved state for run {run_id!r}")

    logger.info("run_resumed", next=list(saved.next))
    return await _finish(graph, None, run_id, settings=settings, toolbox=toolbox)


async def _finish(
    graph: CompiledStateGraph[ResearchState, RunContext, ResearchState, ResearchState],
    graph_input: ResearchState | None,
    run_id: str,
    *,
    settings: Settings | None,
    toolbox: Toolbox | None,
) -> ResearchState:
    """Await the graph and report what the run cost. `None` input means resume."""
    started = time.perf_counter()
    final = cast(
        ResearchState,
        await graph.ainvoke(
            graph_input,
            config=thread(run_id),
            context=RunContext(settings=settings, toolbox=toolbox),
        ),
    )

    logger.info(
        "run_finished",
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
        tokens=final["token_count"],
        reported=bool(final["final_report"]),
        errors=len(final["errors"]),
    )
    return final
