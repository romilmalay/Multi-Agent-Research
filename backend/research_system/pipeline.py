"""One question in, one finished run out: the entry point to the whole system.

Everything above this line — the CLI, the worker, the evaluation harness — wants
the same four things and should not each work out how to do them: build the
starting state, compile the graph, name the run so its checkpoints and its logs
file under one id, and await it. That is what these four functions are, and they
are deliberately the only place that knows the shape of an invocation.

Four, because two questions are independent. A run either starts fresh or carries
on from a checkpoint, and a caller either wants the answer or wants to watch the
run produce it. The worker wants the answer to a fresh run; the CLI wants to watch
one. Neither should have to drain an event stream it does not care about, or poll
for progress that was never emitted.

What none of them do is validate the query. A guardrail rejection has to happen at
the edge, before a run row exists to reject — the API refuses a bad query in the
handler, the CLI refuses it before it prints anything. By the time a query reaches
here it has already been accepted.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, StreamMode

from research_system.domain.state import ResearchState, default_state
from research_system.errors import UnknownRunError
from research_system.graph.builder import compile_graph
from research_system.graph.checkpointer import thread
from research_system.graph.context import RunContext
from research_system.logging import bind_run_id, get_logger
from research_system.settings import Settings
from research_system.tools.toolbox import Toolbox

logger = get_logger(__name__)

Graph = CompiledStateGraph[ResearchState, RunContext, ResearchState, ResearchState]

STREAM_MODES: list[StreamMode] = ["updates", "custom", "values"]
"""What the graph is asked to emit, and why each one is there.

`updates` is a node's own state delta, and arrives when that node finishes.
`custom` is what a node chose to say about itself while still running — the only
way to learn that an agent is working, since an LLM node is silent for tens of
seconds before its update lands. `values` is not progress at all: it is the whole
state after each step, and the last one is the run's result.
"""


@dataclass(frozen=True, slots=True)
class Progress:
    """One thing that happened, reported while the run is still going."""

    agent: str
    detail: str


@dataclass(frozen=True, slots=True)
class Finished:
    """The run is over. Always the last event, and the only one carrying state."""

    state: ResearchState


Event = Progress | Finished


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
    graph, state = _start(query, run_id, settings, checkpointer)
    return await _finish(graph, state, resume=False, settings=settings, toolbox=toolbox)


async def stream_pipeline(
    query: str,
    run_id: str | None = None,
    *,
    settings: Settings | None = None,
    toolbox: Toolbox | None = None,
    checkpointer: Checkpointer = None,
) -> AsyncIterator[Event]:
    """The same run as `run_pipeline`, reported as it happens.

    Yields a `Progress` per thing that occurs and one `Finished` at the end. The
    run is identical either way; only the reporting differs.
    """
    graph, state = _start(query, run_id, settings, checkpointer)
    async for event in _stream(graph, state, resume=False, settings=settings, toolbox=toolbox):
        yield event


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
    graph, saved = await _restart(run_id, settings, checkpointer)
    return await _finish(graph, saved, resume=True, settings=settings, toolbox=toolbox)


async def stream_resume(
    run_id: str,
    *,
    settings: Settings | None = None,
    toolbox: Toolbox | None = None,
    checkpointer: Checkpointer,
) -> AsyncIterator[Event]:
    """The same resume as `resume_pipeline`, reported as it happens.

    Only the steps that still have to run are reported. The work the checkpoint
    already holds produced its events during the run that was interrupted, and
    replaying them would claim work was done that this process never did.
    """
    graph, saved = await _restart(run_id, settings, checkpointer)
    async for event in _stream(graph, saved, resume=True, settings=settings, toolbox=toolbox):
        yield event


def _start(
    query: str,
    run_id: str | None,
    settings: Settings | None,
    checkpointer: Checkpointer,
) -> tuple[Graph, ResearchState]:
    """A graph and the state a new run starts from."""
    state = default_state(query, run_id)
    bind_run_id(state["run_id"])
    logger.info("run_started", query=query)
    return compile_graph(checkpointer=checkpointer, settings=settings), state


async def _restart(
    run_id: str,
    settings: Settings | None,
    checkpointer: Checkpointer,
) -> tuple[Graph, ResearchState]:
    """A graph and the state a saved run had reached, or `UnknownRunError`."""
    bind_run_id(run_id)
    graph = compile_graph(checkpointer=checkpointer, settings=settings)

    saved = await graph.aget_state(thread(run_id))
    if not saved.values:
        raise UnknownRunError(f"no saved state for run {run_id!r}")

    logger.info("run_resumed", next=list(saved.next))
    return graph, cast(ResearchState, saved.values)


async def _finish(
    graph: Graph,
    state: ResearchState,
    *,
    resume: bool,
    settings: Settings | None,
    toolbox: Toolbox | None,
) -> ResearchState:
    """Await the whole run and report what it cost."""
    started = time.perf_counter()
    final = cast(
        ResearchState,
        await graph.ainvoke(
            None if resume else state,
            config=thread(state["run_id"]),
            context=RunContext(settings=settings, toolbox=toolbox),
        ),
    )
    _log_finished(final, started)
    return final


async def _stream(
    graph: Graph,
    state: ResearchState,
    *,
    resume: bool,
    settings: Settings | None,
    toolbox: Toolbox | None,
) -> AsyncIterator[Event]:
    """The same run as `_finish`, yielding what happens on the way.

    `final` starts as the state the run already has, so it is a real answer from
    the first line rather than a placeholder: for a new run the initial state, for
    a resumed one whatever the checkpoint held. Every `values` chunk replaces it,
    and the last one is what the run ended up with.
    """
    started = time.perf_counter()
    final = state

    async for mode, payload in graph.astream(
        None if resume else state,
        config=thread(state["run_id"]),
        context=RunContext(settings=settings, toolbox=toolbox),
        stream_mode=STREAM_MODES,
    ):
        match mode:
            case "values":
                final = cast(ResearchState, payload)
            case "custom":
                said = cast(dict[str, str], payload)
                yield Progress(agent=said["agent"], detail=said["detail"])
            case "updates":
                for update in cast(dict[str, Any], payload).values():
                    for entry in update.get("pipeline_trace", []):
                        yield Progress(agent=entry["agent"], detail=entry["summary"])

    _log_finished(final, started)
    yield Finished(final)


def _log_finished(final: ResearchState, started: float) -> None:
    """One line per run, whichever way it was invoked."""
    logger.info(
        "run_finished",
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
        tokens=final["token_count"],
        reported=bool(final["final_report"]),
        errors=len(final["errors"]),
    )
