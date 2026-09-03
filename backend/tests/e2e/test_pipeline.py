"""The pipeline entry point, end to end. Every agent is stubbed: no model, no network."""

from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from research_system.agents import analyst as analyst_module
from research_system.agents import planner as planner_module
from research_system.agents import quality_gate as quality_gate_module
from research_system.agents import researcher as researcher_module
from research_system.agents import reviewer as reviewer_module
from research_system.agents import synthesizer as synthesizer_module
from research_system.agents import writer as writer_module
from research_system.agents.trace import trace_entry
from research_system.domain.state import ResearchState
from research_system.errors import UnknownRunError
from research_system.graph.checkpointer import thread
from research_system.llm.usage import Usage
from research_system.logging import configure_logging
from research_system.pipeline import (
    Event,
    Finished,
    Progress,
    resume_pipeline,
    run_pipeline,
    stream_pipeline,
    stream_resume,
)
from research_system.settings import Settings
from research_system.tools.toolbox import Toolbox

QUERY = "What are the effects of microplastics on marine life?"
REPORT = "# Microplastics\n\nThey accumulate [1].\n\n[1] https://arxiv.org/abs/1"
SOURCE = {
    "title": "a study",
    "snippet": "measured 1,200 samples over 18 months",
    "url": "https://arxiv.org/abs/1",
    "date": "2026-01-01",
}


USAGE = Usage(input_tokens=8, output_tokens=2)


def _traced(agent: str, **update: Any) -> dict[str, Any]:
    """An agent update carrying the trace entry a real agent would append."""
    return {
        **update,
        "pipeline_trace": [trace_entry(agent, started=0.0, usage=USAGE, summary="stubbed")],
        "token_count": USAGE.total_tokens,
    }


@pytest.fixture(autouse=True)
def stub_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path: one sub-topic, sources that pass the gate, a draft accepted first time."""

    async def plan(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return _traced("planner", sub_topics=["microplastic bioaccumulation"])

    async def research(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return _traced("researcher", sources=[SOURCE], search_queries_used=[state["query"]])

    def assess(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return _traced(
            "quality_gate", quality_passed=True, quality_score=0.9, source_ranking=[SOURCE]
        )

    async def analyse(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return _traced("analyst", key_claims=[{"claim": "they accumulate", "source_idx": 0}])

    async def synthesise(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return _traced("synthesizer", synthesis="the sources agree")

    async def write(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return _traced(
            "writer",
            drafts=[*state["drafts"], REPORT],
            current_draft=REPORT,
            revision_count=state["revision_count"] + 1,
        )

    async def review(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return _traced(
            "reviewer",
            review={"score": 9, "issues": [], "suggestions": [], "passed": True},
            final_report=state["current_draft"],
        )

    for module, name, stub in (
        (planner_module, "plan", plan),
        (researcher_module, "research", research),
        (quality_gate_module, "assess", assess),
        (analyst_module, "analyse", analyse),
        (synthesizer_module, "synthesise", synthesise),
        (writer_module, "write", write),
        (reviewer_module, "review", review),
    ):
        monkeypatch.setattr(module, name, stub)


async def test_a_run_reaches_a_final_report() -> None:
    final = await run_pipeline(QUERY)

    assert final["final_report"] == REPORT
    assert final["review"]["passed"] is True
    assert final["errors"] == []


async def test_the_query_survives_the_fan_out_unchanged() -> None:
    """Researchers are handed a sub-topic as `query`; the run's own question is not lost."""
    final = await run_pipeline(QUERY)

    assert final["query"] == QUERY
    assert final["search_queries_used"] == ["microplastic bioaccumulation"]


async def test_every_agent_runs_once_in_pipeline_order() -> None:
    final = await run_pipeline(QUERY)

    assert [entry["agent"] for entry in final["pipeline_trace"]] == [
        "planner",
        "researcher",
        "quality_gate",
        "analyst",
        "synthesizer",
        "writer",
        "reviewer",
    ]


async def test_tokens_are_summed_across_the_run() -> None:
    final = await run_pipeline(QUERY)

    assert final["token_count"] == 70  # seven agents, ten tokens each


async def test_a_run_without_an_id_is_given_one() -> None:
    final = await run_pipeline(QUERY)

    assert final["run_id"]


async def test_the_callers_run_id_is_the_one_used() -> None:
    """The caller's id, not a new one: the API generates it before the run starts."""
    final = await run_pipeline(QUERY, "run-abc")

    assert final["run_id"] == "run-abc"


async def test_the_run_context_reaches_the_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    settings, toolbox = Settings(), cast(Toolbox, object())

    async def research(
        state: ResearchState,
        *,
        toolbox: Toolbox | None = None,
        settings: Settings | None = None,
    ) -> dict[str, Any]:
        seen["toolbox"], seen["settings"] = toolbox, settings
        return {"sources": []}

    monkeypatch.setattr(researcher_module, "research", research)

    await run_pipeline(QUERY, settings=settings, toolbox=toolbox)

    assert seen["settings"] is settings
    assert seen["toolbox"] is toolbox


async def test_without_context_the_agents_fall_back_to_the_process_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def plan(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
        seen["settings"] = settings
        return {"sub_topics": ["one sub-topic"]}

    monkeypatch.setattr(planner_module, "plan", plan)

    await run_pipeline(QUERY)

    assert seen["settings"] is None


async def test_the_run_is_checkpointed_under_its_own_id() -> None:
    """`thread_id` is the `run_id`, which is what makes a resume possible at all."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await run_pipeline(QUERY, "run-abc", checkpointer=saver)

        checkpoint = await saver.aget(thread("run-abc"))
        assert checkpoint is not None
        assert checkpoint["channel_values"]["final_report"] == REPORT
        assert await saver.aget(thread("another-run")) is None


async def test_a_run_without_a_checkpointer_still_completes() -> None:
    """One-shot runs and tests want no connection open; they just cannot resume."""
    final = await run_pipeline(QUERY, checkpointer=None)

    assert final["final_report"] == REPORT


def crashes_once(module: Any, name: str, monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Make one agent die the first time, and return the switch that heals it.

    The stub wraps the one the fixture installed rather than replacing it, because
    `monkeypatch.undo()` would restore the *real* agent — and a resume test that
    quietly starts calling a live model is worse than no test at all.
    """
    stubbed = getattr(module, name)
    broken = True

    async def flaky(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        if broken:
            raise RuntimeError("the model host went away")
        result: dict[str, Any] = await stubbed(state, **kwargs)
        return result

    monkeypatch.setattr(module, name, flaky)

    def heal() -> None:
        nonlocal broken
        broken = False

    return heal


async def test_a_resumed_run_carries_on_from_the_step_it_stopped_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthesizer dies mid-run; the resume re-runs it and nothing before it.

    The trace is the evidence: seven entries, one per agent. A resume that started
    over would show the planner, the researcher and the analyst twice.
    """
    heal = crashes_once(synthesizer_module, "synthesise", monkeypatch)

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        with pytest.raises(RuntimeError):
            await run_pipeline(QUERY, "run-abc", checkpointer=saver)

        heal()
        final = await resume_pipeline("run-abc", checkpointer=saver)

    assert final["final_report"] == REPORT
    assert [entry["agent"] for entry in final["pipeline_trace"]] == [
        "planner",
        "researcher",
        "quality_gate",
        "analyst",
        "synthesizer",
        "writer",
        "reviewer",
    ]


async def test_a_resumed_run_is_not_billed_twice_for_the_work_it_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resuming passes `None`, not the state: the reducers must not run again."""
    heal = crashes_once(writer_module, "write", monkeypatch)

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        with pytest.raises(RuntimeError):
            await run_pipeline(QUERY, "run-abc", checkpointer=saver)

        heal()
        final = await resume_pipeline("run-abc", checkpointer=saver)

    assert final["token_count"] == 70  # seven agents, ten tokens each, counted once
    assert final["sources"] == [SOURCE]


async def test_resuming_a_finished_run_returns_it_without_running_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed thread has no next step, so the resume is a read of the result."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        first = await run_pipeline(QUERY, "run-abc", checkpointer=saver)

        async def die(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("a finished run must not re-run its agents")

        monkeypatch.setattr(planner_module, "plan", die)
        again = await resume_pipeline("run-abc", checkpointer=saver)

    assert again["final_report"] == first["final_report"]
    assert again["pipeline_trace"] == first["pipeline_trace"]


async def test_resuming_a_run_that_was_never_saved_says_so() -> None:
    """`--resume` on a typo'd id is a caller mistake, and gets a message that says which."""
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        with pytest.raises(UnknownRunError, match="no saved state for run 'nothing-here'"):
            await resume_pipeline("nothing-here", checkpointer=saver)


async def collect(events: AsyncIterator[Event]) -> tuple[list[Progress], ResearchState]:
    """Every progress event, and the state the run finished with."""
    progress: list[Progress] = []
    final: ResearchState | None = None
    async for event in events:
        if isinstance(event, Finished):
            final = event.state
        else:
            progress.append(event)

    assert final is not None, "the stream ended without a Finished event"
    return progress, final


async def test_a_streamed_run_reports_every_agent_starting_and_finishing() -> None:
    """Two events per agent: one when it starts, one when its update lands.

    The gate is the exception, and deliberately: it is arithmetic, so it has
    nothing to say before it has already said everything.
    """
    progress, final = await collect(stream_pipeline(QUERY))

    assert [event.agent for event in progress] == [
        "planner",
        "planner",
        "researcher",
        "researcher",
        "quality_gate",
        "analyst",
        "analyst",
        "synthesizer",
        "synthesizer",
        "writer",
        "writer",
        "reviewer",
        "reviewer",
    ]
    assert final["final_report"] == REPORT


async def test_a_researcher_says_which_sub_topic_it_took() -> None:
    """The fan-out's events are indistinguishable without it: three copies, one name."""
    progress, _ = await collect(stream_pipeline(QUERY))

    assert Progress("researcher", "searching: microplastic bioaccumulation") in progress


async def test_the_starting_event_arrives_before_the_agent_has_produced_anything() -> None:
    """The whole point: `updates` alone is silence until the model answers."""
    progress, _ = await collect(stream_pipeline(QUERY))
    details = [event.detail for event in progress if event.agent == "writer"]

    assert details == ["writing draft 1", "stubbed"]


async def test_a_streamed_run_ends_in_the_same_state_as_an_awaited_one() -> None:
    """Streaming is a reporting choice, not a different run."""
    _, streamed = await collect(stream_pipeline(QUERY, "run-streamed"))
    awaited = await run_pipeline(QUERY, "run-awaited")

    assert streamed["final_report"] == awaited["final_report"]
    assert streamed["token_count"] == awaited["token_count"]
    assert [entry["agent"] for entry in streamed["pipeline_trace"]] == [
        entry["agent"] for entry in awaited["pipeline_trace"]
    ]


async def test_a_streamed_resume_reports_only_what_is_left_to_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replaying the checkpoint's events would claim work this process never did."""
    heal = crashes_once(writer_module, "write", monkeypatch)

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        with pytest.raises(RuntimeError):
            await run_pipeline(QUERY, "run-abc", checkpointer=saver)

        heal()
        progress, final = await collect(stream_resume("run-abc", checkpointer=saver))

    assert [event.agent for event in progress] == ["writer", "writer", "reviewer", "reviewer"]
    assert final["final_report"] == REPORT


async def test_resuming_an_unknown_run_fails_before_it_yields_anything() -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        with pytest.raises(UnknownRunError):
            await collect(stream_resume("nothing-here", checkpointer=saver))


async def test_the_run_id_is_bound_to_the_logs(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True, cache_loggers=False)

    await run_pipeline(QUERY, "run-abc")

    logged = capsys.readouterr().out
    assert '"run_id": "run-abc"' in logged
    assert '"event": "run_started"' in logged
    assert '"event": "run_finished"' in logged
