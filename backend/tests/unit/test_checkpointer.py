"""Checkpointer tests. Real SQLite in a temp directory, no network, no model."""

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from research_system.agents import analyst as analyst_module
from research_system.agents import planner as planner_module
from research_system.agents import quality_gate as quality_gate_module
from research_system.agents import researcher as researcher_module
from research_system.agents import reviewer as reviewer_module
from research_system.agents import synthesizer as synthesizer_module
from research_system.agents import writer as writer_module
from research_system.domain.state import ResearchState, default_state
from research_system.graph.builder import compile_graph
from research_system.graph.checkpointer import checkpointer, thread
from research_system.settings import Settings

QUERY = "What are the effects of microplastics on marine life?"
SOURCE = {
    "title": "a study",
    "snippet": "measured 1,200 samples over 18 months",
    "url": "https://arxiv.org/abs/1",
    "date": "2026-01-01",
}


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings pointing at a checkpoint file this test alone owns."""
    monkeypatch.setenv("CHECKPOINT__PATH", str(tmp_path / "checkpoints" / "runs.sqlite"))
    return Settings()


def stub_agents(monkeypatch: pytest.MonkeyPatch, *, writer_fails: bool = False) -> None:
    """Every agent replaced by a fixed answer, so a run costs nothing and is repeatable."""

    async def plan(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"sub_topics": ["one sub-topic"], "research_plan": "split once"}

    async def research(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"sources": [SOURCE], "search_queries_used": [state["query"]]}

    def assess(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"quality_passed": True, "quality_score": 0.9, "source_ranking": []}

    async def analyse(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"key_claims": [{"claim": "a claim", "source_idx": 0, "confidence": 0.9}]}

    async def synthesise(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {"synthesis": "the sources agree"}

    async def write(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        if writer_fails:
            raise ConnectionError("the model host went away mid-run")
        return {
            "drafts": [*state["drafts"], "the report"],
            "current_draft": "the report",
            "revision_count": state["revision_count"] + 1,
        }

    async def review(state: ResearchState, **kwargs: Any) -> dict[str, Any]:
        return {
            "review": {"score": 9, "issues": [], "suggestions": [], "passed": True},
            "final_report": state["current_draft"],
        }

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


def test_a_thread_is_named_after_the_run() -> None:
    assert thread("run-abc") == {"configurable": {"thread_id": "run-abc"}}


async def test_the_checkpoint_file_is_created_where_configured(settings: Settings) -> None:
    """Including the directory: a fresh deployment has no `.cache/` yet."""
    assert not settings.checkpoint.path.exists()

    async with checkpointer(settings) as saver:
        assert isinstance(saver, AsyncSqliteSaver)

    assert settings.checkpoint.path.exists()


async def test_a_run_leaves_its_state_behind(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written under the run's own id, and readable after the run has finished."""
    stub_agents(monkeypatch)

    async with checkpointer(settings) as saver:
        graph = compile_graph(checkpointer=saver, settings=settings)
        await graph.ainvoke(default_state(QUERY, run_id="run-1"), config=thread("run-1"))

        saved = await graph.aget_state(thread("run-1"))

    assert saved.values["final_report"] == "the report"
    assert saved.next == ()  # nothing left to run: the run finished


async def test_two_runs_do_not_see_each_other(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One database, one thread per run: the id is the whole isolation story."""
    stub_agents(monkeypatch)

    async with checkpointer(settings) as saver:
        graph = compile_graph(checkpointer=saver, settings=settings)
        await graph.ainvoke(default_state("first question", run_id="run-1"), config=thread("run-1"))
        await graph.ainvoke(
            default_state("second question", run_id="run-2"), config=thread("run-2")
        )

        first = await graph.aget_state(thread("run-1"))
        second = await graph.aget_state(thread("run-2"))

    assert first.values["query"] == "first question"
    assert second.values["query"] == "second question"


async def test_a_run_that_died_resumes_from_where_it_stopped(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the whole module: the work before the failure is not repeated."""
    stub_agents(monkeypatch, writer_fails=True)

    async with checkpointer(settings) as saver:
        graph = compile_graph(checkpointer=saver, settings=settings)
        with pytest.raises(ConnectionError):
            await graph.ainvoke(default_state(QUERY, run_id="run-1"), config=thread("run-1"))

        died = await graph.aget_state(thread("run-1"))
        # The synthesis survived the crash, and the writer is what still owes work.
        assert died.values["synthesis"] == "the sources agree"
        assert died.next == ("writer",)

    # A new process: new connection, new compiled graph, same run id.
    stub_agents(monkeypatch)
    async with checkpointer(settings) as saver:
        graph = compile_graph(checkpointer=saver, settings=settings)
        resumed = await graph.ainvoke(None, config=thread("run-1"))

    assert resumed["final_report"] == "the report"
    # The planner never ran again: one sub-topic, not two, and one set of sources.
    assert resumed["sub_topics"] == ["one sub-topic"]
    assert len(resumed["sources"]) == 1


async def test_a_graph_without_a_checkpointer_still_runs(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One-shot runs and most tests want no file on disk at all."""
    stub_agents(monkeypatch)

    final = await compile_graph(settings=settings).ainvoke(default_state(QUERY))

    assert final["final_report"] == "the report"
    assert not settings.checkpoint.path.exists()
