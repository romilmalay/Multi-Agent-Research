"""What a finished run leaves on disk. Real files, in `tmp_path`, never the repo's."""

import json
from pathlib import Path
from typing import Any

import pytest

from research_system.agents.trace import trace_entry
from research_system.artifacts import write_artifact
from research_system.domain.state import ResearchState, SearchResult, default_state
from research_system.llm.usage import Usage
from research_system.settings import Settings

QUERY = "What are the effects of microplastics on marine life?"
REPORT = "# Microplastics\n\nThey accumulate [1]."
SOURCE: SearchResult = {
    "title": "a study",
    "snippet": "measured 1,200 samples",
    "url": "https://arxiv.org/abs/1",
    "date": "2026-01-01",
}

# 100,000 in and 20,000 out at the configured $0.10 / $0.40 per million:
# 0.0100 + 0.0080 = $0.018 exactly. Hand-computed, so the artifact cannot define it.
USAGE = Usage(input_tokens=100_000, output_tokens=20_000)
EXPECTED_COST = 0.018


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Config pointing at a directory that does not exist yet."""
    monkeypatch.setenv("EVALUATION__REPORTS", str(tmp_path / "reports"))
    return Settings()


@pytest.fixture
def final() -> ResearchState:
    state = default_state(QUERY, "run-abc")
    state["final_report"] = REPORT
    state["sources"] = [SOURCE]
    state["quality_score"] = 0.81
    state["review"] = {"score": 8, "issues": [], "suggestions": [], "passed": True}
    state["revision_count"] = 2
    state["errors"] = ["tavily rate-limited, wikipedia answered instead"]
    state["drafts"] = ["the first draft", REPORT]
    state["pipeline_trace"] = [
        trace_entry(
            "planner", started=0.0, usage=USAGE, summary="2 sub-topics", prompt_hash="a1b2"
        ),
        trace_entry("quality_gate", started=0.0, summary="0.81 vs 0.60: pass"),
    ]
    return state


def written(final: ResearchState, settings: Settings, seconds: float = 58.2) -> dict[str, Any]:
    path = write_artifact(final, seconds=seconds, settings=settings)
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def test_the_artifact_is_named_after_the_run(final: ResearchState, settings: Settings) -> None:
    """The same id as the checkpoint thread and the log field: one string finds it."""
    path = write_artifact(final, seconds=58.2, settings=settings)

    assert path.name == "run-abc.json"
    assert path.parent == settings.evaluation.reports


def test_the_reports_directory_is_created_if_it_is_missing(
    final: ResearchState, settings: Settings
) -> None:
    """A fresh clone has no `evals/reports/`, and the first run must not fail on that."""
    assert not settings.evaluation.reports.exists()

    write_artifact(final, seconds=58.2, settings=settings)

    assert settings.evaluation.reports.is_dir()


def test_the_artifact_holds_the_report_and_what_it_cost(
    final: ResearchState, settings: Settings
) -> None:
    record = written(final, settings)

    assert record["run_id"] == "run-abc"
    assert record["query"] == QUERY
    assert record["report"] == REPORT
    assert record["seconds"] == 58.2
    assert record["tokens"] == {"input": 100_000, "output": 20_000, "total": 120_000}
    assert record["cost_usd"] == EXPECTED_COST


def test_the_model_travels_with_the_cost(final: ResearchState, settings: Settings) -> None:
    """The same token counts are a different bill at a different price."""
    record = written(final, settings)

    assert record["model"] == settings.llm.model


def test_the_trace_is_kept_whole(final: ResearchState, settings: Settings) -> None:
    """Phase 13 reads these fields; a summary would throw away what it needs."""
    record = written(final, settings)

    assert [entry["agent"] for entry in record["trace"]] == ["planner", "quality_gate"]
    assert record["trace"][0]["input_tokens"] == 100_000
    assert record["trace"][0]["prompt_hash"] == "a1b2"


def test_the_sources_travel_with_the_report(final: ResearchState, settings: Settings) -> None:
    """A citation is only checkable against the thing it cites."""
    assert written(final, settings)["sources"] == [SOURCE]


def test_what_shaped_the_run_is_recorded(final: ResearchState, settings: Settings) -> None:
    record = written(final, settings)

    assert record["quality_score"] == 0.81
    assert record["review"]["score"] == 8
    assert record["revisions"] == 2
    assert record["retries"] == 0
    assert record["errors"] == ["tavily rate-limited, wikipedia answered instead"]


def test_intermediate_drafts_are_left_out(final: ResearchState, settings: Settings) -> None:
    """Bulk the checkpoint already holds, and no question here is answered by draft 1."""
    record = written(final, settings)

    assert "drafts" not in record
    assert "the first draft" not in json.dumps(record)


def test_a_second_run_does_not_overwrite_the_first(
    final: ResearchState, settings: Settings
) -> None:
    write_artifact(final, seconds=58.2, settings=settings)
    final["run_id"] = "run-def"
    write_artifact(final, seconds=12.0, settings=settings)

    assert sorted(path.name for path in settings.evaluation.reports.iterdir()) == [
        "run-abc.json",
        "run-def.json",
    ]
