"""What a finished run leaves behind on disk.

The terminal scrolls away, and the checkpoint is not a record — it is operational
state, written for resuming and pruned once a run is terminal. Neither one is
somewhere you can look in a month and say what a run cost, or diff two runs of the
same question across a prompt change.

So every run writes one JSON file named after its `run_id`, next to the aggregate
reports phase 13 produces from them. JSON rather than prose because the first
consumer is not a person: the evaluation compares this system against a
single-agent baseline on accuracy, latency and cost, and that reads fields.

What is written is what cannot be recovered later: the report, the trace, the
bill, and the errors that shaped it. Not the intermediate drafts — they are bulky,
the checkpoint has them, and no question here is answered by the third draft of a
report that was rewritten twice.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_system.agents.trace import spent
from research_system.domain.state import ResearchState
from research_system.settings import Settings, get_settings


def write_artifact(
    final: ResearchState,
    *,
    seconds: float,
    settings: Settings | None = None,
) -> Path:
    """Write the run to `evaluation.reports` and return the path it went to."""
    settings = settings or get_settings()
    directory = settings.evaluation.reports
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{final['run_id']}.json"
    path.write_text(json.dumps(artifact(final, seconds=seconds, settings=settings), indent=2))
    return path


def artifact(final: ResearchState, *, seconds: float, settings: Settings) -> dict[str, Any]:
    """The record itself, as a plain dict so a test can read it without a file."""
    usage = spent(final["pipeline_trace"])
    return {
        "run_id": final["run_id"],
        "query": final["query"],
        "finished_at": datetime.now(UTC).isoformat(),
        "seconds": round(seconds, 1),
        # The model is here because the cost cannot be checked without it: the same
        # token counts are a different bill at a different price.
        "model": settings.llm.model,
        "tokens": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "total": usage.total_tokens,
        },
        "cost_usd": round(usage.cost_usd(settings.llm.price), 6),
        "report": final["final_report"],
        "quality_score": final["quality_score"],
        "review": final["review"],
        "revisions": final["revision_count"],
        "retries": final["retry_count"],
        "errors": final["errors"],
        # The sources travel with the report because a citation is only checkable
        # against what it cites, and phase 13 scores citation quality.
        "sources": final["sources"],
        "trace": final["pipeline_trace"],
    }
