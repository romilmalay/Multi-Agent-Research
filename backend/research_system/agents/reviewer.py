"""The reviewer: a draft and the claims it was built from in, a verdict out.

The one node that reads the draft against its own evidence. It is given the
claims, not the sources, because that is what the writer was given: a report can
only be held to the material it was allowed to use, and grounding is checkable
only against that list.

`passed` is decided here in Python from `pipeline.review_pass_score`, not taken
from the model's flag. The model is asked for it — committing to a verdict is
part of what makes the score considered — but the graph routes on the threshold
in config, so config has to be what decides. Otherwise raising the bar from 7 to
9 in YAML would change nothing about which drafts ship.

A review that did not happen leaves `review` empty rather than inventing a score.
An empty verdict is what the writer's revision block keys off, so the next pass
writes a fresh draft instead of being told to fix an unnamed problem in a draft
it has already replaced.

This node also ends the run. When the draft passes, or when the revisions are
spent, the accepted draft is scrubbed of personal data and written to
`final_report` — the only field the user ever reads. The scrub happens here and
not in the writer because a placeholder is text: `[EMAIL]` in a draft would go
back into the next revision prompt as though the writer had put it there. So
`drafts` keeps what was written, and `final_report` is what may be published.
"""

import time
from typing import Any

from research_system.agents.trace import trace_entry
from research_system.domain.schemas import ReviewOutput
from research_system.domain.state import ResearchState
from research_system.guardrails.pii import scrub_pii
from research_system.llm.factory import get_model
from research_system.llm.usage import NOTHING_SPENT, Usage, extract_usage
from research_system.logging import get_logger
from research_system.prompts import load
from research_system.settings import Settings, get_settings

AGENT = "reviewer"

logger = get_logger(__name__)


async def review(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
    """Grade the current draft against the claims, and decide whether it ships."""
    settings = settings or get_settings()
    pass_score = settings.pipeline.review_pass_score
    draft = state["current_draft"]
    started = time.perf_counter()

    if not draft:
        # The writer failed. There is no text to hold to the evidence, and the
        # writer already recorded why, so this is not the reviewer's error.
        logger.warning("reviewer_skipped", reason="no draft")
        return _update(
            started=started,
            verdict={},
            final_report="",
            usage=NOTHING_SPENT,
            errors=[],
            summary="skipped: no draft",
            prompt_hash="",
        )

    prompt = load(AGENT)
    response: Any = None
    try:
        model = get_model(AGENT, settings=settings).with_structured_output(
            ReviewOutput, include_raw=True
        )
        response = await model.ainvoke(
            prompt.render(
                query=state["query"],
                pass_score=pass_score,
                claims=state["key_claims"],
                draft=draft,
            )
        )
        # `include_raw` turns a schema violation into a value, not an exception.
        output, failure = response["parsed"], response["parsing_error"]
    except Exception as exc:
        output, failure = None, exc

    # Tokens are counted even when the answer was unusable: the call was billed.
    usage = extract_usage(response)

    if output is None:
        logger.warning("reviewer_failed", error=str(failure))
        return _update(
            started=started,
            verdict={},
            # An unreviewed draft still ships if this was the last pass. Losing a
            # written report because the grader was down helps nobody, and the
            # error below says the report went out ungraded.
            final_report=_final_report(state, passed=False, settings=settings),
            usage=usage,
            errors=[f"reviewer produced no verdict, the draft is unreviewed: {failure}"],
            summary="failed",
            prompt_hash=prompt.hash,
        )

    passed = output.score >= pass_score
    logger.info("draft_reviewed", score=output.score, passed=passed, issues=len(output.issues))
    return _update(
        started=started,
        verdict={
            "score": output.score,
            "issues": output.issues,
            "suggestions": output.suggestions,
            "passed": passed,
        },
        final_report=_final_report(state, passed=passed, settings=settings),
        usage=usage,
        errors=[],
        summary=f"score {output.score}/10, {'passed' if passed else 'failed'}, "
        f"{len(output.issues)} issues",
        prompt_hash=prompt.hash,
    )


def _final_report(state: ResearchState, *, passed: bool, settings: Settings) -> str:
    """The scrubbed report, once this pass is the last one. Empty while revisions remain.

    Two ways a run ends, and both land here: the draft was accepted, or the
    revisions are spent and this draft is the best there will be. `revision_count`
    counts writer passes and the writer has already run, so the comparison is
    against the pass that produced the draft being judged.
    """
    if not passed and state["revision_count"] < settings.pipeline.max_revisions:
        return ""
    return scrub_pii(state["current_draft"], settings.guardrails)


def _update(
    *,
    started: float,
    verdict: dict[str, Any],
    final_report: str,
    usage: Usage,
    errors: list[str],
    summary: str,
    prompt_hash: str,
) -> dict[str, Any]:
    """The state delta. One shape for all three exits, so none can forget a field.

    `review` is replaced, not merged: a verdict belongs to the draft it judged, so
    carrying an old one past a failed review would point the writer at issues in
    a draft that no longer exists.
    """
    return {
        "review": verdict,
        "final_report": final_report,
        "token_count": usage.total_tokens,
        "errors": errors,
        "pipeline_trace": [
            trace_entry(
                AGENT,
                started=started,
                usage=usage,
                summary=summary,
                prompt_hash=prompt_hash,
            )
        ],
    }
