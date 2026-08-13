"""The writer: claims and a synthesis in, one markdown draft out.

This is the first node that produces prose meant for a person to read, and the
last one with the freedom to invent anything. So it is handed the analyst's
claims and the synthesizer's account and nothing else — never `sources`. Every
fact it can reach is already numbered and already attributed, which is what makes
"[3]" in the final report resolvable back to a URL.

Drafts are appended, never overwritten. The reviewer sends a draft back with
issues, this node writes another, and the run keeps both: the revision history is
what the trace and the evaluation read to see whether review actually improved
anything. `revision_count` counts those passes and bounds the loop.

Every exit appends a draft and increments the count, including the failures. A
pass that quietly did neither would leave the reviewer looking at the previous
draft and returning the same issues forever.
"""

import time
from typing import Any

from research_system.agents.trace import trace_entry
from research_system.domain.state import ResearchState
from research_system.llm.factory import get_model
from research_system.llm.usage import NOTHING_SPENT, Usage, extract_usage
from research_system.logging import get_logger
from research_system.prompts import load
from research_system.settings import Settings, get_settings

AGENT = "writer"

NO_CLAIMS_DRAFT = (
    "No supported claims were found for this question, so there is no report to write. "
    "The errors recorded for this run explain where the research broke down."
)

logger = get_logger(__name__)


async def write(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
    """Write one draft of the report from the claims and the synthesis."""
    settings = settings or get_settings()
    claims = state["key_claims"]
    started = time.perf_counter()

    if not claims:
        # Nothing is attributable, so anything written would be the model's own
        # recollection. A fixed sentence says that, and costs nothing.
        logger.warning("writer_skipped", reason="no claims")
        return _update(
            state,
            started=started,
            draft=NO_CLAIMS_DRAFT,
            usage=NOTHING_SPENT,
            errors=[],
            summary="skipped: no claims",
            prompt_hash="",
        )

    prompt = load(AGENT)
    revision = _revision(state)
    response: Any = None
    try:
        model = get_model(AGENT, settings=settings)
        response = await model.ainvoke(
            prompt.render(
                query=state["query"],
                synthesis=state["synthesis"],
                claims=claims,
                conflicts=state["conflicts"],
                revision=revision,
            )
        )
        draft, failure = response.text.strip(), None
    except Exception as exc:
        draft, failure = "", exc

    # Tokens are counted even when the answer was unusable: the call was billed.
    usage = extract_usage(response)

    if not draft:
        logger.warning("writer_failed", error=str(failure))
        return _update(
            state,
            started=started,
            draft="",
            usage=usage,
            errors=[f"writer produced no draft: {failure}"],
            summary="failed",
            prompt_hash=prompt.hash,
        )

    fixing = f", fixing {len(revision['issues'])} issues" if revision else ""
    logger.info("draft_written", claims=len(claims), chars=len(draft), revising=bool(revision))
    return _update(
        state,
        started=started,
        draft=draft,
        usage=usage,
        errors=[],
        summary=f"draft {state['revision_count'] + 1} from {len(claims)} claims{fixing}",
        prompt_hash=prompt.hash,
    )


def _revision(state: ResearchState) -> dict[str, Any] | None:
    """What the reviewer said about the draft being rewritten, or `None` on the first pass.

    `review` is empty until the reviewer has run, and that emptiness is what the
    template's `{{#revision}}` section keys off: one prompt serves both passes.
    The previous draft travels with the verdict because issues are stated against
    a specific text, and a rewrite from claims alone would lose the wording that
    was already right.
    """
    review = state["review"]
    if not review:
        return None
    return {
        "score": review["score"],
        "issues": review["issues"],
        "suggestions": review["suggestions"],
        "previous_draft": state["current_draft"],
    }


def _update(
    state: ResearchState,
    *,
    started: float,
    draft: str,
    usage: Usage,
    errors: list[str],
    summary: str,
    prompt_hash: str,
) -> dict[str, Any]:
    """The state delta. One shape for all three exits, so none can forget a field.

    `drafts` has no reducer — only this node writes it, and it writes it in
    sequence — so the append happens here against the list already in state.
    """
    return {
        "drafts": [*state["drafts"], draft],
        "current_draft": draft,
        "revision_count": state["revision_count"] + 1,
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
