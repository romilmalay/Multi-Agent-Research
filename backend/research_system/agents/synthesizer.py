"""The synthesizer: a list of separate claims in, one connected account out.

The analyst returns facts that were found independently, in the order it happened
to find them. Nobody has yet asked what they mean together — whether two of them
are the same finding from different literatures, or whether two of them cannot
both be true. That question is this node's only job.

Surfacing conflicts is the half that matters. A model asked to summarise
contradictory sources will smooth them into one confident paragraph, because
fluent prose is what it was trained to produce; the disagreement is exactly the
thing a researcher needed to be told. So `conflicts` is a separate field rather
than a paragraph the writer might drop, and the graph reads it later to decide
whether a human should look at the run.

With no claims there is nothing to cross-reference, so the model is not called.
Handing it an empty list would buy an invented narrative at full price.
"""

import time
from typing import Any

from research_system.agents.trace import trace_entry
from research_system.domain.schemas import SynthesizerOutput
from research_system.domain.state import ResearchState
from research_system.llm.factory import get_model
from research_system.llm.usage import extract_usage
from research_system.logging import get_logger
from research_system.prompts import load
from research_system.settings import Settings, get_settings

AGENT = "synthesizer"

logger = get_logger(__name__)


async def synthesise(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
    """Cross-reference the claims into one account, and name the contradictions."""
    settings = settings or get_settings()
    claims = state["key_claims"]
    started = time.perf_counter()

    if not claims:
        logger.warning("synthesizer_skipped", reason="no claims")
        return _update(
            started=started,
            synthesis="",
            conflicts=[],
            tokens=0,
            errors=[],
            summary="skipped: no claims",
            prompt_hash="",
        )

    prompt = load(AGENT)
    response: Any = None
    try:
        model = get_model(AGENT, settings=settings).with_structured_output(
            SynthesizerOutput, include_raw=True
        )
        response = await model.ainvoke(prompt.render(query=state["query"], claims=claims))
        # `include_raw` turns a schema violation into a value, not an exception.
        output, failure = response["parsed"], response["parsing_error"]
    except Exception as exc:
        output, failure = None, exc

    # Tokens are counted even when the answer was unusable: the call was billed.
    usage = extract_usage(response)

    if output is None:
        logger.warning("synthesizer_failed", error=str(failure))
        # No invented placeholder text: the writer still has the claims, and an
        # empty synthesis says "nothing was cross-referenced" without pretending.
        return _update(
            started=started,
            synthesis="",
            conflicts=[],
            tokens=usage.total_tokens,
            errors=[f"synthesizer failed, the report is built from claims alone: {failure}"],
            summary="failed",
            prompt_hash=prompt.hash,
        )

    logger.info("synthesis_written", claims=len(claims), conflicts=len(output.conflicts))
    return _update(
        started=started,
        synthesis=output.synthesis,
        conflicts=output.conflicts,
        tokens=usage.total_tokens,
        errors=[],
        summary=f"{len(claims)} claims cross-referenced, {len(output.conflicts)} conflicts",
        prompt_hash=prompt.hash,
    )


def _update(
    *,
    started: float,
    synthesis: str,
    conflicts: list[str],
    tokens: int,
    errors: list[str],
    summary: str,
    prompt_hash: str,
) -> dict[str, Any]:
    """The state delta. One shape for all three exits, so none can forget a field."""
    return {
        "synthesis": synthesis,
        "conflicts": conflicts,
        "token_count": tokens,
        "errors": errors,
        "pipeline_trace": [
            trace_entry(
                AGENT,
                started=started,
                tokens=tokens,
                summary=summary,
                prompt_hash=prompt_hash,
            )
        ],
    }
