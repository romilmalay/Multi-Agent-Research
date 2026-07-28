"""The analyst: a pile of sources in, a short list of grounded claims out.

This is where search results stop being text and become the material the report is
built from. Every claim carries the index of the source it came from and the quote
that supports it, so the writer never has to invent provenance and a reader can
follow any sentence back to the page it came from.

Nothing stops the model tagging a claim with source 9 when five sources were
given, so every citation is resolved against the list before the claims enter
state. A claim whose citation does not resolve is dropped and the drop recorded
in `errors` — an uncited sentence in a research report is the exact thing this
pipeline exists to prevent.

The claim count is clamped to `agents.analyst.max_claims` here, in Python, for the
same reason the planner clamps sub-topics: every claim is pasted into the writer's
prompt later, and the prompt asking for a maximum is a request, not a limit. The
prompt orders claims most important first, so the ones a clamp drops are the ones
worth least.

An analyst failure is not fatal. A run with no claims still reaches a report that
says so, which is more useful than a stack trace.
"""

import time
from typing import Any

from research_system.agents.trace import trace_entry
from research_system.domain.schemas import AnalystOutput, ClaimOutput
from research_system.domain.state import ResearchState, SearchResult
from research_system.guardrails.citations import validate_citations
from research_system.llm.factory import get_model
from research_system.llm.usage import extract_usage
from research_system.logging import get_logger
from research_system.prompts import load
from research_system.settings import Settings, get_settings

AGENT = "analyst"

logger = get_logger(__name__)


async def analyse(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
    """Extract claims from the gathered sources, each tied to the source that proves it."""
    settings = settings or get_settings()
    config = settings.agents.analyst
    sources = state["sources"]
    prompt = load(AGENT)
    started = time.perf_counter()

    response: Any = None
    try:
        model = get_model(AGENT, settings=settings).with_structured_output(
            AnalystOutput, include_raw=True
        )
        response = await model.ainvoke(
            prompt.render(
                query=state["query"],
                min_claims=config.min_claims,
                max_claims=config.max_claims,
                sources=_numbered(sources),
            )
        )
        # `include_raw` turns a schema violation into a value, not an exception.
        output, failure = response["parsed"], response["parsing_error"]
    except Exception as exc:
        output, failure = None, exc

    # Tokens are counted even when the answer was unusable: the call was billed.
    usage = extract_usage(response)
    errors: list[str] = []

    if output is None:
        logger.warning("analyst_failed", error=str(failure))
        claims: list[ClaimOutput] = []
        errors.append(f"analyst failed, no claims extracted: {failure}")
    else:
        # Validate before clamping: a claim citing a source that does not exist
        # should not take one of the slots from a claim that does.
        claims, dropped = validate_citations(output.claims, sources)
        claims = claims[: config.max_claims]
        errors.extend(dropped)
        if dropped:
            logger.warning("citations_dropped", claims=len(dropped), sources=len(sources))

    return {
        "key_claims": [claim.model_dump() for claim in claims],
        "token_count": usage.total_tokens,
        "errors": errors,
        "pipeline_trace": [
            trace_entry(
                AGENT,
                started=started,
                tokens=usage.total_tokens,
                summary=f"{len(claims)} claims from {len(sources)} sources",
                prompt_hash=prompt.hash,
            )
        ],
    }


def _numbered(sources: list[SearchResult]) -> list[dict[str, Any]]:
    """The sources with the index the model must cite them by.

    Numbering happens here rather than in the template so that the number the model
    sees is the position in `state["sources"]`, which is what a citation is later
    validated against.
    """
    return [{"idx": idx, **source} for idx, source in enumerate(sources)]
