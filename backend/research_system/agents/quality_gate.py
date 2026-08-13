"""The quality gate: sources in, one score and a verdict out.

The only node in the graph that calls no model. Everything it needs is already in
state, the judgement is arithmetic, and a scorer that can hallucinate is worse
than no scorer at all — so this one cannot. It costs no tokens, adds no latency
worth measuring, and returns the same answer every time it sees the same sources.

Its verdict decides whether the run researches again or starts writing, and the
routing that reads it is bounded by `retry_count`: a run whose sources never
improve degrades to a report rather than looping. Failing here is a signal, not
an error, so nothing is appended to `errors`.

`source_ranking` carries each source's two half-scores alongside its total. A
number nobody can take apart is a number nobody can trust, and being explainable
is the whole reason this gate is arithmetic instead of a model.
"""

import time
from operator import itemgetter
from typing import Any

from research_system.agents.trace import trace_entry
from research_system.domain.scoring import (
    aggregate_score,
    domain_trust,
    snippet_score,
    source_score,
)
from research_system.domain.state import ResearchState, SearchResult
from research_system.logging import get_logger
from research_system.settings import Settings, get_settings

AGENT = "quality_gate"

logger = get_logger(__name__)


def assess(state: ResearchState, *, settings: Settings | None = None) -> dict[str, Any]:
    """Score every source, rank them, and decide whether the research is good enough.

    Sync, unlike every other node: this is pure CPU work with nothing to await.
    """
    settings = settings or get_settings()
    threshold = settings.pipeline.quality_threshold
    started = time.perf_counter()

    ranking = sorted(
        (_scored(source) for source in state["sources"]),
        key=itemgetter("score"),
        reverse=True,
    )
    score = aggregate_score([entry["score"] for entry in ranking])
    passed = score >= threshold

    logger.info(
        "quality_assessed",
        score=round(score, 3),
        passed=passed,
        threshold=threshold,
        sources=len(ranking),
    )
    return {
        "quality_score": score,
        "quality_passed": passed,
        "source_ranking": ranking,
        "pipeline_trace": [
            trace_entry(
                AGENT,
                started=started,
                summary=f"{score:.2f} vs {threshold:.2f} over {len(ranking)} sources: "
                + ("pass" if passed else "fail"),
            )
        ],
    }


def _scored(source: SearchResult) -> dict[str, Any]:
    """One ranking entry: the source, its total, and the two halves behind it."""
    return {
        "url": source["url"],
        "title": source["title"],
        "domain_score": domain_trust(source["url"]),
        "snippet_score": snippet_score(source["snippet"]),
        "score": source_score(source),
    }
