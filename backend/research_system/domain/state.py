"""The single state object every node in the graph reads from and writes to."""

import operator
from typing import Annotated, Any, TypedDict
from uuid import uuid4


class SearchResult(TypedDict):
    """One source, however it was found. Every tool returns this shape."""

    title: str
    snippet: str
    url: str
    date: str


class ResearchState(TypedDict, total=False):
    """State for one research run, accumulated across the 9 graph nodes."""

    # --- input ---
    query: str
    """The user's research question, verbatim."""

    token_count: Annotated[int, operator.add]
    """Total LLM tokens spent so far, summed from `usage_metadata`."""

    errors: Annotated[list[str], operator.add]
    """Non-fatal failures. A run continues; the errors are reported."""

    pipeline_trace: Annotated[list[dict[str, Any]], operator.add]
    """One entry per node: {agent, duration_ms, tokens, summary, prompt_hash}."""

    # --- planner ---
    sub_topics: list[str]
    """1-3 independent sub-questions, one researcher is fanned out per entry."""

    research_plan: str
    """The planner's prose rationale for that split."""

    # --- researchers (parallel) ---
    sources: Annotated[list[SearchResult], operator.add]
    """Search results, merged from every researcher."""

    search_queries_used: Annotated[list[str], operator.add]
    """Every query string actually sent to a tool, for the trace."""

    # --- quality gate ---
    quality_score: float
    """Aggregate trust score, 0.0-1.0, from `domain.scoring`."""

    quality_passed: bool
    """Whether `quality_score` cleared the configured threshold."""

    source_ranking: list[dict[str, Any]]
    """Sources with their individual scores, sorted best first."""

    # --- analyst ---
    key_claims: list[dict[str, Any]]
    """{claim, evidence, source_idx, confidence}, each grounded in one source."""

    conflicts: list[str]
    """Descriptions of sources that contradict each other."""

    # --- synthesizer ---
    synthesis: str
    """Cross-referenced narrative built from the claims."""

    # --- writer ---
    drafts: list[str]
    """Every draft ever written, appended, never overwritten."""

    current_draft: str
    """The draft the reviewer is looking at now."""

    revision_count: int
    """Writer passes completed. Bounded by `max_revisions`."""

    # --- reviewer ---
    review: dict[str, Any]
    """{score, issues, suggestions, passed} from the last review."""

    final_report: str
    """The accepted, PII-scrubbed report. Empty until the run finishes."""

    # --- internal control ---
    retry_count: int
    """Quality-gate retries used. Bounded to stop an infinite research loop."""

    run_id: str
    """Correlation id: the checkpointer thread, the log field, the trace id."""


def default_state(query: str, run_id: str | None = None) -> ResearchState:
    """Every field is set, so nodes index directly instead of guarding with `.get()`.

    `run_id` is generated when the caller has none; the API passes its own.
    """
    return ResearchState(
        query=query,
        token_count=0,
        errors=[],
        pipeline_trace=[],
        sub_topics=[],
        research_plan="",
        sources=[],
        search_queries_used=[],
        quality_score=0.0,
        quality_passed=False,
        source_ranking=[],
        key_claims=[],
        conflicts=[],
        synthesis="",
        drafts=[],
        current_draft="",
        revision_count=0,
        review={},
        final_report="",
        retry_count=0,
        run_id=run_id or uuid4().hex,
    )
