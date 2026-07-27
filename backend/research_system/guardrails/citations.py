"""Checking that every claim points at a source that actually exists.

The analyst is handed a numbered list of sources and returns claims tagged with the
index each one came from. Nothing stops it tagging a claim with index 9 when five
sources were supplied, and that is the failure mode most likely to reach the
reader: the sentence is plausible, the number looks like provenance, and a citation
to a source that was never there is indistinguishable from a real one on the page.

So every index is resolved against the list before the claims enter state. A claim
whose citation does not resolve has no evidence behind it, so it is dropped rather
than kept uncited — an unsupported sentence in a research report is the exact thing
this pipeline exists to prevent. Each drop returns an error string, because a claim
vanishing silently is its own kind of wrong.

Only the upper bound is checked. `ClaimOutput.source_idx` is `ge=0`, so a negative
index cannot survive structured-output parsing and there is nothing left to catch.
"""

from textwrap import shorten

from research_system.domain.schemas import ClaimOutput
from research_system.domain.state import SearchResult

# Enough of a claim to recognise it in a list of errors, not enough to bloat one.
PREVIEW_CHARS = 60


def validate_citations(
    claims: list[ClaimOutput], sources: list[SearchResult]
) -> tuple[list[ClaimOutput], list[str]]:
    """The claims whose citations resolve, plus one error per claim dropped."""
    kept: list[ClaimOutput] = []
    errors: list[str] = []
    for claim in claims:
        if claim.source_idx < len(sources):
            kept.append(claim)
        else:
            preview = shorten(claim.claim, width=PREVIEW_CHARS, placeholder="...")
            errors.append(
                f"dropped claim citing source {claim.source_idx}: "
                f"only {len(sources)} sources available ({preview})"
            )
    return kept, errors
