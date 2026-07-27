from research_system.domain.schemas import ClaimOutput
from research_system.domain.state import SearchResult
from research_system.guardrails.citations import validate_citations


def claim(source_idx: int, text: str = "a factual statement") -> ClaimOutput:
    return ClaimOutput(
        claim=text,
        evidence="the passage that supports it",
        source_idx=source_idx,
        confidence=0.8,
    )


def sources(count: int) -> list[SearchResult]:
    return [
        SearchResult(
            title=f"source {i}",
            snippet="a snippet",
            url=f"https://example.com/{i}",
            date="2026-01-01",
        )
        for i in range(count)
    ]


def test_claims_citing_real_sources_are_all_kept() -> None:
    claims = [claim(0), claim(2), claim(4)]
    kept, errors = validate_citations(claims, sources(5))
    assert kept == claims
    assert errors == []


def test_index_9_of_5_sources_is_dropped() -> None:
    """The named case from the plan: a citation to a source that never existed."""
    kept, errors = validate_citations([claim(9)], sources(5))
    assert kept == []
    assert len(errors) == 1
    assert "source 9" in errors[0]
    assert "only 5 sources" in errors[0]


def test_the_first_index_past_the_end_is_dropped() -> None:
    """Off by one is the likely real failure: an LLM numbering sources from 1."""
    kept, _ = validate_citations([claim(5)], sources(5))
    assert kept == []


def test_the_last_valid_index_is_kept() -> None:
    kept, errors = validate_citations([claim(4)], sources(5))
    assert len(kept) == 1
    assert errors == []


def test_a_single_source_supports_index_zero() -> None:
    kept, errors = validate_citations([claim(0)], sources(1))
    assert len(kept) == 1
    assert errors == []


def test_good_claims_survive_alongside_bad_ones() -> None:
    """One hallucinated citation must not cost the run its valid claims."""
    good_first, bad, good_last = claim(0), claim(7), claim(1)
    kept, errors = validate_citations([good_first, bad, good_last], sources(2))
    assert kept == [good_first, good_last]
    assert len(errors) == 1


def test_every_drop_reports_its_own_error() -> None:
    kept, errors = validate_citations([claim(5), claim(6), claim(7)], sources(3))
    assert kept == []
    assert len(errors) == 3
    assert [e.split("source ")[1].split(":")[0] for e in errors] == ["5", "6", "7"]


def test_no_sources_means_no_claim_can_be_grounded() -> None:
    kept, errors = validate_citations([claim(0)], [])
    assert kept == []
    assert "only 0 sources" in errors[0]


def test_no_claims_is_not_an_error() -> None:
    assert validate_citations([], sources(5)) == ([], [])


def test_the_error_identifies_which_claim_was_dropped() -> None:
    kept, errors = validate_citations([claim(9, "transformers cut error by 40%")], sources(2))
    assert kept == []
    assert "transformers cut error by 40%" in errors[0]


def test_a_long_claim_is_truncated_in_the_error() -> None:
    """An error list stays readable; the preview only has to identify the claim."""
    _, errors = validate_citations([claim(9, "word " * 50)], sources(2))
    assert "..." in errors[0]
    assert len(errors[0]) < 140
