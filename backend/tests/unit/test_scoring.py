import pytest

from research_system.domain.scoring import (
    DOMAIN_TRUST,
    DOMAIN_WEIGHT,
    SHORT_SNIPPET_CHARS,
    SNIPPET_WEIGHT,
    TOP_SOURCES,
    UNKNOWN_DOMAIN_TRUST,
    aggregate_score,
    domain_trust,
    snippet_score,
    source_score,
)
from research_system.domain.state import SearchResult

# A snippet that reports a finding: full length, figures, research vocabulary.
RICH = (
    "The study measured 1,200 participants over 18 months and reported a 12% reduction "
    "in error rates compared with the baseline, a statistically significant result "
    "published in a peer-reviewed journal. Researchers noted the same findings held "
    "across every sample in the dataset they evaluated."
)

# The same length of grammatical English, saying nothing.
FLUFF = (
    "Everything you need to know about staying well, all in one place. Read on for tips "
    "and tricks that will change the way you think about your morning, and discover why "
    "so many people are talking about this right now. You will not want to miss it."
)


def test_a_listed_domain_scores_its_table_value() -> None:
    assert domain_trust("https://arxiv.org/abs/2301.00001") == 0.95


def test_an_unlisted_domain_falls_back_to_unknown() -> None:
    assert domain_trust("https://some-blog.example/post") == UNKNOWN_DOMAIN_TRUST


def test_peer_review_outranks_a_forum() -> None:
    """The ordering is the whole point of the table."""
    assert domain_trust("https://arxiv.org/abs/1") > domain_trust("https://reddit.com/r/ml")


def test_an_unknown_domain_sits_between_the_extremes() -> None:
    assert (
        domain_trust("https://reddit.com/r/ml")
        < UNKNOWN_DOMAIN_TRUST
        < domain_trust("https://arxiv.org/abs/1")
    )


def test_www_is_not_treated_as_a_different_site() -> None:
    assert domain_trust("https://www.nature.com/articles/x") == 0.95


def test_a_subdomain_inherits_its_parent_score() -> None:
    assert domain_trust("https://export.arxiv.org/abs/1") == 0.95
    assert domain_trust("https://en.wikipedia.org/wiki/Transformer") == 0.7


def test_a_lookalike_domain_does_not_inherit() -> None:
    """Suffix matching stops at label boundaries, so it cannot be spoofed."""
    assert domain_trust("https://notreddit.com/x") == UNKNOWN_DOMAIN_TRUST
    assert domain_trust("https://arxiv.org.evil.example/x") == UNKNOWN_DOMAIN_TRUST


def test_a_bare_tld_never_matches() -> None:
    assert domain_trust("https://org/x") == UNKNOWN_DOMAIN_TRUST


def test_the_host_is_matched_case_insensitively() -> None:
    assert domain_trust("https://ARXIV.org/abs/1") == 0.95


def test_a_port_does_not_break_the_lookup() -> None:
    assert domain_trust("https://github.com:443/org/repo") == 0.8


def test_a_malformed_url_scores_unknown_rather_than_raising() -> None:
    assert domain_trust("") == UNKNOWN_DOMAIN_TRUST
    assert domain_trust("not a url") == UNKNOWN_DOMAIN_TRUST


def test_every_table_value_is_a_score() -> None:
    assert all(0.0 <= score <= 1.0 for score in DOMAIN_TRUST.values())


def test_a_snippet_reporting_a_finding_beats_the_same_length_of_fluff() -> None:
    """The whole point: separate specific text from grammatical filler."""
    assert len(RICH) == len(FLUFF) + 45  # comparable length, opposite content
    assert snippet_score(RICH) > snippet_score(FLUFF) * 2


def test_an_empty_snippet_scores_zero() -> None:
    assert snippet_score("") == 0.0


def test_a_whitespace_only_snippet_scores_zero() -> None:
    assert snippet_score("   \n\t  ") == 0.0


def test_every_score_stays_in_range() -> None:
    samples = [RICH, FLUFF, "", "40%", RICH * 3, "a", "the the the the the"]
    assert all(0.0 <= snippet_score(s) <= 1.0 for s in samples)


def test_a_full_length_snippet_of_findings_can_reach_the_top() -> None:
    assert snippet_score(RICH + " " + RICH) == 1.0


def test_a_longer_snippet_scores_higher_than_a_shorter_one() -> None:
    """Length bands, holding content style constant."""
    sentence = "The trial reported results measured across the sample. "
    scores = [snippet_score(sentence * n) for n in (1, 2, 4, 8)]
    assert scores == sorted(scores)


def test_figures_raise_the_score() -> None:
    without = "The trial reported a reduction in error rates across the sample studied here."
    with_figures = "The trial reported a 12% reduction in 1,200 cases over 18 months of study."
    assert snippet_score(with_figures) > snippet_score(without)


def test_research_vocabulary_raises_the_score() -> None:
    plain = "Some people think the new approach is better than the old one for most work."
    technical = "The study measured results across the sample and published its findings now."
    assert snippet_score(technical) > snippet_score(plain)


def test_a_short_snippet_is_scaled_down_even_when_dense() -> None:
    """Three figures in twelve characters is not evidence of a substantial source."""
    dense_but_tiny = "40% 1,200 18"
    assert len(dense_but_tiny) < SHORT_SNIPPET_CHARS
    assert snippet_score(dense_but_tiny) < 0.1


def test_the_short_penalty_stops_applying_once_the_snippet_is_long_enough() -> None:
    at_the_line = "x" * SHORT_SNIPPET_CHARS
    assert snippet_score(at_the_line) == snippet_score("x" * (SHORT_SNIPPET_CHARS + 1))


def test_padding_a_snippet_with_filler_does_not_beat_saying_something() -> None:
    """Length alone is only worth 0.4, so it cannot buy a top score."""
    assert snippet_score("word " * 100) < snippet_score(RICH)


def test_surrounding_whitespace_does_not_change_the_score() -> None:
    assert snippet_score(f"  \n {RICH} \n ") == snippet_score(RICH)


def source(url: str, snippet: str) -> SearchResult:
    return SearchResult(title="a title", snippet=snippet, url=url, date="2026-01-01")


def test_a_source_is_its_two_halves_weighted_together() -> None:
    result = source("https://arxiv.org/abs/1", RICH)
    expected = DOMAIN_WEIGHT * 0.95 + SNIPPET_WEIGHT * snippet_score(RICH)
    assert source_score(result) == expected


def test_the_two_weights_are_a_whole() -> None:
    """Otherwise a source score is not on the same 0-1 scale as the threshold."""
    assert DOMAIN_WEIGHT + SNIPPET_WEIGHT == 1.0


def test_the_best_case_and_worst_case_stay_in_range() -> None:
    """A perfect source scores 0.97, not 1.0: no domain in the table is worth 1.0."""
    best = source_score(source("https://arxiv.org/abs/1", RICH * 2))
    worst = source_score(source("https://reddit.com/r/x", ""))
    assert best == DOMAIN_WEIGHT * 0.95 + SNIPPET_WEIGHT
    assert worst == DOMAIN_WEIGHT * 0.3
    assert 0.0 <= worst < best <= 1.0


def test_a_paper_outranks_a_forum_post() -> None:
    paper = source_score(source("https://arxiv.org/abs/1", RICH))
    forum = source_score(source("https://reddit.com/r/x", FLUFF))
    assert paper > forum


def test_the_domain_half_outweighs_the_snippet_half() -> None:
    """A polished snippet on a weak domain must not beat a plain one on a strong domain."""
    polished_on_weak = source_score(source("https://reddit.com/r/x", RICH))
    plain_on_strong = source_score(source("https://arxiv.org/abs/1", FLUFF))
    assert plain_on_strong > polished_on_weak


def test_no_sources_scores_zero() -> None:
    assert aggregate_score([]) == 0.0


def test_one_source_is_its_own_score() -> None:
    assert aggregate_score([0.42]) == 0.42


def test_fewer_than_five_sources_average_all_of_them() -> None:
    assert aggregate_score([0.2, 0.4, 0.6]) == pytest.approx(0.4)


def test_only_the_best_five_count() -> None:
    assert aggregate_score([1.0] * TOP_SOURCES + [0.0] * 20) == 1.0


def test_a_long_tail_of_weak_sources_does_not_sink_a_good_run() -> None:
    """Five strong sources plus ten weak ones is a good search, not a failed one."""
    strong = [0.9, 0.85, 0.8, 0.8, 0.75]
    assert aggregate_score(strong + [0.1] * 10) == aggregate_score(strong)


def test_order_of_the_input_does_not_matter() -> None:
    scores = [0.3, 0.9, 0.1, 0.7, 0.5, 0.2]
    assert aggregate_score(scores) == aggregate_score(sorted(scores))


def test_the_aggregate_never_exceeds_the_best_source() -> None:
    scores = [0.9, 0.4, 0.4, 0.4, 0.4, 0.4]
    assert aggregate_score(scores) <= max(scores)


def test_a_run_of_weak_sources_falls_below_the_default_threshold() -> None:
    """The gate has to actually fail something: all-forum research does not pass 0.6."""
    weak = [source_score(source("https://reddit.com/r/x", FLUFF)) for _ in range(8)]
    assert aggregate_score(weak) < 0.6


def test_a_run_of_strong_sources_clears_the_default_threshold() -> None:
    strong = [source_score(source("https://arxiv.org/abs/1", RICH)) for _ in range(8)]
    assert aggregate_score(strong) >= 0.6
