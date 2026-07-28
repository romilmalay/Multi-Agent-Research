"""Quality gate tests. No mocks anywhere: the node calls no model and no tool."""

import pytest

from research_system.agents.quality_gate import assess
from research_system.domain.state import ResearchState, SearchResult, default_state
from research_system.settings import Settings

TRACE_KEYS = {"agent", "duration_ms", "tokens", "summary", "prompt_hash"}
RANKING_KEYS = {"url", "title", "domain_score", "snippet_score", "score"}

# A snippet that reports a finding, and one the same length that says nothing.
RICH = (
    "The study measured 1,200 participants over 18 months and reported a 12% reduction "
    "in error rates compared with the baseline, a statistically significant result "
    "published in a peer-reviewed journal. Researchers noted the same findings held "
    "across every sample in the dataset they evaluated."
)
FLUFF = (
    "Everything you need to know about staying well, all in one place. Read on for tips "
    "and tricks that will change the way you think about your morning, and discover why "
    "so many people are talking about this right now. You will not want to miss it."
)


@pytest.fixture
def settings() -> Settings:
    return Settings()


def source(url: str, snippet: str = RICH, title: str = "a source") -> SearchResult:
    return SearchResult(title=title, snippet=snippet, url=url, date="2026-01-01")


def state(*sources: SearchResult) -> ResearchState:
    found = default_state("a research question")
    found["sources"] = list(sources)
    return found


def strong(count: int) -> list[SearchResult]:
    return [source(f"https://arxiv.org/abs/{i}") for i in range(count)]


def weak(count: int) -> list[SearchResult]:
    return [source(f"https://reddit.com/r/x/{i}", FLUFF) for i in range(count)]


def test_no_sources_scores_zero_and_does_not_pass(settings: Settings) -> None:
    update = assess(state(), settings=settings)
    assert update["quality_score"] == 0.0
    assert update["quality_passed"] is False
    assert update["source_ranking"] == []


def test_strong_sources_pass(settings: Settings) -> None:
    update = assess(state(*strong(6)), settings=settings)
    assert update["quality_passed"] is True
    assert update["quality_score"] >= settings.pipeline.quality_threshold


def test_weak_sources_fail(settings: Settings) -> None:
    update = assess(state(*weak(6)), settings=settings)
    assert update["quality_passed"] is False
    assert update["quality_score"] < settings.pipeline.quality_threshold


def test_the_ranking_is_sorted_best_first(settings: Settings) -> None:
    mixed = [
        source("https://medium.com/@a", FLUFF),
        source("https://arxiv.org/abs/1"),
        source("https://reddit.com/r/x", FLUFF),
        source("https://bbc.com/news/1"),
    ]
    ranking = assess(state(*mixed), settings=settings)["source_ranking"]
    scores = [entry["score"] for entry in ranking]
    assert scores == sorted(scores, reverse=True)
    assert [entry["url"] for entry in ranking] == [
        "https://arxiv.org/abs/1",
        "https://bbc.com/news/1",
        "https://medium.com/@a",
        "https://reddit.com/r/x",
    ]


def test_every_source_appears_in_the_ranking(settings: Settings) -> None:
    sources = strong(3) + weak(4)
    ranking = assess(state(*sources), settings=settings)["source_ranking"]
    assert len(ranking) == len(sources)
    assert {entry["url"] for entry in ranking} == {s["url"] for s in sources}


def test_a_ranking_entry_explains_its_own_score(settings: Settings) -> None:
    """The two halves travel with the total: a number nobody can take apart is useless."""
    entry = assess(state(source("https://arxiv.org/abs/1")), settings=settings)["source_ranking"][0]
    assert set(entry) == RANKING_KEYS
    assert entry["domain_score"] == 0.95
    assert entry["snippet_score"] > 0.0
    assert entry["score"] == pytest.approx(
        0.6 * entry["domain_score"] + 0.4 * entry["snippet_score"]
    )


def test_a_known_domain_outranks_an_unknown_one(settings: Settings) -> None:
    pair = [source("https://blog.example/post"), source("https://arxiv.org/abs/1")]
    ranking = assess(state(*pair), settings=settings)["source_ranking"]
    assert ranking[0]["domain_score"] == 0.95
    assert ranking[1]["domain_score"] == 0.5


def test_snippet_quality_breaks_a_tie_between_equal_domains(settings: Settings) -> None:
    pair = [
        source("https://arxiv.org/abs/1", FLUFF, title="fluff"),
        source("https://arxiv.org/abs/2", RICH, title="rich"),
    ]
    ranking = assess(state(*pair), settings=settings)["source_ranking"]
    assert [entry["title"] for entry in ranking] == ["rich", "fluff"]


def test_an_empty_snippet_scores_zero_on_its_snippet_half(settings: Settings) -> None:
    entry = assess(state(source("https://arxiv.org/abs/1", "")), settings=settings)
    assert entry["source_ranking"][0]["snippet_score"] == 0.0


def test_a_short_snippet_ranks_below_a_full_one(settings: Settings) -> None:
    pair = [
        source("https://arxiv.org/abs/1", "Up 40% in 2024.", title="stub"),
        source("https://arxiv.org/abs/2", RICH, title="full"),
    ]
    ranking = assess(state(*pair), settings=settings)["source_ranking"]
    assert [entry["title"] for entry in ranking] == ["full", "stub"]


def test_a_long_tail_of_weak_sources_does_not_fail_a_good_run(settings: Settings) -> None:
    """Five strong sources plus ten weak ones is a successful search."""
    update = assess(state(*(strong(5) + weak(10))), settings=settings)
    assert update["quality_passed"] is True


def test_the_threshold_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same sources pass or fail depending only on the configured threshold."""
    sources = state(*strong(5))
    monkeypatch.setenv("PIPELINE__QUALITY_THRESHOLD", "0.99")
    assert assess(sources, settings=Settings())["quality_passed"] is False
    monkeypatch.setenv("PIPELINE__QUALITY_THRESHOLD", "0.1")
    assert assess(sources, settings=Settings())["quality_passed"] is True


def test_failing_the_gate_is_not_an_error(settings: Settings) -> None:
    """A retry is the plan for weak research, not a fault to report."""
    assert "errors" not in assess(state(*weak(5)), settings=settings)


def test_the_node_spends_no_tokens(settings: Settings) -> None:
    update = assess(state(*strong(3)), settings=settings)
    assert "token_count" not in update
    assert update["pipeline_trace"][0]["tokens"] == 0


def test_the_trace_entry_has_the_standard_shape(settings: Settings) -> None:
    entry = assess(state(*strong(3)), settings=settings)["pipeline_trace"][0]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "quality_gate"
    assert entry["prompt_hash"] == ""
    assert entry["duration_ms"] >= 0.0


def test_the_trace_summary_reports_the_verdict(settings: Settings) -> None:
    passed = assess(state(*strong(5)), settings=settings)["pipeline_trace"][0]["summary"]
    failed = assess(state(*weak(5)), settings=settings)["pipeline_trace"][0]["summary"]
    assert "pass" in passed and "5 sources" in passed
    assert "fail" in failed


def test_the_same_sources_always_score_the_same(settings: Settings) -> None:
    """No model, no clock, no randomness: the gate is reproducible."""
    sources = state(*(strong(3) + weak(3)))
    first = assess(sources, settings=settings)
    second = assess(sources, settings=settings)
    assert first["quality_score"] == second["quality_score"]
    assert first["source_ranking"] == second["source_ranking"]


def test_the_node_does_not_touch_the_sources(settings: Settings) -> None:
    sources = state(*strong(3))
    before = [dict(s) for s in sources["sources"]]
    assess(sources, settings=settings)
    assert sources["sources"] == before
