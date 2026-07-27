import time
from pathlib import Path

import pytest

from research_system.domain.state import SearchResult
from research_system.settings import CacheSettings
from research_system.tools.cache import SearchCache

RESULTS: list[SearchResult] = [
    {
        "title": "Attention Is All You Need",
        "snippet": "The dominant sequence transduction models...",
        "url": "https://arxiv.org/abs/1706.03762",
        "date": "2017-06-12",
    }
]


@pytest.fixture
def settings(tmp_path: Path) -> CacheSettings:
    return CacheSettings(enabled=True, path=tmp_path / "cache" / "search.sqlite", ttl_seconds=3600)


@pytest.fixture
def cache(settings: CacheSettings) -> SearchCache:
    return SearchCache(settings)


def test_a_query_never_searched_is_a_miss(cache: SearchCache) -> None:
    assert cache.get("tavily", "transformers") is None


def test_stored_results_come_back_whole(cache: SearchCache) -> None:
    cache.set("tavily", "transformers", RESULTS)
    assert cache.get("tavily", "transformers") == RESULTS


def test_case_and_spacing_do_not_pay_twice(cache: SearchCache) -> None:
    """ "Transformers" and "transformers" are one search, not two paid calls."""
    cache.set("tavily", "transformer models", RESULTS)
    assert cache.get("tavily", "  Transformer   Models  ") == RESULTS


def test_the_same_query_to_two_tools_is_two_entries(cache: SearchCache) -> None:
    """Tavily and Wikipedia answer the same question differently."""
    cache.set("tavily", "transformers", RESULTS)
    assert cache.get("wikipedia", "transformers") is None


def test_a_repeated_search_replaces_rather_than_duplicates(cache: SearchCache) -> None:
    fresher: list[SearchResult] = [{"title": "newer", "snippet": "s", "url": "u", "date": ""}]
    cache.set("tavily", "transformers", RESULTS)
    cache.set("tavily", "transformers", fresher)
    assert cache.get("tavily", "transformers") == fresher
    assert cache.stats().entries == 1


def test_an_entry_stops_being_used_after_the_ttl(
    cache: SearchCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A day-old answer to "latest AI news" is worse than no answer."""
    cache.set("tavily", "latest AI news", RESULTS)
    later = time.time() + 3601
    monkeypatch.setattr("research_system.tools.cache.time.time", lambda: later)
    assert cache.get("tavily", "latest AI news") is None
    assert cache.stats().entries == 0


def test_stats_count_hits_and_misses(cache: SearchCache) -> None:
    cache.get("tavily", "a")
    cache.set("tavily", "a", RESULTS)
    cache.get("tavily", "a")
    cache.get("tavily", "a")
    stats = cache.stats()
    assert (stats.hits, stats.misses, stats.entries) == (2, 1, 1)


def test_clear_empties_the_store_but_keeps_run_counts(cache: SearchCache) -> None:
    cache.set("tavily", "a", RESULTS)
    cache.get("tavily", "a")
    cache.clear()
    assert cache.get("tavily", "a") is None
    assert cache.stats().entries == 0
    assert cache.stats().hits == 1


def test_the_store_survives_a_restart(settings: CacheSettings) -> None:
    """Re-running the pipeline must not pay for yesterday's searches again."""
    SearchCache(settings).set("tavily", "transformers", RESULTS)
    assert SearchCache(settings).get("tavily", "transformers") == RESULTS


def test_the_database_is_created_where_configured(settings: CacheSettings) -> None:
    SearchCache(settings)
    assert settings.path.exists()


# --- disabled ---


@pytest.fixture
def disabled(tmp_path: Path) -> SearchCache:
    return SearchCache(CacheSettings(enabled=False, path=tmp_path / "off.sqlite", ttl_seconds=3600))


def test_disabled_cache_never_answers(disabled: SearchCache) -> None:
    disabled.set("tavily", "transformers", RESULTS)
    assert disabled.get("tavily", "transformers") is None


def test_disabled_cache_writes_no_file(disabled: SearchCache, tmp_path: Path) -> None:
    disabled.set("tavily", "transformers", RESULTS)
    disabled.clear()
    assert not (tmp_path / "off.sqlite").exists()


def test_disabled_cache_still_reports_stats(disabled: SearchCache) -> None:
    assert disabled.stats().entries == 0
