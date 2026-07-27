import pytest

from research_system.settings import get_settings
from research_system.tools.scraper import find_url
from research_system.tools.selector import SCRAPER, TAVILY, WIKIPEDIA, select_tools


def test_tavily_is_always_primary() -> None:
    """A wrong guess costs a supplement, never the search itself."""
    for query in ["what is a transformer", "latest AI news", "https://example.com/x", "quantum"]:
        assert select_tools(query)[0] == TAVILY


def test_a_plain_query_uses_web_search_alone() -> None:
    assert select_tools("impact of transformers on machine translation") == [TAVILY]


@pytest.mark.parametrize(
    "query",
    [
        "what is retrieval augmented generation",
        "What are diffusion models",
        "who is Geoffrey Hinton",
        "define backpropagation",
        "definition of gradient descent",
        "the meaning of overfitting",
    ],
)
def test_definitional_queries_add_wikipedia(query: str) -> None:
    assert WIKIPEDIA in select_tools(query)


@pytest.mark.parametrize(
    "query",
    [
        "latest developments in LLM agents",
        "newest reasoning models",
        "recent advances in robotics",
        "current state of quantum computing",
        "AI regulation in 2026",
        "what happened this week in AI",
    ],
)
def test_recency_queries_stay_on_the_web(query: str) -> None:
    assert select_tools(query) == [TAVILY]


def test_recency_beats_definitional() -> None:
    """Wikipedia lags, so "what is the latest X" must not consult it."""
    assert select_tools("what is the latest LLM released in 2026") == [TAVILY]


def test_an_old_year_is_not_a_recency_signal() -> None:
    """ "What was the 1969 moon landing" is a history question, not a news one."""
    assert WIKIPEDIA in select_tools("what was the 1969 moon landing")


def test_a_url_routes_to_the_scraper() -> None:
    assert select_tools("summarize https://example.com/paper") == [TAVILY, SCRAPER]


def test_rules_stack() -> None:
    assert select_tools("what is attention, see https://arxiv.org/abs/1706.03762") == [
        TAVILY,
        SCRAPER,
        WIKIPEDIA,
    ]


def test_selected_names_are_the_configured_ones() -> None:
    """A rename here would silently route to a tool the researcher cannot build."""
    assert set(select_tools("what is x, https://a.test/b")) <= set(
        get_settings().agents.researcher.tools
    )


# --- URL extraction ---


def test_url_is_found_inside_a_sentence() -> None:
    assert (
        find_url("please read https://example.com/a/b then summarize") == "https://example.com/a/b"
    )


def test_trailing_sentence_punctuation_is_not_part_of_the_url() -> None:
    assert find_url("see https://example.com/page.") == "https://example.com/page"
    assert find_url("(https://example.com/page)") == "https://example.com/page"


def test_query_parameters_survive() -> None:
    assert find_url("https://example.com/s?q=a&b=2") == "https://example.com/s?q=a&b=2"


def test_a_bare_domain_is_not_a_url() -> None:
    """Without a scheme it is far more likely a topic than a page to fetch."""
    assert find_url("compare example.com and openai.com") is None
    assert select_tools("compare example.com and openai.com") == [TAVILY]
