"""The three tools all satisfy one contract, so they are tested against one."""

import httpx
import pytest
import respx
from pydantic import SecretStr

from research_system.errors import PermanentToolError, TransientToolError
from research_system.settings import SearchSettings, get_settings
from research_system.tools import search, wikipedia
from research_system.tools.base import Tool
from research_system.tools.scraper import ScraperTool
from research_system.tools.search import TavilyTool
from research_system.tools.wikipedia import WikipediaTool

PAGE_URL = "https://example.com/article"

TAVILY_BODY = {
    "results": [
        {
            "title": "Transformer architecture",
            "url": "https://arxiv.org/abs/1706.03762",
            "content": "The dominant sequence transduction models are based on attention.",
            "score": 0.98,
        },
        {
            "title": "Attention explained",
            "url": "https://example.com/attention",
            "content": "A walkthrough of scaled dot-product attention.",
            "score": 0.81,
        },
    ]
}

WIKIPEDIA_BODY = {
    "query": {
        "pages": {
            "61603971": {
                "index": 2,
                "title": "Transformer (deep learning)",
                "fullurl": "https://en.wikipedia.org/wiki/Transformer_(deep_learning)",
                "extract": "In deep learning, the transformer is a family of architectures.",
                "touched": "2026-07-03T05:06:03Z",
            },
            "63999586": {
                "index": 1,
                "title": "Generative pre-trained transformer",
                "fullurl": "https://en.wikipedia.org/wiki/Generative_pre-trained_transformer",
                "extract": "A generative pre-trained transformer is a large language model.",
                "touched": "2026-07-27T04:50:50Z",
            },
        }
    }
}

HTML_PAGE = """
<html>
  <head><title>  Scaling laws  </title><style>body { color: red; }</style></head>
  <body>
    <nav>Home About Contact</nav>
    <p>Model   loss   follows   a power law.</p>
    <script>track();</script>
    <footer>Copyright 2026</footer>
  </body>
</html>
"""


@pytest.fixture
def settings() -> SearchSettings:
    return get_settings().search


@pytest.fixture
def tavily(settings: SearchSettings) -> TavilyTool:
    return TavilyTool(settings, SecretStr("tvly-test-key"))


@pytest.fixture
def wiki(settings: SearchSettings) -> WikipediaTool:
    return WikipediaTool(settings)


@pytest.fixture
def scraper(settings: SearchSettings) -> ScraperTool:
    return ScraperTool(settings)


# --- the uniform contract ---


@respx.mock
async def test_every_tool_returns_the_same_shape(
    tavily: TavilyTool, wiki: WikipediaTool, scraper: ScraperTool
) -> None:
    respx.post(search.API_URL).respond(json=TAVILY_BODY)
    respx.get(wikipedia.API_URL).respond(json=WIKIPEDIA_BODY)
    respx.get(PAGE_URL).respond(html=HTML_PAGE)

    tools: list[Tool] = [tavily, wiki, scraper]
    queries = ["transformers", "transformers", PAGE_URL]
    for tool, query in zip(tools, queries, strict=True):
        results = await tool.search(query)
        assert results, f"{tool.name} returned nothing"
        for result in results:
            assert set(result) == {"title", "snippet", "url", "date"}
            assert result["url"]
            assert result["title"]


@respx.mock
async def test_every_tool_truncates_to_the_configured_length(
    settings: SearchSettings, tavily: TavilyTool, wiki: WikipediaTool, scraper: ScraperTool
) -> None:
    """One long upstream must not crowd the others out of the analyst's context."""
    limit = settings.snippet_max_chars
    long_text = "word " * (limit * 2)
    respx.post(search.API_URL).respond(
        json={"results": [{"title": "t", "url": "https://a.test", "content": long_text}]}
    )
    respx.get(wikipedia.API_URL).respond(
        json={
            "query": {
                "pages": {
                    "1": {
                        "index": 1,
                        "title": "t",
                        "fullurl": "https://b.test",
                        "extract": long_text,
                        "touched": "2026-01-01T00:00:00Z",
                    }
                }
            }
        }
    )
    respx.get(PAGE_URL).respond(html=f"<html><title>t</title><body>{long_text}</body></html>")

    for tool, query in ((tavily, "q"), (wiki, "q"), (scraper, PAGE_URL)):
        results = await tool.search(query)
        assert len(results[0]["snippet"]) == limit


# --- tavily ---


@respx.mock
async def test_tavily_maps_content_to_snippet(tavily: TavilyTool) -> None:
    route = respx.post(search.API_URL).respond(json=TAVILY_BODY)
    results = await tavily.search("transformers")
    assert len(results) == 2
    assert results[0]["title"] == "Transformer architecture"
    assert results[0]["url"] == "https://arxiv.org/abs/1706.03762"
    assert results[0]["snippet"].startswith("The dominant sequence")
    assert route.calls.last.request.headers["authorization"] == "Bearer tvly-test-key"


@respx.mock
async def test_tavily_date_is_empty_when_not_a_news_result(tavily: TavilyTool) -> None:
    respx.post(search.API_URL).respond(json=TAVILY_BODY)
    assert (await tavily.search("transformers"))[0]["date"] == ""


@respx.mock
async def test_tavily_asks_for_the_configured_number_of_results(
    settings: SearchSettings, tavily: TavilyTool
) -> None:
    route = respx.post(search.API_URL).respond(json=TAVILY_BODY)
    await tavily.search("transformers")
    assert route.calls.last.request.read().decode().count(str(settings.results_per_query)) >= 1


# --- wikipedia ---


@respx.mock
async def test_wikipedia_returns_lead_paragraphs_in_relevance_order(wiki: WikipediaTool) -> None:
    """Results are keyed by page id, so `index` is the only thing carrying rank."""
    respx.get(wikipedia.API_URL).respond(json=WIKIPEDIA_BODY)
    results = await wiki.search("transformers")
    assert [r["title"] for r in results] == [
        "Generative pre-trained transformer",
        "Transformer (deep learning)",
    ]
    assert results[0]["date"] == "2026-07-27"


@respx.mock
async def test_wikipedia_no_match_is_empty_not_an_error(wiki: WikipediaTool) -> None:
    """A search that matches nothing omits the "query" block entirely."""
    respx.get(wikipedia.API_URL).respond(json={"batchcomplete": ""})
    assert await wiki.search("qwertyasdf") == []


@respx.mock
async def test_wikipedia_page_without_a_lead_paragraph_still_returns(wiki: WikipediaTool) -> None:
    respx.get(wikipedia.API_URL).respond(
        json={
            "query": {
                "pages": {
                    "1": {
                        "index": 1,
                        "title": "Redirect page",
                        "fullurl": "https://en.wikipedia.org/wiki/Redirect_page",
                        "touched": "2026-02-02T00:00:00Z",
                    }
                }
            }
        }
    )
    assert (await wiki.search("q"))[0]["snippet"] == ""


@respx.mock
async def test_wikipedia_identifies_itself(wiki: WikipediaTool) -> None:
    """Wikimedia answers an anonymous agent with 403."""
    route = respx.get(wikipedia.API_URL).respond(json=WIKIPEDIA_BODY)
    await wiki.search("transformers")
    assert "research-system" in route.calls.last.request.headers["user-agent"]


# --- scraper ---


@respx.mock
async def test_scraper_reads_the_url_it_is_given(scraper: ScraperTool) -> None:
    respx.get(PAGE_URL).respond(html=HTML_PAGE)
    results = await scraper.search(PAGE_URL)
    assert len(results) == 1
    assert results[0]["url"] == PAGE_URL
    assert results[0]["title"] == "Scaling laws"


@respx.mock
async def test_scraper_drops_chrome_and_collapses_whitespace(scraper: ScraperTool) -> None:
    respx.get(PAGE_URL).respond(html=HTML_PAGE)
    snippet = (await scraper.search(PAGE_URL))[0]["snippet"]
    assert snippet == "Model loss follows a power law."


@respx.mock
async def test_scraper_finds_the_url_inside_a_full_query(scraper: ScraperTool) -> None:
    """Every tool gets the raw query, so this one extracts its own target."""
    respx.get(PAGE_URL).respond(html=HTML_PAGE)
    results = await scraper.search(f"summarize {PAGE_URL} for me")
    assert results[0]["url"] == PAGE_URL


async def test_scraper_without_a_url_fails_permanently(scraper: ScraperTool) -> None:
    """Rerunning the same query will not grow a URL, so retrying is pointless."""
    with pytest.raises(PermanentToolError):
        await scraper.search("no link here")


@respx.mock
async def test_scraper_follows_redirects(scraper: ScraperTool) -> None:
    respx.get(PAGE_URL).respond(301, headers={"Location": "https://example.com/moved"})
    respx.get("https://example.com/moved").respond(html=HTML_PAGE)
    assert (await scraper.search(PAGE_URL))[0]["title"] == "Scaling laws"


@respx.mock
async def test_scraper_falls_back_to_the_url_when_the_page_has_no_title(
    scraper: ScraperTool,
) -> None:
    respx.get(PAGE_URL).respond(html="<html><body>text</body></html>")
    assert (await scraper.search(PAGE_URL))[0]["title"] == PAGE_URL


# --- failure translation, uniform across the three ---


@respx.mock
@pytest.mark.parametrize("status", [500, 503, 429])
async def test_server_failure_is_transient_everywhere(
    status: int, tavily: TavilyTool, wiki: WikipediaTool, scraper: ScraperTool
) -> None:
    respx.post(search.API_URL).respond(status)
    respx.get(wikipedia.API_URL).respond(status)
    respx.get(PAGE_URL).respond(status)
    for tool, query in ((tavily, "q"), (wiki, "q"), (scraper, PAGE_URL)):
        with pytest.raises(TransientToolError):
            await tool.search(query)


@respx.mock
async def test_missing_page_is_permanent_everywhere(
    tavily: TavilyTool, wiki: WikipediaTool, scraper: ScraperTool
) -> None:
    respx.post(search.API_URL).respond(404)
    respx.get(wikipedia.API_URL).respond(404)
    respx.get(PAGE_URL).respond(404)
    for tool, query in ((tavily, "q"), (wiki, "q"), (scraper, PAGE_URL)):
        with pytest.raises(PermanentToolError):
            await tool.search(query)


@respx.mock
async def test_timeout_is_transient_everywhere(
    tavily: TavilyTool, wiki: WikipediaTool, scraper: ScraperTool
) -> None:
    """A timeout carries no status code, so it needs its own translation."""
    respx.post(search.API_URL).mock(side_effect=httpx.ReadTimeout("timed out"))
    respx.get(wikipedia.API_URL).mock(side_effect=httpx.ReadTimeout("timed out"))
    respx.get(PAGE_URL).mock(side_effect=httpx.ReadTimeout("timed out"))
    for tool, query in ((tavily, "q"), (wiki, "q"), (scraper, PAGE_URL)):
        with pytest.raises(TransientToolError):
            await tool.search(query)


@respx.mock
async def test_the_failing_tool_names_itself(tavily: TavilyTool) -> None:
    """The researcher records which tool failed, so the error must carry it."""
    respx.post(search.API_URL).respond(503)
    with pytest.raises(TransientToolError) as caught:
        await tavily.search("q")
    assert caught.value.tool == "tavily"


@respx.mock
async def test_configured_timeout_reaches_the_client(
    settings: SearchSettings, scraper: ScraperTool
) -> None:
    route = respx.get(PAGE_URL).respond(html=HTML_PAGE)
    await scraper.search(PAGE_URL)
    timeout = route.calls.last.request.extensions["timeout"]
    assert timeout["read"] == settings.request_timeout_seconds
