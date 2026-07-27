import httpx
import pytest

from research_system.domain.state import SearchResult, default_state
from research_system.errors import PermanentToolError, TransientToolError
from research_system.tools.base import Tool, error_for_status, http_errors, make_result


def status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com/page")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("failed", request=request, response=response)


class FakeTool:
    """A minimal implementation, here to prove the protocol is satisfiable."""

    name = "fake"

    async def search(self, query: str) -> list[SearchResult]:
        return [make_result(title=query, url="https://example.com", snippet="x", max_chars=10)]


def test_result_fits_the_sources_field() -> None:
    """`sources` is what the quality gate and the analyst read; keys must match."""
    result = make_result(
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        snippet="The dominant sequence transduction models...",
        max_chars=100,
        date="2017-06-12",
    )
    assert set(result) == {"title", "snippet", "url", "date"}
    state = default_state("transformers")
    state["sources"].append(result)
    assert state["sources"][0]["url"] == "https://arxiv.org/abs/1706.03762"


def test_snippet_is_cut_to_the_configured_length() -> None:
    result = make_result(title="t", url="u", snippet="a" * 500, max_chars=120)
    assert len(result["snippet"]) == 120


def test_scraped_whitespace_is_stripped_before_truncating() -> None:
    """bs4 hands back text padded with newlines; padding must not eat the budget."""
    result = make_result(title="\n  Title \n", url="u", snippet="\n\n  body text  \n", max_chars=6)
    assert result["title"] == "Title"
    assert result["snippet"] == "body t"


def test_date_is_empty_when_the_upstream_reports_none() -> None:
    assert make_result(title="t", url="u", snippet="s", max_chars=10)["date"] == ""


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retryable_statuses_are_transient(status: int) -> None:
    assert isinstance(error_for_status(status, tool="tavily"), TransientToolError)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_permanent(status: int) -> None:
    assert isinstance(error_for_status(status, tool="wikipedia"), PermanentToolError)


def test_status_error_names_the_tool_and_the_code() -> None:
    error = error_for_status(404, tool="wikipedia")
    assert error.tool == "wikipedia"
    assert "404" in str(error)


def test_server_status_becomes_transient() -> None:
    with pytest.raises(TransientToolError), http_errors("wikipedia"):
        raise status_error(503)


def test_missing_page_becomes_permanent() -> None:
    with pytest.raises(PermanentToolError), http_errors("wikipedia"):
        raise status_error(404)


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ConnectError("connection refused"),
    ],
)
def test_transport_failures_are_transient(exc: httpx.TransportError) -> None:
    """Timeouts carry no status code, so they need translating separately."""
    with pytest.raises(TransientToolError, match="scraper"), http_errors("scraper"):
        raise exc


def test_unrelated_exceptions_pass_through() -> None:
    """A bug in our own parsing must not be disguised as a retryable tool failure."""
    with pytest.raises(KeyError), http_errors("tavily"):
        raise KeyError("results")


def test_success_path_is_untouched() -> None:
    with http_errors("tavily"):
        value = 1
    assert value == 1


async def test_a_tool_satisfies_the_protocol() -> None:
    tool: Tool = FakeTool()
    results = await tool.search("quantum computing")
    assert results[0]["title"] == "quantum computing"
