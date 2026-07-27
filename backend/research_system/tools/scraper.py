"""Page scraper — the tool for when the query names a URL.

Every tool is handed the same raw query, so this one finds the URL inside it.
One page in, one result out; there is nothing to search.
"""

import re

import httpx
from bs4 import BeautifulSoup

from research_system.domain.state import SearchResult
from research_system.errors import PermanentToolError
from research_system.settings import SearchSettings
from research_system.tools.base import USER_AGENT, http_errors, make_result

# Trailing punctuation is far more often a sentence ending than part of the URL.
_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+[^\s<>\".,;:!?)\]]")

# Chrome and navigation text would otherwise dominate a truncated snippet.
_NON_CONTENT_TAGS = ("script", "style", "nav", "header", "footer", "aside")


def find_url(text: str) -> str | None:
    """The first URL in `text`, if it names one."""
    match = _URL_PATTERN.search(text)
    return match.group() if match else None


class ScraperTool:
    """Reads one page the user pointed at."""

    name = "scraper"

    def __init__(self, settings: SearchSettings) -> None:
        self._settings = settings

    async def search(self, query: str) -> list[SearchResult]:
        """Read the page whose URL appears in `query`."""
        url = find_url(query)
        if url is None:
            # Permanent: rerunning the same query will not grow a URL.
            raise PermanentToolError(f"no URL in {query!r}", tool=self.name)
        with http_errors(self.name):
            async with httpx.AsyncClient(
                timeout=self._settings.request_timeout_seconds,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(_NON_CONTENT_TAGS):
            tag.decompose()
        # Body only: the head carries the title, and repeating it in the snippet
        # would spend the character budget on something already recorded.
        body = soup.body or soup
        return [
            make_result(
                title=soup.title.get_text() if soup.title else url,
                url=url,
                # Markup leaves ragged whitespace; collapse it before truncating.
                snippet=" ".join(body.get_text(" ").split()),
                max_chars=self._settings.snippet_max_chars,
            )
        ]
