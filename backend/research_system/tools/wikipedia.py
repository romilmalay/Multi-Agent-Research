"""Wikipedia — the tool for definitional questions.

One request does the whole job: `generator=search` runs the search and
`prop=extracts` returns each hit's lead paragraph as plain text, so a definition
arrives as prose rather than as the fragmented match highlights the plain search
endpoint returns.
"""

import httpx

from research_system.domain.state import SearchResult
from research_system.settings import SearchSettings
from research_system.tools.base import USER_AGENT, http_errors, make_result

API_URL = "https://en.wikipedia.org/w/api.php"


class WikipediaTool:
    """Encyclopaedic background, one lead paragraph per matching article."""

    name = "wikipedia"

    def __init__(self, settings: SearchSettings) -> None:
        self._settings = settings

    async def search(self, query: str) -> list[SearchResult]:
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": self._settings.results_per_query,
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
        }
        with http_errors(self.name):
            async with httpx.AsyncClient(
                timeout=self._settings.request_timeout_seconds,
                headers={"User-Agent": USER_AGENT},
            ) as client:
                response = await client.get(API_URL, params=params)
                response.raise_for_status()
        # A query matching no article comes back without a "query" block at all.
        pages = response.json().get("query", {}).get("pages", {}).values()
        return [
            make_result(
                title=page["title"],
                url=page["fullurl"],
                # Redirects and disambiguation pages carry no lead paragraph.
                snippet=page.get("extract", ""),
                date=page["touched"][:10],
                max_chars=self._settings.snippet_max_chars,
            )
            # `pages` is keyed by page id, so relevance order lives in `index`.
            for page in sorted(pages, key=lambda page: page["index"])
        ]
