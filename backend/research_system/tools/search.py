"""Tavily web search — the primary tool.

Calls the Tavily REST API over httpx rather than going through
`langchain_tavily.TavilySearch`, which returns `{"error": exc}` instead of
raising, raises when a search legitimately finds nothing, and issues its
requests with aiohttp on a timeout of its own. All three fight the contract in
`tools.base`.
"""

import httpx
from pydantic import SecretStr

from research_system.domain.state import SearchResult
from research_system.settings import SearchSettings
from research_system.tools.base import http_errors, make_result

API_URL = "https://api.tavily.com/search"


class TavilyTool:
    """General web search across the open internet."""

    name = "tavily"

    def __init__(self, settings: SearchSettings, api_key: SecretStr) -> None:
        self._settings = settings
        self._api_key = api_key

    async def search(self, query: str) -> list[SearchResult]:
        payload = {"query": query, "max_results": self._settings.results_per_query}
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        with http_errors(self.name):
            async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
                response = await client.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
        # `published_date` is only present for the news topic; absent elsewhere.
        return [
            make_result(
                title=item["title"],
                url=item["url"],
                snippet=item["content"],
                date=item.get("published_date", ""),
                max_chars=self._settings.snippet_max_chars,
            )
            for item in response.json()["results"]
        ]
