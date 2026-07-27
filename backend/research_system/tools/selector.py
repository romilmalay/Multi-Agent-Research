"""Which tools to run for a given query.

A pure function of the query text: names in, names out. Building the tools needs
an API key and a settings object, and that is the researcher's job — keeping it
out of here is what makes every routing rule testable with a one-line string.

Tavily is always primary, so the cost of a wrong guess is a missing supplement,
never a missing search.
"""

import re

from research_system.tools.scraper import find_url

TAVILY = "tavily"
WIKIPEDIA = "wikipedia"
SCRAPER = "scraper"

# "What is X" wants a definition, and an encyclopaedia answers that better than
# a page of search results.
_DEFINITIONAL = re.compile(
    r"\b(what (is|are|was|were)|who (is|are|was|were)|define|definition of|meaning of)\b"
)

# Wikipedia lags on anything current, so these send the query to the web alone.
_RECENCY = re.compile(r"\b(latest|newest|recent|current|today|this (year|month|week)|20[2-9]\d)\b")


def select_tools(query: str) -> list[str]:
    """Tool names to run for `query`, primary first.

    Rules stack: a query can name a URL *and* ask for a definition, and both
    tools then run alongside the web search.
    """
    text = query.lower()
    tools = [TAVILY]
    if find_url(query):
        tools.append(SCRAPER)
    if _DEFINITIONAL.search(text) and not _RECENCY.search(text):
        tools.append(WIKIPEDIA)
    return tools
