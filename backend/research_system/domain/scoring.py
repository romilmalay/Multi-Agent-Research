"""How much a source is worth, judged without an LLM.

The quality gate is the one place in the pipeline that decides whether the
research is good enough to write from, so it deliberately uses no model: a
scorer that hallucinates is worse than no scorer, and the judgement here is
cheap enough to make in pure Python. Two things can be computed about a search
result without reading it - where it came from, and what its snippet looks
like - so those are the two signals.

`domain_trust` is where it came from. The numbers are a ranking, not a
measurement. What matters is the order - peer-reviewed above encyclopedic,
encyclopedic above forum - and the gap between bands, which has to be wide
enough that a strong snippet on a weak domain cannot outrank a weak snippet on
a strong one.

An unrecognised domain scores `UNKNOWN_DOMAIN_TRUST`, mid-scale. Zero would
mean the table is a whitelist, and a run whose sources are all specialist
sites nobody thought to list would fail the gate for no real reason; scoring
them high would make the table pointless. Mid-scale says only what is true:
nothing is known about this host, so the snippet decides.

`snippet_score` is what the text looks like. It cannot tell whether a claim is
true - nothing here reads for meaning - so it measures the one thing surface
text does reveal: whether the source is *saying something specific*. Length,
figures, and research vocabulary all separate "we studied 1,200 patients over
18 months and found a 12% reduction" from "Everything you need to know about
health. Read more." Both are grammatical English; only one carries content.

Short snippets scale the whole score down. Every signal below is a count or a
density, and a twelve-character string can max one out on a single token, so a
snippet too short to judge is not allowed to score as if it had been judged.

`source_score` weights the two together and `aggregate_score` reduces a whole
run to one number, over the best few sources rather than all of them - the
report is written from the best material, so that is what the gate should
judge.
"""

import re
from urllib.parse import urlparse

from research_system.domain.state import SearchResult

DOMAIN_TRUST = {
    # Peer review, or a preprint archive that feeds it.
    "arxiv.org": 0.95,
    "nature.com": 0.95,
    "science.org": 0.95,
    "pubmed.ncbi.nlm.nih.gov": 0.95,
    "acm.org": 0.9,
    "ieee.org": 0.9,
    "nih.gov": 0.9,
    "who.int": 0.9,
    # Primary sources: the org itself, the code itself, the wire.
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "github.com": 0.8,
    "openai.com": 0.8,
    "anthropic.com": 0.8,
    # Edited, but secondhand.
    "bbc.com": 0.75,
    "nytimes.com": 0.75,
    "economist.com": 0.75,
    "wikipedia.org": 0.7,
    # Community answers: often right, never reviewed.
    "stackoverflow.com": 0.65,
    "techcrunch.com": 0.6,
    "substack.com": 0.45,
    "medium.com": 0.4,
    "reddit.com": 0.3,
    "quora.com": 0.3,
}

UNKNOWN_DOMAIN_TRUST = 0.5
"""Score for a host the table has never heard of. See the module docstring."""


def domain_trust(url: str) -> float:
    """Trust score for the host in `url`, matched on whole domain labels.

    Sub-domains inherit their parent's score, so `export.arxiv.org` scores as
    `arxiv.org`, while `notreddit.com` matches nothing and stays unknown.
    """
    host = urlparse(url).netloc.lower().partition(":")[0]
    labels = host.split(".")
    for i in range(len(labels) - 1):
        score = DOMAIN_TRUST.get(".".join(labels[i:]))
        if score is not None:
            return score
    return UNKNOWN_DOMAIN_TRUST


# Genre markers, not jargon: the words research prose uses when it reports a
# finding. A subject-specific term list would only score the subjects someone
# happened to list, and would rate a paper on medicine below one on transformers.
TECHNICAL_TERMS = frozenset(
    {
        "according",
        "analysis",
        "average",
        "baseline",
        "benchmark",
        "compared",
        "correlation",
        "data",
        "dataset",
        "estimated",
        "evaluation",
        "evidence",
        "experiment",
        "experiments",
        "findings",
        "journal",
        "measured",
        "median",
        "method",
        "methods",
        "participants",
        "peer-reviewed",
        "percent",
        "published",
        "reported",
        "research",
        "researchers",
        "results",
        "sample",
        "significant",
        "statistically",
        "studies",
        "study",
        "survey",
        "trial",
    }
)

# Snippet length in characters, and what each band is worth. A search API caps
# snippets around 500 chars, so 300 is a full-length one, not an outlier.
LENGTH_BANDS = ((300, 1.0), (150, 0.75), (60, 0.4))
SHORTEST_BAND = 0.1

# Three figures is a snippet reporting a result, not one that mentions a year.
NUMERIC_TOKENS_FOR_FULL = 3

# One word in twenty. Higher would demand denser prose than a real abstract has.
TECHNICAL_DENSITY_FOR_FULL = 0.05

# Below this, the counts above are too easy to max out to be believed.
SHORT_SNIPPET_CHARS = 80

LENGTH_WEIGHT = 0.4
NUMERIC_WEIGHT = 0.3
TECHNICAL_WEIGHT = 0.3

_HAS_DIGIT = re.compile(r"\d")
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")  # internal hyphens only, so a dash is not a word


def _length_score(length: int) -> float:
    """Which length band the snippet falls in, 0.1 to 1.0."""
    for minimum, score in LENGTH_BANDS:
        if length >= minimum:
            return score
    return SHORTEST_BAND


def _numeric_score(snippet: str) -> float:
    """How much of the snippet is figures: percentages, counts, dates, sizes."""
    hits = sum(1 for token in snippet.split() if _HAS_DIGIT.search(token))
    return min(hits / NUMERIC_TOKENS_FOR_FULL, 1.0)


def _technical_score(snippet: str) -> float:
    """Share of words that belong to research prose, normalised to a full mark."""
    words = _WORD.findall(snippet.lower())
    if not words:
        return 0.0
    density = sum(1 for word in words if word in TECHNICAL_TERMS) / len(words)
    return min(density / TECHNICAL_DENSITY_FOR_FULL, 1.0)


def snippet_score(snippet: str) -> float:
    """How substantial a source's text looks, 0.0-1.0, judged on surface signals only.

    Length, figures and research vocabulary are weighted together, then scaled
    down for a snippet too short for those signals to mean anything.
    """
    text = snippet.strip()
    score = (
        LENGTH_WEIGHT * _length_score(len(text))
        + NUMERIC_WEIGHT * _numeric_score(text)
        + TECHNICAL_WEIGHT * _technical_score(text)
    )
    short_penalty = min(len(text) / SHORT_SNIPPET_CHARS, 1.0)
    return score * short_penalty


# Where a source was published outweighs how its preview reads, because the
# publisher is the harder signal to fake: anyone can write a snippet full of
# figures, nobody can publish on arxiv.org by choosing to.
DOMAIN_WEIGHT = 0.6
SNIPPET_WEIGHT = 0.4

# How many sources the aggregate is taken over. See `aggregate_score`.
TOP_SOURCES = 5


def source_score(source: SearchResult) -> float:
    """One source's overall worth, 0.0-1.0: where it came from, and how it reads."""
    return DOMAIN_WEIGHT * domain_trust(source["url"]) + SNIPPET_WEIGHT * snippet_score(
        source["snippet"]
    )


def aggregate_score(scores: list[float]) -> float:
    """The run's quality: the mean of its best `TOP_SOURCES` scores, or 0.0 with none.

    Only the best few count because that is what the report is written from. A
    search returning five strong sources and ten weak ones is a good search, and
    a mean over all fifteen would call it a failure and trigger a pointless retry.
    Taking the top five instead asks the question that matters: is there enough
    good material here to write from?
    """
    if not scores:
        return 0.0
    best = sorted(scores, reverse=True)[:TOP_SOURCES]
    return sum(best) / len(best)
