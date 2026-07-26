"""The five agent prompts: variable contract, shape, and rendered structure."""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from research_system.prompts import registry

CLAIMS = [
    {
        "claim": "Costs fell 40%.",
        "evidence": "costs fell 40% YoY",
        "source_idx": 1,
        "confidence": 0.9,
    },
    {"claim": "Adoption doubled.", "evidence": "2x users", "source_idx": 2, "confidence": 0.6},
]

SOURCES = [
    {
        "idx": 1,
        "title": "Report & Review",
        "url": "https://arxiv.org/abs/1",
        "date": "2026-01-02",
        "snippet": 'costs fell 40% YoY, "materially" so',
    },
    {
        "idx": 2,
        "title": "Blog <b>",
        "url": "https://example.com/p",
        "date": "",
        "snippet": "2x users",
    },
]

# Each prompt's full variable contract. The template may use no more and no less.
FIXTURES: dict[str, dict[str, Any]] = {
    "planner": {"query": "state of LLM agents", "max_sub_topics": 3},
    "analyst": {"query": "q", "min_claims": 5, "max_claims": 8, "sources": SOURCES},
    "synthesizer": {"query": "q", "claims": CLAIMS},
    "writer": {"query": "q", "synthesis": "S", "claims": CLAIMS, "conflicts": [], "revision": None},
    "reviewer": {"query": "q", "pass_score": 7, "claims": CLAIMS, "draft": "D"},
}


@pytest.fixture(params=sorted(FIXTURES))
def name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_prompt_declares_exactly_the_variables_its_fixture_supplies(name: str) -> None:
    """Catches a template typo and an unused fixture key from the same assertion."""
    assert registry.load(name).variables == set(FIXTURES[name])


def test_prompt_renders_a_system_human_pair_with_content(name: str) -> None:
    messages = registry.load(name).render(**FIXTURES[name])

    assert [type(m) for m in messages] == [SystemMessage, HumanMessage]
    assert all(str(m.content).strip() for m in messages)


def test_no_mustache_delimiters_survive_rendering(name: str) -> None:
    rendered = "".join(str(m.content) for m in registry.load(name).render(**FIXTURES[name]))

    assert "{{" not in rendered and "}}" not in rendered


def test_prompts_never_html_escape_source_text() -> None:
    """`{{var}}` would turn quotes and ampersands into entities; `{{{var}}}` must not."""
    _, human = registry.load("analyst").render(**FIXTURES["analyst"])

    assert '"materially"' in human.content
    assert "Report & Review" in human.content
    assert "&amp;" not in human.content
    assert "&quot;" not in human.content


def test_analyst_numbers_every_source_for_unambiguous_citation() -> None:
    _, human = registry.load("analyst").render(**FIXTURES["analyst"])

    assert "[1] Report & Review" in human.content
    assert "[2] Blog <b>" in human.content
    assert "https://arxiv.org/abs/1" in human.content


def test_writer_lists_claims_by_their_source_number() -> None:
    _, human = registry.load("writer").render(**FIXTURES["writer"])

    assert "[1] Costs fell 40%." in human.content
    assert "[2] Adoption doubled." in human.content


def test_writer_omits_the_revision_block_on_a_first_draft() -> None:
    _, human = registry.load("writer").render(**FIXTURES["writer"])

    assert "This is a revision" not in human.content
    assert "Problems to fix" not in human.content


SCORE = 5
ISSUES = ["Claim 2 is uncited.", "Summary buries the finding."]
SUGGESTIONS = ["Cite source 2 or cut it.", "Lead with the 40% drop."]
PREVIOUS_DRAFT = "# Findings\nAdoption doubled.\n"

REVISION: dict[str, Any] = {
    "score": SCORE,
    "issues": ISSUES,
    "suggestions": SUGGESTIONS,
    "previous_draft": PREVIOUS_DRAFT,
}


def test_writer_revision_block_carries_the_reviewer_verdict() -> None:
    _, human = registry.load("writer").render(**(FIXTURES["writer"] | {"revision": REVISION}))

    assert f"previous draft scored {SCORE} out of 10" in human.content
    for text in ISSUES + SUGGESTIONS:
        assert f"- {text}" in human.content


def test_writer_revision_block_shows_the_draft_being_revised() -> None:
    """Reviewer issues quote the draft, so the writer has to be able to see it."""
    _, human = registry.load("writer").render(**(FIXTURES["writer"] | {"revision": REVISION}))

    assert PREVIOUS_DRAFT in human.content


def test_writer_issues_and_suggestions_stay_paired_in_order() -> None:
    """The reviewer promises fix i answers issue i; scrambling that breaks the contract."""
    _, human = registry.load("writer").render(**(FIXTURES["writer"] | {"revision": REVISION}))
    positions = [str(human.content).index(text) for text in ISSUES + SUGGESTIONS]

    assert positions == sorted(positions)


def test_writer_reports_conflicts_only_when_there_are_some() -> None:
    without = registry.load("writer").render(**FIXTURES["writer"])[1]
    assert "Sources disagree" not in without.content

    conflicts = ["[1] says costs fell while [2] says they rose."]
    with_them = registry.load("writer").render(**(FIXTURES["writer"] | {"conflicts": conflicts}))[1]
    assert f"Sources disagree, report it: {conflicts[0]}" in with_them.content


def test_synthesizer_shows_evidence_and_confidence_per_claim() -> None:
    _, human = registry.load("synthesizer").render(**FIXTURES["synthesizer"])

    assert "[1] Costs fell 40%." in human.content
    assert "evidence: costs fell 40% YoY" in human.content
    assert "confidence: 0.9" in human.content


def test_reviewer_sees_the_draft_and_the_claims_behind_it() -> None:
    _, human = registry.load("reviewer").render(**FIXTURES["reviewer"])

    assert "Draft under review:" in human.content
    assert "[2] Adoption doubled." in human.content


def test_each_prompt_has_its_own_hash() -> None:
    hashes = {n: registry.load(n).hash for n in FIXTURES}

    assert len(set(hashes.values())) == len(FIXTURES)
