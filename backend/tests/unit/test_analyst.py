"""Analyst tests. No network: `get_model` is replaced with a stub."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from research_system.agents import analyst as analyst_module
from research_system.agents.analyst import analyse
from research_system.domain.schemas import AnalystOutput, ClaimOutput
from research_system.domain.state import ResearchState, SearchResult, default_state
from research_system.prompts import load
from research_system.settings import Settings

QUERY = "What are the effects of microplastics on marine life?"
TRACE_KEYS = {
    "agent",
    "duration_ms",
    "tokens",
    "input_tokens",
    "output_tokens",
    "summary",
    "prompt_hash",
}


def source(
    url: str, title: str = "A study", snippet: str = "Fish ingest particles."
) -> SearchResult:
    return SearchResult(title=title, snippet=snippet, url=url, date="2024-01-01")


def state_with(sources: list[SearchResult]) -> ResearchState:
    state = default_state(QUERY)
    state["sources"] = sources
    return state


def claim(text: str = "Fish ingest microplastics.", source_idx: int = 0) -> ClaimOutput:
    return ClaimOutput(
        claim=text,
        evidence="Fish ingest particles.",
        source_idx=source_idx,
        confidence=0.9,
    )


def raw(input_tokens: int = 900, output_tokens: int = 200) -> AIMessage:
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def structured_response(parsed: AnalystOutput | None, **kwargs: Any) -> dict[str, Any]:
    """What `with_structured_output(..., include_raw=True)` hands back."""
    return {
        "raw": kwargs.pop("message", raw()),
        "parsed": parsed,
        "parsing_error": kwargs.pop("parsing_error", None),
    }


class FakeStructuredModel:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.messages: list[Any] = []

    async def ainvoke(self, messages: Any) -> Any:
        self.messages.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeModel:
    """Records what the analyst asked for, and answers with a canned response."""

    def __init__(self, response: Any) -> None:
        self.structured = FakeStructuredModel(response)
        self.schema: Any = None
        self.include_raw: bool | None = None

    def with_structured_output(self, schema: Any, include_raw: bool = False) -> FakeStructuredModel:
        self.schema, self.include_raw = schema, include_raw
        return self.structured


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a stub model whose response each test sets."""
    holder: dict[str, FakeModel] = {}

    def fake_get_model(agent: str, *, settings: Settings | None = None) -> FakeModel:
        holder["model"] = FakeModel(holder["response"])
        return holder["model"]

    monkeypatch.setattr(analyst_module, "get_model", fake_get_model)
    return holder


def answers(model: dict[str, Any], response: Any) -> None:
    model["response"] = response


async def test_claims_reach_state_as_plain_dicts(model: dict[str, Any]) -> None:
    answers(model, structured_response(AnalystOutput(claims=[claim(), claim("Corals bleach.", 1)])))
    update = await analyse(
        state_with([source("https://a.org"), source("https://b.org")]), settings=Settings()
    )

    assert update["key_claims"] == [
        {
            "claim": "Fish ingest microplastics.",
            "evidence": "Fish ingest particles.",
            "source_idx": 0,
            "confidence": 0.9,
        },
        {
            "claim": "Corals bleach.",
            "evidence": "Fish ingest particles.",
            "source_idx": 1,
            "confidence": 0.9,
        },
    ]
    assert update["errors"] == []


async def test_the_sources_are_numbered_by_their_position_in_state(model: dict[str, Any]) -> None:
    """The number the model cites is the index a citation is later checked against."""
    answers(model, structured_response(AnalystOutput(claims=[claim()])))
    await analyse(
        state_with(
            [source("https://a.org", title="First"), source("https://b.org", title="Second")]
        ),
        settings=Settings(),
    )

    _system, human = model["model"].structured.messages[0]
    assert "[0] First" in human.content
    assert "[1] Second" in human.content
    assert "https://b.org" in human.content


async def test_the_query_and_the_claim_range_are_rendered_into_the_prompt(
    model: dict[str, Any],
) -> None:
    answers(model, structured_response(AnalystOutput(claims=[claim()])))
    await analyse(state_with([source("https://a.org")]), settings=Settings())

    system, human = model["model"].structured.messages[0]
    assert QUERY in human.content
    assert "between 5 and 8 claims" in system.content


async def test_structured_output_is_requested_with_the_raw_message(model: dict[str, Any]) -> None:
    """Without `include_raw` the token counts are discarded with the message."""
    answers(model, structured_response(AnalystOutput(claims=[claim()])))
    await analyse(state_with([source("https://a.org")]), settings=Settings())

    assert model["model"].schema is AnalystOutput
    assert model["model"].include_raw is True


async def test_too_many_claims_are_clamped_to_the_configured_maximum(model: dict[str, Any]) -> None:
    """The prompt asks for a maximum; this is the enforcement."""
    answers(
        model,
        structured_response(AnalystOutput(claims=[claim(f"claim {i}") for i in range(12)])),
    )
    update = await analyse(state_with([source("https://a.org")]), settings=Settings())

    assert len(update["key_claims"]) == 8
    assert update["key_claims"][0]["claim"] == "claim 0"


async def test_a_lower_configured_maximum_clamps_further(
    model: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTS__ANALYST__MIN_CLAIMS", "1")
    monkeypatch.setenv("AGENTS__ANALYST__MAX_CLAIMS", "2")
    answers(
        model,
        structured_response(AnalystOutput(claims=[claim(f"claim {i}") for i in range(6)])),
    )
    update = await analyse(state_with([source("https://a.org")]), settings=Settings())

    assert [entry["claim"] for entry in update["key_claims"]] == ["claim 0", "claim 1"]


async def test_a_claim_citing_a_source_that_does_not_exist_is_dropped(
    model: dict[str, Any],
) -> None:
    """Two sources were given, so source 7 is invented and the claim has no evidence."""
    answers(
        model,
        structured_response(
            AnalystOutput(claims=[claim("Fish ingest microplastics.", 0), claim("Invented.", 7)])
        ),
    )
    update = await analyse(
        state_with([source("https://a.org"), source("https://b.org")]), settings=Settings()
    )

    assert [entry["claim"] for entry in update["key_claims"]] == ["Fish ingest microplastics."]
    assert len(update["errors"]) == 1
    assert "dropped claim citing source 7" in update["errors"][0]
    assert "Invented." in update["errors"][0]


async def test_citations_are_validated_before_the_clamp(model: dict[str, Any]) -> None:
    """A hallucinated citation must not take a slot from a claim that has evidence."""
    answers(
        model,
        structured_response(
            AnalystOutput(
                claims=[claim(f"claim {i}", 7) for i in range(4)]
                + [claim(f"claim {i}", 0) for i in range(4, 12)]
            )
        ),
    )
    update = await analyse(state_with([source("https://a.org")]), settings=Settings())

    assert [entry["claim"] for entry in update["key_claims"]] == [
        f"claim {i}" for i in range(4, 12)
    ]
    assert len(update["errors"]) == 4


async def test_every_citation_resolving_records_no_error(model: dict[str, Any]) -> None:
    answers(model, structured_response(AnalystOutput(claims=[claim("Fish ingest.", 1)])))
    update = await analyse(
        state_with([source("https://a.org"), source("https://b.org")]), settings=Settings()
    )

    assert len(update["key_claims"]) == 1
    assert update["errors"] == []


async def test_an_llm_exception_yields_no_claims_and_one_error(model: dict[str, Any]) -> None:
    answers(model, RuntimeError("provider is down"))
    update = await analyse(state_with([source("https://a.org")]), settings=Settings())

    assert update["key_claims"] == []
    assert len(update["errors"]) == 1
    assert "provider is down" in update["errors"][0]


async def test_an_unparsable_answer_still_counts_its_tokens(model: dict[str, Any]) -> None:
    """A schema violation is billed like any other call."""
    answers(model, structured_response(None, parsing_error=ValueError("claims is not a list")))
    update = await analyse(state_with([source("https://a.org")]), settings=Settings())

    assert update["key_claims"] == []
    assert "claims is not a list" in update["errors"][0]
    assert update["token_count"] == 1_100


async def test_tokens_come_from_the_response(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(
            AnalystOutput(claims=[claim()]), message=raw(input_tokens=3_000, output_tokens=400)
        ),
    )
    update = await analyse(state_with([source("https://a.org")]), settings=Settings())

    assert update["token_count"] == 3_400


async def test_a_failed_call_reports_no_tokens(model: dict[str, Any]) -> None:
    answers(model, RuntimeError("timeout"))
    update = await analyse(state_with([source("https://a.org")]), settings=Settings())

    assert update["token_count"] == 0


async def test_the_trace_entry_has_the_agreed_shape(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(
            AnalystOutput(claims=[claim(), claim("Corals bleach.", 1)]),
            message=raw(input_tokens=900, output_tokens=200),
        ),
    )
    update = await analyse(
        state_with([source("https://a.org"), source("https://b.org")]), settings=Settings()
    )

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "analyst"
    assert entry["tokens"] == 1_100
    assert entry["summary"] == "2 claims from 2 sources"
    assert entry["prompt_hash"] == load("analyst").hash
    assert entry["duration_ms"] >= 0.0


async def test_a_failure_is_still_traced(model: dict[str, Any]) -> None:
    """A node that failed still has to account for the time it spent."""
    answers(model, RuntimeError("provider is down"))
    update = await analyse(state_with([source("https://a.org")]), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "analyst"
    assert entry["summary"] == "0 claims from 1 sources"
