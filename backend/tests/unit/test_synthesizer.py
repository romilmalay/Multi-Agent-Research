"""Synthesizer tests. No network: `get_model` is replaced with a stub."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from research_system.agents import synthesizer as synthesizer_module
from research_system.agents.synthesizer import synthesise
from research_system.domain.schemas import SynthesizerOutput
from research_system.domain.state import ResearchState, default_state
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


def claim(text: str, source_idx: int = 0, confidence: float = 0.9) -> dict[str, Any]:
    """One entry as the analyst leaves it in `key_claims`: a plain dict."""
    return {
        "claim": text,
        "evidence": f"quoted support for {text}",
        "source_idx": source_idx,
        "confidence": confidence,
    }


def state_with(claims: list[dict[str, Any]]) -> ResearchState:
    state = default_state(QUERY)
    state["key_claims"] = claims
    return state


def raw(input_tokens: int = 700, output_tokens: int = 300) -> AIMessage:
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def structured_response(parsed: SynthesizerOutput | None, **kwargs: Any) -> dict[str, Any]:
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
    """Records what the synthesizer asked for, and answers with a canned response."""

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
    holder: dict[str, Any] = {"calls": 0}

    def fake_get_model(agent: str, *, settings: Settings | None = None) -> FakeModel:
        holder["calls"] += 1
        built = FakeModel(holder["response"])
        holder["model"] = built
        return built

    monkeypatch.setattr(synthesizer_module, "get_model", fake_get_model)
    return holder


def answers(model: dict[str, Any], response: Any) -> None:
    model["response"] = response


async def test_the_synthesis_and_conflicts_reach_state(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(
            SynthesizerOutput(
                synthesis="Both studies find ingestion in wild fish [0][1].",
                conflicts=["Source 0 reports 38% ingestion, source 1 reports 4% [0] vs [1]."],
            )
        ),
    )
    update = await synthesise(
        state_with([claim("38% of fish contained particles"), claim("4% of fish did", 1)]),
        settings=Settings(),
    )

    assert update["synthesis"] == "Both studies find ingestion in wild fish [0][1]."
    assert update["conflicts"] == [
        "Source 0 reports 38% ingestion, source 1 reports 4% [0] vs [1]."
    ]
    assert update["errors"] == []


async def test_agreeing_claims_produce_no_conflicts(model: dict[str, Any]) -> None:
    """An empty conflicts list is the normal answer, not a missing one."""
    answers(
        model,
        structured_response(
            SynthesizerOutput(synthesis="The sources agree on ingestion [0].", conflicts=[])
        ),
    )
    update = await synthesise(
        state_with([claim("38% of fish contained particles")]), settings=Settings()
    )

    assert update["conflicts"] == []
    assert update["errors"] == []


async def test_the_claims_are_rendered_with_their_source_numbers(model: dict[str, Any]) -> None:
    """The citation number has to survive into the synthesis, so it goes into the prompt."""
    answers(model, structured_response(SynthesizerOutput(synthesis="s", conflicts=[])))
    await synthesise(
        state_with([claim("first finding", 0), claim("second finding", 2, confidence=0.4)]),
        settings=Settings(),
    )

    _system, human = model["model"].structured.messages[0]
    assert QUERY in human.content
    assert "[0] first finding" in human.content
    assert "[2] second finding" in human.content
    assert "evidence: quoted support for second finding" in human.content
    assert "confidence: 0.4" in human.content


async def test_structured_output_is_requested_with_the_raw_message(model: dict[str, Any]) -> None:
    """Without `include_raw` the token counts are discarded with the message."""
    answers(model, structured_response(SynthesizerOutput(synthesis="s", conflicts=[])))
    await synthesise(state_with([claim("a finding")]), settings=Settings())

    assert model["model"].schema is SynthesizerOutput
    assert model["model"].include_raw is True


async def test_no_claims_means_no_model_call(model: dict[str, Any]) -> None:
    """Nothing to cross-reference: an invented narrative is not worth paying for."""
    answers(model, structured_response(SynthesizerOutput(synthesis="invented", conflicts=[])))
    update = await synthesise(state_with([]), settings=Settings())

    assert model["calls"] == 0
    assert update["synthesis"] == ""
    assert update["conflicts"] == []
    assert update["token_count"] == 0
    assert update["errors"] == []


async def test_an_llm_exception_leaves_an_empty_synthesis_and_one_error(
    model: dict[str, Any],
) -> None:
    """The writer still has the claims, so the run continues without a synthesis."""
    answers(model, RuntimeError("provider is down"))
    update = await synthesise(state_with([claim("a finding")]), settings=Settings())

    assert update["synthesis"] == ""
    assert update["conflicts"] == []
    assert len(update["errors"]) == 1
    assert "provider is down" in update["errors"][0]


async def test_an_unparsable_answer_still_counts_its_tokens(model: dict[str, Any]) -> None:
    """A schema violation is billed like any other call."""
    answers(model, structured_response(None, parsing_error=ValueError("conflicts is not a list")))
    update = await synthesise(state_with([claim("a finding")]), settings=Settings())

    assert update["synthesis"] == ""
    assert "conflicts is not a list" in update["errors"][0]
    assert update["token_count"] == 1_000


async def test_tokens_come_from_the_response(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(
            SynthesizerOutput(synthesis="s", conflicts=[]),
            message=raw(input_tokens=2_000, output_tokens=500),
        ),
    )
    update = await synthesise(state_with([claim("a finding")]), settings=Settings())

    assert update["token_count"] == 2_500


async def test_the_trace_entry_has_the_agreed_shape(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(
            SynthesizerOutput(synthesis="s", conflicts=["one real disagreement"]),
            message=raw(input_tokens=700, output_tokens=300),
        ),
    )
    update = await synthesise(state_with([claim("first"), claim("second", 1)]), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "synthesizer"
    assert entry["tokens"] == 1_000
    assert entry["summary"] == "2 claims cross-referenced, 1 conflicts"
    assert entry["prompt_hash"] == load("synthesizer").hash
    assert entry["duration_ms"] >= 0.0


async def test_a_failure_is_still_traced(model: dict[str, Any]) -> None:
    """A node that failed still has to account for the time it spent."""
    answers(model, RuntimeError("provider is down"))
    update = await synthesise(state_with([claim("a finding")]), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "synthesizer"
    assert entry["summary"] == "failed"


async def test_a_skipped_run_is_traced_without_a_prompt_hash(model: dict[str, Any]) -> None:
    """No prompt was loaded, so there is no prompt to name."""
    answers(model, structured_response(SynthesizerOutput(synthesis="s", conflicts=[])))
    update = await synthesise(state_with([]), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["summary"] == "skipped: no claims"
    assert entry["prompt_hash"] == ""
