"""Planner tests. No network: `get_model` is replaced with a stub."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from research_system.agents import planner as planner_module
from research_system.agents.planner import FALLBACK_PLAN, plan
from research_system.domain.schemas import PlannerOutput
from research_system.domain.state import default_state
from research_system.prompts import load
from research_system.settings import Settings

QUERY = "What are the effects of microplastics on marine life?"
TRACE_KEYS = {"agent", "duration_ms", "tokens", "summary", "prompt_hash"}


def raw(input_tokens: int = 300, output_tokens: int = 60) -> AIMessage:
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def structured_response(parsed: PlannerOutput | None, **kwargs: Any) -> dict[str, Any]:
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
    """Records what the planner asked for, and answers with a canned response."""

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

    monkeypatch.setattr(planner_module, "get_model", fake_get_model)
    return holder


def answers(model: dict[str, Any], response: Any) -> None:
    model["response"] = response


async def test_a_good_plan_reaches_state(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(
            PlannerOutput(
                sub_topics=["How do microplastics enter the ocean?", "What do they do to fish?"],
                research_plan="Sources and effects are answered by different literatures.",
            )
        ),
    )
    update = await plan(default_state(QUERY), settings=Settings())

    assert update["sub_topics"] == [
        "How do microplastics enter the ocean?",
        "What do they do to fish?",
    ]
    assert update["research_plan"].startswith("Sources and effects")
    assert update["errors"] == []


async def test_the_query_and_the_cap_are_rendered_into_the_prompt(model: dict[str, Any]) -> None:
    answers(model, structured_response(PlannerOutput(sub_topics=[QUERY], research_plan="one")))
    await plan(default_state(QUERY), settings=Settings())

    system, human = model["model"].structured.messages[0]
    assert QUERY in human.content
    assert "3" in system.content


async def test_structured_output_is_requested_with_the_raw_message(model: dict[str, Any]) -> None:
    """Without `include_raw` the token counts are discarded with the message."""
    answers(model, structured_response(PlannerOutput(sub_topics=[QUERY], research_plan="one")))
    await plan(default_state(QUERY), settings=Settings())

    assert model["model"].schema is PlannerOutput
    assert model["model"].include_raw is True


async def test_five_sub_topics_are_clamped_to_three(model: dict[str, Any]) -> None:
    """The schema caps at 3; this is the clamp for when a model gets past it."""
    over = PlannerOutput.model_construct(
        sub_topics=[f"sub-question {i}" for i in range(5)],
        research_plan="five ways to split it",
    )
    answers(model, structured_response(over))
    update = await plan(default_state(QUERY), settings=Settings())

    assert update["sub_topics"] == ["sub-question 0", "sub-question 1", "sub-question 2"]


async def test_a_lower_configured_cap_clamps_further(
    model: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIPELINE__MAX_SUB_TOPICS", "1")
    answers(
        model,
        structured_response(
            PlannerOutput(sub_topics=["first", "second", "third"], research_plan="three ways")
        ),
    )
    update = await plan(default_state(QUERY), settings=Settings())

    assert update["sub_topics"] == ["first"]


async def test_an_llm_exception_falls_back_to_the_query(model: dict[str, Any]) -> None:
    answers(model, RuntimeError("provider is down"))
    update = await plan(default_state(QUERY), settings=Settings())

    assert update["sub_topics"] == [QUERY]
    assert update["research_plan"] == FALLBACK_PLAN
    assert len(update["errors"]) == 1
    assert "provider is down" in update["errors"][0]


async def test_an_unparsable_answer_falls_back_and_still_counts_its_tokens(
    model: dict[str, Any],
) -> None:
    """A schema violation is billed like any other call."""
    answers(
        model,
        structured_response(None, parsing_error=ValueError("sub_topics is not a list")),
    )
    update = await plan(default_state(QUERY), settings=Settings())

    assert update["sub_topics"] == [QUERY]
    assert "sub_topics is not a list" in update["errors"][0]
    assert update["token_count"] == 360


async def test_tokens_come_from_the_response(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(
            PlannerOutput(sub_topics=[QUERY], research_plan="one"),
            message=raw(input_tokens=1_200, output_tokens=80),
        ),
    )
    update = await plan(default_state(QUERY), settings=Settings())

    assert update["token_count"] == 1_280


async def test_a_failed_call_reports_no_tokens(model: dict[str, Any]) -> None:
    answers(model, RuntimeError("timeout"))
    assert (await plan(default_state(QUERY), settings=Settings()))["token_count"] == 0


async def test_the_trace_entry_has_the_agreed_shape(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(
            PlannerOutput(sub_topics=["a", "b"], research_plan="two"),
            message=raw(input_tokens=300, output_tokens=60),
        ),
    )
    update = await plan(default_state(QUERY), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "planner"
    assert entry["tokens"] == 360
    assert entry["summary"] == "2 sub-topics"
    assert entry["duration_ms"] >= 0.0


async def test_the_trace_names_the_prompt_that_produced_the_plan(model: dict[str, Any]) -> None:
    """The report can be tied back to the exact prompt text."""
    answers(model, structured_response(PlannerOutput(sub_topics=["a"], research_plan="one")))
    update = await plan(default_state(QUERY), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert entry["prompt_hash"] == load("planner").hash
    assert entry["summary"] == "1 sub-topic"


async def test_a_fallback_is_still_traced(model: dict[str, Any]) -> None:
    """A node that failed still has to account for the time it spent."""
    answers(model, RuntimeError("provider is down"))
    update = await plan(default_state(QUERY), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "planner"
