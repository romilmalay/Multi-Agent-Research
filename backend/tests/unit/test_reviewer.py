"""Reviewer tests. No network: `get_model` is replaced with a stub."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from research_system.agents import reviewer as reviewer_module
from research_system.agents.reviewer import review
from research_system.domain.schemas import ReviewOutput
from research_system.domain.state import ResearchState, default_state
from research_system.prompts import load
from research_system.settings import Settings

QUERY = "What are the effects of microplastics on marine life?"
DRAFT = "## Summary\nIngestion is documented in wild fish [0]."
TRACE_KEYS = {
    "agent",
    "duration_ms",
    "tokens",
    "input_tokens",
    "output_tokens",
    "summary",
    "prompt_hash",
}


def claim(text: str, source_idx: int = 0) -> dict[str, Any]:
    """One entry as the analyst leaves it in `key_claims`: a plain dict."""
    return {
        "claim": text,
        "evidence": f"quoted support for {text}",
        "source_idx": source_idx,
        "confidence": 0.9,
    }


def state_with(
    draft: str = DRAFT,
    claims: list[dict[str, Any]] | None = None,
    *,
    revision_count: int = 1,
) -> ResearchState:
    """State as the writer leaves it. `revision_count` is the pass that wrote this draft."""
    state = default_state(QUERY)
    state["key_claims"] = (
        claims if claims is not None else [claim("38% of fish contained particles")]
    )
    state["current_draft"] = draft
    state["drafts"] = [draft] if draft else []
    state["revision_count"] = revision_count
    return state


def verdict(score: int = 8, issues: list[str] | None = None, passed: bool = True) -> ReviewOutput:
    """What the model returns. `passed` is the model's flag, not the decision."""
    return ReviewOutput(
        score=score,
        issues=issues if issues is not None else [],
        suggestions=["Cite source 2."] if issues else [],
        passed=passed,
    )


def raw(input_tokens: int = 800, output_tokens: int = 200) -> AIMessage:
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def structured_response(parsed: ReviewOutput | None, **kwargs: Any) -> dict[str, Any]:
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
    """Records what the reviewer asked for, and answers with a canned response."""

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

    monkeypatch.setattr(reviewer_module, "get_model", fake_get_model)
    return holder


def answers(model: dict[str, Any], response: Any) -> None:
    model["response"] = response


async def test_a_score_at_the_threshold_passes(model: dict[str, Any]) -> None:
    """review_pass_score is 7, so 8 ships."""
    answers(model, structured_response(verdict(score=8)))
    update = await review(state_with(), settings=Settings())

    assert update["review"]["score"] == 8
    assert update["review"]["passed"] is True
    assert update["review"]["issues"] == []
    assert update["errors"] == []


async def test_a_score_below_the_threshold_fails_with_its_issues(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(verdict(score=5, issues=["The 4% figure is uncited."], passed=False)),
    )
    update = await review(state_with(), settings=Settings())

    assert update["review"]["score"] == 5
    assert update["review"]["passed"] is False
    assert update["review"]["issues"] == ["The 4% figure is uncited."]
    assert update["review"]["suggestions"] == ["Cite source 2."]


async def test_the_threshold_decides_not_the_models_flag(model: dict[str, Any]) -> None:
    """Config has to be authoritative, or changing review_pass_score would do nothing."""
    answers(model, structured_response(verdict(score=4, issues=["unsupported"], passed=True)))
    update = await review(state_with(), settings=Settings())

    assert update["review"]["passed"] is False


async def test_a_raised_threshold_fails_a_draft_that_would_have_shipped(
    model: dict[str, Any],
) -> None:
    """The same score, two configs, two verdicts."""
    answers(model, structured_response(verdict(score=8)))
    settings = Settings(pipeline={**Settings().pipeline.model_dump(), "review_pass_score": 9})
    update = await review(state_with(), settings=settings)

    assert update["review"]["passed"] is False


async def test_a_passed_draft_becomes_the_final_report(model: dict[str, Any]) -> None:
    answers(model, structured_response(verdict(score=8)))
    update = await review(state_with(), settings=Settings())

    assert update["final_report"] == DRAFT


async def test_a_failed_draft_writes_no_report_while_revisions_remain(
    model: dict[str, Any],
) -> None:
    """max_revisions is 2, so a first draft that failed is going back to the writer."""
    answers(model, structured_response(verdict(score=5, issues=["uncited"], passed=False)))
    update = await review(state_with(revision_count=1), settings=Settings())

    assert update["final_report"] == ""


async def test_the_last_draft_ships_even_though_it_failed(model: dict[str, Any]) -> None:
    """Revisions are spent: this draft is the best there will be, so it is the report."""
    answers(model, structured_response(verdict(score=5, issues=["uncited"], passed=False)))
    update = await review(state_with(revision_count=2), settings=Settings())

    assert update["final_report"] == DRAFT
    assert update["review"]["passed"] is False


async def test_an_unreviewed_last_draft_still_ships(model: dict[str, Any]) -> None:
    """Losing a written report because the grader was down helps nobody."""
    answers(model, RuntimeError("provider is down"))
    update = await review(state_with(revision_count=2), settings=Settings())

    assert update["final_report"] == DRAFT
    assert update["review"] == {}
    assert "unreviewed" in update["errors"][0]


async def test_the_final_report_is_pii_scrubbed(model: dict[str, Any]) -> None:
    """The last thing written before a person reads it, and the last chance to catch this."""
    draft = "Contact the author at ana.silva@example.org or +44 20 7123 4567 [0]."
    answers(model, structured_response(verdict(score=8)))
    update = await review(state_with(draft), settings=Settings())

    assert update["final_report"] == "Contact the author at [EMAIL] or [PHONE] [0]."


async def test_the_draft_itself_is_left_unscrubbed(model: dict[str, Any]) -> None:
    """A placeholder in `drafts` would reach the next revision prompt as the writer's own text."""
    state = state_with("Reach ana.silva@example.org [0].", revision_count=1)
    answers(model, structured_response(verdict(score=5, issues=["uncited"], passed=False)))
    update = await review(state, settings=Settings())

    assert "drafts" not in update
    assert "current_draft" not in update
    assert state["current_draft"] == "Reach ana.silva@example.org [0]."


async def test_no_draft_means_no_report(model: dict[str, Any]) -> None:
    answers(model, structured_response(verdict()))
    update = await review(state_with(draft="", revision_count=2), settings=Settings())

    assert update["final_report"] == ""


async def test_the_draft_claims_and_threshold_reach_the_prompt(model: dict[str, Any]) -> None:
    """Grounding is only checkable against the claims the writer was given."""
    answers(model, structured_response(verdict()))
    await review(
        state_with(claims=[claim("first finding", 0), claim("second finding", 2)]),
        settings=Settings(),
    )

    _system, human = model["model"].structured.messages[0]
    assert QUERY in human.content
    assert "[0] first finding" in human.content
    assert "[2] second finding" in human.content
    assert DRAFT in human.content


async def test_the_pass_score_is_stated_in_the_system_message(model: dict[str, Any]) -> None:
    """The model is told the bar it is scoring against."""
    answers(model, structured_response(verdict()))
    await review(state_with(), settings=Settings())

    system, _human = model["model"].structured.messages[0]
    assert "Set passed true only at 7 or above." in system.content


async def test_structured_output_is_requested_with_the_raw_message(model: dict[str, Any]) -> None:
    """Without `include_raw` the token counts are discarded with the message."""
    answers(model, structured_response(verdict()))
    await review(state_with(), settings=Settings())

    assert model["model"].schema is ReviewOutput
    assert model["model"].include_raw is True


async def test_no_draft_means_no_model_call(model: dict[str, Any]) -> None:
    """The writer failed; there is no text to hold to the evidence."""
    answers(model, structured_response(verdict()))
    update = await review(state_with(draft=""), settings=Settings())

    assert model["calls"] == 0
    assert update["review"] == {}
    assert update["token_count"] == 0
    assert update["errors"] == []


async def test_an_llm_exception_leaves_no_verdict_and_one_error(model: dict[str, Any]) -> None:
    """No invented score: an empty verdict is what says the draft is unreviewed."""
    answers(model, RuntimeError("provider is down"))
    update = await review(state_with(), settings=Settings())

    assert update["review"] == {}
    assert len(update["errors"]) == 1
    assert "provider is down" in update["errors"][0]


async def test_an_unparsable_answer_still_counts_its_tokens(model: dict[str, Any]) -> None:
    """A schema violation is billed like any other call."""
    answers(model, structured_response(None, parsing_error=ValueError("score is not an int")))
    update = await review(state_with(), settings=Settings())

    assert update["review"] == {}
    assert "score is not an int" in update["errors"][0]
    assert update["token_count"] == 1_000


async def test_tokens_come_from_the_response(model: dict[str, Any]) -> None:
    answers(
        model, structured_response(verdict(), message=raw(input_tokens=2_000, output_tokens=500))
    )
    update = await review(state_with(), settings=Settings())

    assert update["token_count"] == 2_500


async def test_the_trace_entry_has_the_agreed_shape(model: dict[str, Any]) -> None:
    answers(
        model,
        structured_response(
            verdict(score=5, issues=["uncited figure", "buried conflict"], passed=False),
            message=raw(input_tokens=800, output_tokens=200),
        ),
    )
    update = await review(state_with(), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "reviewer"
    assert entry["tokens"] == 1_000
    assert entry["summary"] == "score 5/10, failed, 2 issues"
    assert entry["prompt_hash"] == load("reviewer").hash
    assert entry["duration_ms"] >= 0.0


async def test_a_failure_is_still_traced(model: dict[str, Any]) -> None:
    """A node that failed still has to account for the time it spent."""
    answers(model, RuntimeError("provider is down"))
    update = await review(state_with(), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "reviewer"
    assert entry["summary"] == "failed"


async def test_a_skipped_run_is_traced_without_a_prompt_hash(model: dict[str, Any]) -> None:
    """No prompt was loaded, so there is no prompt to name."""
    answers(model, structured_response(verdict()))
    update = await review(state_with(draft=""), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["summary"] == "skipped: no draft"
    assert entry["prompt_hash"] == ""
