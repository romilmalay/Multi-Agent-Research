"""Writer tests. No network: `get_model` is replaced with a stub."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from research_system.agents import writer as writer_module
from research_system.agents.writer import NO_CLAIMS_DRAFT, write
from research_system.domain.state import ResearchState, default_state
from research_system.prompts import load
from research_system.settings import Settings

QUERY = "What are the effects of microplastics on marine life?"
TRACE_KEYS = {"agent", "duration_ms", "tokens", "summary", "prompt_hash"}


def claim(text: str, source_idx: int = 0, confidence: float = 0.9) -> dict[str, Any]:
    """One entry as the analyst leaves it in `key_claims`: a plain dict."""
    return {
        "claim": text,
        "evidence": f"quoted support for {text}",
        "source_idx": source_idx,
        "confidence": confidence,
    }


def state_with(
    claims: list[dict[str, Any]],
    *,
    synthesis: str = "The sources agree on ingestion [0].",
    conflicts: list[str] | None = None,
    drafts: list[str] | None = None,
    review: dict[str, Any] | None = None,
) -> ResearchState:
    state = default_state(QUERY)
    state["key_claims"] = claims
    state["synthesis"] = synthesis
    state["conflicts"] = conflicts or []
    state["drafts"] = drafts or []
    state["current_draft"] = state["drafts"][-1] if state["drafts"] else ""
    state["revision_count"] = len(state["drafts"])
    state["review"] = review or {}
    return state


def review(
    *,
    score: int = 5,
    issues: list[str] | None = None,
    suggestions: list[str] | None = None,
) -> dict[str, Any]:
    """One verdict as the reviewer leaves it in `review`."""
    return {
        "score": score,
        "issues": issues if issues is not None else ["The 4% figure is uncited."],
        "suggestions": suggestions if suggestions is not None else ["Cite source 2 for the 4%."],
        "passed": False,
    }


def message(content: str, input_tokens: int = 900, output_tokens: int = 600) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class FakeModel:
    """Records what the writer asked for, and answers with a canned response."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.messages: list[Any] = []

    async def ainvoke(self, messages: Any) -> Any:
        self.messages.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a stub model whose response each test sets."""
    holder: dict[str, Any] = {"calls": 0}

    def fake_get_model(agent: str, *, settings: Settings | None = None) -> FakeModel:
        holder["calls"] += 1
        built = FakeModel(holder["response"])
        holder["model"] = built
        return built

    monkeypatch.setattr(writer_module, "get_model", fake_get_model)
    return holder


def answers(model: dict[str, Any], response: Any) -> None:
    model["response"] = response


async def test_the_draft_reaches_state(model: dict[str, Any]) -> None:
    answers(model, message("## Summary\nIngestion is documented in wild fish [0]."))
    state = state_with([claim("38% of fish contained particles")])
    update = await write(state, settings=Settings())

    assert update["current_draft"] == "## Summary\nIngestion is documented in wild fish [0]."
    assert update["drafts"] == ["## Summary\nIngestion is documented in wild fish [0]."]
    assert update["revision_count"] == 1
    assert update["errors"] == []


async def test_drafts_append_and_never_overwrite(model: dict[str, Any]) -> None:
    """The revision history is what shows whether review improved anything."""
    answers(model, message("second draft [0]"))
    update = await write(
        state_with([claim("a finding")], drafts=["first draft [0]"]), settings=Settings()
    )

    assert update["drafts"] == ["first draft [0]", "second draft [0]"]
    assert update["current_draft"] == "second draft [0]"


async def test_revision_count_increments_once_per_pass(model: dict[str, Any]) -> None:
    answers(model, message("third draft [0]"))
    update = await write(
        state_with([claim("a finding")], drafts=["first", "second"]), settings=Settings()
    )

    assert update["revision_count"] == 3
    assert len(update["drafts"]) == 3


async def test_the_claims_synthesis_and_conflicts_all_reach_the_prompt(
    model: dict[str, Any],
) -> None:
    """The citation numbers have to survive into the draft, so they go into the prompt."""
    answers(model, message("a draft"))
    await write(
        state_with(
            [claim("first finding", 0), claim("second finding", 2)],
            synthesis="Both seas show ingestion [0][2].",
            conflicts=["Source 0 reports 38%, source 2 reports 4%."],
        ),
        settings=Settings(),
    )

    _system, human = model["model"].messages[0]
    assert QUERY in human.content
    assert "Both seas show ingestion [0][2]." in human.content
    assert "[0] first finding" in human.content
    assert "[2] second finding" in human.content
    assert "Source 0 reports 38%, source 2 reports 4%." in human.content


async def test_the_first_pass_carries_no_revision_block(model: dict[str, Any]) -> None:
    """Nothing has been reviewed yet, so the revision half of the template stays closed."""
    answers(model, message("a draft"))
    await write(state_with([claim("a finding")]), settings=Settings())

    _system, human = model["model"].messages[0]
    assert "This is a revision" not in human.content


async def test_a_revision_carries_the_reviewers_verdict_into_the_prompt(
    model: dict[str, Any],
) -> None:
    """The issues are why this pass exists; without them it rewrites the same draft."""
    answers(model, message("a better draft"))
    await write(
        state_with(
            [claim("a finding")],
            drafts=["the draft that scored 5 [0]"],
            review=review(
                score=5,
                issues=["The 4% figure is uncited.", "The summary buries the disagreement."],
                suggestions=["Cite source 2 for the 4%.", "Move the conflict into the summary."],
            ),
        ),
        settings=Settings(),
    )

    _system, human = model["model"].messages[0]
    assert "This is a revision. The previous draft scored 5 out of 10." in human.content
    assert "- The 4% figure is uncited." in human.content
    assert "- The summary buries the disagreement." in human.content
    assert "- Cite source 2 for the 4%." in human.content
    assert "- Move the conflict into the summary." in human.content
    assert "the draft that scored 5 [0]" in human.content


async def test_a_revision_still_gets_the_claims_and_synthesis(model: dict[str, Any]) -> None:
    """The fixes have to come from the same evidence, so the facts are sent again."""
    answers(model, message("a better draft"))
    await write(
        state_with(
            [claim("first finding", 2)],
            synthesis="Both seas show ingestion [2].",
            drafts=["the previous draft"],
            review=review(),
        ),
        settings=Settings(),
    )

    _system, human = model["model"].messages[0]
    assert "[2] first finding" in human.content
    assert "Both seas show ingestion [2]." in human.content


async def test_a_revision_appends_rather_than_replacing_the_draft_it_fixes(
    model: dict[str, Any],
) -> None:
    answers(model, message("the revised draft"))
    update = await write(
        state_with([claim("a finding")], drafts=["the draft that scored 5"], review=review()),
        settings=Settings(),
    )

    assert update["drafts"] == ["the draft that scored 5", "the revised draft"]
    assert update["current_draft"] == "the revised draft"
    assert update["revision_count"] == 2


async def test_the_revision_trace_names_how_many_issues_it_was_fixing(
    model: dict[str, Any],
) -> None:
    answers(model, message("the revised draft"))
    update = await write(
        state_with(
            [claim("a finding")],
            drafts=["the draft that scored 5"],
            review=review(issues=["one", "two"], suggestions=["fix one", "fix two"]),
        ),
        settings=Settings(),
    )

    (entry,) = update["pipeline_trace"]
    assert entry["summary"] == "draft 2 from 1 claims, fixing 2 issues"


async def test_no_claims_means_no_model_call(model: dict[str, Any]) -> None:
    """Nothing is attributable, so anything written would be the model's own recollection."""
    answers(model, message("invented prose"))
    update = await write(state_with([]), settings=Settings())

    assert model["calls"] == 0
    assert update["current_draft"] == NO_CLAIMS_DRAFT
    assert update["drafts"] == [NO_CLAIMS_DRAFT]
    assert update["revision_count"] == 1
    assert update["token_count"] == 0
    assert update["errors"] == []


async def test_an_llm_exception_leaves_an_empty_draft_and_one_error(model: dict[str, Any]) -> None:
    """The pass still counts, or the reviewer would send the same draft back forever."""
    answers(model, RuntimeError("provider is down"))
    update = await write(state_with([claim("a finding")]), settings=Settings())

    assert update["current_draft"] == ""
    assert update["drafts"] == [""]
    assert update["revision_count"] == 1
    assert len(update["errors"]) == 1
    assert "provider is down" in update["errors"][0]


async def test_an_empty_answer_is_a_failure_not_a_draft(model: dict[str, Any]) -> None:
    """Whitespace back from the model is nothing to review, and is billed."""
    answers(model, message("   \n  "))
    update = await write(state_with([claim("a finding")]), settings=Settings())

    assert update["current_draft"] == ""
    assert update["errors"][0].startswith("writer produced no draft")
    assert update["token_count"] == 1_500


async def test_a_block_content_answer_is_read_as_text(model: dict[str, Any]) -> None:
    """Some providers answer in content blocks; the draft is the text, not the list."""
    answers(
        model,
        AIMessage(
            content=[{"type": "text", "text": "## Summary"}, {"type": "text", "text": " [0]"}]
        ),
    )
    update = await write(state_with([claim("a finding")]), settings=Settings())

    assert update["current_draft"] == "## Summary [0]"


async def test_tokens_come_from_the_response(model: dict[str, Any]) -> None:
    answers(model, message("a draft", input_tokens=2_000, output_tokens=500))
    update = await write(state_with([claim("a finding")]), settings=Settings())

    assert update["token_count"] == 2_500


async def test_the_trace_entry_has_the_agreed_shape(model: dict[str, Any]) -> None:
    answers(model, message("a draft", input_tokens=900, output_tokens=600))
    update = await write(
        state_with([claim("first"), claim("second", 1)], drafts=["first draft"]),
        settings=Settings(),
    )

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "writer"
    assert entry["tokens"] == 1_500
    assert entry["summary"] == "draft 2 from 2 claims"
    assert entry["prompt_hash"] == load("writer").hash
    assert entry["duration_ms"] >= 0.0


async def test_a_failure_is_still_traced(model: dict[str, Any]) -> None:
    """A node that failed still has to account for the time it spent."""
    answers(model, RuntimeError("provider is down"))
    update = await write(state_with([claim("a finding")]), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["agent"] == "writer"
    assert entry["summary"] == "failed"


async def test_a_skipped_run_is_traced_without_a_prompt_hash(model: dict[str, Any]) -> None:
    """No prompt was loaded, so there is no prompt to name."""
    answers(model, message("a draft"))
    update = await write(state_with([]), settings=Settings())

    (entry,) = update["pipeline_trace"]
    assert set(entry) == TRACE_KEYS
    assert entry["summary"] == "skipped: no claims"
    assert entry["prompt_hash"] == ""
