"""Phase 2's live validation: one real call reports real tokens and a real cost.

Excluded by default. Run with a key in the environment:

    uv run pytest -m live -s
"""

from typing import Any, cast

import pytest

from research_system.domain.schemas import PlannerOutput
from research_system.llm.factory import get_model
from research_system.llm.usage import extract_usage
from research_system.settings import get_settings

pytestmark = pytest.mark.live


def test_a_real_call_reports_tokens_and_cost() -> None:
    settings = get_settings()
    if settings.llm_api_key is None:
        pytest.skip(f"no API key set for provider {settings.llm.provider}")

    model = get_model("planner", settings=settings)
    response = model.invoke("Reply with one word: ok")

    usage = extract_usage(response)
    cost = usage.cost_usd(settings.llm.price)
    print(
        f"\nmodel={settings.llm.model} provider={settings.llm.provider} "
        f"input={usage.input_tokens} output={usage.output_tokens} "
        f"total={usage.total_tokens} cost=${cost:.8f}"
    )

    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert cost > 0


def test_structured_output_fits_the_configured_token_cap() -> None:
    """The pinned model must parse into a schema without exhausting max_output_tokens.

    This is what rules a model out: a model that spends its completion budget on
    reasoning tokens hits the cap and never returns parseable output.
    """
    settings = get_settings()
    if settings.llm_api_key is None:
        pytest.skip(f"no API key set for provider {settings.llm.provider}")

    model = get_model("planner", settings=settings).with_structured_output(
        PlannerOutput, include_raw=True
    )
    # include_raw widens the return type to `dict | BaseModel`; it is the dict.
    result = cast(
        dict[str, Any],
        model.invoke("Split this into sub-topics: effects of caffeine on sleep quality"),
    )

    plan = result["parsed"]
    usage = extract_usage(result)
    print(
        f"\nmodel={settings.llm.model} sub_topics={len(plan.sub_topics)} "
        f"output={usage.output_tokens}/{settings.agents.planner.max_output_tokens} "
        f"cost=${usage.cost_usd(settings.llm.price):.8f}"
    )

    assert result["parsing_error"] is None
    assert 1 <= len(plan.sub_topics) <= settings.pipeline.max_sub_topics
    assert usage.output_tokens < settings.agents.planner.max_output_tokens
