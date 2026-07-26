from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from research_system.settings import (
    BASE_YAML,
    AnalystSettings,
    GuardrailsSettings,
    LoggingSettings,
    RateLimitSettings,
    Settings,
    get_settings,
)

AGENTS = {
    "planner",
    "researcher",
    "quality_gate",
    "analyst",
    "synthesizer",
    "writer",
    "reviewer",
}

LLM_AGENTS = {"planner", "analyst", "synthesizer", "writer", "reviewer"}


@pytest.fixture
def raw() -> dict[str, Any]:
    """The YAML as data, so tests compare against it instead of restating defaults."""
    loaded: dict[str, Any] = yaml.safe_load(BASE_YAML.read_text())
    return loaded


def test_yaml_is_the_source_of_values(raw: dict[str, Any]) -> None:
    settings = get_settings()
    assert settings.llm.model == raw["llm"]["model"]
    assert settings.pipeline.quality_threshold == raw["pipeline"]["quality_threshold"]
    assert settings.budget.max_tokens_per_run == raw["budget"]["max_tokens_per_run"]
    assert (
        settings.guardrails.rate_limit.requests_per_minute
        == (raw["guardrails"]["rate_limit"]["requests_per_minute"])
    )


def test_all_seven_agents_have_blocks(raw: dict[str, Any]) -> None:
    assert raw["agents"].keys() == AGENTS
    settings = get_settings()
    for name in AGENTS:
        assert getattr(settings.agents, name).uses_llm is (name in LLM_AGENTS)


def test_llm_lookup_returns_generation_settings() -> None:
    agents = get_settings().agents
    for name in LLM_AGENTS:
        block = agents.llm(name)
        assert 0.0 <= block.temperature <= 2.0
        assert block.max_output_tokens > 0


def test_llm_lookup_rejects_a_non_llm_agent() -> None:
    with pytest.raises(KeyError):
        get_settings().agents.llm("quality_gate")


def test_default_model_is_priced() -> None:
    settings = get_settings()
    assert settings.llm.price is settings.llm.prices[settings.llm.model]


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch, raw: dict[str, Any]) -> None:
    monkeypatch.setenv("PIPELINE__QUALITY_THRESHOLD", "0.95")
    monkeypatch.setenv("AGENTS__WRITER__TEMPERATURE", "0.1")
    settings = Settings()
    assert settings.pipeline.quality_threshold == 0.95
    assert settings.agents.writer.temperature == 0.1
    # A partial override of a block leaves its siblings on the YAML value.
    assert settings.agents.writer.max_output_tokens == raw["agents"]["writer"]["max_output_tokens"]


def test_secrets_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings()
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(settings)


def test_out_of_range_value_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    # PlannerOutput caps sub_topics at 3; a higher config value must not load.
    monkeypatch.setenv("PIPELINE__MAX_SUB_TOPICS", "4")
    with pytest.raises(ValidationError):
        Settings()


def test_unpriced_model_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__MODEL", "not-a-real-model")
    with pytest.raises(ValidationError, match="no price entry"):
        Settings()


def test_inverted_claim_range_fails_at_startup() -> None:
    with pytest.raises(ValidationError, match="below min_claims"):
        AnalystSettings(
            uses_llm=True,
            temperature=0.2,
            max_output_tokens=2048,
            min_claims=8,
            max_claims=5,
        )


def test_inverted_query_length_range_fails_at_startup() -> None:
    with pytest.raises(ValidationError, match="below query_min_chars"):
        GuardrailsSettings(
            query_min_chars=500,
            query_max_chars=10,
            injection_enabled=True,
            pii_enabled=True,
            rate_limit=RateLimitSettings(requests_per_minute=30, burst=5),
        )


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LoggingSettings(level="INFO", format="json", verbosity=3)  # type: ignore[call-arg]


def test_settings_are_immutable() -> None:
    with pytest.raises(ValidationError):
        get_settings().pipeline.max_revisions = 9  # type: ignore[misc]
