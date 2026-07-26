"""Factory tests. No provider package is exercised: `init_chat_model` is mocked."""

from typing import Any

import pytest

from research_system.llm import factory
from research_system.settings import Settings


class FakeModel:
    """Stands in for whatever chat model the provider package would return."""


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Record every `init_chat_model` call instead of making one."""
    recorded: list[tuple[str, dict[str, Any]]] = []

    def fake_init(model: str, **kwargs: Any) -> FakeModel:
        recorded.append((model, kwargs))
        return FakeModel()

    monkeypatch.setattr(factory, "init_chat_model", fake_init)
    return recorded


def test_agent_config_reaches_the_model(calls: list[tuple[str, dict[str, Any]]]) -> None:
    settings = Settings()
    factory.get_model("writer", settings=settings)

    model, kwargs = calls[0]
    writer = settings.agents.writer
    assert model == settings.llm.model
    assert kwargs["temperature"] == writer.temperature
    assert kwargs["max_tokens"] == writer.max_output_tokens
    assert kwargs["timeout"] == settings.llm.timeout_seconds
    assert kwargs["max_retries"] == settings.llm.max_retries


def test_each_agent_gets_its_own_generation_settings(
    calls: list[tuple[str, dict[str, Any]]],
) -> None:
    settings = Settings()
    for agent in ("planner", "analyst", "synthesizer", "writer", "reviewer"):
        factory.get_model(agent, settings=settings)

    temperatures = [kwargs["temperature"] for _, kwargs in calls]
    assert temperatures == [
        settings.agents.llm(a).temperature
        for a in ("planner", "analyst", "synthesizer", "writer", "reviewer")
    ]


@pytest.mark.parametrize("provider", ["openai", "google_genai", "anthropic"])
def test_provider_comes_from_config(
    calls: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    monkeypatch.setenv("LLM__PROVIDER", provider)
    factory.get_model("planner", settings=Settings())

    _, kwargs = calls[0]
    assert kwargs["model_provider"] == provider


def test_api_key_is_forwarded_from_settings(
    calls: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    factory.get_model("planner", settings=Settings())

    _, kwargs = calls[0]
    assert kwargs["api_key"] == "sk-test"


def test_api_key_is_omitted_when_unset(
    calls: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a key the provider client falls back to its own env lookup."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory.get_model("planner", settings=Settings(_env_file=None))

    _, kwargs = calls[0]
    assert "api_key" not in kwargs


def test_non_llm_agent_is_rejected(calls: list[tuple[str, dict[str, Any]]]) -> None:
    with pytest.raises(KeyError):
        factory.get_model("quality_gate")
    assert calls == []
