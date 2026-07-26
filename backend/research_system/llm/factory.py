"""The one place that builds a chat model.

Agents call `get_model(name)` and never name a provider or a model id, so
switching providers is a config change and nothing else.
"""

from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from research_system.settings import Settings, get_settings


def get_model(agent: str, *, settings: Settings | None = None) -> BaseChatModel:
    """Build the chat model for one LLM agent.

    Generation settings come from that agent's config block; the provider,
    model id and transport settings are global. The API key is passed
    explicitly because a key read from `.env` never reaches `os.environ`,
    where the provider clients look for it.
    """
    settings = settings or get_settings()
    llm = settings.llm
    agent_config = settings.agents.llm(agent)

    kwargs: dict[str, Any] = {
        "model_provider": llm.provider,
        "temperature": agent_config.temperature,
        "max_tokens": agent_config.max_output_tokens,
        "timeout": llm.timeout_seconds,
        "max_retries": llm.max_retries,
    }
    api_key = settings.llm_api_key
    if api_key is not None:
        kwargs["api_key"] = api_key.get_secret_value()

    return cast(BaseChatModel, init_chat_model(llm.model, **kwargs))
