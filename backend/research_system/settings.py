"""Typed configuration.

`configs/base.yaml` holds every default value; environment variables override it
using `__` for nesting (`PIPELINE__QUALITY_THRESHOLD=0.7`). Secrets live only in
the environment. Nothing here restates a YAML default, so a value is defined once.

Invalid or unknown config raises on first `get_settings()` call, not mid-run.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

BASE_YAML = Path(__file__).parent / "configs" / "base.yaml"


class Section(BaseModel):
    """A config block: immutable, and an unknown key is an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelPrice(Section):
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class LLMSettings(Section):
    # These are init_chat_model's provider ids, not friendly names.
    provider: Literal["openai", "google_genai", "anthropic"]
    model: str
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    prices: dict[str, ModelPrice]

    @model_validator(mode="after")
    def _model_must_be_priced(self) -> "LLMSettings":
        """An unpriced model would silently report a run as costing nothing."""
        if self.model not in self.prices:
            raise ValueError(f"no price entry for model {self.model!r}")
        return self

    @property
    def price(self) -> ModelPrice:
        return self.prices[self.model]


class LLMAgentSettings(Section):
    uses_llm: Literal[True]
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(gt=0)


class AnalystSettings(LLMAgentSettings):
    min_claims: int = Field(gt=0)
    max_claims: int = Field(gt=0)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "AnalystSettings":
        if self.max_claims < self.min_claims:
            raise ValueError("max_claims is below min_claims")
        return self


class ResearcherSettings(Section):
    uses_llm: Literal[False]
    max_sources_per_topic: int = Field(gt=0)
    tools: list[str] = Field(min_length=1)


class QualityGateSettings(Section):
    uses_llm: Literal[False]


class AgentsSettings(Section):
    """Named fields, so a missing agent block fails at startup."""

    planner: LLMAgentSettings
    researcher: ResearcherSettings
    quality_gate: QualityGateSettings
    analyst: AnalystSettings
    synthesizer: LLMAgentSettings
    writer: LLMAgentSettings
    reviewer: LLMAgentSettings

    def llm(self, agent: str) -> LLMAgentSettings:
        """Generation settings for one LLM agent, by name."""
        block = getattr(self, agent, None)
        if not isinstance(block, LLMAgentSettings):
            raise KeyError(f"{agent!r} is not an LLM agent")
        return block


class PipelineSettings(Section):
    # Capped at 3 to match PlannerOutput.sub_topics.
    max_sub_topics: int = Field(ge=1, le=3)
    quality_threshold: float = Field(ge=0.0, le=1.0)
    max_quality_retries: int = Field(ge=0)
    max_revisions: int = Field(ge=0)
    review_pass_score: int = Field(ge=1, le=10)
    recursion_limit: int = Field(gt=0)


class CheckpointSettings(Section):
    path: Path


class BudgetSettings(Section):
    max_tokens_per_run: int = Field(gt=0)
    degrade_at_fraction: float = Field(gt=0.0, le=1.0)

    @property
    def degrade_at_tokens(self) -> int:
        """The token count where degradation starts, derived once from the cap."""
        return int(self.max_tokens_per_run * self.degrade_at_fraction)


class CacheSettings(Section):
    enabled: bool
    path: Path
    ttl_seconds: int = Field(gt=0)


class RetrySettings(Section):
    max_attempts: int = Field(ge=1)
    initial_backoff_seconds: float = Field(gt=0)
    max_backoff_seconds: float = Field(gt=0)
    budget_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "RetrySettings":
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds is below initial_backoff_seconds")
        return self


class SearchSettings(Section):
    results_per_query: int = Field(gt=0)
    snippet_max_chars: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    retry: RetrySettings
    cache: CacheSettings


class RateLimitSettings(Section):
    requests_per_minute: int = Field(gt=0)
    burst: int = Field(gt=0)


class GuardrailsSettings(Section):
    query_min_chars: int = Field(gt=0)
    query_max_chars: int = Field(gt=0)
    injection_enabled: bool
    pii_enabled: bool
    rate_limit: RateLimitSettings

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "GuardrailsSettings":
        if self.query_max_chars < self.query_min_chars:
            raise ValueError("query_max_chars is below query_min_chars")
        return self


class EvaluationSettings(Section):
    dataset: Path
    reports: Path
    judge_model: str
    score_max: int = Field(gt=0)
    target_improvement_pct: float = Field(ge=0)


class LoggingSettings(Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    format: Literal["json", "console"]


class Settings(BaseSettings):
    """The whole configuration. Built from YAML, overridden by the environment."""

    model_config = SettingsConfigDict(
        yaml_file=BASE_YAML,
        env_nested_delimiter="__",
        env_file=".env",
        extra="forbid",
        frozen=True,
    )

    llm: LLMSettings
    agents: AgentsSettings
    pipeline: PipelineSettings
    checkpoint: CheckpointSettings
    budget: BudgetSettings
    search: SearchSettings
    guardrails: GuardrailsSettings
    evaluation: EvaluationSettings
    logging: LoggingSettings

    # Secrets: environment only, never YAML.
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    @property
    def llm_api_key(self) -> SecretStr | None:
        """The key for the configured provider, if one is set."""
        keys: dict[str, SecretStr | None] = {
            "openai": self.openai_api_key,
            "google_genai": self.google_api_key,
            "anthropic": self.anthropic_api_key,
        }
        return keys[self.llm.provider]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Highest precedence first: explicit args, env, .env, then the YAML base."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache
def get_settings() -> Settings:
    """Load and validate once per process."""
    return Settings()
