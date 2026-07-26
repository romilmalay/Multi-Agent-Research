"""Token accounting.

Counts come from the provider's `usage_metadata` and nowhere else. When a
response carries none, the answer is zero rather than an estimate: a fabricated
number would make the budget cap and the cost report quietly wrong.
"""

from dataclasses import dataclass
from typing import Any

from research_system.settings import ModelPrice

# Prices in `configs/base.yaml` are quoted per million tokens.
TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens consumed by one LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, price: ModelPrice) -> float:
        """What this call cost at the configured price for the model."""
        billed = self.input_tokens * price.input + self.output_tokens * price.output
        return billed / TOKENS_PER_PRICE_UNIT


def extract_usage(response: Any) -> Usage:
    """Read token counts off an LLM response.

    Accepts a message, or the `{"raw", "parsed", "parsing_error"}` mapping that
    `with_structured_output(..., include_raw=True)` returns — structured output
    otherwise discards the metadata along with the message.
    """
    message = response.get("raw") if isinstance(response, dict) else response
    metadata = getattr(message, "usage_metadata", None)
    if not metadata:
        return Usage()
    # Both keys are required by langchain-core's UsageMetadata, so index them.
    return Usage(
        input_tokens=metadata["input_tokens"],
        output_tokens=metadata["output_tokens"],
    )
