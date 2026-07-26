from langchain_core.messages import AIMessage

from research_system.llm.usage import Usage, extract_usage
from research_system.settings import ModelPrice, get_settings


def message(input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def test_counts_come_from_usage_metadata() -> None:
    usage = extract_usage(message(1200, 340))
    assert usage.input_tokens == 1200
    assert usage.output_tokens == 340
    assert usage.total_tokens == 1540


def test_structured_output_raw_message_is_read() -> None:
    """`include_raw=True` hides the message under "raw"; the tokens are still there."""
    response = {"raw": message(90, 10), "parsed": {"score": 8}, "parsing_error": None}
    assert extract_usage(response).total_tokens == 100


def test_missing_metadata_degrades_to_zero() -> None:
    assert extract_usage(AIMessage(content="no metadata")).total_tokens == 0
    assert extract_usage({"parsed": {"score": 8}}).total_tokens == 0
    assert extract_usage(None).total_tokens == 0


def test_cost_is_priced_per_million_tokens() -> None:
    price = ModelPrice(input=0.15, output=0.60)
    # 1M in at $0.15 plus 0.5M out at $0.60 = 0.15 + 0.30
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000)
    assert usage.cost_usd(price) == 0.45


def test_a_realistic_run_costs_a_believable_amount() -> None:
    usage = Usage(input_tokens=40_000, output_tokens=8_000)
    cost = usage.cost_usd(get_settings().llm.price)
    assert 0.0 < cost < 0.05


def test_zero_usage_is_free() -> None:
    assert Usage().cost_usd(ModelPrice(input=99.0, output=99.0)) == 0.0
