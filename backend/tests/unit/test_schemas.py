import pytest
from pydantic import BaseModel

from research_system.domain.schemas import (
    AnalystOutput,
    ClaimOutput,
    PlannerOutput,
    ReviewOutput,
    SynthesizerOutput,
)

ALL_SCHEMAS = [PlannerOutput, ClaimOutput, AnalystOutput, SynthesizerOutput, ReviewOutput]

CLAIM_JSON = {
    "claim": "Transformers replaced RNNs for sequence modelling.",
    "evidence": "The Transformer outperforms recurrent architectures on WMT 2014.",
    "source_idx": 2,
    "confidence": 0.9,
}


@pytest.mark.parametrize("schema", ALL_SCHEMAS)
def test_every_field_has_a_description(schema: type[BaseModel]) -> None:
    """The description is what the LLM sees. A field without one is unsteered."""
    missing = [name for name, f in schema.model_fields.items() if not f.description]
    assert missing == []


def test_planner_parses() -> None:
    out = PlannerOutput(sub_topics=["a", "b"], research_plan="split by era")
    assert out.sub_topics == ["a", "b"]


def test_analyst_parses_a_claim_with_evidence_and_source_index() -> None:
    out = AnalystOutput.model_validate({"claims": [CLAIM_JSON]})
    claim = out.claims[0]
    assert claim.source_idx == 2
    assert claim.evidence.startswith("The Transformer")
    assert claim.confidence == 0.9


def test_claim_round_trips_through_json() -> None:
    """What the LLM returns, and what we hand to the writer prompt, must match."""
    assert ClaimOutput.model_validate(CLAIM_JSON).model_dump() == CLAIM_JSON


def test_synthesizer_parses_conflicts() -> None:
    out = SynthesizerOutput(synthesis="broadly agree", conflicts=["source 1 contradicts source 3"])
    assert len(out.conflicts) == 1


def test_synthesizer_accepts_no_conflicts() -> None:
    assert SynthesizerOutput(synthesis="all agree", conflicts=[]).conflicts == []


def test_reviewer_parses() -> None:
    out = ReviewOutput(
        score=8, issues=["thin on citations"], suggestions=["cite source 4"], passed=True
    )
    assert out.score == 8
    assert out.passed is True


def test_missing_required_field_is_rejected() -> None:
    """A truncated LLM response must fail loudly, not silently lose a field."""
    with pytest.raises(ValueError):
        ClaimOutput.model_validate({"claim": "x", "evidence": "y"})


def test_planner_accepts_three_sub_topics() -> None:
    assert len(PlannerOutput(sub_topics=["a", "b", "c"], research_plan="p").sub_topics) == 3


def test_planner_rejects_four_sub_topics() -> None:
    """One researcher is fanned out per sub-topic, so 4 means an unbudgeted extra run."""
    with pytest.raises(ValueError):
        PlannerOutput(sub_topics=["a", "b", "c", "d"], research_plan="p")


def test_planner_rejects_zero_sub_topics() -> None:
    """An empty split would fan out to no researchers at all."""
    with pytest.raises(ValueError):
        PlannerOutput(sub_topics=[], research_plan="p")


@pytest.mark.parametrize("score", [1, 10])
def test_reviewer_accepts_the_bounds(score: int) -> None:
    assert ReviewOutput(score=score, issues=[], suggestions=[], passed=True).score == score


@pytest.mark.parametrize("score", [0, 11])
def test_reviewer_rejects_out_of_range_scores(score: int) -> None:
    """The pass threshold is a number comparison; a score of 0 or 11 makes it meaningless."""
    with pytest.raises(ValueError):
        ReviewOutput(score=score, issues=[], suggestions=[], passed=False)


def test_claim_rejects_a_negative_source_index() -> None:
    with pytest.raises(ValueError):
        ClaimOutput.model_validate(CLAIM_JSON | {"source_idx": -1})


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_claim_rejects_confidence_outside_zero_to_one(confidence: float) -> None:
    with pytest.raises(ValueError):
        ClaimOutput.model_validate(CLAIM_JSON | {"confidence": confidence})
