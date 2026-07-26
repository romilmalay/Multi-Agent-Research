"""What each LLM agent must return.

These models are passed to `model.with_structured_output(...)`, so every
`description` is sent to the LLM as part of the schema and steers the answer.
Write them as instructions to the model, not as notes to a reader.
"""

from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    """Planner: split the question into independent sub-topics."""

    sub_topics: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "Between 1 and 3 independent sub-questions that together answer the query. "
            "Each must be searchable on its own, with no overlap between them."
        ),
    )
    research_plan: str = Field(
        description="One short paragraph explaining why the query splits this way."
    )


class ClaimOutput(BaseModel):
    """One factual claim the analyst extracted, tied to the source that supports it."""

    claim: str = Field(description="A single factual statement, one sentence.")
    evidence: str = Field(description="The quote or passage from the source that supports it.")
    source_idx: int = Field(
        ge=0,
        description="Index of the supporting source in the numbered list you were given.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How strongly the evidence supports the claim, from 0.0 to 1.0.",
    )


class AnalystOutput(BaseModel):
    """Analyst: turn raw sources into grounded claims."""

    claims: list[ClaimOutput] = Field(
        description="The most important claims found in the sources, each with its evidence."
    )


class SynthesizerOutput(BaseModel):
    """Synthesizer: cross-reference the claims and surface disagreement."""

    synthesis: str = Field(
        description="A connected narrative built only from the claims, noting where they agree."
    )
    conflicts: list[str] = Field(
        description=(
            "Pairs of claims that contradict each other, each described in one sentence. "
            "Empty if the sources agree. Do not invent conflicts."
        )
    )


class ReviewOutput(BaseModel):
    """Reviewer: grade the draft and decide whether it ships."""

    score: int = Field(
        ge=1,
        le=10,
        description="Overall quality of the draft, 1 (poor) to 10 (excellent).",
    )
    issues: list[str] = Field(
        description="Specific problems with the draft. Empty only if there are none."
    )
    suggestions: list[str] = Field(description="Concrete fixes for the issues, in the same order.")
    passed: bool = Field(description="True if the draft is ready to publish without changes.")
