"""The token budget: how much of the run's allowance is left, and what to do next.

Three states, not two. Under the cap, near it, and past it — and the middle one is
the point. A run that finds out it is out of tokens between the analyst and the
writer has paid for research it will never report. Degrading at 80% (fewer
sub-topics, fewer sources, no quality retry) spends the last fifth on finishing.

Checks return a status and never raise. Every caller has something useful to do
with "spend less", and an exception would only let it abort.
"""

from dataclasses import dataclass

from research_system.settings import BudgetSettings


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    """Where a run stands against its token cap."""

    spent: int
    limit: int
    remaining: int
    degraded: bool
    """At or past the degradation threshold: do the cheap version of the work."""

    exhausted: bool
    """At or past the cap: make no further LLM calls."""


def check_budget(token_count: int, settings: BudgetSettings) -> BudgetStatus:
    """The budget state at `token_count` tokens spent.

    Both thresholds are inclusive. Landing exactly on the cap means the next call
    would cross it, and there is no useful difference between "at" and "over" once
    the decision is whether to spend again.
    """
    return BudgetStatus(
        spent=token_count,
        limit=settings.max_tokens_per_run,
        # Overspend is reported by `exhausted`; there is no negative remainder.
        remaining=max(0, settings.max_tokens_per_run - token_count),
        degraded=token_count >= settings.degrade_at_tokens,
        exhausted=token_count >= settings.max_tokens_per_run,
    )
