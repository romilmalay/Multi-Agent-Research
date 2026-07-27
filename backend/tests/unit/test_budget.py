import pytest

from research_system.guardrails.budget import check_budget
from research_system.settings import BudgetSettings, get_settings


@pytest.fixture
def budget() -> BudgetSettings:
    return get_settings().budget


def test_the_configured_budget_is_the_50k_cap(budget: BudgetSettings) -> None:
    assert budget.max_tokens_per_run == 50_000
    assert budget.degrade_at_tokens == 40_000


def test_well_under_the_cap_is_neither_degraded_nor_exhausted(budget: BudgetSettings) -> None:
    status = check_budget(1_000, budget)
    assert (status.degraded, status.exhausted) == (False, False)
    assert status.remaining == 49_000
    assert (status.spent, status.limit) == (1_000, 50_000)


def test_just_below_the_threshold_is_not_yet_degraded(budget: BudgetSettings) -> None:
    assert check_budget(budget.degrade_at_tokens - 1, budget).degraded is False


def test_at_the_threshold_degrades_but_still_spends(budget: BudgetSettings) -> None:
    """Inclusive: at 80% the run does the cheap version of the remaining work."""
    status = check_budget(budget.degrade_at_tokens, budget)
    assert (status.degraded, status.exhausted) == (True, False)


def test_at_the_cap_is_exhausted(budget: BudgetSettings) -> None:
    """Landing exactly on the cap means the next call would cross it."""
    status = check_budget(budget.max_tokens_per_run, budget)
    assert (status.degraded, status.exhausted) == (True, True)
    assert status.remaining == 0


def test_over_the_cap_reports_no_negative_remainder(budget: BudgetSettings) -> None:
    status = check_budget(budget.max_tokens_per_run * 2, budget)
    assert status.exhausted is True
    assert status.remaining == 0
    assert status.spent == 100_000


def test_a_fresh_run_has_the_whole_budget(budget: BudgetSettings) -> None:
    status = check_budget(0, budget)
    assert status.remaining == budget.max_tokens_per_run
    assert (status.degraded, status.exhausted) == (False, False)


def test_a_custom_budget_moves_both_thresholds() -> None:
    small = BudgetSettings(max_tokens_per_run=1_000, degrade_at_fraction=0.5)
    assert small.degrade_at_tokens == 500
    assert check_budget(499, small).degraded is False
    assert check_budget(500, small).degraded is True
    assert check_budget(999, small).exhausted is False
    assert check_budget(1_000, small).exhausted is True


def test_degrading_only_at_the_cap_is_expressible() -> None:
    """`degrade_at_fraction: 1.0` turns graceful degradation off."""
    strict = BudgetSettings(max_tokens_per_run=1_000, degrade_at_fraction=1.0)
    assert check_budget(999, strict).degraded is False
    status = check_budget(1_000, strict)
    assert (status.degraded, status.exhausted) == (True, True)


def test_the_status_is_frozen(budget: BudgetSettings) -> None:
    """A caller cannot talk itself into more budget."""
    with pytest.raises(AttributeError):
        check_budget(0, budget).exhausted = False  # type: ignore[misc]
