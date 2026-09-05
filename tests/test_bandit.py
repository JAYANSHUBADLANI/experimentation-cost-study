"""The bandit, its regret, and the validity of the two intervals."""

import numpy as np
import pytest

from expcost.bandit import coverage_table, run_bandit


@pytest.fixture(scope="module")
def null_run():
    """Both arms identical, so any apparent effect is an error."""
    return run_bandit((0.04, 0.04), rounds=120, batch=400, replications=1500, label="test_null", floor=0.01)


@pytest.fixture(scope="module")
def lift_run():
    return run_bandit((0.04, 0.05), rounds=120, batch=400, replications=800, label="test_lift", floor=0.01)


def test_the_bandit_finds_the_better_arm(lift_run):
    assert lift_run.share_to_best.mean() > 0.7


def test_the_bandit_earns_less_regret_than_a_balanced_split(lift_run):
    total_users = lift_run.rounds * lift_run.batch
    balanced_regret = 0.5 * total_users * lift_run.true_effect
    assert lift_run.regret.mean() < 0.6 * balanced_regret


def test_no_regret_and_no_effect_under_the_null(null_run):
    assert null_run.true_effect == 0.0
    assert null_run.regret.max() == 0.0


def test_the_naive_interval_undercovers_under_adaptive_sampling(null_run):
    """The central claim about adaptively collected data, measured.

    With the exploration floor at one percent the assignment is driven hard by
    noise, and the ordinary interval computed on the resulting data does not
    cover at its nominal rate.
    """
    rows = {row["interval"]: row for row in coverage_table(null_run)}
    naive = rows["bandit_naive"]
    assert naive["coverage_hi"] < 0.95


def test_the_adaptively_weighted_interval_covers_better(null_run):
    rows = {row["interval"]: row for row in coverage_table(null_run)}
    assert rows["bandit_adaptively_weighted"]["coverage"] > rows["bandit_naive"]["coverage"]


def test_the_balanced_design_covers_at_its_nominal_rate(null_run):
    rows = {row["interval"]: row for row in coverage_table(null_run)}
    fixed = rows["fixed_horizon_balanced"]
    assert fixed["coverage_lo"] <= 0.95 <= fixed["coverage_hi"]


def test_validity_is_bought_with_a_wider_interval(null_run):
    rows = {row["interval"]: row for row in coverage_table(null_run)}
    assert (
        rows["bandit_adaptively_weighted"]["mean_half_width"]
        > rows["bandit_naive"]["mean_half_width"]
    )
    assert (
        rows["bandit_adaptively_weighted"]["mean_half_width"]
        > rows["fixed_horizon_balanced"]["mean_half_width"]
    )


def test_the_run_is_deterministic():
    a = run_bandit((0.04, 0.05), 30, 200, 50, label="determinism", floor=0.01)
    b = run_bandit((0.04, 0.05), 30, 200, 50, label="determinism", floor=0.01)
    assert np.array_equal(a.regret, b.regret)
    assert np.array_equal(a.weighted_effect, b.weighted_effect)
