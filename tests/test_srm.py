"""SRM detection against closed form power and hand computable cases."""

import numpy as np
import pytest
from scipy import stats

from expcost.srm import (
    DISCARD_THRESHOLD,
    analytic_detection_rate,
    simulate_detection_rate,
    smallest_detectable_mismatch,
    srm_test,
)


def test_a_perfect_split_gives_a_statistic_of_zero():
    statistic, p_value = srm_test(5000, 5000, 0.5)
    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_statistic_by_hand():
    # 600 vs 400 against an expected 500/500: (100^2)/500 + (100^2)/500 = 40.
    statistic, p_value = srm_test(600, 400, 0.5)
    assert statistic == pytest.approx(40.0)
    assert p_value == pytest.approx(stats.chi2.sf(40.0, 1))


def test_a_deliberate_imbalance_is_not_a_mismatch():
    """An 85/15 design checked against 85/15 must not fire.

    This is the distinction the whole module exists to make: imbalance by
    design is not the same thing as a sample ratio mismatch.
    """
    treated, control = 850_000, 150_000
    _, against_intended = srm_test(treated, control, 0.85)
    _, against_fifty_fifty = srm_test(treated, control, 0.5)
    assert against_intended > 0.05
    assert against_fifty_fifty < 1e-100


def test_a_drift_away_from_a_deliberate_imbalance_does_fire():
    # intended 85/15, actually delivered 84/16 at a million users
    _, p_value = srm_test(840_000, 160_000, 0.85)
    assert p_value < DISCARD_THRESHOLD


def test_false_positive_rate_matches_the_threshold():
    rate, reps = simulate_detection_rate(200_000, 0.5, 0.5, replications=40_000)
    error = np.sqrt(DISCARD_THRESHOLD * (1 - DISCARD_THRESHOLD) / reps)
    assert abs(rate - DISCARD_THRESHOLD) < 4 * error


@pytest.mark.parametrize("n,allocation", [(50_000, 0.49), (100_000, 0.495), (1_000_000, 0.498)])
def test_simulated_power_matches_the_closed_form(n, allocation):
    measured, reps = simulate_detection_rate(n, allocation, replications=6000)
    predicted = analytic_detection_rate(n, allocation)
    error = np.sqrt(max(predicted * (1 - predicted), 1e-6) / reps)
    assert abs(measured - predicted) < 4 * error


def test_smallest_detectable_mismatch_shrinks_with_sample_size():
    small = smallest_detectable_mismatch(10_000)
    large = smallest_detectable_mismatch(1_000_000)
    assert small > large
    # and the closed form agrees that the returned drift sits at the target
    assert analytic_detection_rate(1_000_000, 0.5 + large) == pytest.approx(0.80, abs=0.01)
