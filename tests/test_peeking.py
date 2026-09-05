"""Daily monitoring: the statistics, and what each reading rule does."""

import numpy as np
import pytest

from expcost.estimators import diff_in_means
from expcost.peeking import build_peek_chunks, daily_statistics, run_peek_chunk, tau_squared_for
from expcost.seeds import generator
from expcost.simulate import MetricSpec, simulate_experiment


@pytest.fixture(scope="module")
def spec():
    return MetricSpec("conversion", 0.04, name="conversion_p04")


def test_last_look_reproduces_the_ordinary_test(spec):
    """The final daily look must be the plain two sample test on all the data."""
    experiment = simulate_experiment(spec, 40_000, 0.0, 0.0, generator("peek_last_look"))
    effect, variance, z, p = daily_statistics(experiment.outcome, experiment.arm, 4, 10_000)
    reference = diff_in_means(experiment.outcome, experiment.arm)
    assert effect[-1] == pytest.approx(reference.effect, rel=1e-10)
    assert np.sqrt(variance[-1]) == pytest.approx(reference.se, rel=1e-10)
    assert p[-1] == pytest.approx(reference.p_value, rel=1e-6)


def test_cumulative_counts_grow_by_a_day_each_look(spec):
    experiment = simulate_experiment(spec, 30_000, 0.0, 0.0, generator("peek_counts"))
    effect, variance, _, _ = daily_statistics(experiment.outcome, experiment.arm, 3, 10_000)
    # variance of the difference must fall roughly like one over sample size
    assert variance[0] > variance[1] > variance[2]
    assert variance[0] / variance[2] == pytest.approx(3.0, rel=0.15)


def test_tau_squared_is_the_squared_absolute_effect(spec):
    tau_squared = tau_squared_for(spec, 0.05)
    assert tau_squared == pytest.approx((0.05 * 0.04) ** 2, rel=1e-9)


def test_peeking_inflates_the_error_rate_and_the_corrections_do_not(spec):
    """The headline of the peeking study, at a replication count a test can afford."""
    chunks = build_peek_chunks("conversion_p04", spec, 0.0, 10, 8_000, 600, 0.05, block=600)
    result = run_peek_chunk(chunks[0])
    fixed = result["fixed_horizon"].mean()
    naive = result["naive_peeking"].mean()
    assert fixed < 0.09
    assert naive > 2.0 * fixed
    assert result["msprt"].mean() <= 0.05
    assert result["alpha_spending"].mean() < 0.09


def test_naive_peeking_never_stops_later_than_it_rejects(spec):
    chunks = build_peek_chunks("conversion_p04", spec, 0.0, 6, 5_000, 200, 0.05, block=200)
    result = run_peek_chunk(chunks[0])
    rejected = result["naive_peeking"]
    assert np.all(result["naive_stop_day"][rejected] <= 6)
    assert np.all(result["naive_stop_day"][~rejected] == 6)
