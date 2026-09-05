"""The simulator must reproduce the calibrated shape and the injected effect."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from expcost.calibration import RevenueShape, fit_revenue_shape
from expcost.seeds import generator
from expcost.simulate import (
    MetricSpec,
    _continuous_moments,
    _positive_part_moments,
    achieved_correlation,
    expected_correlation,
    marginal_ppf,
    metric_moments,
    positive_part_ppf,
    simulate_experiment,
    solve_copula_r,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def shape():
    return RevenueShape.from_json(ROOT / "results/revenue_shape.json")


@pytest.fixture(scope="module")
def revenue_spec(shape):
    return MetricSpec("revenue", 0.10, shape=shape, name="test_revenue")


def test_quantile_function_is_monotone(shape):
    values = positive_part_ppf(np.linspace(0.001, 0.999, 2000), shape)
    assert np.all(np.diff(values) > 0)


def test_splice_is_continuous_at_the_threshold(shape):
    below = positive_part_ppf(np.array([shape.tail_quantile - 1e-6]), shape)[0]
    above = positive_part_ppf(np.array([shape.tail_quantile + 1e-6]), shape)[0]
    assert below == pytest.approx(shape.threshold, rel=1e-3)
    assert above == pytest.approx(shape.threshold, rel=1e-3)


def test_simulated_positive_part_matches_the_committed_sample(shape):
    real = pd.read_csv(ROOT / "data/sample/olist_order_values_sample.csv")["order_value"].to_numpy()
    q = (np.arange(400_000) + 0.5) / 400_000
    sim = positive_part_ppf(q, shape)
    for level in [0.25, 0.5, 0.75, 0.9, 0.99]:
        assert np.quantile(sim, level) == pytest.approx(np.quantile(real, level), rel=0.10)
    assert sim.std() / sim.mean() == pytest.approx(real.std() / real.mean(), rel=0.15)


def test_a_single_lognormal_would_understate_the_spread(shape):
    """The reason the spliced model exists, asserted rather than claimed."""
    q = (np.arange(400_000) + 0.5) / 400_000
    spliced = positive_part_ppf(q, shape)
    lognormal_only = stats.lognorm.ppf(q, shape.log_sigma, scale=np.exp(shape.log_mu))
    cv_spliced = spliced.std() / spliced.mean()
    assert lognormal_only.std() / lognormal_only.mean() < 0.8 * cv_spliced
    assert cv_spliced == pytest.approx(shape.empirical_cv, rel=0.15)


def test_zero_rate_matches_the_base_rate(revenue_spec):
    exp = simulate_experiment(revenue_spec, 200_000, 0.0, 0.0, generator("test_zero_rate"))
    assert (exp.outcome > 0).mean() == pytest.approx(0.10, abs=0.005)


def test_conversion_marginal_is_bernoulli():
    spec = MetricSpec("conversion", 0.04)
    y = marginal_ppf((np.arange(100_000) + 0.5) / 100_000, spec, 0.04)
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert y.mean() == pytest.approx(0.04, abs=1e-3)


def test_sessions_are_overdispersed():
    spec = MetricSpec("sessions", 3.0, dispersion=0.8)
    y = marginal_ppf((np.arange(400_000) + 0.5) / 400_000, spec, 3.0)
    assert y.mean() == pytest.approx(3.0, rel=0.02)
    assert y.var() > 2.0 * y.mean()


def test_metric_moments_agree_with_simulation(revenue_spec):
    mean, var = metric_moments(revenue_spec)
    exp = simulate_experiment(revenue_spec, 500_000, 0.0, 0.0, generator("test_moments"))
    assert exp.outcome.mean() == pytest.approx(mean, rel=0.05)
    assert exp.outcome.var() == pytest.approx(var, rel=0.20)


def test_quadrature_matches_the_closed_form_moments(shape, revenue_spec):
    """The zero inflated mean and variance have an exact closed form."""
    first, second = _positive_part_moments(shape)
    rate = revenue_spec.base_rate
    q_mean, q_second, _ = _continuous_moments(revenue_spec, 0.0)
    # the quadrature returns the zero inflated moments, the closed form the
    # positive part, so they differ by exactly the rate.
    assert q_mean == pytest.approx(rate * first, rel=1e-4)
    assert q_second == pytest.approx(rate * second, rel=1e-4)
    mean, var = metric_moments(revenue_spec)
    assert mean == pytest.approx(rate * first, rel=1e-10)
    assert var == pytest.approx(rate * second - (rate * first) ** 2, rel=1e-10)


def test_closed_form_positive_part_matches_the_real_data(shape):
    first, second = _positive_part_moments(shape)
    sd = np.sqrt(second - first**2)
    assert first == pytest.approx(shape.empirical_mean, rel=0.06)
    assert sd / first == pytest.approx(shape.empirical_cv, rel=0.06)


@pytest.mark.parametrize("target", [0.2, 0.5, 0.7])
def test_copula_solve_hits_the_target_exactly(revenue_spec, target):
    """The solve is quadrature, not simulation, so it is exact to solver tolerance."""
    r = solve_copula_r(revenue_spec, target)
    assert expected_correlation(revenue_spec, r) == pytest.approx(target, abs=1e-4)


@pytest.mark.parametrize("family,base", [("revenue", 0.10), ("conversion", 0.04), ("sessions", 3.0)])
def test_quadrature_agrees_with_simulation(shape, family, base):
    spec = MetricSpec(family, base, shape=shape if family == "revenue" else None, name=f"q_{family}")
    r = solve_copula_r(spec, 0.5)
    assert achieved_correlation(spec, r) == pytest.approx(expected_correlation(spec, r), abs=0.02)


@pytest.mark.parametrize("target", [0.2, 0.5, 0.7])
def test_simulated_data_shows_the_target_correlation(revenue_spec, target):
    """Sampling tolerance is wide on purpose.

    With a Pareto tail at xi close to 0.25 the fourth moment barely exists, so
    the sample correlation of a single experiment has a standard deviation of
    about 0.012 at n = 300,000. The tolerance below is roughly three of those.
    """
    r = solve_copula_r(revenue_spec, target)
    exp = simulate_experiment(
        revenue_spec, 300_000, 0.0, target, generator("test_corr", target), copula_r=r
    )
    assert np.corrcoef(exp.pre, exp.outcome)[0, 1] == pytest.approx(target, abs=0.04)


def test_true_effect_is_what_was_injected(revenue_spec):
    true_effect = 0.05
    lifts = []
    for rep in range(60):
        exp = simulate_experiment(
            revenue_spec, 200_000, true_effect, 0.3, generator("test_effect", rep)
        )
        lifts.append(exp.outcome[exp.arm].mean() / exp.outcome[~exp.arm].mean() - 1.0)
    se = float(np.std(lifts, ddof=1) / np.sqrt(len(lifts)))
    assert abs(float(np.mean(lifts)) - true_effect) < 3 * se


def test_no_effect_means_no_effect(revenue_spec):
    from expcost.estimators import diff_in_means

    exp = simulate_experiment(revenue_spec, 400_000, 0.0, 0.5, generator("test_null"))
    assert diff_in_means(exp.outcome, exp.arm).p_value > 0.001


def test_stratified_assignment_balances_every_stratum(revenue_spec):
    exp = simulate_experiment(
        revenue_spec, 50_000, 0.0, 0.3, generator("test_strat"), stratified_assignment=True
    )
    for s in np.unique(exp.stratum):
        assert exp.arm[exp.stratum == s].mean() == pytest.approx(0.5, abs=0.001)


def test_simulation_is_deterministic(revenue_spec):
    a = simulate_experiment(revenue_spec, 10_000, 0.02, 0.4, generator("determinism", 7))
    b = simulate_experiment(revenue_spec, 10_000, 0.02, 0.4, generator("determinism", 7))
    assert np.array_equal(a.outcome, b.outcome)
    assert np.array_equal(a.arm, b.arm)
    assert np.array_equal(a.pre, b.pre)


def test_fit_recovers_known_lognormal_parameters():
    values = generator("test_fit_recovery").lognormal(4.0, 0.9, 200_000)
    fitted = fit_revenue_shape(values, source="synthetic")
    assert fitted.log_mu == pytest.approx(4.0, abs=0.02)
    assert fitted.log_sigma == pytest.approx(0.9, abs=0.02)
