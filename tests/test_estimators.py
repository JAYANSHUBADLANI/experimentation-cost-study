"""Estimators against hand computable cases and against statsmodels or scipy."""

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from expcost.estimators import (
    ESTIMATORS,
    cuped,
    cuped_theta,
    diff_in_means,
    lin_regression_adjustment,
    post_stratified,
)
from expcost.seeds import generator
from expcost.simulate import MetricSpec, simulate_experiment


def test_diff_in_means_by_hand():
    # treated mean 4, control mean 2, difference 2.
    y = np.array([3.0, 5.0, 1.0, 3.0])
    arm = np.array([True, True, False, False])
    est = diff_in_means(y, arm)
    assert est.effect == pytest.approx(2.0)
    # each arm has variance 2 with ddof=1 and n=2, so se = sqrt(2/2 + 2/2) = sqrt(2).
    assert est.se == pytest.approx(np.sqrt(2.0))
    assert est.t_stat == pytest.approx(2.0 / np.sqrt(2.0))
    assert est.control_mean == pytest.approx(2.0)


def test_diff_in_means_matches_scipy_welch():
    rng = generator("test_welch")
    y = rng.lognormal(0, 1, 4000)
    arm = rng.random(4000) < 0.5
    est = diff_in_means(y, arm)
    ref = stats.ttest_ind(y[arm], y[~arm], equal_var=False)
    assert est.t_stat == pytest.approx(ref.statistic, rel=1e-12)
    assert est.p_value == pytest.approx(ref.pvalue, rel=1e-12)


def test_cuped_theta_is_the_regression_slope():
    rng = generator("test_theta")
    x = rng.normal(size=5000)
    y = 3.0 * x + rng.normal(size=5000)
    assert cuped_theta(y, x) == pytest.approx(3.0, abs=0.05)
    assert cuped_theta(y, x) == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-10)


def test_cuped_removes_all_variance_when_the_covariate_is_the_outcome():
    # y = 2x exactly, so the adjusted outcome is constant and the se is zero.
    rng = generator("test_cuped_exact")
    x = rng.normal(size=500)
    arm = rng.random(500) < 0.5
    est = cuped(2.0 * x, arm, x)
    assert est.se == pytest.approx(0.0, abs=1e-9)
    assert est.effect == pytest.approx(0.0, abs=1e-9)


def test_cuped_matches_statsmodels_ols_at_scale():
    rng = generator("test_cuped_ols")
    n = 200_000
    x = rng.normal(size=n)
    arm = rng.random(n) < 0.5
    y = 1.0 + 0.5 * arm + 2.0 * x + rng.normal(size=n)
    est = cuped(y, arm, x)
    design = sm.add_constant(np.column_stack([arm.astype(float), x - x.mean()]))
    ols = sm.OLS(y, design).fit(cov_type="HC2")
    assert est.effect == pytest.approx(ols.params[1], rel=2e-3)
    assert est.se == pytest.approx(ols.bse[1], rel=2e-3)


def test_lin_adjustment_matches_statsmodels_hc2():
    rng = generator("test_lin")
    n = 20_000
    x = rng.normal(size=n)
    arm = rng.random(n) < 0.5
    y = 1.0 + 0.3 * arm + 2.0 * x + 0.7 * arm * x + rng.normal(size=n)
    est = lin_regression_adjustment(y, arm, x)
    centred = x - x.mean()
    design = sm.add_constant(np.column_stack([arm.astype(float), centred, arm * centred]))
    ols = sm.OLS(y, design).fit(cov_type="HC2")
    assert est.effect == pytest.approx(ols.params[1], rel=1e-10)
    assert est.se == pytest.approx(ols.bse[1], rel=1e-8)


def test_post_stratified_by_hand():
    # stratum 0: treated mean 10, control mean 8, difference 2, weight 4/8.
    # stratum 1: treated mean 30, control mean 26, difference 4, weight 4/8.
    # weighted difference = 0.5 * 2 + 0.5 * 4 = 3.
    y = np.array([9.0, 11.0, 7.0, 9.0, 29.0, 31.0, 25.0, 27.0])
    arm = np.array([True, True, False, False, True, True, False, False])
    stratum = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    est = post_stratified(y, arm, stratum)
    assert est.effect == pytest.approx(3.0)
    assert est.control_mean == pytest.approx(0.5 * 8.0 + 0.5 * 26.0)
    # each cell has variance 2 with n=2, so each stratum contributes 2/2 + 2/2 = 2.
    assert est.se == pytest.approx(np.sqrt(0.25 * 2 + 0.25 * 2))


def test_post_stratified_with_one_stratum_is_diff_in_means():
    rng = generator("test_ps_single")
    y = rng.lognormal(0, 1, 2000)
    arm = rng.random(2000) < 0.5
    a = post_stratified(y, arm, np.zeros(2000, dtype=int))
    b = diff_in_means(y, arm)
    assert a.effect == pytest.approx(b.effect, rel=1e-12)
    assert a.se == pytest.approx(b.se, rel=1e-12)


@pytest.mark.parametrize("name", ["diff_in_means", "cuped", "lin_adjusted", "post_stratified"])
def test_estimators_are_unbiased(name):
    spec = MetricSpec("conversion", 0.05, name="unbiased_check")
    true_effect = 0.10
    effects = [
        ESTIMATORS[name](
            simulate_experiment(spec, 20_000, true_effect, 0.4, generator("unbiased", name, rep))
        ).effect
        for rep in range(200)
    ]
    truth = 0.05 * true_effect
    se = float(np.std(effects, ddof=1) / np.sqrt(len(effects)))
    assert abs(float(np.mean(effects)) - truth) < 3 * se


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_estimators_hold_their_nominal_false_positive_rate(name):
    """Under the null every estimator must reject at about 5 percent."""
    spec = MetricSpec("conversion", 0.05, name="null_check")
    rejects = [
        ESTIMATORS[name](
            simulate_experiment(spec, 20_000, 0.0, 0.4, generator("null_rate", name, rep))
        ).significant()
        for rep in range(600)
    ]
    rate = float(np.mean(rejects))
    se = float(np.sqrt(0.05 * 0.95 / len(rejects)))
    assert abs(rate - 0.05) < 3 * se
