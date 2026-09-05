"""Treatment effect estimators, implemented directly rather than imported.

All of them return the same Estimate record so the power study can treat them
interchangeably. Each is cross checked against statsmodels or scipy in the test
suite where an equivalent exists, and against a hand computable case where it
does not.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Estimate:
    """One estimated treatment effect."""

    effect: float
    se: float
    df: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    control_mean: float
    method: str

    @property
    def relative_effect(self) -> float:
        if self.control_mean == 0:
            return float("nan")
        return self.effect / self.control_mean

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def _welch(mean_t, var_t, n_t, mean_c, var_c, n_c, method, alpha=0.05):
    se = float(np.sqrt(var_t / n_t + var_c / n_c))
    effect = float(mean_t - mean_c)
    if se == 0:
        return Estimate(
            effect=effect, se=0.0, df=float(n_t + n_c - 2),
            t_stat=0.0, p_value=1.0, ci_low=effect, ci_high=effect,
            control_mean=float(mean_c), method=method,
        )
    num = (var_t / n_t + var_c / n_c) ** 2
    den = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    df = float(num / den)
    t_stat = effect / se
    half = float(stats.t.ppf(1 - alpha / 2, df) * se)
    return Estimate(
        effect=effect,
        se=se,
        df=df,
        t_stat=float(t_stat),
        p_value=float(2 * stats.t.sf(abs(t_stat), df)),
        ci_low=effect - half,
        ci_high=effect + half,
        control_mean=float(mean_c),
        method=method,
    )


def diff_in_means(outcome: np.ndarray, arm: np.ndarray, alpha: float = 0.05) -> Estimate:
    """Plain two sample difference in means with a Welch t-test."""
    y_t, y_c = outcome[arm], outcome[~arm]
    return _welch(
        y_t.mean(), y_t.var(ddof=1), y_t.size,
        y_c.mean(), y_c.var(ddof=1), y_c.size,
        "diff_in_means", alpha,
    )


def cuped_theta(outcome: np.ndarray, covariate: np.ndarray) -> float:
    """Variance minimising coefficient, cov(Y, X) / var(X) on the pooled sample."""
    centred_x = covariate - covariate.mean()
    denominator = float(centred_x @ centred_x)
    if denominator == 0:
        return 0.0
    return float((centred_x @ (outcome - outcome.mean())) / denominator)


def cuped(
    outcome: np.ndarray, arm: np.ndarray, covariate: np.ndarray, alpha: float = 0.05
) -> Estimate:
    """CUPED: subtract theta times the centred pre period covariate, then test.

    theta is estimated on the pooled sample as in Deng, Xu, Kohavi and Walker
    (2013). The extra uncertainty from estimating theta is O(1/n) relative to
    the leading term and is not included, which is the usual convention and is
    checked against the regression form in the tests.
    """
    theta = cuped_theta(outcome, covariate)
    adjusted = outcome - theta * (covariate - covariate.mean())
    est = diff_in_means(adjusted, arm, alpha)
    # Report the control mean on the original scale, the adjustment is mean zero.
    return replace(est, method="cuped", control_mean=float(outcome[~arm].mean()))


def lin_regression_adjustment(
    outcome: np.ndarray, arm: np.ndarray, covariate: np.ndarray, alpha: float = 0.05
) -> Estimate:
    """Lin (2013) interacted regression adjustment with HC2 standard errors.

    This is CUPED's stronger sibling: it fits a separate slope in each arm, so
    it is never asymptotically worse than the unadjusted estimator even when
    the covariate relationship differs by arm. Included so that the adjusted
    estimator being compared against is not a straw man.
    """
    n = outcome.size
    x = covariate - covariate.mean()

    # The interacted model [1, T, X, T X] spans the same column space as two
    # separate within arm regressions, so the estimator, the residuals and the
    # leverages are all identical. Computing it that way is O(n) in a handful
    # of vector passes instead of forming an n by 4 design and multiplying
    # through it four times.
    contribution = np.zeros(n)
    resid = np.zeros(n)
    leverage = np.zeros(n)
    effect = 0.0
    for is_treated, sign in ((True, 1.0), (False, -1.0)):
        idx = arm if is_treated else ~arm
        x_a, y_a = x[idx], outcome[idx]
        n_a = x_a.size
        x_bar, y_bar = x_a.mean(), y_a.mean()
        x_dev = x_a - x_bar
        sxx = float(x_dev @ x_dev)
        slope = float(x_dev @ y_a) / sxx if sxx > 0 else 0.0
        effect += sign * (y_bar - slope * x_bar)
        contribution[idx] = sign * (1.0 / n_a - x_bar * x_dev / sxx if sxx > 0 else 1.0 / n_a)
        resid[idx] = y_a - (y_bar + slope * x_dev)
        leverage[idx] = 1.0 / n_a + (x_dev * x_dev / sxx if sxx > 0 else 0.0)

    hc2 = resid**2 / np.clip(1.0 - leverage, 1e-12, None)
    variance = float((contribution * contribution) @ hc2)
    effect = float(effect)
    se = float(np.sqrt(variance))
    df = float(n - 4)
    t_stat = effect / se if se > 0 else 0.0
    half = float(stats.t.ppf(1 - alpha / 2, df) * se)
    return Estimate(
        effect=effect,
        se=se,
        df=df,
        t_stat=float(t_stat),
        p_value=float(2 * stats.t.sf(abs(t_stat), df)),
        ci_low=effect - half,
        ci_high=effect + half,
        control_mean=float(outcome[~arm].mean()),
        method="lin_adjusted",
    )


def post_stratified(
    outcome: np.ndarray,
    arm: np.ndarray,
    stratum: np.ndarray,
    alpha: float = 0.05,
    method: str = "post_stratified",
) -> Estimate:
    """Stratum weighted difference in means.

    Weights are the observed stratum shares. With stratified assignment this is
    the stratified estimator, with simple assignment it is post stratification.
    The formula is identical, the difference is in the design that produced the
    data, so the same code serves both and the caller names which it is.
    """
    n = outcome.size
    n_strata = int(stratum.max()) + 1
    # Grouped reductions over the 2 * n_strata cells, rather than a Python loop
    # with boolean masks over the whole vector once per stratum.
    cell = stratum.astype(np.intp) * 2 + arm.astype(np.intp)
    size = 2 * n_strata
    count = np.bincount(cell, minlength=size).astype(float)
    total = np.bincount(cell, weights=outcome, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_mean = total / count
    resid = outcome - cell_mean[cell]
    sum_sq = np.bincount(cell, weights=resid * resid, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_var = sum_sq / (count - 1.0)

    control, treated = np.arange(0, size, 2), np.arange(1, size, 2)
    usable = (count[control] >= 2) & (count[treated] >= 2)
    weight = (count[control] + count[treated]) / n
    effect = float(np.sum(weight[usable] * (cell_mean[treated][usable] - cell_mean[control][usable])))
    var = float(
        np.sum(
            weight[usable] ** 2
            * (
                cell_var[treated][usable] / count[treated][usable]
                + cell_var[control][usable] / count[control][usable]
            )
        )
    )
    control_mean = float(np.sum(weight[usable] * cell_mean[control][usable]))
    se = float(np.sqrt(var))
    df = float(n - 2 * n_strata)
    t_stat = effect / se if se > 0 else 0.0
    half = float(stats.t.ppf(1 - alpha / 2, df) * se) if se > 0 else 0.0
    return Estimate(
        effect=float(effect),
        se=se,
        df=df,
        t_stat=float(t_stat),
        p_value=float(2 * stats.t.sf(abs(t_stat), df)),
        ci_low=float(effect - half),
        ci_high=float(effect + half),
        control_mean=float(control_mean),
        method=method,
    )


def stratified(outcome, arm, stratum, alpha: float = 0.05) -> Estimate:
    """Stratified estimator, for data that was assigned within strata."""
    return post_stratified(outcome, arm, stratum, alpha, method="stratified")


def cuped_plus_strata(
    outcome: np.ndarray,
    arm: np.ndarray,
    covariate: np.ndarray,
    stratum: np.ndarray,
    alpha: float = 0.05,
) -> Estimate:
    """Both adjustments at once, to see whether they stack."""
    theta = cuped_theta(outcome, covariate)
    adjusted = outcome - theta * (covariate - covariate.mean())
    est = post_stratified(adjusted, arm, stratum, alpha, method="cuped_plus_strata")
    return replace(est, control_mean=float(outcome[~arm].mean()))


ESTIMATORS = {
    "diff_in_means": lambda e: diff_in_means(e.outcome, e.arm),
    "cuped": lambda e: cuped(e.outcome, e.arm, e.pre),
    "lin_adjusted": lambda e: lin_regression_adjustment(e.outcome, e.arm, e.pre),
    "post_stratified": lambda e: post_stratified(e.outcome, e.arm, e.stratum),
    "cuped_plus_strata": lambda e: cuped_plus_strata(e.outcome, e.arm, e.pre, e.stratum),
}
