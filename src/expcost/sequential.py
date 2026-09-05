"""Two named sequential procedures, implemented rather than imported.

"A sequential test" is a family, not a method, so both procedures used here
are named, cited and implemented in full.

1. The mixture sequential probability ratio test (mSPRT) with a normal mixing
   density, as in Johari, Pekelis and Walsh, "Always Valid Inference:
   Continuous Monitoring of A/B Tests" (KDD 2017, Management Science 2022).
   This is an always valid test: the experimenter may look as often as they
   like, for as long as they like, and stop whenever they please, and the type
   I error over the whole sequence is still bounded by alpha. It needs no
   pre-committed horizon, which is what makes it the right comparison for a
   team that peeks daily and stops when it likes.

2. Group sequential boundaries from the Lan and DeMets (1983) alpha spending
   framework using the O'Brien and Fleming (1979) spending function. This one
   does require a pre-committed maximum sample size, and spends very little
   alpha early, so it is strict at the first looks and close to the fixed
   horizon critical value at the last.

The two differ in what they assume, so they are reported separately and never
averaged together.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def msprt_likelihood_ratio(
    effect: np.ndarray, variance: np.ndarray, tau_squared: float
) -> np.ndarray:
    """Mixture likelihood ratio against the point null of no effect.

    With an estimate distributed as normal(theta, V) and a normal(0, tau^2)
    mixing density over the alternative, the mixture likelihood ratio is

        sqrt(V / (V + tau^2)) * exp(effect^2 tau^2 / (2 V (V + tau^2)))

    which is the expression used throughout. tau^2 has to be chosen before the
    experiment starts, and it sets which effect sizes the test is most
    sensitive to.
    """
    effect = np.asarray(effect, dtype=float)
    variance = np.asarray(variance, dtype=float)
    total = variance + tau_squared
    return np.sqrt(variance / total) * np.exp(effect**2 * tau_squared / (2.0 * variance * total))


def always_valid_p_value(
    effect: np.ndarray, variance: np.ndarray, tau_squared: float
) -> np.ndarray:
    """Running always valid p value, the minimum of 1/LR so far.

    Taking the running minimum is what makes the quantity monotone, so that
    "stop the first time it drops below alpha" is a valid stopping rule.
    """
    ratio = msprt_likelihood_ratio(effect, variance, tau_squared)
    with np.errstate(divide="ignore"):
        p = np.minimum(1.0, 1.0 / ratio)
    return np.minimum.accumulate(p, axis=-1)


def obrien_fleming_spending(information_fraction: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Cumulative alpha spent by the O'Brien-Fleming spending function."""
    t = np.clip(np.asarray(information_fraction, dtype=float), 1e-12, 1.0)
    return 2.0 * (1.0 - stats.norm.cdf(stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(t)))


def classical_obrien_fleming_boundaries(n_looks: int, alpha: float = 0.05) -> np.ndarray:
    """The original O'Brien and Fleming (1979) boundaries, equally spaced looks.

    Constant on the score scale, so on the z scale the boundary falls as
    c * sqrt(K / k). One constant is solved so that the overall two sided error
    is alpha. Kept because its values are tabulated in the literature, which
    gives the recursion below something external to be checked against.
    """
    fractions = np.arange(1, n_looks + 1) / n_looks

    def crossing_probability(c: float) -> float:
        boundaries = c * np.sqrt(n_looks / np.arange(1, n_looks + 1))
        return _crossing_probability(fractions, boundaries)

    lo, hi = 0.5, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if crossing_probability(mid) > alpha:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    return c * np.sqrt(n_looks / np.arange(1, n_looks + 1))


def _grid(limit: float, points: int):
    grid = np.linspace(-limit, limit, points)
    return grid, float(grid[1] - grid[0])


def _propagate(density, grid, step, sd):
    """Convolve the sub density with the next Brownian increment."""
    half = min(int(np.ceil(8.0 * sd / step)), grid.size - 1)
    offsets = np.arange(-half, half + 1) * step
    kernel = stats.norm.pdf(offsets, scale=sd) * step
    return np.convolve(density, kernel, mode="same")


def _mass_outside(density, grid, step, cut: float) -> float:
    """Probability mass outside [-cut, cut], interpolated between grid points.

    Interpolating matters: if the tail is read off whole grid cells the solved
    boundary cannot be more accurate than one cell, and at the first look the
    quantity being resolved is around 1e-5.
    """
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1])) * step])
    total = cumulative[-1]
    below = float(np.interp(-cut, grid, cumulative))
    above = total - float(np.interp(cut, grid, cumulative))
    return below + above


def _crossing_probability(information_fractions, z_boundaries, points: int = 8001, limit: float = 14.0) -> float:
    """Probability a Brownian path crosses any of these z boundaries."""
    t = np.asarray(information_fractions, dtype=float)
    increments = np.diff(np.concatenate([[0.0], t]))
    grid, step = _grid(limit, points)
    density = None
    crossed = 0.0
    for k in range(t.size):
        sd = np.sqrt(increments[k])
        density = stats.norm.pdf(grid, scale=sd) if k == 0 else _propagate(density, grid, step, sd)
        cut = z_boundaries[k] * np.sqrt(t[k])
        crossed += _mass_outside(density, grid, step, cut)
        density = np.where(np.abs(grid) >= cut, 0.0, density)
    return float(crossed)


def alpha_spending_boundaries(
    information_fractions,
    alpha: float = 0.05,
    points: int = 8001,
    limit: float = 14.0,
) -> np.ndarray:
    """Group sequential boundaries on the z scale, by exact recursion.

    The score process has independent increments, so the joint law of the
    interim statistics is a Brownian motion observed at the information
    fractions. The sub density of the score on "not yet stopped" is carried
    forward, convolved with the next increment, and the next boundary is solved
    so that the newly spent alpha matches the spending function. This is the
    Armitage, McPherson and Rowe recursion that Lan-DeMets is built on, rather
    than an approximation to it.
    """
    t = np.asarray(information_fractions, dtype=float)
    spent = obrien_fleming_spending(t, alpha)
    increments = np.diff(np.concatenate([[0.0], t]))
    grid, step = _grid(limit, points)

    density = None
    boundaries = np.empty(t.size)
    for k in range(t.size):
        sd = np.sqrt(increments[k])
        density = stats.norm.pdf(grid, scale=sd) if k == 0 else _propagate(density, grid, step, sd)
        target = spent[k] - (spent[k - 1] if k > 0 else 0.0)
        lo, hi = 0.0, limit / max(np.sqrt(t[k]), 1e-12)
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if _mass_outside(density, grid, step, mid * np.sqrt(t[k])) > target:
                lo = mid
            else:
                hi = mid
        boundaries[k] = 0.5 * (lo + hi)
        density = np.where(np.abs(grid) >= boundaries[k] * np.sqrt(t[k]), 0.0, density)
    return boundaries


def fixed_horizon_boundary(alpha: float = 0.05) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def inflation_factor(
    information_fractions, alpha: float = 0.05, power: float = 0.80, points: int = 4001
) -> float:
    """How much larger the maximum sample size must be to keep the same power.

    A group sequential design pays for its early looks with a larger maximum
    horizon. This solves for the multiplier on the fixed horizon sample size
    that restores the target power, by simulating the Brownian path against
    the boundaries.
    """
    t = np.asarray(information_fractions, dtype=float)
    boundaries = alpha_spending_boundaries(t, alpha, points=points)
    drift_fixed = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)

    def attained_power(multiplier: float, draws: int = 200_000) -> float:
        rng = np.random.default_rng(12345)
        increments = np.diff(np.concatenate([[0.0], t]))
        noise = rng.standard_normal((draws, t.size)) * np.sqrt(increments)
        drift = drift_fixed * np.sqrt(multiplier)
        score = np.cumsum(noise, axis=1) + drift * t
        z = score / np.sqrt(t)
        return float((np.abs(z) >= boundaries).any(axis=1).mean())

    lo, hi = 1.0, 1.6
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if attained_power(mid) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
