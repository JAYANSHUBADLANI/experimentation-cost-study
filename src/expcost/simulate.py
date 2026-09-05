"""The metric simulator.

Generates user level outcomes with a known injected effect, a pre period
covariate whose correlation with the outcome is controllable on the Pearson
scale, and a discrete stratum. Three metric families are supported:

  revenue      zero inflated, positive part spliced lognormal / generalized
               Pareto, calibrated to Olist order values
  conversion   Bernoulli at a low base rate
  sessions     negative binomial, overdispersed

Dependence between the pre period value and the outcome is induced with a
Gaussian copula, so the marginal distribution stays exactly the calibrated one
whatever the correlation is. The copula parameter needed to hit a target
Pearson correlation is solved by quadrature rather than by simulation: for a
heavy tailed marginal the sample correlation converges slowly, and a solver
run against a simulated objective lands on a root of the noise. The exact
route is worth the extra code, it moved the solution at rho = 0.7 by 0.035,
which is far more than the Monte Carlo standard error of 0.002.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import optimize, special, stats

from .calibration import RevenueShape
from .seeds import generator

Family = Literal["revenue", "conversion", "sessions"]
Mechanism = Literal["rate", "value"]

N_STRATA = 5
_NB_TABLE_CACHE: dict[tuple[float, float], np.ndarray] = {}
_COPULA_PILOT_N = 400_000
_CACHE_PATH = Path("results/copula_cache.json")
# Bumped whenever the quadrature changes, so a stale cache cannot survive a
# change of method and silently feed old numbers into a new run.
_QUADRATURE_VERSION = "legendre-positive-region-v1"


@dataclass(frozen=True)
class MetricSpec:
    """One metric to run an experiment on."""

    family: Family
    base_rate: float
    shape: RevenueShape | None = None
    dispersion: float = 0.8
    mechanism: Mechanism = "rate"
    name: str = ""

    def label(self) -> str:
        return self.name or f"{self.family}_{self.base_rate:g}"


def positive_part_ppf(q: np.ndarray, shape: RevenueShape) -> np.ndarray:
    """Quantile function of the spliced lognormal / generalized Pareto part.

    Written out in closed form rather than through scipy's distribution
    objects. The generic .ppf wrappers cost two to three times as much, and
    this runs inside every replication of the grid.
    """
    q = np.asarray(q, dtype=float)
    tail_q = shape.tail_quantile
    body_mass = special.ndtr((np.log(shape.threshold) - shape.log_mu) / shape.log_sigma)
    out = np.empty_like(q)
    is_body = q <= tail_q
    body_q = np.clip(q[is_body] / tail_q * body_mass, 1e-300, 1 - 1e-16)
    out[is_body] = np.exp(shape.log_mu + shape.log_sigma * special.ndtri(body_q))
    tail_u = (q[~is_body] - tail_q) / (1.0 - tail_q)
    xi, scale = shape.gpd_xi, shape.gpd_scale
    out[~is_body] = shape.threshold + scale / xi * ((1.0 - tail_u) ** (-xi) - 1.0)
    return out


def marginal_ppf(u: np.ndarray, spec: MetricSpec, rate: float) -> np.ndarray:
    """Quantile function of the metric marginal at the given rate parameter."""
    u = np.asarray(u, dtype=float)
    if spec.family == "conversion":
        return (u >= 1.0 - rate).astype(float)
    if spec.family == "sessions":
        return _nbinom_ppf(u, spec.dispersion, rate)
    if spec.family == "revenue":
        assert spec.shape is not None, "revenue metric needs a fitted RevenueShape"
        out = np.zeros_like(u)
        positive = u >= 1.0 - rate
        if positive.any():
            q = (u[positive] - (1.0 - rate)) / rate
            out[positive] = positive_part_ppf(np.clip(q, 1e-12, 1 - 1e-12), spec.shape)
        return out
    raise ValueError(f"unknown family {spec.family}")


def _nbinom_ppf(u: np.ndarray, dispersion: float, rate: float) -> np.ndarray:
    """Negative binomial quantiles by table lookup.

    scipy's generic discrete ppf costs about 3 microseconds per draw, which is
    eight times the rest of a replication put together. The support here is a
    few dozen integers, so tabulating the cdf once and binary searching it is
    exact and roughly sixty times faster.
    """
    key = (dispersion, rate)
    table = _NB_TABLE_CACHE.get(key)
    if table is None:
        nb = stats.nbinom(dispersion, dispersion / (dispersion + rate))
        top = int(nb.ppf(1 - 1e-15)) + 2
        table = nb.cdf(np.arange(top))
        table[-1] = 1.0
        _NB_TABLE_CACHE[key] = table
    return np.searchsorted(table, u, side="left").astype(float)


GL_NODES = 300
GL_ZMAX = 9.0


def _positive_part_moments(shape: RevenueShape) -> tuple[float, float]:
    """First two moments of the spliced positive part, in closed form.

    Body piece: the partial lognormal moments up to the splice threshold.
    Tail piece: the generalized Pareto moments, which exist because the fitted
    xi is below 1/2. Used as the exact reference the quadrature is checked
    against in the tests.
    """
    mu, sigma, u = shape.log_mu, shape.log_sigma, shape.threshold
    tail_q, xi, scale = shape.tail_quantile, shape.gpd_xi, shape.gpd_scale
    body_mass = stats.lognorm.cdf(u, sigma, scale=np.exp(mu))
    ratio = tail_q / body_mass
    m1_body = np.exp(mu + sigma**2 / 2) * stats.norm.cdf((np.log(u) - mu - sigma**2) / sigma)
    m2_body = np.exp(2 * mu + 2 * sigma**2) * stats.norm.cdf((np.log(u) - mu - 2 * sigma**2) / sigma)
    mean_excess = scale / (1.0 - xi)
    second_excess = 2.0 * scale**2 / ((1.0 - xi) * (1.0 - 2.0 * xi))
    m1_tail = (1.0 - tail_q) * (u + mean_excess)
    m2_tail = (1.0 - tail_q) * (u**2 + 2.0 * u * mean_excess + second_excess)
    return float(ratio * m1_body + m1_tail), float(ratio * m2_body + m2_tail)


def _positive_region_nodes(rate: float):
    """Gauss-Legendre nodes on the region where the outcome is non zero.

    A zero inflated outcome is a step function of the latent normal, and
    Gauss-Hermite over the whole line converges badly across that step: the
    estimated mean swung by 0.9 across node counts. Integrating only over the
    smooth positive region fixes it, and reproduces the closed form above to
    five significant figures.
    """
    cut = stats.norm.ppf(1.0 - rate)
    x, w = np.polynomial.legendre.leggauss(GL_NODES)
    half = 0.5 * (GL_ZMAX - cut)
    return half * x + 0.5 * (GL_ZMAX + cut), half * w, cut


def _continuous_moments(spec: MetricSpec, r: float, rate: float | None = None):
    """Mean, second moment and E[Y1 Y2] for a zero inflated continuous metric."""
    rate = spec.base_rate if rate is None else rate
    z, w, cut = _positive_region_nodes(rate)
    q = np.clip((special.ndtr(z) - (1.0 - rate)) / rate, 1e-15, 1 - 1e-15)
    h = positive_part_ppf(q, spec.shape)
    phi = stats.norm.pdf(z)
    mean = float(w @ (phi * h))
    second = float(w @ (phi * h * h))
    if r <= 0.0:
        return mean, second, mean * mean
    root = np.sqrt(max(1.0 - r * r, 0.0))
    joint = stats.norm.pdf((z[None, :] - r * z[:, None]) / root) / root
    density = phi[:, None] * joint
    cross = float(np.einsum("i,i,ij,j,j->", w, h, density, h, w))
    return mean, second, cross

def _bernoulli_correlation(p: float, r: float) -> float:
    """Exact correlation of two Bernoulli marginals under a Gaussian copula."""
    cut = stats.norm.ppf(1.0 - p)
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]])
    joint = float(mvn.cdf([-cut, -cut]))
    return (joint - p * p) / (p * (1.0 - p))


def _discrete_correlation(spec: MetricSpec, r: float, upper_q: float = 1 - 1e-7) -> float:
    """Exact correlation for a count marginal, via upper orthant probabilities.

    For non negative integers, E[XY] = sum_{i>=1} sum_{j>=1} P(X >= i, Y >= j),
    and each of those is a bivariate normal upper orthant probability.
    """
    k = spec.dispersion
    nb = stats.nbinom(k, k / (k + spec.base_rate))
    top = int(nb.ppf(upper_q)) + 1
    levels = np.arange(1, top + 1)
    cuts = stats.norm.ppf(np.clip(nb.cdf(levels - 1), 1e-15, 1 - 1e-15))
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]])
    grid = np.stack(np.meshgrid(-cuts, -cuts, indexing="ij"), axis=-1).reshape(-1, 2)
    cross = float(mvn.cdf(grid).sum())
    mean, var = float(nb.mean()), float(nb.var())
    return (cross - mean * mean) / var


def expected_correlation(spec: MetricSpec, copula_r: float) -> float:
    """Exact Pearson correlation produced by a Gaussian copula parameter."""
    if copula_r <= 0.0:
        return 0.0
    if spec.family == "conversion":
        return _bernoulli_correlation(spec.base_rate, copula_r)
    if spec.family == "sessions":
        return _discrete_correlation(spec, copula_r)
    mean, second, cross = _continuous_moments(spec, copula_r)
    var = second - mean * mean
    if var <= 0:
        return 0.0
    return (cross - mean * mean) / var


def achieved_correlation(spec: MetricSpec, copula_r: float, n: int = _COPULA_PILOT_N) -> float:
    """Monte Carlo check on expected_correlation, used only in the tests."""
    rng = generator("copula_pilot", spec.label(), f"{copula_r:.6f}")
    z1 = rng.standard_normal(n)
    z2 = copula_r * z1 + np.sqrt(max(1.0 - copula_r**2, 0.0)) * rng.standard_normal(n)
    pre = marginal_ppf(stats.norm.cdf(z1), spec, spec.base_rate)
    post = marginal_ppf(stats.norm.cdf(z2), spec, spec.base_rate)
    if pre.std() == 0 or post.std() == 0:
        return 0.0
    return float(np.corrcoef(pre, post)[0, 1])


def _cache_key(spec: MetricSpec, target_rho: float) -> str:
    bits = [_QUADRATURE_VERSION, spec.family, f"{spec.base_rate:g}", f"{spec.dispersion:g}", f"{target_rho:g}"]
    if spec.shape is not None:
        bits.append(f"{spec.shape.log_mu:.6f}:{spec.shape.log_sigma:.6f}:{spec.shape.gpd_xi:.6f}")
    return "|".join(bits)


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def solve_copula_r(spec: MetricSpec, target_rho: float, use_cache: bool = True) -> float:
    """Copula correlation that yields the requested Pearson correlation."""
    if target_rho <= 0.0:
        return 0.0
    key = _cache_key(spec, target_rho)
    cache = _load_cache() if use_cache else {}
    if key in cache:
        return float(cache[key])

    def gap(r: float) -> float:
        return expected_correlation(spec, r) - target_rho

    hi = 0.9995
    solution = hi if gap(hi) < 0 else float(optimize.brentq(gap, 1e-6, hi, xtol=1e-6, rtol=1e-8))
    if use_cache:
        cache[key] = solution
        _save_cache(cache)
    return solution


@dataclass(frozen=True)
class Experiment:
    """One simulated experiment's user level data."""

    arm: np.ndarray
    outcome: np.ndarray
    pre: np.ndarray
    stratum: np.ndarray
    true_effect: float
    spec: MetricSpec

    @property
    def n(self) -> int:
        return int(self.arm.size)


def simulate_experiment(
    spec: MetricSpec,
    n: int,
    relative_effect: float,
    rho: float,
    rng: np.random.Generator,
    allocation: float = 0.5,
    copula_r: float | None = None,
    stratified_assignment: bool = False,
) -> Experiment:
    """Simulate one experiment.

    relative_effect is the true relative lift applied to the treated arm's
    outcome mean. rho is the target Pearson correlation between the pre period
    covariate and the control outcome.
    """
    if copula_r is None:
        copula_r = solve_copula_r(spec, rho)

    z1 = rng.standard_normal(n)
    z2 = copula_r * z1 + np.sqrt(max(1.0 - copula_r**2, 0.0)) * rng.standard_normal(n)
    pre = marginal_ppf(special.ndtr(z1), spec, spec.base_rate)

    # Strata are pre period quintiles, which is what a team actually has.
    edges = np.quantile(z1, np.linspace(0, 1, N_STRATA + 1)[1:-1])
    stratum = np.searchsorted(edges, z1)

    if stratified_assignment:
        arm = np.zeros(n, dtype=bool)
        for s in range(N_STRATA):
            idx = rng.permutation(np.flatnonzero(stratum == s))
            arm[idx[: int(round(allocation * idx.size))]] = True
    else:
        arm = rng.random(n) < allocation

    u2 = special.ndtr(z2)
    outcome = marginal_ppf(u2, spec, spec.base_rate)
    if relative_effect != 0.0 and arm.any():
        if spec.family == "revenue" and spec.mechanism == "value":
            outcome[arm] *= 1.0 + relative_effect
        else:
            treated_rate = spec.base_rate * (1.0 + relative_effect)
            if spec.family == "conversion":
                treated_rate = min(treated_rate, 1.0 - 1e-9)
            # only the treated half needs the second quantile evaluation
            outcome[arm] = marginal_ppf(u2[arm], spec, treated_rate)

    return Experiment(
        arm=arm,
        outcome=outcome,
        pre=pre,
        stratum=stratum,
        true_effect=relative_effect,
        spec=spec,
    )


def metric_moments(spec: MetricSpec, rate: float | None = None) -> tuple[float, float]:
    """Exact mean and variance of the metric marginal, by quadrature.

    Used to predict required sample sizes without simulating them.
    """
    rate = spec.base_rate if rate is None else rate
    if spec.family == "conversion":
        return rate, rate * (1.0 - rate)
    if spec.family == "sessions":
        nb = stats.nbinom(spec.dispersion, spec.dispersion / (spec.dispersion + rate))
        return float(nb.mean()), float(nb.var())
    first, second = _positive_part_moments(spec.shape)
    return rate * first, rate * second - (rate * first) ** 2
