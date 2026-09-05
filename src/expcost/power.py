"""Power and required sample size, measured rather than assumed.

For each scenario the study simulates experiments at a ladder of sample sizes
and records, for every estimator, how often it reaches significance. The
sample size needed for 80 percent power is then read off the measured power
curve, and separately predicted from the variance each estimator achieved. The
two are reported side by side: they are different routes to the same number
and disagreeing would mean something is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from .estimators import ESTIMATORS
from .seeds import generator
from .simulate import MetricSpec, metric_moments, simulate_experiment, solve_copula_r

ESTIMATOR_ORDER = list(ESTIMATORS)


def z_sum(alpha: float, power: float) -> float:
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)


def analytic_n(
    spec: MetricSpec, relative_effect: float, alpha: float = 0.05, power: float = 0.80,
    variance: float | None = None,
) -> float:
    """Total users, both arms, for a balanced two sample test.

    Uses the exact metric variance from quadrature unless a measured variance
    is supplied, which is how the adjusted estimators get their prediction.
    """
    mean, var = metric_moments(spec)
    if variance is not None:
        var = variance
    delta = relative_effect * mean
    return float(4.0 * z_sum(alpha, power) ** 2 * var / delta**2)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95):
    """Wilson score interval, which behaves at rates near 0 and 1."""
    if trials == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return (float(centre - half), float(centre + half))


@dataclass(frozen=True)
class Chunk:
    """One unit of parallel work: a block of replications at one design point."""

    metric_key: str
    spec: MetricSpec
    effect: float
    rho: float
    copula_r: float
    n: int
    rep_start: int
    rep_stop: int
    alpha: float = 0.05
    stratified_assignment: bool = False


def run_chunk(chunk: Chunk) -> tuple[str, float, int, np.ndarray, np.ndarray]:
    """Run a block of replications and return per replication p values and ses."""
    reps = chunk.rep_stop - chunk.rep_start
    p_values = np.empty((len(ESTIMATOR_ORDER), reps))
    standard_errors = np.empty_like(p_values)
    label = f"power|{chunk.metric_key}|{chunk.effect:g}|{chunk.rho:g}|{chunk.n}"
    for j, rep in enumerate(range(chunk.rep_start, chunk.rep_stop)):
        experiment = simulate_experiment(
            chunk.spec,
            chunk.n,
            chunk.effect,
            chunk.rho,
            generator(label, rep),
            copula_r=chunk.copula_r,
            stratified_assignment=chunk.stratified_assignment,
        )
        for i, name in enumerate(ESTIMATOR_ORDER):
            estimate = ESTIMATORS[name](experiment)
            p_values[i, j] = estimate.p_value
            standard_errors[i, j] = estimate.se
    return chunk.metric_key, chunk.effect, chunk.n, p_values, standard_errors


def build_chunks(
    metric_key: str,
    spec: MetricSpec,
    effects,
    ladder,
    replications: int,
    rho: float,
    alpha: float = 0.05,
    users_per_chunk: int = 2_000_000,
) -> list[Chunk]:
    """Split the work into chunks of roughly equal cost.

    Chunks are sized by simulated users rather than by replication count, so a
    design point at n = 400,000 does not hand one worker sixteen times the work
    of a design point at n = 25,000 and leave the rest idle at the end.
    """
    copula_r = solve_copula_r(spec, rho)
    chunks: list[Chunk] = []
    for effect in effects:
        base = analytic_n(spec, effect, alpha=alpha)
        for multiplier in ladder:
            n = int(round(base * multiplier / 2) * 2)
            block = int(np.clip(users_per_chunk // max(n, 1), 1, replications))
            for start in range(0, replications, block):
                chunks.append(
                    Chunk(
                        metric_key=metric_key,
                        spec=spec,
                        effect=effect,
                        rho=rho,
                        copula_r=copula_r,
                        n=n,
                        rep_start=start,
                        rep_stop=min(start + block, replications),
                        alpha=alpha,
                    )
                )
    return chunks


def collect(results, alpha: float = 0.05) -> pd.DataFrame:
    """Turn raw chunk output into one row per design point and estimator."""
    buckets: dict[tuple[str, float, int, str], dict[str, list]] = {}
    for metric_key, effect, n, p_values, standard_errors in results:
        for i, name in enumerate(ESTIMATOR_ORDER):
            key = (metric_key, effect, n, name)
            bucket = buckets.setdefault(key, {"p": [], "se": []})
            bucket["p"].append(p_values[i])
            bucket["se"].append(standard_errors[i])
    rows = []
    for (metric_key, effect, n, name), bucket in buckets.items():
        p = np.concatenate(bucket["p"])
        se = np.concatenate(bucket["se"])
        successes = int((p < alpha).sum())
        low, high = wilson_interval(successes, p.size)
        rows.append(
            {
                "metric": metric_key,
                "true_relative_effect": effect,
                "n_total": n,
                "estimator": name,
                "replications": int(p.size),
                "power": successes / p.size,
                "power_lo": low,
                "power_hi": high,
                "mean_se": float(se.mean()),
                "effective_variance": float((se.mean() ** 2) * n / 4.0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["metric", "true_relative_effect", "estimator", "n_total"]
    ).reset_index(drop=True)


def interpolate_required_n(
    n_values: np.ndarray,
    powers: np.ndarray,
    replications: int,
    alpha: float = 0.05,
    target: float = 0.80,
) -> float:
    """Read the sample size at the target power off the measured curve.

    Power for a two sample z test is Phi(k sqrt(n) - z_alpha/2), so on the
    probit scale the curve is a straight line through the origin in sqrt(n).
    Fitting that one parameter uses every ladder point instead of only the two
    that happen to bracket the target, and it extrapolates in the right shape.
    Points are weighted by the sampling variance of the probit transform.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    clipped = np.clip(powers, 0.5 / replications, 1 - 0.5 / replications)
    y = stats.norm.ppf(clipped) + z_alpha
    x = np.sqrt(n_values)
    density = stats.norm.pdf(stats.norm.ppf(clipped))
    variance = clipped * (1 - clipped) / (replications * np.clip(density, 1e-12, None) ** 2)
    weight = 1.0 / np.clip(variance, 1e-12, None)
    usable = (powers > 0.02) & (powers < 0.999)
    if usable.sum() < 2:
        usable = np.ones_like(powers, dtype=bool)
    k = float((weight[usable] * x[usable] * y[usable]).sum() / (weight[usable] * x[usable] ** 2).sum())
    if k <= 0:
        return float("nan")
    return float(((stats.norm.ppf(target) + z_alpha) / k) ** 2)


def summarise(curve: pd.DataFrame, specs: dict[str, MetricSpec], alpha=0.05, target=0.80) -> pd.DataFrame:
    """One row per metric, effect and estimator: the sample size it needs."""
    rows = []
    for (metric_key, effect, name), group in curve.groupby(
        ["metric", "true_relative_effect", "estimator"], sort=False
    ):
        group = group.sort_values("n_total")
        replications = int(group["replications"].iloc[0])
        measured = interpolate_required_n(
            group["n_total"].to_numpy(float),
            group["power"].to_numpy(float),
            replications,
            alpha=alpha,
            target=target,
        )
        variance = float(group["effective_variance"].mean())
        predicted = analytic_n(specs[metric_key], effect, alpha, target, variance=variance)
        rows.append(
            {
                "metric": metric_key,
                "true_relative_effect": effect,
                "estimator": name,
                "n_required_measured": measured,
                "n_required_from_variance": predicted,
                "measured_variance": variance,
                "replications": replications,
            }
        )
    out = pd.DataFrame(rows)
    baseline = out[out["estimator"] == "diff_in_means"].set_index(
        ["metric", "true_relative_effect"]
    )["n_required_measured"]
    keys = list(zip(out["metric"], out["true_relative_effect"]))
    out["n_required_baseline"] = [baseline.get(k, np.nan) for k in keys]
    out["sample_size_ratio"] = out["n_required_measured"] / out["n_required_baseline"]
    out["users_saved_pct"] = 100.0 * (1.0 - out["sample_size_ratio"])
    return out.sort_values(["metric", "true_relative_effect", "estimator"]).reset_index(drop=True)


def run(chunks: list[Chunk], workers: int) -> list:
    if workers <= 1:
        return [run_chunk(c) for c in chunks]
    with Pool(workers) as pool:
        return pool.map(run_chunk, chunks, chunksize=1)
