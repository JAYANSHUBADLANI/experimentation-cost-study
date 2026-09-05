"""How much variance CUPED removes, as a function of the pre/post correlation.

Reporting a single number for "CUPED reduces variance by X percent" is
meaningless without saying at what correlation, because the whole effect is
driven by that correlation. Under the standard theory the adjusted variance is
(1 - rho^2) times the unadjusted one, so the sample size ratio is (1 - rho^2)
and the saving vanishes quadratically as rho falls. This module measures the
curve instead of assuming it, and reports the measured curve against that
theoretical line.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .estimators import ESTIMATORS
from .seeds import generator
from .simulate import MetricSpec, simulate_experiment, solve_copula_r

# Below this correlation the sample size saving is under ten percent, which is
# the point argued in the guide as the level where the plumbing stops paying.
WORTH_IT_THRESHOLD = 0.10


@dataclass(frozen=True)
class CurvePoint:
    metric_key: str
    spec: MetricSpec
    rho: float
    n: int
    replications: int
    effect: float = 0.0


def run_curve_point(point: CurvePoint) -> dict:
    copula_r = solve_copula_r(point.spec, point.rho)
    label = f"cuped_curve|{point.metric_key}|{point.rho:g}|{point.n}"
    names = list(ESTIMATORS)
    standard_errors = {name: np.empty(point.replications) for name in names}
    effects = {name: np.empty(point.replications) for name in names}
    realised = np.empty(point.replications)

    for rep in range(point.replications):
        experiment = simulate_experiment(
            point.spec, point.n, point.effect, point.rho, generator(label, rep), copula_r=copula_r
        )
        if experiment.pre.std() > 0 and experiment.outcome.std() > 0:
            realised[rep] = np.corrcoef(experiment.pre, experiment.outcome)[0, 1]
        else:
            realised[rep] = 0.0
        for name in names:
            estimate = ESTIMATORS[name](experiment)
            standard_errors[name][rep] = estimate.se
            effects[name][rep] = estimate.effect

    baseline = float(np.mean(standard_errors["diff_in_means"] ** 2))
    rows = []
    for name in names:
        variance = float(np.mean(standard_errors[name] ** 2))
        rows.append(
            {
                "metric": point.metric_key,
                "target_rho": point.rho,
                "realised_rho_mean": float(realised.mean()),
                "realised_rho_sd": float(realised.std(ddof=1)),
                "n_total": point.n,
                "estimator": name,
                "replications": point.replications,
                "variance": variance,
                "variance_ratio": variance / baseline,
                "theoretical_ratio": 1.0 - point.rho**2,
                "empirical_variance_of_estimates": float(np.var(effects[name], ddof=1)),
                "sample_size_ratio": variance / baseline,
                "users_saved_pct": 100.0 * (1.0 - variance / baseline),
            }
        )
    return rows


def run_curve(points: list[CurvePoint], workers: int = 7) -> pd.DataFrame:
    if workers <= 1:
        collected = [run_curve_point(p) for p in points]
    else:
        with Pool(workers) as pool:
            collected = pool.map(run_curve_point, points, chunksize=1)
    rows = [row for group in collected for row in group]
    return pd.DataFrame(rows).sort_values(["metric", "estimator", "target_rho"]).reset_index(drop=True)


def break_even_correlation(threshold: float = WORTH_IT_THRESHOLD) -> float:
    """The correlation at which the theoretical saving reaches the threshold."""
    return float(np.sqrt(threshold))
