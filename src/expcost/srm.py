"""Sample ratio mismatch: the check, its power, and when to discard a result.

A sample ratio mismatch is a discrepancy between the allocation the experiment
was designed to have and the allocation the data actually shows. The important
distinction, and the one this module is built around, is that an imbalanced
design is not a mismatch. An 85/15 split is perfectly valid if 85/15 is what
was intended. The test always compares the observed split against the intended
one, whatever that intended one is, so a deliberate imbalance is not flagged
and a one percent drift away from a deliberate 85/15 still is.

Only the arm counts matter here, so the counts are drawn directly from their
binomial law rather than by simulating users and counting them. That is exact
and it costs nothing, which is what makes a four thousand replication power
curve over five sample sizes affordable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .seeds import generator

# A conventional threshold. Chosen so that at the sample sizes experimentation
# platforms run at, a healthy experiment almost never trips it, while the kinds
# of bug that cause a mismatch almost always do.
DISCARD_THRESHOLD = 0.0005


def srm_test(treated: int, control: int, intended_allocation: float = 0.5) -> tuple[float, float]:
    """Chi square goodness of fit of the observed split to the intended one."""
    total = treated + control
    if total == 0:
        return float("nan"), float("nan")
    expected_treated = total * intended_allocation
    expected_control = total * (1.0 - intended_allocation)
    statistic = (treated - expected_treated) ** 2 / expected_treated + (
        control - expected_control
    ) ** 2 / expected_control
    return float(statistic), float(stats.chi2.sf(statistic, df=1))


def analytic_detection_rate(
    n: int, true_allocation: float, intended_allocation: float = 0.5, alpha: float = DISCARD_THRESHOLD
) -> float:
    """Probability the check fires, from the non central chi square.

    The chi square statistic has one degree of freedom and non centrality
    n (p - p0)^2 / (p0 (1 - p0)), so the detection rate is available in closed
    form and the simulation below has something to be checked against.
    """
    p0 = intended_allocation
    noncentrality = n * (true_allocation - p0) ** 2 / (p0 * (1.0 - p0))
    critical = stats.chi2.ppf(1.0 - alpha, df=1)
    return float(stats.ncx2.sf(critical, df=1, nc=noncentrality))


def simulate_detection_rate(
    n: int,
    true_allocation: float,
    intended_allocation: float = 0.5,
    replications: int = 4000,
    alpha: float = DISCARD_THRESHOLD,
) -> tuple[float, int]:
    """Measured detection rate over independent experiments."""
    rng = generator("srm", n, f"{true_allocation:g}", f"{intended_allocation:g}")
    treated = rng.binomial(n, true_allocation, size=replications)
    total = np.full(replications, n)
    expected_treated = total * intended_allocation
    expected_control = total * (1.0 - intended_allocation)
    statistic = (treated - expected_treated) ** 2 / expected_treated + (
        (total - treated) - expected_control
    ) ** 2 / expected_control
    p_values = stats.chi2.sf(statistic, df=1)
    return float((p_values < alpha).mean()), int(replications)


def smallest_detectable_mismatch(
    n: int, intended_allocation: float = 0.5, alpha: float = DISCARD_THRESHOLD, power: float = 0.80
) -> float:
    """The allocation drift this sample size can detect at the target power."""
    lo, hi = intended_allocation, intended_allocation + 0.5 * (1.0 - intended_allocation)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if analytic_detection_rate(n, mid, intended_allocation, alpha) < power:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi) - intended_allocation)


def run_srm_study(
    n_grid, true_allocations, intended_allocation: float = 0.5, replications: int = 4000
) -> pd.DataFrame:
    rows = []
    for n in n_grid:
        for allocation in true_allocations:
            measured, reps = simulate_detection_rate(
                n, allocation, intended_allocation, replications
            )
            rows.append(
                {
                    "n_total": n,
                    "intended_allocation": intended_allocation,
                    "true_allocation": allocation,
                    "drift": allocation - intended_allocation,
                    "detection_rate_measured": measured,
                    "detection_rate_analytic": analytic_detection_rate(
                        n, allocation, intended_allocation
                    ),
                    "replications": reps,
                    "threshold": DISCARD_THRESHOLD,
                }
            )
    return pd.DataFrame(rows)
