"""What peeking actually costs, measured as a false positive rate.

The scenario is the one teams actually run: users arrive at a steady rate, the
analyst looks at the dashboard once a day, and stops the moment the test goes
significant. Under the null nothing is happening, so every stop is a false
positive, and the rate of those stops is the number this module measures.

Three reading rules are compared on identical data:

  fixed_horizon   read once, at the planned end. The reference, should be alpha.
  naive_peeking   the same fixed horizon test read every day, stop at the first
                  significant result. This is the thing being measured.
  msprt           the always valid mixture test, read every day.
  alpha_spending  Lan-DeMets O'Brien-Fleming boundaries, read every day.

Note carefully that this is multiplicity over looks in time. Multiplicity
across several metrics at one look is a different problem with a different
correction, and lives in multiplicity.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .seeds import generator
from .sequential import alpha_spending_boundaries, always_valid_p_value
from .simulate import MetricSpec, metric_moments, simulate_experiment, solve_copula_r


@dataclass(frozen=True)
class PeekChunk:
    """A block of replications of one daily monitoring scenario."""

    metric_key: str
    spec: MetricSpec
    effect: float
    days: int
    users_per_day: int
    tau_squared: float
    boundaries: np.ndarray
    copula_r: float
    rep_start: int
    rep_stop: int
    alpha: float = 0.05
    rho: float = 0.0


def daily_statistics(outcome: np.ndarray, arm: np.ndarray, days: int, users_per_day: int):
    """Cumulative Welch statistics at the end of each day.

    The arriving users are cut into daily blocks, each block reduced once, and
    the block totals accumulated. That gives all the interim looks in a couple
    of passes instead of re-reading the whole history once per day.
    """
    block_outcome = outcome[: days * users_per_day].reshape(days, users_per_day)
    block_arm = arm[: days * users_per_day].reshape(days, users_per_day).astype(float)

    weighted = block_outcome * block_arm
    n_t = np.cumsum(block_arm.sum(axis=1))
    sum_t = np.cumsum(weighted.sum(axis=1))
    sq_t = np.cumsum((weighted * block_outcome).sum(axis=1))

    control = 1.0 - block_arm
    weighted_c = block_outcome * control
    n_c = np.cumsum(control.sum(axis=1))
    sum_c = np.cumsum(weighted_c.sum(axis=1))
    sq_c = np.cumsum((weighted_c * block_outcome).sum(axis=1))

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_t, mean_c = sum_t / n_t, sum_c / n_c
        var_t = (sq_t - n_t * mean_t**2) / np.maximum(n_t - 1.0, 1.0)
        var_c = (sq_c - n_c * mean_c**2) / np.maximum(n_c - 1.0, 1.0)
        variance = var_t / n_t + var_c / n_c
        effect = mean_t - mean_c
        z = effect / np.sqrt(variance)
        numerator = variance**2
        denominator = (var_t / n_t) ** 2 / np.maximum(n_t - 1.0, 1.0) + (var_c / n_c) ** 2 / np.maximum(
            n_c - 1.0, 1.0
        )
        df = numerator / denominator
        p = 2.0 * stats.t.sf(np.abs(z), np.maximum(df, 1.0))
    return effect, variance, z, p


def run_peek_chunk(chunk: PeekChunk) -> dict:
    """Return, per replication, what each reading rule concluded."""
    reps = chunk.rep_stop - chunk.rep_start
    n = chunk.days * chunk.users_per_day
    label = f"peek|{chunk.metric_key}|{chunk.effect:g}|{chunk.users_per_day}|{chunk.days}"

    out = {
        "fixed_horizon": np.zeros(reps, dtype=bool),
        "naive_peeking": np.zeros(reps, dtype=bool),
        "msprt": np.zeros(reps, dtype=bool),
        "alpha_spending": np.zeros(reps, dtype=bool),
        "naive_stop_day": np.full(reps, chunk.days, dtype=int),
        "msprt_stop_day": np.full(reps, chunk.days, dtype=int),
        "alpha_spending_stop_day": np.full(reps, chunk.days, dtype=int),
    }

    for j, rep in enumerate(range(chunk.rep_start, chunk.rep_stop)):
        experiment = simulate_experiment(
            chunk.spec, n, chunk.effect, chunk.rho, generator(label, rep), copula_r=chunk.copula_r
        )
        effect, variance, z, p = daily_statistics(
            experiment.outcome, experiment.arm, chunk.days, chunk.users_per_day
        )
        significant = p < chunk.alpha
        out["fixed_horizon"][j] = bool(significant[-1])
        if significant.any():
            out["naive_peeking"][j] = True
            out["naive_stop_day"][j] = int(np.argmax(significant)) + 1

        valid_p = always_valid_p_value(effect, variance, chunk.tau_squared)
        crossed = valid_p < chunk.alpha
        if crossed.any():
            out["msprt"][j] = True
            out["msprt_stop_day"][j] = int(np.argmax(crossed)) + 1

        spent = np.abs(z) >= chunk.boundaries
        if spent.any():
            out["alpha_spending"][j] = True
            out["alpha_spending_stop_day"][j] = int(np.argmax(spent)) + 1

    return {"metric": chunk.metric_key, "effect": chunk.effect, "reps": reps, **out}


def tau_squared_for(spec: MetricSpec, target_relative_effect: float) -> float:
    """Mixing variance for the mSPRT, fixed before the experiment.

    Set to the square of the absolute effect the team says it cares about. This
    is a design choice, not something fitted to the data, and it controls which
    effect sizes the always valid test detects soonest.
    """
    mean, _ = metric_moments(spec)
    return float((target_relative_effect * mean) ** 2)


def build_peek_chunks(
    metric_key: str,
    spec: MetricSpec,
    effect: float,
    days: int,
    users_per_day: int,
    replications: int,
    target_relative_effect: float,
    alpha: float = 0.05,
    block: int = 250,
) -> list[PeekChunk]:
    boundaries = alpha_spending_boundaries(np.arange(1, days + 1) / days, alpha)
    tau_squared = tau_squared_for(spec, target_relative_effect)
    return [
        PeekChunk(
            metric_key=metric_key,
            spec=spec,
            effect=effect,
            days=days,
            users_per_day=users_per_day,
            tau_squared=tau_squared,
            boundaries=boundaries,
            copula_r=solve_copula_r(spec, 0.0),
            rep_start=start,
            rep_stop=min(start + block, replications),
            alpha=alpha,
        )
        for start in range(0, replications, block)
    ]
