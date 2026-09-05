"""Thompson sampling against a fixed horizon test, and what the interval costs.

The algorithm is batched Thompson sampling on a Beta-Bernoulli model. In each
round the assignment probability for the better looking arm is computed from
the current posterior, every user in the round is assigned by an independent
draw at that probability, and the posteriors are updated at the end of the
round. A floor on the assignment probability keeps some exploration alive,
which is both what a platform actually does and what keeps the inverse
propensity weights below.

The point of the comparison is the tradeoff, not the win. A bandit earns less
regret. In exchange its data is adaptively collected, which means the ordinary
confidence interval computed on it is not valid: arms that looked good early
get sampled more, and the sample mean of an arm is conditioned on the very
outcomes that caused it to be sampled. Both intervals are reported here, the
naive one so that its undercoverage can be seen, and a valid one so the price
of validity can be seen.

The valid interval uses the adaptively weighted augmented estimator of Hadad,
Hirshberg, Zhan, Wager and Athey, "Confidence intervals for policy evaluation
in adaptive experiments" (PNAS 2021). Each round contributes an augmented
inverse propensity score, and rounds are weighted by the square root of the
propensity product, which is the variance stabilising choice for a contrast
between two arms.

Because the outcomes are Bernoulli and the users inside a round are
exchangeable, a whole round is summarised by four counts. Every quantity below
is computed from those counts, so all replications run in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .seeds import generator

PROPENSITY_FLOOR = 0.05
POSTERIOR_DRAWS = 400


@dataclass(frozen=True)
class BanditOutcome:
    """Everything the comparison needs, one entry per replication."""

    regret: np.ndarray
    naive_effect: np.ndarray
    naive_se: np.ndarray
    weighted_effect: np.ndarray
    weighted_se: np.ndarray
    fixed_effect: np.ndarray
    fixed_se: np.ndarray
    share_to_best: np.ndarray
    true_effect: float
    replications: int
    rounds: int
    batch: int


def run_bandit(
    arm_rates: tuple[float, float],
    rounds: int,
    batch: int,
    replications: int,
    label: str = "bandit",
    floor: float = PROPENSITY_FLOOR,
    posterior_draws: int = POSTERIOR_DRAWS,
) -> BanditOutcome:
    rng = generator(label, rounds, batch, replications)
    rate_control, rate_treated = arm_rates
    true_effect = rate_treated - rate_control

    successes = np.zeros((2, replications))
    pulls = np.zeros((2, replications))

    weight_total = np.zeros(replications)
    score_total = np.zeros(replications)
    # squared deviations are accumulated after the estimate is known, so the
    # per round score values are kept
    round_weights = np.empty((rounds, replications))
    round_scores = np.empty((4, rounds, replications))
    round_counts = np.empty((4, rounds, replications))

    regret = np.zeros(replications)

    for t in range(rounds):
        alpha = 1.0 + successes
        beta = 1.0 + (pulls - successes)
        draws_control = rng.beta(alpha[0], beta[0], size=(posterior_draws, replications))
        draws_treated = rng.beta(alpha[1], beta[1], size=(posterior_draws, replications))
        probability = (draws_treated > draws_control).mean(axis=0)
        probability = np.clip(probability, floor, 1.0 - floor)

        # posterior means from data strictly before this round, so the
        # augmentation term is predictable given the past
        mean_control = alpha[0] / (alpha[0] + beta[0])
        mean_treated = alpha[1] / (alpha[1] + beta[1])

        k_treated = rng.binomial(batch, probability)
        k_control = batch - k_treated
        s_treated = rng.binomial(k_treated, rate_treated)
        s_control = rng.binomial(k_control, rate_control)

        regret += k_control * true_effect

        e_treated = probability
        e_control = 1.0 - probability
        # the four distinct augmented scores in this round
        round_scores[0, t] = mean_treated + (1.0 - mean_treated) / e_treated - mean_control
        round_scores[1, t] = mean_treated + (0.0 - mean_treated) / e_treated - mean_control
        round_scores[2, t] = mean_treated - (mean_control + (1.0 - mean_control) / e_control)
        round_scores[3, t] = mean_treated - (mean_control + (0.0 - mean_control) / e_control)
        round_counts[0, t] = s_treated
        round_counts[1, t] = k_treated - s_treated
        round_counts[2, t] = s_control
        round_counts[3, t] = k_control - s_control

        weight = np.sqrt(e_treated * e_control)
        round_weights[t] = weight
        weight_total += weight * batch
        score_total += weight * (round_scores[:, t] * round_counts[:, t]).sum(axis=0)

        successes[1] += s_treated
        successes[0] += s_control
        pulls[1] += k_treated
        pulls[0] += k_control

    weighted_effect = score_total / weight_total
    squared = np.zeros(replications)
    for t in range(rounds):
        deviation = round_scores[:, t] - weighted_effect
        squared += round_weights[t] ** 2 * (round_counts[:, t] * deviation**2).sum(axis=0)
    weighted_se = np.sqrt(squared) / weight_total

    with np.errstate(invalid="ignore", divide="ignore"):
        p_treated = successes[1] / pulls[1]
        p_control = successes[0] / pulls[0]
        naive_effect = p_treated - p_control
        naive_se = np.sqrt(
            p_treated * (1 - p_treated) / pulls[1] + p_control * (1 - p_control) / pulls[0]
        )

    total = rounds * batch
    half = total // 2
    fixed_successes_t = rng.binomial(half, rate_treated, size=replications)
    fixed_successes_c = rng.binomial(total - half, rate_control, size=replications)
    fixed_p_t = fixed_successes_t / half
    fixed_p_c = fixed_successes_c / (total - half)
    fixed_effect = fixed_p_t - fixed_p_c
    fixed_se = np.sqrt(
        fixed_p_t * (1 - fixed_p_t) / half + fixed_p_c * (1 - fixed_p_c) / (total - half)
    )

    return BanditOutcome(
        regret=regret,
        naive_effect=naive_effect,
        naive_se=naive_se,
        weighted_effect=weighted_effect,
        weighted_se=weighted_se,
        fixed_effect=fixed_effect,
        fixed_se=fixed_se,
        share_to_best=pulls[1] / (pulls[0] + pulls[1]),
        true_effect=true_effect,
        replications=replications,
        rounds=rounds,
        batch=batch,
    )


def coverage_table(outcome: BanditOutcome, confidence: float = 0.95):
    """Coverage, width and bias of each interval, with a binomial interval."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    rows = []
    for name, effect, se in [
        ("bandit_naive", outcome.naive_effect, outcome.naive_se),
        ("bandit_adaptively_weighted", outcome.weighted_effect, outcome.weighted_se),
        ("fixed_horizon_balanced", outcome.fixed_effect, outcome.fixed_se),
    ]:
        good = np.isfinite(effect) & np.isfinite(se) & (se > 0)
        low, high = effect[good] - z * se[good], effect[good] + z * se[good]
        covered = (low <= outcome.true_effect) & (outcome.true_effect <= high)
        rate = float(covered.mean())
        n = int(good.sum())
        half_width = z * float(np.mean(se[good]))
        error = float(np.sqrt(rate * (1 - rate) / n))
        rows.append(
            {
                "interval": name,
                "coverage": rate,
                "coverage_lo": rate - 1.96 * error,
                "coverage_hi": rate + 1.96 * error,
                "mean_half_width": half_width,
                "mean_estimate": float(np.mean(effect[good])),
                "bias": float(np.mean(effect[good])) - outcome.true_effect,
                "replications": n,
                "true_effect": outcome.true_effect,
            }
        )
    return rows
