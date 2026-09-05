"""The frozen scenario grid.

This file is fixed before the study runs. If anything here changes after
results have been seen, the change and the reason for it are recorded in the
README, as required by the project's own honesty rules.

Effect sizes were chosen before running, on one stated criterion: the sample
size a plain difference in means needs for 80 percent power has to land inside
a range that can actually be simulated many thousands of times, which is
roughly 20,000 to 400,000 users. The realistic small effects that a team
actually argues about, a 1 percent lift on conversion say, need millions of
users and are reported from the measured variance rather than simulated
directly. That distinction is stated everywhere those numbers appear.
"""

from __future__ import annotations

from pathlib import Path

from expcost.calibration import RevenueShape
from expcost.simulate import MetricSpec

ALPHA = 0.05
TARGET_POWER = 0.80

# Correlation between the pre period covariate and the outcome used in the
# headline power study. The CUPED curve sweeps this, question 2 is precisely
# about how much it matters.
RHO_MAIN = 0.5

# Ladder of sample sizes, as multiples of the sample size a plain difference in
# means is predicted to need. Spans both the adjusted estimators, which need
# less, and the noise around the target.
N_LADDER = (0.35, 0.5, 0.7, 1.0, 1.4, 1.9)
POWER_REPLICATIONS = 2000

# CUPED curve
RHO_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CUPED_CURVE_N = 100_000
CUPED_CURVE_REPLICATIONS = 800

# Peeking study
PEEK_DAYS = 14
PEEK_USERS_PER_DAY = 10_000
PEEK_REPLICATIONS = 20_000

# Sample ratio mismatch
SRM_TRUE_ALLOCATIONS = (0.500, 0.499, 0.498, 0.495, 0.49, 0.48, 0.45)
SRM_N_GRID = (10_000, 50_000, 100_000, 500_000, 1_000_000)
SRM_REPLICATIONS = 4000

# Bandit
BANDIT_ROUNDS = 200
BANDIT_BATCH = 500
BANDIT_REPLICATIONS = 2000
BANDIT_ARM_RATES = (0.040, 0.044)

_SHAPE_PATH = Path(__file__).resolve().parents[1] / "results" / "revenue_shape.json"


def revenue_shape() -> RevenueShape:
    return RevenueShape.from_json(_SHAPE_PATH)


def metrics() -> dict[str, MetricSpec]:
    """The three metric families, each at a realistic base rate."""
    shape = revenue_shape()
    return {
        "revenue_p10": MetricSpec(
            family="revenue", base_rate=0.10, shape=shape, name="revenue_p10"
        ),
        "conversion_p04": MetricSpec(
            family="conversion", base_rate=0.04, name="conversion_p04"
        ),
        "sessions_m3": MetricSpec(
            family="sessions", base_rate=3.0, dispersion=0.8, name="sessions_m3"
        ),
    }


# True relative lifts, per metric. Stated with every power number they produce.
EFFECTS: dict[str, tuple[float, ...]] = {
    "revenue_p10": (0.05, 0.075, 0.10),
    "conversion_p04": (0.05, 0.075, 0.10),
    "sessions_m3": (0.02, 0.03, 0.05),
}

# Effects too small to simulate at this replication count, reported from the
# measured variance instead and labelled as such wherever they appear.
EXTRAPOLATED_EFFECTS = (0.005, 0.01, 0.02, 0.03)

METRIC_LABELS = {
    "revenue_p10": "revenue per user, 10 percent purchase",
    "conversion_p04": "conversion, 4 percent base",
    "sessions_m3": "sessions per user, mean 3",
}

ESTIMATOR_LABELS = {
    "diff_in_means": "difference in means",
    "cuped": "CUPED",
    "lin_adjusted": "Lin regression adjustment",
    "post_stratified": "post stratified",
    "cuped_plus_strata": "CUPED plus strata",
}
