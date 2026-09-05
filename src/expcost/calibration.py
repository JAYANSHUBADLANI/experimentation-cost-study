"""Fit the simulated metric distribution to a real one, and record the evidence.

Source of shape: the Olist Brazilian E-Commerce public dataset, order payment
values aggregated to one value per order. That gives a real, heavy tailed
money-per-event distribution. What is taken from it is the shape of the
positive part of revenue. What is NOT taken from it is the zero rate or the
pre/post correlation, for reasons stated in the README: Olist's repeat
purchase rate is about 3 percent, so its per-user monthly panel is not
representative of the populations this study is about.

The fitted model is a spliced distribution: lognormal below a high threshold,
generalized Pareto above it. A single lognormal was tried first and rejected,
the evidence for that rejection is in the calibration table and figure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TAIL_QUANTILE = 0.95
HIST_BINS = 60
HIST_MAX = 1500.0
SUMMARY_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]


@dataclass(frozen=True)
class RevenueShape:
    """Parameters of the spliced lognormal / generalized Pareto positive part."""

    log_mu: float
    log_sigma: float
    threshold: float
    tail_quantile: float
    gpd_xi: float
    gpd_scale: float
    source: str
    n_observations: int
    empirical_mean: float
    empirical_sd: float
    empirical_cv: float

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> "RevenueShape":
        return cls(**json.loads(Path(path).read_text()))


def load_olist_order_values(olist_dir: Path) -> np.ndarray:
    """One payment value per non cancelled order."""
    olist_dir = Path(olist_dir)
    orders = pd.read_csv(
        olist_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "order_status"],
    )
    payments = pd.read_csv(
        olist_dir / "olist_order_payments_dataset.csv",
        usecols=["order_id", "payment_value"],
    )
    per_order = payments.groupby("order_id", as_index=False)["payment_value"].sum()
    merged = orders.merge(per_order, on="order_id")
    merged = merged[merged["order_status"] != "canceled"]
    values = merged["payment_value"].to_numpy(dtype=float)
    return np.sort(values[values > 0])


def fit_revenue_shape(values: np.ndarray, source: str) -> RevenueShape:
    """Fit the spliced model by MLE on each piece."""
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    log_values = np.log(values)
    threshold = float(np.quantile(values, TAIL_QUANTILE))
    excess = values[values > threshold] - threshold
    xi, _, scale = stats.genpareto.fit(excess, floc=0.0)
    return RevenueShape(
        log_mu=float(log_values.mean()),
        log_sigma=float(log_values.std(ddof=1)),
        threshold=threshold,
        tail_quantile=TAIL_QUANTILE,
        gpd_xi=float(xi),
        gpd_scale=float(scale),
        source=source,
        n_observations=int(values.size),
        empirical_mean=float(values.mean()),
        empirical_sd=float(values.std(ddof=1)),
        empirical_cv=float(values.std(ddof=1) / values.mean()),
    )


def empirical_summary(values: np.ndarray) -> pd.DataFrame:
    """Committed evidence: quantiles and histogram counts, no raw rows."""
    values = np.asarray(values, dtype=float)
    rows = [
        {"statistic": f"q{q:g}", "value": float(np.quantile(values, q))}
        for q in SUMMARY_QUANTILES
    ]
    rows += [
        {"statistic": "n", "value": float(values.size)},
        {"statistic": "mean", "value": float(values.mean())},
        {"statistic": "sd", "value": float(values.std(ddof=1))},
        {"statistic": "cv", "value": float(values.std(ddof=1) / values.mean())},
        {"statistic": "skew", "value": float(stats.skew(values))},
        {"statistic": "max", "value": float(values.max())},
    ]
    counts, edges = np.histogram(values, bins=HIST_BINS, range=(0.0, HIST_MAX))
    rows += [
        {"statistic": f"hist_count_{i:02d}", "value": float(c)} for i, c in enumerate(counts)
    ]
    rows += [
        {"statistic": f"hist_edge_{i:02d}", "value": float(e)} for i, e in enumerate(edges)
    ]
    rows.append({"statistic": "hist_above_range", "value": float((values > HIST_MAX).sum())})
    return pd.DataFrame(rows)
