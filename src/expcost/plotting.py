"""Figures. Every number plotted comes from a results table on disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .calibration import RevenueShape
from .simulate import positive_part_ppf

FIGURES = Path("figures")
RESULTS = Path("results")

COLOURS = {
    "diff_in_means": "#3b3b3b",
    "cuped": "#1f77b4",
    "lin_adjusted": "#6baed6",
    "post_stratified": "#2ca02c",
    "cuped_plus_strata": "#9467bd",
}
LABELS = {
    "diff_in_means": "difference in means",
    "cuped": "CUPED",
    "lin_adjusted": "Lin adjustment",
    "post_stratified": "post stratified",
    "cuped_plus_strata": "CUPED plus strata",
}
METRIC_TITLES = {
    "revenue_p10": "revenue per user\n(10 percent purchase, heavy tail)",
    "conversion_p04": "conversion\n(4 percent base rate)",
    "sessions_m3": "sessions per user\n(mean 3, overdispersed)",
}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_calibration(shape: RevenueShape, summary: pd.DataFrame) -> Path:
    """The realism evidence: fitted against empirical, and why lognormal alone fails."""
    counts = summary[summary["statistic"].str.startswith("hist_count_")]["value"].to_numpy()
    edges = summary[summary["statistic"].str.startswith("hist_edge_")]["value"].to_numpy()
    stat = dict(zip(summary["statistic"], summary["value"]))
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    density = counts / (counts.sum() + stat["hist_above_range"]) / width

    grid = (np.arange(400_000) + 0.5) / 400_000
    spliced = positive_part_ppf(grid, shape)
    lognormal = stats.lognorm.ppf(grid, shape.log_sigma, scale=np.exp(shape.log_mu))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    ax.bar(centres, density, width=width, color="#c6dbef", edgecolor="none", label="Olist order values")
    x = np.linspace(1, edges[-1], 800)
    body = stats.lognorm.pdf(x, shape.log_sigma, scale=np.exp(shape.log_mu))
    ax.plot(x, body, color="#d62728", lw=1.6, ls="--", label="lognormal only")
    hist_fit, _ = np.histogram(spliced, bins=edges, density=False)
    ax.plot(centres, hist_fit / spliced.size / width, color="#08519c", lw=1.8, label="spliced fit")
    ax.axvline(shape.threshold, color="#666666", lw=1.0, ls=":")
    ax.text(shape.threshold * 1.05, ax.get_ylim()[1] * 0.75,
            f"splice at\nq{shape.tail_quantile:g} = {shape.threshold:,.0f}", fontsize=7.5, color="#444444")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "the body of the distribution", "order value", "density")

    ax = axes[1]
    levels = np.array([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    depth = -np.log10(1.0 - levels)
    empirical = np.array([stat[f"q{q:g}"] for q in levels])
    ax.plot(depth, empirical, "o-", color="#08519c", lw=1.8, ms=6, label="Olist, empirical")
    ax.plot(depth, np.quantile(spliced, levels), "s--", color="#2ca02c", lw=1.5, ms=5, label="spliced fit")
    ax.plot(depth, np.quantile(lognormal, levels), "^--", color="#d62728", lw=1.5, ms=5, label="lognormal only")
    ax.set_yscale("log")
    ax.set_xticks(depth)
    ax.set_xticklabels([f"{q:g}" for q in levels], fontsize=7.5)
    ax.annotate(
        f"lognormal misses the\n99.9th percentile by\n{100 * (1 - np.quantile(lognormal, 0.999) / stat['q0.999']):.0f} percent",
        xy=(depth[-1], np.quantile(lognormal, 0.999)),
        xytext=(depth[-3], np.quantile(lognormal, 0.999) * 1.05),
        fontsize=7.5, color="#a01010",
        arrowprops={"arrowstyle": "->", "color": "#a01010", "lw": 0.9},
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax, "the tail, where sample size is decided", "quantile (stretched towards the tail)",
           "order value (log scale)")

    ax = axes[2]
    names = ["mean", "sd", "cv", "p99", "p99.9"]
    empirical_stats = [stat["mean"], stat["sd"], stat["cv"], stat["q0.99"], stat["q0.999"]]
    spliced_stats = [spliced.mean(), spliced.std(), spliced.std() / spliced.mean(),
                     np.quantile(spliced, 0.99), np.quantile(spliced, 0.999)]
    lognormal_stats = [lognormal.mean(), lognormal.std(), lognormal.std() / lognormal.mean(),
                       np.quantile(lognormal, 0.99), np.quantile(lognormal, 0.999)]
    position = np.arange(len(names))
    ax.bar(position - 0.26, np.array(spliced_stats) / empirical_stats, 0.26, color="#2ca02c", label="spliced fit")
    ax.bar(position + 0.0, np.ones(len(names)), 0.26, color="#08519c", label="Olist, empirical")
    ax.bar(position + 0.26, np.array(lognormal_stats) / empirical_stats, 0.26, color="#d62728", label="lognormal only")
    ax.axhline(1.0, color="#333333", lw=0.8)
    ax.set_xticks(position)
    ax.set_xticklabels(names)
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "ratio to the real data (1.0 is perfect)", None, "fitted / empirical")

    fig.suptitle(
        f"Calibrating the revenue metric to real order values ({shape.n_observations:,} orders). "
        "A single lognormal understates the spread, which would understate every sample size in this study.",
        fontsize=10, y=1.04,
    )
    return _save(fig, "fig01_calibration.png")


def figure_sample_size(summary: pd.DataFrame) -> Path:
    """The headline: users needed for 80 percent power, by estimator."""
    metrics = list(dict.fromkeys(summary["metric"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    estimators = [e for e in COLOURS if e in set(summary["estimator"])]

    for ax, metric in zip(axes, metrics):
        block = summary[summary["metric"] == metric]
        effects = sorted(block["true_relative_effect"].unique())
        position = np.arange(len(effects))
        width = 0.8 / len(estimators)
        for i, estimator in enumerate(estimators):
            values = [
                block[(block["true_relative_effect"] == e) & (block["estimator"] == estimator)]
                ["n_required_measured"].iloc[0]
                for e in effects
            ]
            ax.bar(position + i * width - 0.4 + width / 2, values, width,
                   color=COLOURS[estimator], label=LABELS[estimator])
        ax.set_xticks(position)
        ax.set_xticklabels([f"{e:.1%}" for e in effects])
        ax.set_yscale("log")
        _style(ax, METRIC_TITLES.get(metric, metric), "true relative lift", "users needed for 80 percent power")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Users needed for 80 percent power, measured. Lower is cheaper.", fontsize=11, y=1.02)
    return _save(fig, "fig02_sample_size.png")


def figure_cuped_curve(curve: pd.DataFrame) -> Path:
    """Variance reduction against the pre and post correlation."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    rho = np.linspace(0, 0.95, 200)
    ax.plot(rho, 1 - rho**2, color="#333333", lw=1.4, ls="--", label="theory, 1 minus rho squared")
    for metric, block in curve[curve["estimator"] == "cuped"].groupby("metric"):
        block = block.sort_values("target_rho")
        ax.plot(block["target_rho"], block["variance_ratio"], "o-", ms=5, lw=1.5,
                label=METRIC_TITLES.get(metric, metric).replace("\n", " "))
    ax.axvline(np.sqrt(0.10), color="#d62728", lw=1.0, ls=":")
    ax.text(np.sqrt(0.10) + 0.01, 0.92, "below rho = 0.32\nthe saving is under\n10 percent of users",
            fontsize=7.5, color="#a01010")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "CUPED variance ratio against correlation", "correlation between pre period and outcome",
           "variance relative to no adjustment")

    ax = axes[1]
    for metric, block in curve[curve["estimator"] == "cuped"].groupby("metric"):
        block = block.sort_values("target_rho")
        ax.plot(block["target_rho"], block["users_saved_pct"], "o-", ms=5, lw=1.5,
                label=METRIC_TITLES.get(metric, metric).replace("\n", " "))
    ax.axhline(10, color="#d62728", lw=1.0, ls=":")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "the same thing as users saved", "correlation between pre period and outcome",
           "percent of users saved")

    fig.suptitle("CUPED buys exactly what the correlation gives it, and nothing more.", fontsize=11, y=1.02)
    return _save(fig, "fig03_cuped_curve.png")


def figure_peeking(peek: pd.DataFrame) -> Path:
    """False positive rate by reading rule."""
    rules = ["fixed_horizon", "naive_peeking", "msprt", "alpha_spending"]
    nice = {"fixed_horizon": "read once\nat the end", "naive_peeking": "read daily,\nstop when significant",
            "msprt": "mSPRT\n(always valid)", "alpha_spending": "alpha spending\n(O'Brien-Fleming)"}
    colours = {"fixed_horizon": "#3b3b3b", "naive_peeking": "#d62728",
               "msprt": "#1f77b4", "alpha_spending": "#2ca02c"}
    metrics = list(dict.fromkeys(peek["metric"]))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    position = np.arange(len(metrics))
    width = 0.8 / len(rules)
    for i, rule in enumerate(rules):
        block = peek[peek["reading_rule"] == rule].set_index("metric")
        values = [block.loc[m, "rejection_rate"] for m in metrics]
        errors = [
            [block.loc[m, "rejection_rate"] - block.loc[m, "rate_lo"] for m in metrics],
            [block.loc[m, "rate_hi"] - block.loc[m, "rejection_rate"] for m in metrics],
        ]
        ax.bar(position + i * width - 0.4 + width / 2, values, width, yerr=errors,
               color=colours[rule], label=nice[rule], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0.05, color="#333333", lw=1.2, ls="--")
    ax.text(len(metrics) - 0.45, 0.053, "nominal 5 percent", fontsize=8, color="#333333")
    ax.set_xticks(position)
    ax.set_xticklabels([METRIC_TITLES.get(m, m).replace("\n", " ") for m in metrics], fontsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    reps = int(peek["replications"].iloc[0])
    days = int(peek["days"].iloc[0])
    _style(ax, None, None, "false positive rate under the null")
    fig.suptitle(
        f"What peeking costs. Simulated under a true effect of zero, {reps:,} replications per bar, "
        f"{days} daily looks. Bars are 95 percent Wilson intervals.",
        fontsize=10, y=1.0,
    )
    return _save(fig, "fig04_peeking.png")


def figure_srm(power: pd.DataFrame, detectable: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for n, block in power.groupby("n_total"):
        block = block.sort_values("drift")
        ax.plot(100 * block["drift"].abs(), block["detection_rate_measured"], "o-", ms=4, lw=1.4,
                label=f"n = {int(n):,}")
    ax.axhline(0.8, color="#333333", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "detection rate against the size of the mismatch",
           "allocation drift (percentage points)", "probability the check fires")

    ax = axes[1]
    ax.plot(detectable["n_total"], 100 * detectable["smallest_detectable_drift"], "o-",
            color="#08519c", ms=5, lw=1.6)
    for _, row in detectable.iterrows():
        ax.annotate(f"{100 * row['smallest_detectable_drift']:.2f} pp",
                    (row["n_total"], 100 * row["smallest_detectable_drift"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "smallest mismatch worth looking for", "total users", "detectable drift at 80 percent power")

    threshold = float(power["threshold"].iloc[0])
    fig.suptitle(
        f"Sample ratio mismatch, tested against the intended allocation at a p value threshold of {threshold}. "
        "An imbalanced design is not a mismatch, only a departure from the intended ratio is.",
        fontsize=10, y=1.02,
    )
    return _save(fig, "fig05_srm.png")


def figure_bandit(coverage: pd.DataFrame, regret: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    nice = {"bandit_naive": "bandit, naive interval",
            "bandit_adaptively_weighted": "bandit, adaptively weighted",
            "fixed_horizon_balanced": "fixed horizon, balanced"}
    colours = {"bandit_naive": "#d62728", "bandit_adaptively_weighted": "#1f77b4",
               "fixed_horizon_balanced": "#3b3b3b"}

    null = coverage[coverage["scenario"] == "null"]
    ax = axes[0]
    floors = sorted(null["exploration_floor"].unique())
    position = np.arange(len(floors))
    width = 0.8 / 3
    for i, interval in enumerate(nice):
        block = null[null["interval"] == interval].set_index("exploration_floor")
        values = [block.loc[f, "coverage"] for f in floors]
        errors = [
            [block.loc[f, "coverage"] - block.loc[f, "coverage_lo"] for f in floors],
            [block.loc[f, "coverage_hi"] - block.loc[f, "coverage"] for f in floors],
        ]
        ax.bar(position + i * width - 0.4 + width / 2, values, width, yerr=errors,
               color=colours[interval], label=nice[interval], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0.95, color="#333333", lw=1.2, ls="--")
    ax.set_ylim(0.75, 1.0)
    ax.set_xticks(position)
    ax.set_xticklabels([f"{f:g}" for f in floors])
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax, "interval coverage under the null", "exploration floor", "coverage (nominal 0.95)")

    ax = axes[1]
    for i, interval in enumerate(nice):
        block = null[null["interval"] == interval].set_index("exploration_floor")
        ax.plot(floors, [block.loc[f, "mean_half_width"] for f in floors], "o-",
                color=colours[interval], ms=5, lw=1.5, label=nice[interval])
    ax.set_xscale("log")
    _style(ax, "what the interval costs in width", "exploration floor", "mean half width")

    ax = axes[2]
    lift = regret[regret["scenario"] == "lift"].sort_values("exploration_floor")
    position = np.arange(len(lift))
    ax.bar(position - 0.2, lift["balanced_regret"], 0.4, color="#3b3b3b", label="balanced split")
    ax.bar(position + 0.2, lift["bandit_regret"], 0.4, color="#1f77b4", label="bandit")
    ax.set_xticks(position)
    ax.set_xticklabels([f"{f:g}" for f in lift["exploration_floor"]])
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "regret, the thing the bandit buys", "exploration floor", "expected conversions forgone")

    fig.suptitle(
        "A bandit trades regret for inference. The naive interval on adaptively collected data "
        "does not cover, and the valid alternative is wider than a balanced test.",
        fontsize=10, y=1.02,
    )
    return _save(fig, "fig06_bandit.png")


def figure_power_curves(curve: pd.DataFrame) -> Path:
    """The measured power curves the required sample sizes were read off."""
    metrics = list(dict.fromkeys(curve["metric"]))
    effects = {m: sorted(curve[curve["metric"] == m]["true_relative_effect"].unique()) for m in metrics}
    n_cols = max(len(v) for v in effects.values())
    fig, axes = plt.subplots(len(metrics), n_cols, figsize=(4.2 * n_cols, 3.5 * len(metrics)))
    axes = np.atleast_2d(axes)
    for row, metric in enumerate(metrics):
        for col, effect in enumerate(effects[metric]):
            ax = axes[row, col]
            block = curve[(curve["metric"] == metric) & (curve["true_relative_effect"] == effect)]
            for estimator in COLOURS:
                sub = block[block["estimator"] == estimator].sort_values("n_total")
                if sub.empty:
                    continue
                ax.plot(sub["n_total"], sub["power"], "o-", ms=4, lw=1.4,
                        color=COLOURS[estimator], label=LABELS[estimator])
                ax.fill_between(sub["n_total"], sub["power_lo"], sub["power_hi"],
                                color=COLOURS[estimator], alpha=0.15, linewidth=0)
            ax.axhline(0.8, color="#333333", lw=1.0, ls="--")
            ax.set_ylim(0, 1.02)
            _style(ax, f"{METRIC_TITLES.get(metric, metric).splitlines()[0]}, true lift {effect:.1%}",
                   "total users", "power")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Measured power curves. Shaded bands are 95 percent Wilson intervals.", fontsize=11, y=1.0)
    fig.tight_layout()
    return _save(fig, "fig07_power_curves.png")
