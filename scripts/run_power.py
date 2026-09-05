"""Run the power and sample size study over the frozen grid."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config.scenarios import ALPHA, EFFECTS, N_LADDER, POWER_REPLICATIONS, RHO_MAIN, TARGET_POWER, metrics
from expcost.power import build_chunks, collect, run, summarise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=POWER_REPLICATIONS)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()

    specs = metrics()
    if args.metrics:
        wanted = set(args.metrics.split(","))
        specs = {k: v for k, v in specs.items() if k in wanted}

    chunks = []
    for key, spec in specs.items():
        chunks += build_chunks(
            key, spec, EFFECTS[key], N_LADDER, args.replications, RHO_MAIN, ALPHA
        )
    users = sum((c.rep_stop - c.rep_start) * c.n for c in chunks)
    print(f"{len(chunks)} chunks, {args.replications} replications, {users:,} simulated users")

    start = time.perf_counter()
    results = run(chunks, args.workers)
    elapsed = time.perf_counter() - start

    curve = collect(results, ALPHA)
    summary = summarise(curve, specs, ALPHA, TARGET_POWER)
    prefix = args.out_prefix
    curve.to_csv(ROOT / f"results/{prefix}power_curve.csv", index=False)
    summary.to_csv(ROOT / f"results/{prefix}sample_size.csv", index=False)
    print(f"ran in {elapsed:.1f} s on {args.workers} workers ({users / elapsed / 1e6:.2f} M users/s)")
    print(f"wrote results/{prefix}power_curve.csv and results/{prefix}sample_size.csv")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        show = summary[
            ["metric", "true_relative_effect", "estimator", "n_required_measured",
             "n_required_from_variance", "users_saved_pct"]
        ].copy()
        show["true_relative_effect"] = show["true_relative_effect"].map(lambda v: f"{v:.3g}")
        for col in ("n_required_measured", "n_required_from_variance"):
            show[col] = show[col].map(lambda v: f"{v:,.0f}")
        show["users_saved_pct"] = show["users_saved_pct"].map(lambda v: f"{v:.1f}")
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
