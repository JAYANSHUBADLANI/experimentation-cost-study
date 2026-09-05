"""Measure the false positive rate under daily peeking, and what fixing it costs."""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config.scenarios import ALPHA, PEEK_DAYS, PEEK_REPLICATIONS, PEEK_USERS_PER_DAY, metrics
from expcost.peeking import build_peek_chunks, run_peek_chunk
from expcost.power import wilson_interval
from expcost.sequential import inflation_factor

RULES = ["fixed_horizon", "naive_peeking", "msprt", "alpha_spending"]
TAU_RELATIVE_EFFECT = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=PEEK_REPLICATIONS)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--effect", type=float, default=0.0)
    args = parser.parse_args()

    chunks = []
    for key, spec in metrics().items():
        chunks += build_peek_chunks(
            key, spec, args.effect, PEEK_DAYS, PEEK_USERS_PER_DAY,
            args.replications, TAU_RELATIVE_EFFECT, ALPHA,
        )
    total_users = args.replications * PEEK_DAYS * PEEK_USERS_PER_DAY * len(metrics())
    print(f"{len(chunks)} chunks, {args.replications} replications per metric, "
          f"{PEEK_DAYS} daily looks at {PEEK_USERS_PER_DAY:,} users a day, {total_users:,} simulated users")

    start = time.perf_counter()
    with Pool(args.workers) as pool:
        results = pool.map(run_peek_chunk, chunks, chunksize=1)
    elapsed = time.perf_counter() - start

    buckets: dict[str, dict[str, list]] = {}
    for result in results:
        bucket = buckets.setdefault(result["metric"], {rule: [] for rule in RULES} | {"stop": []})
        for rule in RULES:
            bucket[rule].append(result[rule])
        bucket["stop"].append(result["naive_stop_day"])

    rows = []
    for metric, bucket in buckets.items():
        for rule in RULES:
            flags = np.concatenate(bucket[rule])
            successes = int(flags.sum())
            low, high = wilson_interval(successes, flags.size)
            rows.append({
                "metric": metric,
                "true_relative_effect": args.effect,
                "reading_rule": rule,
                "days": PEEK_DAYS,
                "users_per_day": PEEK_USERS_PER_DAY,
                "replications": int(flags.size),
                "rejection_rate": successes / flags.size,
                "rate_lo": low,
                "rate_hi": high,
                "nominal_alpha": ALPHA,
            })
    table = pd.DataFrame(rows)
    suffix = "" if args.effect == 0 else f"_effect{args.effect:g}"
    table.to_csv(ROOT / f"results/peeking{suffix}.csv", index=False)

    fractions = np.arange(1, PEEK_DAYS + 1) / PEEK_DAYS
    cost = pd.DataFrame([{
        "procedure": "alpha_spending_obrien_fleming",
        "looks": PEEK_DAYS,
        "max_sample_size_inflation": inflation_factor(fractions, ALPHA, 0.80),
        "note": "multiplier on the fixed horizon sample size to keep 80 percent power",
    }])
    cost.to_csv(ROOT / f"results/sequential_cost{suffix}.csv", index=False)

    print(f"ran in {elapsed:.1f} s")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(cost.to_string(index=False))


if __name__ == "__main__":
    main()
