"""Thompson sampling against a fixed horizon test: regret, width and validity."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config.scenarios import BANDIT_ARM_RATES, BANDIT_BATCH, BANDIT_REPLICATIONS, BANDIT_ROUNDS
from expcost.bandit import coverage_table, run_bandit

FLOORS = (0.05, 0.01, 0.002)
HEADLINE_FLOOR = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=BANDIT_REPLICATIONS)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    start = time.perf_counter()
    rows = []
    regret_rows = []
    for floor in FLOORS:
        for label, rates in [("null", (BANDIT_ARM_RATES[0], BANDIT_ARM_RATES[0])),
                             ("lift", BANDIT_ARM_RATES)]:
            outcome = run_bandit(
                rates, BANDIT_ROUNDS, BANDIT_BATCH, args.replications,
                label=f"bandit|{label}|{floor:g}", floor=floor,
            )
            for row in coverage_table(outcome):
                rows.append({"scenario": label, "exploration_floor": floor,
                             "arm_rates": f"{rates[0]:g}/{rates[1]:g}", **row})
            total_users = BANDIT_ROUNDS * BANDIT_BATCH
            regret_rows.append({
                "scenario": label,
                "exploration_floor": floor,
                "arm_rates": f"{rates[0]:g}/{rates[1]:g}",
                "total_users": total_users,
                "bandit_regret": float(outcome.regret.mean()),
                "balanced_regret": 0.5 * total_users * outcome.true_effect,
                "regret_saved": 0.5 * total_users * outcome.true_effect - float(outcome.regret.mean()),
                "share_to_best_arm": float(outcome.share_to_best.mean()),
                "replications": args.replications,
            })

    coverage = pd.DataFrame(rows)
    regret = pd.DataFrame(regret_rows)
    coverage.to_csv(ROOT / "results/bandit_intervals.csv", index=False)
    regret.to_csv(ROOT / "results/bandit_regret.csv", index=False)
    print(f"ran in {time.perf_counter() - start:.1f} s")
    print(coverage.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
    print(regret.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
