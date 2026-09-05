"""Sample ratio mismatch: detection power and the discard threshold."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config.scenarios import SRM_N_GRID, SRM_REPLICATIONS, SRM_TRUE_ALLOCATIONS
from expcost.srm import DISCARD_THRESHOLD, run_srm_study, smallest_detectable_mismatch


def main() -> None:
    table = run_srm_study(SRM_N_GRID, SRM_TRUE_ALLOCATIONS, 0.5, SRM_REPLICATIONS)
    table.to_csv(ROOT / "results/srm_power.csv", index=False)

    detectable = pd.DataFrame([
        {
            "n_total": n,
            "intended_allocation": 0.5,
            "smallest_detectable_drift": smallest_detectable_mismatch(n),
            "as_percent_of_users": 100 * smallest_detectable_mismatch(n),
            "threshold": DISCARD_THRESHOLD,
            "power": 0.80,
        }
        for n in SRM_N_GRID
    ])
    detectable.to_csv(ROOT / "results/srm_detectable.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
    print(detectable.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    main()
