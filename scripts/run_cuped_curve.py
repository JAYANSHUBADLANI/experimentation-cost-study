"""Measure the CUPED variance reduction curve across the correlation grid."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config.scenarios import CUPED_CURVE_N, CUPED_CURVE_REPLICATIONS, RHO_GRID, metrics
from expcost.cuped_curve import CurvePoint, break_even_correlation, run_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=CUPED_CURVE_REPLICATIONS)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    points = [
        CurvePoint(key, spec, rho, CUPED_CURVE_N, args.replications)
        for key, spec in metrics().items()
        for rho in RHO_GRID
    ]
    print(f"{len(points)} curve points, {args.replications} replications at n={CUPED_CURVE_N:,}")
    start = time.perf_counter()
    curve = run_curve(points, args.workers)
    curve.to_csv(ROOT / "results/cuped_curve.csv", index=False)
    print(f"ran in {time.perf_counter() - start:.1f} s, wrote results/cuped_curve.csv")
    print(f"theoretical break even correlation at a 10 percent saving: {break_even_correlation():.3f}")

    view = curve[curve["estimator"] == "cuped"]
    print(view[["metric", "target_rho", "realised_rho_mean", "variance_ratio", "theoretical_ratio", "users_saved_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
