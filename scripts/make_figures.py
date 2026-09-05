"""Build every figure from the committed results tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from expcost.calibration import RevenueShape
from expcost import plotting


def _read(name: str):
    path = ROOT / "results" / name
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    wanted = set(args.only.split(",")) if args.only else None
    built = []

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    if want("calibration"):
        shape = RevenueShape.from_json(ROOT / "results/revenue_shape.json")
        summary = _read("calibration_empirical_summary.csv")
        built.append(plotting.figure_calibration(shape, summary))

    if want("sample_size"):
        summary = _read("sample_size.csv")
        if summary is not None:
            built.append(plotting.figure_sample_size(summary))

    if want("power_curves"):
        curve = _read("power_curve.csv")
        if curve is not None:
            built.append(plotting.figure_power_curves(curve))

    if want("cuped"):
        curve = _read("cuped_curve.csv")
        if curve is not None:
            built.append(plotting.figure_cuped_curve(curve))

    if want("peeking"):
        peek = _read("peeking.csv")
        if peek is not None:
            built.append(plotting.figure_peeking(peek))

    if want("srm"):
        power, detectable = _read("srm_power.csv"), _read("srm_detectable.csv")
        if power is not None and detectable is not None:
            built.append(plotting.figure_srm(power, detectable))

    if want("bandit"):
        coverage, regret = _read("bandit_intervals.csv"), _read("bandit_regret.csv")
        if coverage is not None and regret is not None:
            built.append(plotting.figure_bandit(coverage, regret))

    for path in built:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
