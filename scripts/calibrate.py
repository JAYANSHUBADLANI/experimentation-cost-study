"""Fit the revenue shape to real order values and write the committed evidence.

Needs the Olist raw CSVs. Point at them with --olist-dir or the environment
variable EXPCOST_OLIST_DIR. The outputs of this script are committed, so the
rest of the study runs without the raw data present.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from expcost.calibration import empirical_summary, fit_revenue_shape, load_olist_order_values
from expcost.seeds import generator

SAMPLE_SIZE = 20_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--olist-dir", default=os.environ.get("EXPCOST_OLIST_DIR", ""))
    args = parser.parse_args()
    if not args.olist_dir:
        raise SystemExit("set --olist-dir or EXPCOST_OLIST_DIR to the Olist CSV directory")

    values = load_olist_order_values(Path(args.olist_dir))
    shape = fit_revenue_shape(values, source="olist_brazilian_ecommerce_order_payment_value")
    shape.to_json(Path("results/revenue_shape.json"))
    empirical_summary(values).to_csv("results/calibration_empirical_summary.csv", index=False)

    rng = generator("calibration_sample")
    sample = np.sort(rng.choice(values, size=min(SAMPLE_SIZE, values.size), replace=False))
    pd.DataFrame({"order_value": np.round(sample, 2)}).to_csv(
        "data/sample/olist_order_values_sample.csv", index=False
    )

    print(f"fitted on {shape.n_observations} order values from {shape.source}")
    print(f"  lognormal body   log_mu={shape.log_mu:.4f} log_sigma={shape.log_sigma:.4f}")
    print(f"  splice threshold {shape.threshold:.2f} at q{shape.tail_quantile}")
    print(f"  Pareto tail      xi={shape.gpd_xi:.4f} scale={shape.gpd_scale:.2f}")
    print(f"  empirical mean {shape.empirical_mean:.2f} sd {shape.empirical_sd:.2f} cv {shape.empirical_cv:.3f}")
    print(f"  committed sample of {sample.size} values for the test suite")


if __name__ == "__main__":
    main()
