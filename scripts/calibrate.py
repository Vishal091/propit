"""
Reliability curve. Takes a set of predicted probabilities and observed binary
outcomes and bins them. If 70% predictions only hit 55% of the time, you have
a calibration problem, not an edge.

Usage:
    python scripts/calibrate.py predictions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def reliability(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Compute calibration table for cols `p_over` and `actual_over`."""
    df = df.dropna(subset=["p_over", "actual_over"]).copy()
    df["bin"] = pd.cut(df["p_over"], bins=np.linspace(0, 1, n_bins + 1))
    table = (
        df.groupby("bin", observed=True)
        .agg(
            count=("p_over", "size"),
            mean_predicted=("p_over", "mean"),
            mean_actual=("actual_over", "mean"),
        )
        .reset_index()
    )
    table["gap"] = table["mean_actual"] - table["mean_predicted"]
    return table


if __name__ == "__main__":
    path = Path(sys.argv[1])
    out = reliability(pd.read_csv(path))
    print(out.to_string(index=False))
