"""
Inference: given features for a single matchup, return p_over, expected value,
credible interval, and SHAP-ish driver attributions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.build import feature_columns
from src.model.train import TrainedModel

ARTIFACTS = Path("artifacts")


@dataclass
class Prediction:
    target: str
    line: float
    p_over: float
    expected_value: float
    interval_80: tuple[float, float]
    drivers: list[dict[str, Any]]


def _normal_cdf(x: float) -> float:
    """Standard normal CDF. No scipy dependency required."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def predict_one(
    feature_row: pd.Series,
    line: float,
    target: str,
    bundle: TrainedModel | None = None,
) -> Prediction:
    """Run the full prediction pipeline on a single feature row."""
    if bundle is None:
        bundle = TrainedModel.load(ARTIFACTS / f"model_{target}.joblib")

    cols = feature_columns(target)
    X = pd.DataFrame([feature_row[cols].astype(float).values], columns=cols)

    p10 = float(bundle.q10.predict(X)[0])
    p50 = float(bundle.q50.predict(X)[0])
    p90 = float(bundle.q90.predict(X)[0])

    # P(actual > line) under a Gaussian residual model around the median forecast.
    z = (line - p50) / bundle.residual_std
    p_over = float(1.0 - _normal_cdf(z))

    # Driver attribution via gain importance combined with the row deviation.
    importance = bundle.q50.feature_importances_
    means = X.mean()
    drivers = sorted(
        (
            {
                "feature": cols[i],
                "value": float(X.iloc[0, i]),
                "weight": float(importance[i]),
                "score": float(importance[i] * abs(X.iloc[0, i] - means[i])),
            }
            for i in range(len(cols))
        ),
        key=lambda d: d["score"],
        reverse=True,
    )[:3]

    return Prediction(
        target=target,
        line=line,
        p_over=p_over,
        expected_value=p50,
        interval_80=(p10, p90),
        drivers=drivers,
    )
