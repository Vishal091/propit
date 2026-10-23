"""Inference utilities for expected stat values and over/under probabilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.build import feature_columns
from src.model.train import TrainedModel

ARTIFACTS = Path("artifacts")


@dataclass
class Prediction:
    target: str
    line: float | None
    p_over: float | None
    expected_value: float
    interval_80: tuple[float, float]
    drivers: list[dict[str, Any]]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def predict_one(
    feature_row: pd.Series,
    line: float | None,
    target: str,
    bundle: TrainedModel | None = None,
) -> Prediction:
    """Predict expected stat and interval; calculate P(over) only with a line."""
    if bundle is None:
        bundle = TrainedModel.load(ARTIFACTS / f"model_{target}.joblib")
    cols = feature_columns(target)
    X = pd.DataFrame([feature_row[cols].astype(float).values], columns=cols)
    p10 = float(bundle.q10.predict(X)[0])
    p50 = float(bundle.q50.predict(X)[0])
    p90 = float(bundle.q90.predict(X)[0])
    p_over: float | None = None
    if line is not None:
        z = (float(line) - p50) / bundle.residual_std
        p_over = float(1.0 - _normal_cdf(z))
    importance = bundle.q50.feature_importances_
    baselines = getattr(bundle, "feature_means", None) or {}
    drivers = []
    for index, col in enumerate(cols):
        value = float(X.iloc[0, index])
        baseline = float(baselines.get(col, value))
        weight = float(importance[index])
        drivers.append({"feature": col, "value": value, "weight": weight, "score": weight * abs(value - baseline)})
    drivers.sort(key=lambda item: item["score"], reverse=True)
    return Prediction(
        target=target,
        line=float(line) if line is not None else None,
        p_over=p_over,
        expected_value=p50,
        interval_80=(p10, p90),
        drivers=drivers[:3],
    )
