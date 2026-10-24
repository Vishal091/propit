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
_MIN_PROBABILITY = 0.02
_MAX_PROBABILITY = 0.98


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


def _probability_over(line: float, p10: float, p50: float, p90: float, fallback_std: float) -> float:
    """Estimate P(stat > line) from the row-specific quantile interval.

    This is a distributional approximation, not a calibrated probability. The
    quantile spread lets uncertainty vary by player/game context instead of
    applying one global residual standard deviation to every prediction.
    """
    spread_std = (p90 - p10) / (2.0 * 1.2815515655446004)
    scale = spread_std if math.isfinite(spread_std) and spread_std > 1e-6 else fallback_std
    scale = max(float(scale), 1e-6)
    raw = 1.0 - _normal_cdf((line - p50) / scale)
    return float(min(_MAX_PROBABILITY, max(_MIN_PROBABILITY, raw)))


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
        p_over = _probability_over(float(line), p10, p50, p90, float(bundle.residual_std))
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
