"""Inference utilities for expected stat values and over/under probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.build import feature_columns
from src.model.prop_probability import estimate_prop_outcomes
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
    # Conditional on a decisive result (pushes excluded), matching calibration
    # targets used by the chronological settled-prop workflow.
    p_push: float | None = None
    probability_distribution: str | None = None


def _probability_over(
    line: float,
    p10: float,
    p50: float,
    p90: float,
    fallback_std: float,
    target: str = "PTS",
) -> tuple[float, float, str]:
    """Return P(over | no push), P(push), and the discrete distribution family."""
    outcomes = estimate_prop_outcomes(
        mean=max(float(p50), 0.0),
        p10=float(p10),
        p90=float(p90),
        line=float(line),
        target=target,
        fallback_std=float(fallback_std),
    )
    decisive = outcomes.p_over + outcomes.p_under
    conditional_over = outcomes.p_over / decisive if decisive > 1e-9 else 0.5
    return float(conditional_over), float(outcomes.p_push), outcomes.distribution


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
    p_push: float | None = None
    distribution: str | None = None
    if line is not None:
        p_over, p_push, distribution = _probability_over(
            float(line), p10, p50, p90, float(bundle.residual_std), target=target
        )
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
        p_push=p_push,
        probability_distribution=distribution,
    )
