"""
Train a quantile XGBoost ensemble for a given prop target.

Three regressors are fit at quantiles 0.1, 0.5, 0.9 to produce a credible interval
plus a point estimate. The 0.5 model also acts as the basis for the over/under
probability calculation; the empirical residuals from the 0.5 model are used to
fit a Gaussian and convert any (line, prediction) pair into P(stat > line).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.features.build import feature_columns

log = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainedModel:
    """Bundle of models + residual scale for a single target."""

    target: str
    q10: XGBRegressor
    q50: XGBRegressor
    q90: XGBRegressor
    residual_std: float

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        return joblib.load(path)


def _fit_quantile(X: pd.DataFrame, y: pd.Series, q: float) -> XGBRegressor:
    """Fit one quantile-loss XGBoost regressor."""
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=q,
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model


def train(features_df: pd.DataFrame, target: str) -> TrainedModel:
    """
    Train quantile ensemble. Splits 80/20 by date so we never train on the future.
    """
    df = features_df.sort_values("GAME_DATE").reset_index(drop=True)
    cols = feature_columns(target)
    X = df[cols].astype(float)
    y = df[target].astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    q10 = _fit_quantile(X_tr, y_tr, 0.1)
    q50 = _fit_quantile(X_tr, y_tr, 0.5)
    q90 = _fit_quantile(X_tr, y_tr, 0.9)

    pred_50 = q50.predict(X_te)
    mae = mean_absolute_error(y_te, pred_50)
    residual_std = float(np.std(y_te - pred_50))
    log.info("target=%s mae=%.3f residual_std=%.3f", target, mae, residual_std)

    bundle = TrainedModel(
        target=target,
        q10=q10,
        q50=q50,
        q90=q90,
        residual_std=max(residual_std, 1e-6),
    )
    bundle.save(ARTIFACTS / f"model_{target}.joblib")
    return bundle
