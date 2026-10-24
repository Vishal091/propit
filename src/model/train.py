"""Train quantile XGBoost models with a strictly chronological holdout split."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.features.build import feature_columns

log = logging.getLogger(__name__)
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainedModel:
    """Bundle of quantile models and residual statistics for a target."""
    target: str
    q10: XGBRegressor
    q50: XGBRegressor
    q90: XGBRegressor
    residual_std: float
    feature_means: dict[str, float] | None = None
    feature_stds: dict[str, float] | None = None
    feature_names: tuple[str, ...] | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        return joblib.load(path)


def _fit_quantile(X: pd.DataFrame, y: pd.Series, q: float) -> XGBRegressor:
    """Fit one quantile-loss boosted-tree regressor."""
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=q,
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _chronological_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Split by unique game date so a date never appears in both partitions."""
    dates = pd.Series(pd.to_datetime(df["GAME_DATE"], errors="coerce").dropna().unique()).sort_values().reset_index(drop=True)
    if len(dates) < 2:
        raise ValueError("at least two distinct game dates are required for a chronological split")
    split_at = min(max(int(len(dates) * 0.8), 1), len(dates) - 1)
    cutoff = dates.iloc[split_at]
    game_dates = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return game_dates < cutoff, game_dates >= cutoff


def train(features_df: pd.DataFrame, target: str) -> TrainedModel:
    """Fit three quantile models and report MAE on a later-date holdout."""
    if features_df.empty:
        raise ValueError(f"no feature rows available for {target}")
    df = features_df.sort_values("GAME_DATE", kind="stable").reset_index(drop=True)
    cols = feature_columns(target)
    X = df[cols].astype(float)
    y = df[target].astype(float)
    train_mask, test_mask = _chronological_masks(df)
    if not train_mask.any() or not test_mask.any():
        raise ValueError("chronological split produced an empty train or test set")
    X_tr, X_te = X.loc[train_mask], X.loc[test_mask]
    y_tr, y_te = y.loc[train_mask], y.loc[test_mask]
    q10 = _fit_quantile(X_tr, y_tr, 0.1)
    q50 = _fit_quantile(X_tr, y_tr, 0.5)
    q90 = _fit_quantile(X_tr, y_tr, 0.9)
    pred_50 = q50.predict(X_te)
    mae = mean_absolute_error(y_te, pred_50)
    residual_std = float(np.std(y_te - pred_50))
    log.info("target=%s chronological_holdout_mae=%.3f residual_std=%.3f", target, mae, residual_std)
    bundle = TrainedModel(
        target=target, q10=q10, q50=q50, q90=q90,
        residual_std=max(residual_std, 1e-6),
        feature_means={column: float(X_tr[column].mean()) for column in cols},
        feature_stds={column: float(X_tr[column].std(ddof=0)) for column in cols},
        feature_names=tuple(cols),
    )
    bundle.save(ARTIFACTS / f"model_{target}.joblib")
    return bundle
