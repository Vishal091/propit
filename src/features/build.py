"""Leakage-aware, player-isolated feature engineering for NBA prop models."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20)
PLAYER_KEY_CANDIDATES: tuple[str, ...] = ("PLAYER_ID", "player_id", "player_name", "PLAYER_NAME")


def build_player_features(
    games: pd.DataFrame,
    target: str,
    opponent_team_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build prior-game features, isolating histories for different players."""
    if target not in {"PTS", "REB", "AST"}:
        raise ValueError(f"target must be PTS, REB, or AST; got {target}")
    if games.empty:
        return games.copy()
    df = games.copy()
    group_col = next(
        (col for col in PLAYER_KEY_CANDIDATES if col in df.columns and df[col].nunique(dropna=False) > 1),
        None,
    )
    if group_col is not None:
        built: list[pd.DataFrame] = []
        for _, player_games in df.groupby(group_col, sort=False, dropna=False):
            frame = _build_single_player_features(player_games, target, opponent_team_metrics)
            if not frame.empty:
                built.append(frame)
        if not built:
            return df.iloc[0:0].copy()
        out = pd.concat(built, ignore_index=True)
        sort_cols = ["GAME_DATE"]
        if group_col in out.columns:
            sort_cols.append(group_col)
        if "GAME_ID" in out.columns:
            sort_cols.append("GAME_ID")
        return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return _build_single_player_features(df, target, opponent_team_metrics)


def _build_single_player_features(
    games: pd.DataFrame,
    target: str,
    opponent_team_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = games.copy()
    required = {"GAME_DATE", "MATCHUP", target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"game log missing required columns: {', '.join(sorted(missing))}")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE", kind="stable").reset_index(drop=True)
    if df.empty:
        return df
    if "OPPONENT_TEAM_ID" not in df:
        df["OPPONENT_TEAM_ID"] = 0
    df["OPPONENT_TEAM_ID"] = pd.to_numeric(df["OPPONENT_TEAM_ID"], errors="coerce").fillna(0)
    df["rest_days"] = df["GAME_DATE"].diff().dt.days.fillna(7).clip(upper=14)
    df["is_home"] = df["MATCHUP"].astype(str).str.contains("vs", case=False, na=False).astype(int)
    for window in ROLLING_WINDOWS:
        df[f"last{window}_{target}_mean"] = df[target].shift(1).rolling(window=window, min_periods=1).mean()
    df[f"ema_{target}"] = df[target].shift(1).ewm(alpha=0.3, adjust=False).mean()
    df[f"season_{target}_mean"] = df[target].shift(1).expanding(min_periods=1).mean()
    if opponent_team_metrics is not None and len(opponent_team_metrics):
        metrics = opponent_team_metrics.drop_duplicates("TEAM_ID").set_index("TEAM_ID")
        df = df.drop(columns=["opp_def_rating", "opp_pace"], errors="ignore").merge(
            metrics[["DEF_RATING", "PACE"]].rename(columns={"DEF_RATING": "opp_def_rating", "PACE": "opp_pace"}),
            left_on="OPPONENT_TEAM_ID", right_index=True, how="left", sort=False,
        )
    else:
        df["opp_def_rating"] = np.nan
        df["opp_pace"] = np.nan
    df["opp_def_rating"] = df["opp_def_rating"].fillna(112.0)
    df["opp_pace"] = df["opp_pace"].fillna(99.0)
    df = df.dropna(subset=[f"last{ROLLING_WINDOWS[0]}_{target}_mean", f"ema_{target}"])
    return df.reset_index(drop=True)


def feature_columns(target: str) -> list[str]:
    """Return feature columns for one supported prop target."""
    if target not in {"PTS", "REB", "AST"}:
        raise ValueError(f"target must be PTS, REB, or AST; got {target}")
    static = ["rest_days", "is_home", "opp_def_rating", "opp_pace"]
    rolling = [f"last{window}_{target}_mean" for window in ROLLING_WINDOWS]
    rolling += [f"ema_{target}", f"season_{target}_mean"]
    return static + rolling
