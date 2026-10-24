"""Leakage-aware, player-isolated feature engineering for NBA prop models."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20)
TEAM_CONTEXT_WINDOW = 10
PLAYER_KEY_CANDIDATES: tuple[str, ...] = ("PLAYER_ID", "player_id", "player_name", "PLAYER_NAME")
TEAM_GAME_REQUIRED = {
    "GAME_ID", "GAME_DATE", "TEAM_ID", "OPPONENT_TEAM_ID",
    "PTS", "FGA", "FTA", "OREB", "TOV",
}


def build_team_game_metrics(games: pd.DataFrame, window: int = TEAM_CONTEXT_WINDOW) -> pd.DataFrame:
    """Build time-aware team defense and possession-volume context from past box scores.

    Team rows are reconstructed from player game logs. Estimated possessions use
    FGA + 0.44 * FTA - OREB + TOV; defensive rating uses opponent points divided
    by opponent estimated possessions. Returned values are rolling means through
    each team game. Callers must join them strictly before the target game date
    so no target-game statistics leak into a pregame feature.
    """
    columns = ["TEAM_ID", "GAME_DATE", "DEF_RATING", "PACE"]
    if games.empty or not TEAM_GAME_REQUIRED.issubset(games.columns):
        return pd.DataFrame(columns=columns)
    if window < 1:
        raise ValueError("window must be at least 1")

    source = games[
        ["GAME_ID", "GAME_DATE", "TEAM_ID", "OPPONENT_TEAM_ID",
         "PTS", "FGA", "FTA", "OREB", "TOV"]
    ].copy()
    source["GAME_DATE"] = pd.to_datetime(source["GAME_DATE"], errors="coerce")
    for column in ("TEAM_ID", "OPPONENT_TEAM_ID", "PTS", "FGA", "FTA", "OREB", "TOV"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=["GAME_ID", "GAME_DATE", "TEAM_ID", "OPPONENT_TEAM_ID"])
    if source.empty:
        return pd.DataFrame(columns=columns)

    team_games = (
        source.groupby(["TEAM_ID", "GAME_ID", "GAME_DATE"], as_index=False, sort=False)
        .agg(
            OPPONENT_TEAM_ID=("OPPONENT_TEAM_ID", "first"),
            PTS=("PTS", lambda values: values.sum(min_count=1)),
            FGA=("FGA", lambda values: values.sum(min_count=1)),
            FTA=("FTA", lambda values: values.sum(min_count=1)),
            OREB=("OREB", lambda values: values.sum(min_count=1)),
            TOV=("TOV", lambda values: values.sum(min_count=1)),
        )
    )
    team_games["ESTIMATED_POSSESSIONS"] = (
        team_games["FGA"] + 0.44 * team_games["FTA"] - team_games["OREB"] + team_games["TOV"]
    ).clip(lower=1.0)

    opponent_stats = team_games[
        ["GAME_ID", "TEAM_ID", "PTS", "ESTIMATED_POSSESSIONS"]
    ].rename(columns={
        "TEAM_ID": "OPPONENT_TEAM_ID",
        "PTS": "OPPONENT_POINTS",
        "ESTIMATED_POSSESSIONS": "OPPONENT_POSSESSIONS",
    })
    team_games = team_games.merge(
        opponent_stats,
        on=["GAME_ID", "OPPONENT_TEAM_ID"],
        how="left",
        validate="one_to_one",
    )
    team_games["DEF_RATING"] = (
        100.0 * team_games["OPPONENT_POINTS"] / team_games["OPPONENT_POSSESSIONS"]
    )
    team_games["PACE"] = team_games["ESTIMATED_POSSESSIONS"]
    team_games = team_games.dropna(subset=["DEF_RATING", "PACE"]).sort_values(
        ["TEAM_ID", "GAME_DATE", "GAME_ID"], kind="stable"
    ).reset_index(drop=True)

    # Include each game in the metric stored on its date; a strict prior-date
    # join later ensures that a game cannot influence its own pregame features.
    for column in ("DEF_RATING", "PACE"):
        team_games[column] = team_games.groupby("TEAM_ID", sort=False)[column].transform(
            lambda values: values.rolling(window=window, min_periods=1).mean()
        )
    return team_games[columns].sort_values(["GAME_DATE", "TEAM_ID"], kind="stable").reset_index(drop=True)


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


def _attach_opponent_metrics(
    df: pd.DataFrame,
    opponent_team_metrics: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach actual opponent metrics known strictly before each row's game date."""
    result = df.copy()
    if opponent_team_metrics is not None and not opponent_team_metrics.empty:
        metrics = opponent_team_metrics.copy()
        timed_columns = {"TEAM_ID", "GAME_DATE", "DEF_RATING", "PACE"}
        if timed_columns.issubset(metrics.columns):
            left = result.copy()
            left["_FEATURE_ORDER"] = np.arange(len(left))
            left["OPPONENT_TEAM_ID"] = pd.to_numeric(
                left["OPPONENT_TEAM_ID"], errors="coerce"
            ).fillna(0).astype("int64")
            right = metrics[["TEAM_ID", "GAME_DATE", "DEF_RATING", "PACE"]].copy()
            right["TEAM_ID"] = pd.to_numeric(right["TEAM_ID"], errors="coerce")
            right["GAME_DATE"] = pd.to_datetime(right["GAME_DATE"], errors="coerce")
            right["DEF_RATING"] = pd.to_numeric(right["DEF_RATING"], errors="coerce")
            right["PACE"] = pd.to_numeric(right["PACE"], errors="coerce")
            right = right.dropna(subset=["TEAM_ID", "GAME_DATE"]).rename(columns={
                "TEAM_ID": "_METRIC_TEAM_ID",
                "GAME_DATE": "_METRIC_DATE",
                "DEF_RATING": "_OPP_DEF_RATING",
                "PACE": "_OPP_PACE",
            })
            right["_METRIC_TEAM_ID"] = right["_METRIC_TEAM_ID"].astype("int64")
            left_sorted = left.sort_values("GAME_DATE", kind="stable")
            right_sorted = right.sort_values("_METRIC_DATE", kind="stable")
            joined = pd.merge_asof(
                left_sorted,
                right_sorted,
                left_on="GAME_DATE",
                right_on="_METRIC_DATE",
                left_by="OPPONENT_TEAM_ID",
                right_by="_METRIC_TEAM_ID",
                direction="backward",
                allow_exact_matches=False,
            )
            joined = joined.sort_values("_FEATURE_ORDER", kind="stable")
            result["opp_def_rating"] = joined["_OPP_DEF_RATING"].to_numpy()
            result["opp_pace"] = joined["_OPP_PACE"].to_numpy()
        elif {"TEAM_ID", "DEF_RATING", "PACE"}.issubset(metrics.columns):
            # Backward-compatible support for an explicitly static metric table.
            static = metrics.drop_duplicates("TEAM_ID").set_index("TEAM_ID")
            result = result.drop(columns=["opp_def_rating", "opp_pace"], errors="ignore").merge(
                static[["DEF_RATING", "PACE"]].rename(
                    columns={"DEF_RATING": "opp_def_rating", "PACE": "opp_pace"}
                ),
                left_on="OPPONENT_TEAM_ID",
                right_index=True,
                how="left",
                sort=False,
            )
        else:
            result["opp_def_rating"] = np.nan
            result["opp_pace"] = np.nan
    else:
        result["opp_def_rating"] = np.nan
        result["opp_pace"] = np.nan

    result["opp_def_rating"] = pd.to_numeric(result["opp_def_rating"], errors="coerce").fillna(112.0)
    result["opp_pace"] = pd.to_numeric(result["opp_pace"], errors="coerce").fillna(99.0)
    return result


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
    df["is_home"] = df["MATCHUP"].astype(str).str.contains(r"\bvs\b", case=False, na=False).astype(int)
    for window in ROLLING_WINDOWS:
        df[f"last{window}_{target}_mean"] = df[target].shift(1).rolling(window=window, min_periods=1).mean()
    df[f"ema_{target}"] = df[target].shift(1).ewm(alpha=0.3, adjust=False).mean()
    df[f"season_{target}_mean"] = df[target].shift(1).expanding(min_periods=1).mean()
    df = _attach_opponent_metrics(df, opponent_team_metrics)
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
