"""Tests for feature engineering and player-history isolation."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import build_player_features, feature_columns


def make_fake_games(n: int = 30, player_id: int | None = None, points_start: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2024-10-25", periods=n, freq="2D")
    frame = pd.DataFrame({
        "GAME_DATE": dates, "MATCHUP": ["LAL vs. BOS"] * n, "MIN": [34] * n,
        "PTS": list(range(points_start, points_start + n)), "REB": [7] * n,
        "AST": [8] * n, "OPPONENT_TEAM_ID": [1610612738] * n,
    })
    if player_id is not None:
        frame["PLAYER_ID"] = player_id
        frame["PLAYER_NAME"] = f"Player {player_id}"
    return frame


def test_features_have_expected_columns() -> None:
    frame = build_player_features(make_fake_games(), target="PTS")
    for column in feature_columns("PTS"):
        assert column in frame.columns


def test_rolling_mean_does_not_include_current_game() -> None:
    frame = build_player_features(make_fake_games(20), target="PTS")
    last = frame.iloc[-1]
    assert last["last5_PTS_mean"] == pytest.approx(sum(range(34, 39)) / 5)
    assert last["last5_PTS_mean"] < last["PTS"]


def test_league_table_keeps_each_players_history_separate() -> None:
    player_a = make_fake_games(n=3, player_id=1, points_start=10)
    player_b = make_fake_games(n=3, player_id=2, points_start=100)
    player_a["GAME_DATE"] = pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05"])
    player_b["GAME_DATE"] = pd.to_datetime(["2025-01-02", "2025-01-04", "2025-01-06"])
    both = pd.concat([player_a, player_b], ignore_index=True).sort_values("GAME_DATE")
    features = build_player_features(both, target="PTS")
    b_last = features.loc[features["PLAYER_ID"].eq(2)].sort_values("GAME_DATE").iloc[-1]
    assert b_last["last5_PTS_mean"] == pytest.approx((100 + 101) / 2)


def test_target_validation() -> None:
    with pytest.raises(ValueError):
        build_player_features(make_fake_games(), target="STL")
