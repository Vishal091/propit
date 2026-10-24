"""Tests for feature engineering and player-history isolation."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import (
    build_player_features,
    build_team_game_metrics,
    feature_columns,
)


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


def _team_game(game_id, day, team_id, opponent_id, pts, fga=80, fta=20, oreb=10, tov=10):
    return {
        "GAME_ID": str(game_id),
        "GAME_DATE": pd.Timestamp(day),
        "TEAM_ID": team_id,
        "OPPONENT_TEAM_ID": opponent_id,
        "PTS": pts,
        "FGA": fga,
        "FTA": fta,
        "OREB": oreb,
        "TOV": tov,
    }


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


def test_team_context_is_rolled_from_real_games_not_fixed_defaults():
    rows = [
        _team_game("g1", "2024-01-01", 1, 2, 100, 80, 20, 10, 10),
        _team_game("g1", "2024-01-01", 2, 1, 90, 70, 20, 8, 12),
        _team_game("g2", "2024-01-03", 1, 2, 110, 85, 18, 9, 11),
        _team_game("g2", "2024-01-03", 2, 1, 95, 75, 22, 7, 10),
    ]
    metrics = build_team_game_metrics(pd.DataFrame(rows))
    assert len(metrics) == 4
    first_team_2 = metrics.loc[(metrics["TEAM_ID"] == 2) & (metrics["GAME_DATE"] == pd.Timestamp("2024-01-01"))].iloc[0]
    second_team_2 = metrics.loc[(metrics["TEAM_ID"] == 2) & (metrics["GAME_DATE"] == pd.Timestamp("2024-01-03"))].iloc[0]
    assert first_team_2["DEF_RATING"] != pytest.approx(112.0)
    assert second_team_2["PACE"] != pytest.approx(99.0)


def test_historical_opponent_metrics_are_from_prior_games_only():
    games = pd.DataFrame([
        {**_team_game("g1", "2024-01-01", 1, 2, 100), "PLAYER_ID": 11, "PLAYER_NAME": "A", "MATCHUP": "AAA vs. BBB", "REB": 5, "AST": 3},
        {**_team_game("g1", "2024-01-01", 2, 1, 90), "PLAYER_ID": 22, "PLAYER_NAME": "B", "MATCHUP": "BBB @ AAA", "REB": 6, "AST": 4},
        {**_team_game("g2", "2024-01-03", 1, 2, 110), "PLAYER_ID": 11, "PLAYER_NAME": "A", "MATCHUP": "AAA vs. BBB", "REB": 6, "AST": 4},
        {**_team_game("g2", "2024-01-03", 2, 1, 95), "PLAYER_ID": 22, "PLAYER_NAME": "B", "MATCHUP": "BBB @ AAA", "REB": 5, "AST": 3},
    ])
    metrics = build_team_game_metrics(games)
    features = build_player_features(
        games.loc[games["PLAYER_ID"].eq(11)].copy(),
        target="PTS",
        opponent_team_metrics=metrics,
    )
    game_two = features.loc[features["GAME_DATE"] == pd.Timestamp("2024-01-03")].iloc[0]
    team_two_prior = metrics.loc[
        (metrics["TEAM_ID"] == 2) & (metrics["GAME_DATE"] == pd.Timestamp("2024-01-01"))
    ].iloc[0]
    assert game_two["opp_def_rating"] == pytest.approx(team_two_prior["DEF_RATING"])
    assert game_two["opp_pace"] == pytest.approx(team_two_prior["PACE"])



def test_minutes_role_and_usage_features_use_only_prior_games():
    games = make_fake_games(n=7)
    games["MIN"] = [20, 20, 30, 40, 50, 60, 65]
    games["FGA"] = [5, 6, 7, 8, 9, 10, 20]
    games["FTA"] = [1, 2, 3, 4, 5, 6, 15]
    games["TOV"] = [0, 1, 0, 1, 2, 3, 9]
    games["REB"] = [4, 5, 6, 7, 8, 9, 99]
    games["AST"] = [2, 3, 4, 5, 6, 7, 99]

    features = build_player_features(games, target="PTS")
    # On the final row, all new rolling inputs must end at game six, not game seven.
    row = features.iloc[-1]
    expected_minutes = sum([20, 30, 40, 50, 60]) / 5
    expected_fga = sum([6, 7, 8, 9, 10]) / 5
    expected_fta = sum([2, 3, 4, 5, 6]) / 5
    expected_tov = sum([1, 0, 1, 2, 3]) / 5
    expected_usage = sum([6 + 0.44 * 2 + 1, 7 + 0.44 * 3 + 0, 8 + 0.44 * 4 + 1, 9 + 0.44 * 5 + 2, 10 + 0.44 * 6 + 3]) / 5

    assert row["last5_MIN_mean"] == pytest.approx(expected_minutes)
    assert row["last10_MIN_mean"] == pytest.approx(sum([20, 20, 30, 40, 50, 60]) / 6)
    assert row["minutes_trend"] == pytest.approx(expected_minutes - sum([20, 20, 30, 40, 50, 60]) / 6)
    assert row["minutes_role_ratio"] == pytest.approx(expected_minutes / sum([20, 20, 30, 40, 50, 60]) * 6)
    assert row["last5_FGA_mean"] == pytest.approx(expected_fga)
    assert row["last5_FTA_mean"] == pytest.approx(expected_fta)
    assert row["last5_TOV_mean"] == pytest.approx(expected_tov)
    assert row["last5_usage_proxy_mean"] == pytest.approx(expected_usage)
    assert row["last5_REB_mean"] == pytest.approx(sum([5, 6, 7, 8, 9]) / 5)
    assert row["last5_AST_mean"] == pytest.approx(sum([3, 4, 5, 6, 7]) / 5)


def test_minutes_parser_accepts_mm_ss_values():
    games = make_fake_games(n=3)
    games["MIN"] = ["30:00", "35:30", "40:15"]
    games["FGA"] = [10, 10, 10]
    games["FTA"] = [2, 2, 2]
    games["TOV"] = [1, 1, 1]
    features = build_player_features(games, target="PTS")
    row = features.iloc[-1]
    assert row["last5_MIN_mean"] == pytest.approx((30.0 + 35.5) / 2)
