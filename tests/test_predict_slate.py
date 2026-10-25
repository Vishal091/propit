"""Regression tests for upcoming-game feature construction."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import build_team_game_metrics, feature_columns
from src.predict_slate import _build_latest_features, _decorate_and_rank_candidates, _validate_model_feature_context


def _row(game_id, day, player_id, player_name, team_id, opponent_id, points, matchup):
    return {
        "GAME_ID": str(game_id),
        "GAME_DATE": pd.Timestamp(day),
        "PLAYER_ID": player_id,
        "PLAYER_NAME": player_name,
        "TEAM_ID": team_id,
        "TEAM_ABBREVIATION": "AAA" if team_id == 1 else "BBB",
        "OPPONENT_TEAM_ID": opponent_id,
        "MATCHUP": matchup,
        "PTS": points,
        "REB": 6,
        "AST": 4,
        "FGA": 75 if team_id == 1 else 72,
        "FTA": 20,
        "OREB": 8,
        "TOV": 12,
    }


def make_games():
    rows = []
    for game_id, day, player_points, opponent_points in [
        ("g1", "2026-10-19", 10, 92),
        ("g2", "2026-10-21", 20, 96),
        ("g3", "2026-10-23", 30, 101),
    ]:
        rows.append(_row(game_id, day, 1, "Alpha Player", 1, 2, player_points, "AAA vs. BBB"))
        rows.append(_row(game_id, day, 2, "Beta Player", 2, 1, opponent_points, "BBB @ AAA"))
    return pd.DataFrame(rows)


def test_next_game_features_include_latest_completed_game_and_actual_schedule():
    games = make_games()
    team_metrics = build_team_game_metrics(games)
    player_teams = {
        1: {"team": "Alpha Team", "team_abbreviation": "AAA", "team_id": 1},
    }
    next_games = {
        "AAA": {
            "team": "Alpha Team",
            "team_abbreviation": "AAA",
            "match_date": "2026-10-25",
            "opponent_team": "Beta Team",
            "opponent_abbreviation": "BBB",
            "matchup": "AAA vs BBB",
            "home_away": "HOME",
        }
    }

    lookup = _build_latest_features(
        games,
        {1: "Alpha Player"},
        player_teams=player_teams,
        next_games=next_games,
        opponent_team_metrics=team_metrics,
        team_id_by_abbreviation={"AAA": 1, "BBB": 2},
    )

    feature_row, name, last_game_date = lookup[(1, "PTS")]
    assert name == "Alpha Player"
    assert last_game_date == "2026-10-23"
    # The forecast row follows all three completed scores: mean is (10+20+30)/3.
    assert feature_row["last5_PTS_mean"] == pytest.approx(20.0)
    assert feature_row["rest_days"] == pytest.approx(2.0)
    assert feature_row["is_home"] == 1
    expected_opponent = team_metrics.loc[
        (team_metrics["TEAM_ID"] == 2) & (team_metrics["GAME_DATE"] == pd.Timestamp("2026-10-23"))
    ].iloc[0]
    assert feature_row["opp_def_rating"] == pytest.approx(expected_opponent["DEF_RATING"])
    assert feature_row["opp_pace"] == pytest.approx(expected_opponent["PACE"])


def test_next_game_features_skip_players_without_a_future_schedule_match():
    games = make_games()
    lookup = _build_latest_features(
        games,
        {1: "Alpha Player"},
        player_teams={1: {"team": "Alpha Team", "team_abbreviation": "AAA", "team_id": 1}},
        next_games={},
        opponent_team_metrics=build_team_game_metrics(games),
        team_id_by_abbreviation={"AAA": 1, "BBB": 2},
    )
    assert not any(key[0] == 1 for key in lookup)




def test_old_model_artifact_is_rejected_before_prediction():
    class LegacyBundle:
        pass

    with pytest.raises(RuntimeError, match="no saved feature schema and is outdated"):
        _validate_model_feature_context(LegacyBundle(), "PTS")


def test_phase_one_artifact_with_old_feature_schema_is_rejected():
    class PhaseOneBundle:
        feature_names = (
            "rest_days", "is_home", "opp_def_rating", "opp_pace",
            "last5_REB_mean", "last10_REB_mean", "last20_REB_mean",
            "ema_REB", "season_REB_mean",
        )
        feature_stds = {"opp_def_rating": 4.5, "opp_pace": 3.2}

    with pytest.raises(RuntimeError, match="older feature schema"):
        _validate_model_feature_context(PhaseOneBundle(), "REB")


def test_model_artifact_requires_varying_opponent_features():
    class FixedContextBundle:
        feature_names = tuple(feature_columns("REB"))
        feature_stds = {
            "opp_def_rating": 0.0,
            "opp_pace": 2.1,
        }

    with pytest.raises(RuntimeError, match="without varying opponent context"):
        _validate_model_feature_context(FixedContextBundle(), "REB")


def test_retrained_model_with_varying_opponent_features_is_accepted():
    class CurrentBundle:
        feature_names = tuple(feature_columns("AST"))
        feature_stds = {
            "opp_def_rating": 4.5,
            "opp_pace": 3.2,
        }

    _validate_model_feature_context(CurrentBundle(), "AST")


def test_empty_shortlist_preserves_schema_for_dashboard_csv():
    result, shortlist = _decorate_and_rank_candidates(
        pd.DataFrame(columns=["player_id", "player_name", "target"])
    )

    assert result.empty
    assert shortlist.empty
    assert "rank" in shortlist.columns
    assert {"player_id", "shortlist_status", "shortlist_reason"} <= set(shortlist.columns)
    assert {
        "suggested_stake_fraction",
        "suggested_stake_percent",
        "daily_exposure_after_pick_percent",
    } <= set(shortlist.columns)
    assert shortlist.to_csv(index=False).startswith("rank,")

