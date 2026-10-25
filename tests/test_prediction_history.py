from __future__ import annotations

import pandas as pd
import pytest

from scripts.archive_predictions import prepare_archive_rows
from scripts.settle_predictions import settle_history_rows


def test_prepare_archive_rows_captures_pregame_prices_and_probabilities():
    source = pd.DataFrame({
        "player_id": [11],
        "player_name": ["Demo Player"],
        "target": ["reb"],
        "line": [7.5],
        "team": ["Toronto Raptors"],
        "team_abbreviation": ["TOR"],
        "match_date": ["2026-10-25"],
        "opponent_team": ["New York Knicks"],
        "opponent_abbreviation": ["NYK"],
        "matchup": ["TOR @ NYK"],
        "p_over": [0.42],
        "odds_over_american": [-110],
        "odds_under_american": [100],
        "bookmaker": ["Example book"],
        "bookmaker_over": ["DraftKings"],
        "bookmaker_under": ["DraftKings"],
        "line_source": ["user_supplied"],
        "market_source": ["manual snapshot"],
        "market_snapshot_at_utc": ["2026-10-24T00:00:00Z"],
        "opening_line": [6.5],
        "source_line_move": [1.0],
        "previous_line": [7.0],
        "line_move_from_previous": [0.5],
    })
    rows = prepare_archive_rows(source, "2026-10-24", "2026-10-24T00:00:00+00:00")
    assert rows.iloc[0]["snapshot_date"] == "2026-10-24"
    assert rows.iloc[0]["game_date"] == "2026-10-25"
    assert rows.iloc[0]["match_date"] == "2026-10-25"
    assert rows.iloc[0]["team"] == "Toronto Raptors"
    assert rows.iloc[0]["opponent_team"] == "New York Knicks"
    assert rows.iloc[0]["target"] == "REB"
    assert rows.iloc[0]["p_over"] == pytest.approx(0.42)
    assert rows.iloc[0]["odds_under_american"] == 100
    assert rows.iloc[0]["market_source"] == "manual snapshot"
    assert rows.iloc[0]["market_snapshot_at_utc"] == "2026-10-24T00:00:00Z"
    assert rows.iloc[0]["opening_line"] == pytest.approx(6.5)
    assert rows.iloc[0]["line_move_from_previous"] == pytest.approx(0.5)
    assert rows.iloc[0]["bookmaker_over"] == "DraftKings"
    assert rows.iloc[0]["settlement_status"] == "PENDING"


def test_settlement_marks_over_under_and_pushes():
    history = pd.DataFrame({
        "game_date": ["2026-10-24"] * 3,
        "player_id": [11, 22, 33],
        "target": ["REB", "AST", "PTS"],
        "line": [7.5, 5.5, 20.0],
        "actual_stat": [None, None, None],
        "actual_over": [None, None, None],
        "settlement_status": ["PENDING"] * 3,
    })
    games = pd.DataFrame({
        "GAME_DATE": ["2026-10-24"] * 3,
        "PLAYER_ID": [11, 22, 33],
        "PTS": [18, 12, 20],
        "REB": [8, 4, 6],
        "AST": [2, 5, 7],
    })
    updated, settled, pushes = settle_history_rows(history, games, "2026-10-24")
    assert settled == 2
    assert pushes == 1
    assert updated.loc[0, "actual_over"] == 1
    assert updated.loc[1, "actual_over"] == 0
    assert updated.loc[2, "settlement_status"] == "PUSH"
    assert pd.isna(updated.loc[2, "actual_over"])


def test_settlement_uses_match_date_not_snapshot_date():
    history = pd.DataFrame({
        "snapshot_date": ["2026-10-24"],
        "game_date": ["2026-10-24"],
        "match_date": ["2026-10-25"],
        "player_id": [11],
        "target": ["REB"],
        "line": [7.5],
        "actual_stat": [None],
        "actual_over": [None],
        "settlement_status": ["PENDING"],
    })
    games = pd.DataFrame({
        "GAME_DATE": ["2026-10-25"],
        "PLAYER_ID": [11],
        "PTS": [18],
        "REB": [8],
        "AST": [2],
    })
    updated, settled, pushes = settle_history_rows(history, games, "2026-10-25")
    assert settled == 1
    assert pushes == 0
    assert updated.loc[0, "actual_stat"] == 8
    assert updated.loc[0, "actual_over"] == 1


def test_archive_does_not_fake_match_date_when_schedule_lookup_is_blank():
    source = pd.DataFrame({
        "player_id": [11],
        "player_name": ["Demo Player"],
        "team": ["Toronto Raptors"],
        "team_abbreviation": ["TOR"],
        "target": ["PTS"],
        "line": [18.5],
        "match_date": [""],
        "p_over": [0.55],
    })
    rows = prepare_archive_rows(source, "2026-10-24", "2026-10-24T00:00:00+00:00")
    assert rows.iloc[0]["snapshot_date"] == "2026-10-24"
    assert rows.iloc[0]["match_date"] == ""
    assert rows.iloc[0]["game_date"] == ""



def test_archive_separates_raw_and_market_shrunk_probabilities_and_minutes_context():
    source = pd.DataFrame({
        "player_id": [11],
        "player_name": ["Demo Player"],
        "target": ["PTS"],
        "line": [20.5],
        "match_date": ["2026-10-25"],
        "model_p_over_raw": [0.61],
        "model_p_under_raw": [0.39],
        "p_over": [0.54],
        "calibrated_p_over": [0.54],
        "calibrated_p_under": [0.46],
        "selected_model_probability": [0.54],
        "selected_market_no_vig_probability": [0.50],
        "selected_model_edge": [0.04],
        "selected_ev_per_1_unit": [0.03],
        "recommended_side": ["OVER"],
        "shortlist_status": ["SHORTLIST_CANDIDATE"],
        "probability_gate_status": ["CALIBRATED"],
        "p_push": [0.0],
        "minutes_projection_mean": [32.5],
        "minutes_projection_std": [3.2],
        "minutes_projection_p10": [28.4],
        "minutes_projection_p90": [36.6],
        "minutes_sample_size": [18],
        "minutes_role_change_detected": [False],
        "minutes_stable": [True],
        "minutes_projection_status": ["HISTORY_SUPPORTED"],
        "minutes_uncertainty_reason": ["Historical minutes are stable."],
        "lineup_status": ["STARTER"],
        "lineup_gate": ["CLEAR"],
        "lineup_reported_at_utc": ["2026-10-25T14:00:00Z"],
        "availability_status": ["AVAILABLE"],
        "availability_gate": ["CLEAR"],
        "availability_reported_at_utc": ["2026-10-25T14:00:00Z"],
        "availability_retrieved_at_utc": ["2026-10-25T14:05:00Z"],
        "odds_over_american": [-110],
        "odds_under_american": [-110],
        "bookmaker": ["Book A"],
    })
    row = prepare_archive_rows(source, "2026-10-25", "2026-10-25T14:10:00Z").iloc[0]
    assert row["p_over"] == pytest.approx(0.61)  # raw input reserved for chronological calibration
    assert row["model_p_over_raw"] == pytest.approx(0.61)
    assert row["market_shrunk_p_over"] == pytest.approx(0.54)
    assert row["selected_model_probability"] == pytest.approx(0.54)
    assert row["minutes_projection_mean"] == pytest.approx(32.5)
    assert row["minutes_sample_size"] == 18
    assert row["lineup_status"] == "STARTER"
    assert row["availability_retrieved_at_utc"] == "2026-10-25T14:05:00Z"
    assert row["recommended_side"] == "OVER"


def test_archive_keeps_insufficient_data_rows_without_probability():
    source = pd.DataFrame({
        "player_id": [11],
        "player_name": ["Demo Player"],
        "target": ["REB"],
        "line": [7.5],
        "match_date": ["2026-10-25"],
        "p_over": [float("nan")],
        "probability_gate_status": ["INSUFFICIENT_LINEUP_DATA"],
        "shortlist_status": ["INSUFFICIENT_LINEUP_DATA"],
        "shortlist_reason": ["No fresh confirmed starter/bench snapshot."],
    })
    rows = prepare_archive_rows(source, "2026-10-25", "2026-10-25T14:10:00Z")
    assert len(rows) == 1
    assert pd.isna(rows.iloc[0]["p_over"])
    assert rows.iloc[0]["shortlist_status"] == "INSUFFICIENT_LINEUP_DATA"
