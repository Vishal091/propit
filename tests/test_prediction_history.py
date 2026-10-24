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
        "line_source": ["user_supplied"],
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

