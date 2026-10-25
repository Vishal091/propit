from __future__ import annotations

import pandas as pd
import pytest

from scripts.capture_lineup_snapshot import build_snapshot_rows, write_snapshot


def _input():
    return pd.DataFrame([
        {
            "player_id": 1628369, "player_name": "Jayson Tatum",
            "team_abbreviation": "BOS", "starting_status": "STARTER",
            "reason": "Official starting lineup", "minutes_limit": "",
        },
        {
            "player_id": "", "player_name": "Derrick White",
            "team_abbreviation": "BOS", "starting_status": "BENCH",
            "reason": "Listed as reserve", "minutes_limit": 20,
        },
    ])


def test_build_snapshot_rows_stamps_source_and_limits():
    rows = build_snapshot_rows(
        _input(), game_date="2026-10-25", source="https://www.nba.com/starting-lineups",
        reported_at_utc="2026-10-25T14:30:00Z",
    )
    assert len(rows) == 2
    assert set(rows["starting_status"]) == {"STARTER", "BENCH"}
    assert rows["reported_at_utc"].nunique() == 1
    assert rows["reported_at_utc"].iloc[0] == "2026-10-25T14:30:00Z"
    assert pd.isna(rows.loc[rows["player_name"] == "Jayson Tatum", "minutes_limit"].iloc[0])
    assert rows.loc[rows["player_name"] == "Derrick White", "minutes_limit"].iloc[0] == 20


def test_unknown_or_projected_starter_status_is_rejected():
    bad = _input()
    bad.loc[0, "starting_status"] = "Projected starter"
    with pytest.raises(ValueError, match="explicit STARTER or BENCH"):
        build_snapshot_rows(bad, game_date="2026-10-25", source="trusted source")


def test_duplicate_player_team_rows_are_rejected():
    bad = pd.concat([_input(), _input().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        build_snapshot_rows(bad, game_date="2026-10-25", source="trusted source")


def test_snapshot_upserts_current_file_but_appends_history(tmp_path):
    input_path = tmp_path / "lineups_input.csv"
    _input().to_csv(input_path, index=False)
    current_path = tmp_path / "player_lineups.csv"
    history_path = tmp_path / "player_lineups_history.csv"
    first = write_snapshot(
        input_path, game_date="2026-10-25", source="trusted source",
        output_path=current_path, history_path=history_path,
        reported_at_utc="2026-10-25T14:30:00Z",
    )
    second = write_snapshot(
        input_path, game_date="2026-10-25", source="trusted source",
        output_path=current_path, history_path=history_path,
        reported_at_utc="2026-10-25T15:00:00Z",
    )
    assert first["rows"] == second["rows"] == 2
    assert len(pd.read_csv(current_path)) == 2
    assert len(pd.read_csv(history_path)) == 4
    assert set(pd.read_csv(current_path)["reported_at_utc"]) == {"2026-10-25T15:00:00Z"}
