from __future__ import annotations

import pandas as pd
import pytest

from src.lineups import attach_lineups, normalize_lineup_status


def test_normalizes_lineup_status_without_guessing():
    assert normalize_lineup_status("Confirmed Starter") == "STARTER"
    assert normalize_lineup_status("Bench") == "BENCH"
    assert normalize_lineup_status("Likely starter") == "UNKNOWN"


def test_exact_fresh_lineup_matches_by_name_team_date(tmp_path):
    path = tmp_path / "lineups.csv"
    pd.DataFrame([{
        "player_id": "",
        "player_name": "Tatum, Jayson",
        "team_abbreviation": "BOS",
        "match_date": "2026-10-25",
        "starting_status": "Starter",
        "reported_at_utc": "2026-10-25T14:30:00Z",
        "source": "https://www.nba.com/starting-lineups",
        "minutes_limit": "",
    }]).to_csv(path, index=False)
    rows = pd.DataFrame([{
        "player_id": 1628369,
        "player_name": "Jayson Tatum",
        "team_abbreviation": "BOS",
        "match_date": "2026-10-25",
    }])
    result = attach_lineups(rows, path, now_utc="2026-10-25T15:00:00Z")
    assert result.iloc[0]["lineup_status"] == "STARTER"
    assert result.iloc[0]["lineup_gate"] == "CLEAR"
    assert result.iloc[0]["lineup_age_hours"] == pytest.approx(0.5)


def test_stale_or_conflicting_lineup_is_not_confirmed(tmp_path):
    path = tmp_path / "lineups.csv"
    pd.DataFrame([{
        "player_id": 11, "player_name": "Test Player", "team_abbreviation": "NYK",
        "match_date": "2026-10-25", "starting_status": "STARTER",
        "reported_at_utc": "2026-10-25T10:00:00Z", "source": "manual",
    }]).to_csv(path, index=False)
    rows = pd.DataFrame([{
        "player_id": 11, "player_name": "Test Player", "team_abbreviation": "BOS",
        "match_date": "2026-10-25",
    }])
    result = attach_lineups(rows, path, now_utc="2026-10-25T15:00:00Z")
    assert result.iloc[0]["lineup_gate"] == "UNVERIFIED"
    assert result.iloc[0]["lineup_status"] == "UNKNOWN"
