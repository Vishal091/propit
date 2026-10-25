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


def test_projected_auto_lineup_is_visible_but_does_not_clear_strict_gate(tmp_path):
    auto_path = tmp_path / "player_lineups_auto.csv"
    pd.DataFrame([{
        "player_id": 11,
        "player_name": "Test Player",
        "team_abbreviation": "BOS",
        "match_date": "2026-10-25",
        "starting_status": "STARTER",
        "confirmation_status": "PROJECTED",
        "confidence_label": "PROJECTED",
        "source_status": "Expected Lineup",
        "reported_at_utc": "2026-10-25T14:30:00Z",
        "source": "https://www.rotowire.com/basketball/nba-lineups.php",
        "reason": "Expected lineup only",
    }]).to_csv(auto_path, index=False)
    rows = pd.DataFrame([{
        "player_id": 11, "player_name": "Test Player", "team_abbreviation": "BOS",
        "match_date": "2026-10-25",
    }])

    result = attach_lineups(
        rows, tmp_path / "player_lineups.csv", auto_path=auto_path,
        now_utc="2026-10-25T15:00:00Z",
    )
    row = result.iloc[0]
    assert row["lineup_status"] == "STARTER"
    assert row["lineup_confirmation_status"] == "PROJECTED"
    assert row["lineup_confidence_label"] == "PROJECTED"
    assert row["lineup_gate"] == "UNVERIFIED"
    assert "projected" in row["lineup_reason"].lower()


def test_fresh_confirmed_manual_row_outranks_newer_projection(tmp_path):
    confirmed_path = tmp_path / "player_lineups.csv"
    auto_path = tmp_path / "player_lineups_auto.csv"
    pd.DataFrame([{
        "player_id": 11, "player_name": "Test Player", "team_abbreviation": "BOS",
        "match_date": "2026-10-25", "starting_status": "BENCH",
        "reported_at_utc": "2026-10-25T14:20:00Z", "source": "Official team announcement",
    }]).to_csv(confirmed_path, index=False)
    pd.DataFrame([{
        "player_id": 11, "player_name": "Test Player", "team_abbreviation": "BOS",
        "match_date": "2026-10-25", "starting_status": "STARTER",
        "confirmation_status": "PROJECTED", "confidence_label": "PROJECTED",
        "source_status": "Expected Lineup",
        "reported_at_utc": "2026-10-25T14:55:00Z", "source": "RotoWire",
    }]).to_csv(auto_path, index=False)
    rows = pd.DataFrame([{
        "player_id": 11, "player_name": "Test Player", "team_abbreviation": "BOS",
        "match_date": "2026-10-25",
    }])

    result = attach_lineups(
        rows, confirmed_path, auto_path=auto_path,
        now_utc="2026-10-25T15:00:00Z",
    )
    row = result.iloc[0]
    assert row["lineup_status"] == "BENCH"
    assert row["lineup_confirmation_status"] == "CONFIRMED"
    assert row["lineup_gate"] == "CLEAR"


def test_fresh_projection_is_displayed_when_only_confirmed_row_is_stale(tmp_path):
    confirmed_path = tmp_path / "player_lineups.csv"
    auto_path = tmp_path / "player_lineups_auto.csv"
    pd.DataFrame([{
        "player_id": 11, "player_name": "Test Player", "team_abbreviation": "BOS",
        "match_date": "2026-10-25", "starting_status": "BENCH",
        "reported_at_utc": "2026-10-25T10:00:00Z", "source": "Official team announcement",
    }]).to_csv(confirmed_path, index=False)
    pd.DataFrame([{
        "player_id": 11, "player_name": "Test Player", "team_abbreviation": "BOS",
        "match_date": "2026-10-25", "starting_status": "STARTER",
        "confirmation_status": "PROJECTED", "confidence_label": "PROJECTED",
        "source_status": "Expected Lineup",
        "reported_at_utc": "2026-10-25T14:55:00Z", "source": "RotoWire",
    }]).to_csv(auto_path, index=False)
    rows = pd.DataFrame([{
        "player_id": 11, "player_name": "Test Player", "team_abbreviation": "BOS",
        "match_date": "2026-10-25",
    }])

    result = attach_lineups(
        rows, confirmed_path, auto_path=auto_path,
        now_utc="2026-10-25T15:00:00Z", max_age_hours=2,
    )
    row = result.iloc[0]
    assert row["lineup_status"] == "STARTER"
    assert row["lineup_confirmation_status"] == "PROJECTED"
    assert row["lineup_gate"] == "UNVERIFIED"


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
