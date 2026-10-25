from __future__ import annotations

import json
import pandas as pd
import pytest

from src.availability import attach_availability, normalize_status
from src.predict_slate import _decorate_and_rank_candidates


def _row():
    return {
        "player_id": 11,
        "player_name": "Test Player",
        "team_abbreviation": "BOS",
        "opponent_abbreviation": "NYK",
        "opponent_team": "New York Knicks",
        "match_date": "2026-10-25",
        "target": "PTS",
        "line": 20.5,
        "last_game_date": "2026-10-24",
        "line_source": "user_supplied",
        "bookmaker": "Book A",
        "model_p_over": 0.56,
        "model_p_under": 0.44,
        "market_no_vig_p_over": 0.5,
        "market_no_vig_p_under": 0.5,
        "model_edge_over_vs_market": 0.06,
        "model_edge_under_vs_market": -0.06,
        "ev_over_per_1_unit": 0.07,
        "ev_under_per_1_unit": -0.2,
        "odds_over_american": -110,
        "odds_under_american": -110,
    }


def _write_snapshot(path, status, stamp="2026-10-25T12:00:00Z", team="BOS"):
    pd.DataFrame([{
        "player_id": 11,
        "player_name": "Test Player",
        "team_abbreviation": team,
        "match_date": "2026-10-25",
        "status": status,
        "reason": "Test report note",
        "source": "NBA official injury report (manual check)",
        "reported_at_utc": stamp,
    }]).to_csv(path, index=False)


def test_status_normalization_does_not_guess_unknown_labels():
    assert normalize_status("Active") == "AVAILABLE"
    assert normalize_status("Game Time Decision") == "QUESTIONABLE"
    assert normalize_status("Questionable") == "QUESTIONABLE"
    assert normalize_status("some unfamiliar label") == "UNKNOWN"


def test_fresh_available_report_is_matched_by_player_date_and_team(tmp_path):
    path = tmp_path / "availability.csv"
    _write_snapshot(path, "Available")
    attached = attach_availability(
        pd.DataFrame([_row()]), path,
        now_utc="2026-10-25T15:00:00Z", max_age_hours=6,
    )
    row = attached.iloc[0]
    assert row["availability_status"] == "AVAILABLE"
    assert row["availability_gate"] == "CLEAR"
    assert row["availability_age_hours"] == pytest.approx(3.0)
    assert "NBA official" in row["availability_source"]


def test_out_report_blocks_the_market_from_shortlist(tmp_path):
    path = tmp_path / "availability.csv"
    _write_snapshot(path, "Out", stamp="2026-10-25T14:30:00Z")
    attached = attach_availability(
        pd.DataFrame([_row()]), path, now_utc="2026-10-25T15:00:00Z",
    )
    ranked, shortlist = _decorate_and_rank_candidates(
        attached, min_edge=0, min_ev=0, min_probability=0,
        today=pd.Timestamp("2026-10-25"),
    )
    assert ranked.iloc[0]["availability_gate"] == "BLOCKED"
    assert ranked.iloc[0]["shortlist_status"] == "AVAILABILITY_BLOCKED"
    assert shortlist.empty


def test_questionable_report_requires_review_and_is_not_shortlisted(tmp_path):
    path = tmp_path / "availability.csv"
    _write_snapshot(path, "Questionable", stamp="2026-10-25T14:30:00Z")
    attached = attach_availability(
        pd.DataFrame([_row()]), path, now_utc="2026-10-25T15:00:00Z",
    )
    ranked, shortlist = _decorate_and_rank_candidates(
        attached, min_edge=0, min_ev=0, min_probability=0,
        today=pd.Timestamp("2026-10-25"),
    )
    assert ranked.iloc[0]["shortlist_status"] == "AVAILABILITY_REVIEW"
    assert shortlist.empty


def _write_report_coverage(path, teams_not_submitted):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "game_date": "2026-10-25",
        "reported_at_utc": "2026-10-25T14:30:00Z",
        "retrieved_at_utc": "2026-10-25T14:31:00Z",
        "report_url": "https://official.nba.com/report.pdf",
        "team_report_coverage_parsed": True,
        "team_submission_markers_resolved": True,
        "teams_not_yet_submitted_json": json.dumps(teams_not_submitted),
    }]).to_csv(path, index=False)


def test_unlisted_player_is_not_called_available_when_team_report_is_submitted(tmp_path):
    data_dir = tmp_path / "data"
    availability_path = data_dir / "player_availability.csv"
    _write_report_coverage(data_dir / "injury_report_imports.csv", ["NYK"])

    attached = attach_availability(
        pd.DataFrame([_row()]), availability_path,
        now_utc="2026-10-25T15:00:00Z", max_age_hours=2,
    )
    row = attached.iloc[0]
    assert row["availability_status"] == "NOT_LISTED"
    assert row["availability_gate"] == "CLEAR"
    assert "not an explicit Available designation" in row["availability_reason"]
    assert row["availability_source"] == "https://official.nba.com/report.pdf"


def test_unlisted_player_on_team_marked_not_yet_submitted_stays_unverified(tmp_path):
    data_dir = tmp_path / "data"
    availability_path = data_dir / "player_availability.csv"
    _write_report_coverage(data_dir / "injury_report_imports.csv", ["BOS"])

    attached = attach_availability(
        pd.DataFrame([_row()]), availability_path,
        now_utc="2026-10-25T15:00:00Z", max_age_hours=2,
    )
    row = attached.iloc[0]
    assert row["availability_status"] == "NOT_YET_SUBMITTED"
    assert row["availability_gate"] == "UNVERIFIED"
    assert "NOT YET SUBMITTED" in row["availability_reason"]


def test_shortlist_explains_unlisted_status_is_not_explicitly_available(tmp_path):
    attached = pd.DataFrame([{
        **_row(),
        "availability_status": "NOT_LISTED",
        "availability_gate": "CLEAR",
        "availability_reason": (
            "No player-specific issue is listed in the fresh official report; "
            "this is not an explicit Available designation."
        ),
        "availability_source": "https://official.nba.com/report.pdf",
        "availability_reported_at_utc": "2026-10-25T14:30:00Z",
        "availability_age_hours": 0.5,
    }])
    ranked, shortlist = _decorate_and_rank_candidates(
        attached, min_edge=0, min_ev=0, min_probability=0,
        require_verified_availability=True, today=pd.Timestamp("2026-10-25"),
    )

    assert ranked.iloc[0]["shortlist_status"] == "SHORTLIST_CANDIDATE"
    assert len(shortlist) == 1
    assert "not an explicit Available designation" in shortlist.iloc[0]["shortlist_reason"]


def test_stale_report_is_not_treated_as_clear(tmp_path):
    path = tmp_path / "availability.csv"
    _write_snapshot(path, "Available", stamp="2026-10-24T01:00:00Z")
    attached = attach_availability(
        pd.DataFrame([_row()]), path, now_utc="2026-10-25T15:00:00Z",
        max_age_hours=6,
    )
    assert attached.iloc[0]["availability_gate"] == "STALE"


def test_wrong_match_date_or_team_does_not_match_status(tmp_path):
    path = tmp_path / "availability.csv"
    _write_snapshot(path, "Out", team="NYK")
    attached = attach_availability(
        pd.DataFrame([_row()]), path, now_utc="2026-10-25T15:00:00Z",
    )
    assert attached.iloc[0]["availability_gate"] == "UNVERIFIED"
    assert attached.iloc[0]["availability_status"] == "UNKNOWN"


def test_missing_availability_file_is_explicitly_unverified(tmp_path):
    attached = attach_availability(
        pd.DataFrame([_row()]), tmp_path / "missing.csv",
        now_utc="2026-10-25T15:00:00Z",
    )
    assert attached.iloc[0]["availability_gate"] == "UNVERIFIED"
    assert "No availability record" in attached.iloc[0]["availability_reason"]


def test_strict_mode_blocks_unverified_status_but_opt_in_only():
    attached = attach_availability(
        pd.DataFrame([_row()]), "/path/that/does/not/exist.csv",
        now_utc="2026-10-25T15:00:00Z",
    )
    permissive, shortlist = _decorate_and_rank_candidates(
        attached, min_edge=0, min_ev=0, min_probability=0,
        today=pd.Timestamp("2026-10-25"),
    )
    assert permissive.iloc[0]["shortlist_status"] == "SHORTLIST_CANDIDATE"
    assert len(shortlist) == 1

    strict, strict_shortlist = _decorate_and_rank_candidates(
        attached, min_edge=0, min_ev=0, min_probability=0,
        require_verified_availability=True, today=pd.Timestamp("2026-10-25"),
    )
    assert strict.iloc[0]["shortlist_status"] == "AVAILABILITY_UNVERIFIED"
    assert strict_shortlist.empty



def test_archive_preserves_availability_audit_fields():
    from scripts.archive_predictions import prepare_archive_rows

    predictions = pd.DataFrame([{
        **_row(),
        "p_over": 0.7,
        "availability_status": "AVAILABLE",
        "availability_gate": "CLEAR",
        "availability_reason": "Latest matched report status: AVAILABLE.",
        "availability_source": "Official NBA injury report",
        "availability_reported_at_utc": "2026-10-25T12:00:00Z",
        "availability_age_hours": 3.0,
    }])
    archived = prepare_archive_rows(
        predictions, "2026-10-25", "2026-10-25T15:00:00Z"
    )
    row = archived.iloc[0]
    assert row["availability_status"] == "AVAILABLE"
    assert row["availability_gate"] == "CLEAR"
    assert row["availability_source"] == "Official NBA injury report"
    assert row["availability_age_hours"] == pytest.approx(3.0)



def test_official_report_name_format_matches_when_player_id_missing(tmp_path):
    path = tmp_path / "availability.csv"
    pd.DataFrame([{
        "player_id": "",
        "player_name": "Tatum III, Jayson",
        "team_abbreviation": "BOS",
        "match_date": "2026-10-25",
        "status": "Available",
        "reason": "Official report row",
        "source": "https://official.nba.com/report.pdf",
        "reported_at_utc": "2026-10-25T14:30:00Z",
        "retrieved_at_utc": "2026-10-25T14:35:00Z",
    }]).to_csv(path, index=False)
    rows = pd.DataFrame([{
        "player_id": 1628369,
        "player_name": "Jayson Tatum III",
        "team_abbreviation": "BOS",
        "match_date": "2026-10-25",
    }])
    result = attach_availability(rows, path, now_utc="2026-10-25T15:00:00Z")
    assert result.iloc[0]["availability_status"] == "AVAILABLE"
    assert result.iloc[0]["availability_gate"] == "CLEAR"
    assert result.iloc[0]["availability_retrieved_at_utc"] == "2026-10-25T14:35:00Z"
