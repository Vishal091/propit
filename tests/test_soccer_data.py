from __future__ import annotations

import csv
from pathlib import Path
import pytest

from scripts.import_soccer_data import upsert_matches
from src.soccer.providers.openfootball import parse_openfootball_fixtures
from src.soccer.providers.statsbomb import summarize_shot_events


def test_statsbomb_aggregates_xg_and_shot_counts_by_team():
    events = [
        {"type": {"name": "Shot"}, "team": {"id": 10}, "shot": {"statsbomb_xg": 0.25, "outcome": {"name": "Goal"}}},
        {"type": {"name": "Shot"}, "team": {"id": 10}, "shot": {"statsbomb_xg": 0.10, "outcome": {"name": "Saved"}}},
        {"type": {"name": "Shot"}, "team": {"id": 20}, "shot": {"statsbomb_xg": 0.05, "outcome": {"name": "Blocked"}}},
        {"type": {"name": "Pass"}, "team": {"id": 10}},
        {"type": {"name": "Shot"}, "team": {"id": 99}, "shot": {"statsbomb_xg": 0.7, "outcome": {"name": "Goal"}}},
    ]
    result = summarize_shot_events(events, home_team_id=10, away_team_id=20)
    assert result["home_xg"] == pytest.approx(0.35)
    assert result["away_xg"] == pytest.approx(0.05)
    assert result["home_shots"] == 2
    assert result["away_shots"] == 1
    assert result["home_shots_on_target"] == 2
    assert result["away_shots_on_target"] == 0
    assert result["xg_status"] == "complete"


def test_statsbomb_withholds_partial_xg_but_preserves_shot_counts():
    events = [
        {"type": {"name": "Shot"}, "team": {"id": 10}, "shot": {"statsbomb_xg": 0.25, "outcome": {"name": "Goal"}}},
        {"type": {"name": "Shot"}, "team": {"id": 20}, "shot": {"outcome": {"name": "Saved"}}},
    ]
    result = summarize_shot_events(events, home_team_id=10, away_team_id=20)
    assert result["home_shots"] == 1 and result["away_shots"] == 1
    assert result["home_xg"] is None and result["away_xg"] is None
    assert result["xg_status"] == "partial"


def test_openfootball_parses_finished_and_future_rows_and_year_rollover():
    text = """= England | Premier League 2025/26
▪ Regular Season - 1
Fri Dec 19 2025
  20:00   Liverpool  2-1 (1-0)  Bournemouth
Sat Jan 3
  15:00   Arsenal v Chelsea
"""
    rows = parse_openfootball_fixtures(text, league="england", season="2025-26", source_url="https://example.invalid/fixtures.txt", source_retrieved_at_utc="2026-01-01T00:00:00Z")
    assert len(rows) == 2
    finished, upcoming = rows
    assert finished.match_date == "2025-12-19"
    assert finished.home_team == "Liverpool" and finished.away_team == "Bournemouth"
    assert (finished.home_score, finished.away_score) == (2, 1)
    assert finished.match_state == "finished" and finished.match_week == 1
    assert upcoming.match_date == "2026-01-03"
    assert upcoming.home_team == "Arsenal" and upcoming.away_team == "Chelsea"
    assert upcoming.home_score is None and upcoming.match_state == "scheduled"
    assert upcoming.kickoff_time_local == "15:00" and upcoming.kickoff_utc is None
    assert upcoming.home_xg is None and upcoming.xg_status == "not_provided"


def test_openfootball_supports_result_rows_without_time():
    text = """= English Premier League 2025/26
▪ Matchday 1
Sun Aug 17 2025
    Arsenal 1-0 (0-0) Manchester United
"""
    rows = parse_openfootball_fixtures(text, "england", "2025-26", "https://example.invalid/fixtures.txt")
    assert len(rows) == 1
    assert rows[0].kickoff_time_local is None
    assert (rows[0].home_score, rows[0].away_score) == (1, 0)


def test_openfootball_rejects_unknown_league_and_bad_season():
    with pytest.raises(ValueError):
        parse_openfootball_fixtures("", "unknown", "2025-26", "https://example.invalid/file.txt")
    with pytest.raises(ValueError):
        parse_openfootball_fixtures("", "england", "last-season", "https://example.invalid/file.txt")


def test_upsert_matches_merges_feeds_and_updates_same_provider_record(tmp_path: Path):
    from src.soccer.schemas import SoccerMatch
    base = dict(provider="openfootball", match_id="openfootball:abc", provider_match_id="abc", competition_id="openfootball:england:premierleague", competition_name="English Premier League", season_id="2026-27", season_name="2026/27", match_date="2026-08-21", kickoff_time_local="20:00", kickoff_timezone=None, kickoff_utc=None, match_week=1, home_team_id=None, home_team="Arsenal", away_team_id=None, away_team="Chelsea", home_score=None, away_score=None, match_state="scheduled", home_xg=None, away_xg=None, home_shots=None, away_shots=None, home_shots_on_target=None, away_shots_on_target=None, xg_status="not_provided", source_url="https://example.invalid/a.txt", source_updated_at=None, source_retrieved_at_utc="2026-08-01T00:00:00Z", source_cache_status="fresh", observed_at_utc="2026-08-01T00:00:00Z")
    path = tmp_path / "matches.csv"
    assert upsert_matches(path, [SoccerMatch(**base)]) == 1
    updated = SoccerMatch(**{**base, "home_score": 2, "away_score": 0, "match_state": "finished"})
    second = SoccerMatch(**{**base, "provider": "statsbomb", "match_id": "statsbomb:123", "provider_match_id": "123"})
    assert upsert_matches(path, [updated, second]) == 2
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    row = next(item for item in rows if item["provider"] == "openfootball")
    assert row["home_score"] == "2" and row["match_state"] == "finished"
