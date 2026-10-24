from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.soccer.market_baseline import build_market_implied_comparisons
from src.soccer.odds import parse_odds_payload


def odds_payload(kickoff="2026-11-01T16:30:00Z"):
    return [{
        "id": "event-1",
        "sport_key": "soccer_epl",
        "sport_title": "EPL",
        "commence_time": kickoff,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {"key": "book-a", "title": "Book A", "markets": [
                {"key": "h2h", "last_update": "2026-11-01T14:00:00Z", "outcomes": [
                    {"name": "Arsenal", "price": 2.0},
                    {"name": "Draw", "price": 3.5},
                    {"name": "Chelsea", "price": 4.0},
                ]},
                {"key": "totals", "last_update": "2026-11-01T14:01:00Z", "outcomes": [
                    {"name": "Over", "price": 1.9, "point": 2.5},
                    {"name": "Under", "price": 2.0, "point": 2.5},
                ]},
            ]},
        ],
    }]


def prediction(**overrides):
    row = {
        "provider": "openfootball",
        "competition_id": "openfootball:england:premierleague",
        "competition_name": "Premier League",
        "season_id": "2026-27",
        "season_name": "2026/27",
        "match_id": "match-1",
        "match_date": "2026-11-01",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "actual_home_score": 2,
        "actual_away_score": 0,
        "actual_outcome": "H",
        "expected_home_goals": 1.7,
        "expected_away_goals": 1.1,
        "total_goals_line": 2.5,
        "poisson_home_win": 0.52,
        "poisson_draw": 0.25,
        "poisson_away_win": 0.23,
        "poisson_over": 0.53,
        "poisson_under": 0.47,
        "dixon_coles_home_win": 0.51,
        "dixon_coles_draw": 0.26,
        "dixon_coles_away_win": 0.23,
    }
    row.update(overrides)
    return row


def test_market_implied_baseline_uses_same_book_no_vig_snapshots_before_kickoff():
    odds = parse_odds_payload(
        {"timestamp": "2026-11-01T14:02:00Z", "data": odds_payload()},
        sport_key="soccer_epl", competition_name="Premier League", snapshot_mode="historical",
        requested_snapshot_at_utc="2026-11-01T14:02:00Z",
        retrieved_at_utc="2026-11-01T14:03:00Z",
    )
    comparisons, summary = build_market_implied_comparisons([prediction()], odds)
    assert summary["available"] is True
    assert summary["prediction_rows"] == 1
    assert summary["matched_1x2_predictions"] == 1
    assert summary["matched_totals_predictions"] == 1
    assert len(comparisons) == 1
    row = comparisons[0]
    assert row["market_match_status"] == "MATCHED_PRE_KICKOFF"
    assert row["market_implied_home_win"] + row["market_implied_draw"] + row["market_implied_away_win"] == pytest.approx(1)
    assert row["market_implied_over"] + row["market_implied_under"] == pytest.approx(1)


def test_market_implied_baseline_rejects_snapshots_after_kickoff_and_fuzzy_teams():
    odds_after = parse_odds_payload(
        {"timestamp": "2026-11-01T14:02:00Z", "data": odds_payload(kickoff="2026-11-01T13:00:00Z")},
        sport_key="soccer_epl", competition_name="Premier League", snapshot_mode="historical",
        requested_snapshot_at_utc="2026-11-01T14:02:00Z",
        retrieved_at_utc="2026-11-01T14:03:00Z",
    )
    comparisons, summary = build_market_implied_comparisons([prediction()], odds_after)
    assert comparisons == []
    assert summary["matched_1x2_predictions"] == 0

    odds = parse_odds_payload(
        {"timestamp": "2026-11-01T14:02:00Z", "data": odds_payload()},
        sport_key="soccer_epl", competition_name="Premier League", snapshot_mode="historical",
        requested_snapshot_at_utc="2026-11-01T14:02:00Z",
        retrieved_at_utc="2026-11-01T14:03:00Z",
    )
    mismatched = prediction(home_team="Arsenal FC Reserves")
    comparisons, summary = build_market_implied_comparisons([mismatched], odds)
    assert comparisons == []
    assert summary["matched_1x2_predictions"] == 0



def test_market_baseline_rejects_quote_updates_newer_than_the_snapshot():
    odds = parse_odds_payload(
        {"timestamp": "2026-11-01T14:02:00Z", "data": odds_payload()},
        sport_key="soccer_epl", competition_name="Premier League", snapshot_mode="historical",
        requested_snapshot_at_utc="2026-11-01T14:02:00Z",
        retrieved_at_utc="2026-11-01T14:03:00Z",
    )
    for quote in odds:
        quote["market_last_update_utc"] = "2026-11-01T14:03:00Z"
    comparisons, summary = build_market_implied_comparisons([prediction()], odds)
    assert comparisons == []
    assert summary["matched_1x2_predictions"] == 0
