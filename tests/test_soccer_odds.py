from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pytest
import requests

from src.soccer.insights import attach_market_odds
from src.soccer.odds import (
    ODDS_COLUMNS,
    OddsProviderError,
    decimal_implied_probability,
    fetch_league_odds,
    parse_odds_payload,
    upsert_odds_snapshots,
)


def live_payload():
    return [
        {
            "id": "event-1",
            "sport_key": "soccer_epl",
            "sport_title": "EPL",
            "commence_time": "2026-11-01T16:30:00Z",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [
                {
                    "key": "book-a",
                    "title": "Book A",
                    "markets": [
                        {"key": "h2h", "last_update": "2026-11-01T14:00:00Z", "outcomes": [
                            {"name": "Arsenal", "price": 2.0},
                            {"name": "Draw", "price": 3.5},
                            {"name": "Chelsea", "price": 4.0},
                        ]},
                        {"key": "totals", "last_update": "2026-11-01T14:01:00Z", "outcomes": [
                            {"name": "Over", "price": 1.9, "point": 2.5},
                            {"name": "Under", "price": 2.0, "point": 2.5},
                            {"name": "Over", "price": 1.4, "point": 1.5},
                            {"name": "Under", "price": 3.2, "point": 1.5},
                        ]},
                    ],
                }
            ],
        }
    ]


def test_decimal_probability_validates_odds():
    assert decimal_implied_probability(2.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        decimal_implied_probability(1.0)
    with pytest.raises(ValueError):
        decimal_implied_probability(float("inf"))


def test_parse_live_odds_normalizes_same_book_1x2_and_totals_lines():
    rows = parse_odds_payload(
        live_payload(), sport_key="soccer_epl", competition_name="Premier League",
        retrieved_at_utc="2026-11-01T14:02:00Z",
    )
    assert len(rows) == 3
    h2h = next(row for row in rows if row["market_key"] == "h2h")
    outcomes = {item["key"]: item for item in json.loads(h2h["outcomes_json"])}
    assert outcomes["home"]["no_vig_probability"] + outcomes["draw"]["no_vig_probability"] + outcomes["away"]["no_vig_probability"] == pytest.approx(1)
    assert float(h2h["overround"]) > 0
    totals = [row for row in rows if row["market_key"] == "totals"]
    assert {row["market_point"] for row in totals} == {"1.5", "2.5"}
    line25 = next(row for row in totals if row["market_point"] == "2.5")
    outcomes = {item["key"]: item for item in json.loads(line25["outcomes_json"])}
    assert set(outcomes) == {"over", "under"}
    assert outcomes["over"]["no_vig_probability"] + outcomes["under"]["no_vig_probability"] == pytest.approx(1)
    assert all(row["snapshot_at_utc"] == "2026-11-01T14:02:00Z" for row in rows)


def test_parse_historical_snapshot_uses_provider_snapshot_timestamp():
    payload = {"timestamp": "2026-11-01T13:55:00Z", "data": live_payload()}
    rows = parse_odds_payload(
        payload, sport_key="soccer_epl", competition_name="Premier League", snapshot_mode="historical",
        requested_snapshot_at_utc="2026-11-01T14:00:00Z", retrieved_at_utc="2026-11-01T14:02:00Z",
    )
    assert rows
    assert {row["snapshot_at_utc"] for row in rows} == {"2026-11-01T13:55:00Z"}
    assert {row["requested_snapshot_at_utc"] for row in rows} == {"2026-11-01T14:00:00Z"}
    assert {row["snapshot_mode"] for row in rows} == {"historical"}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return self.response


def test_fetch_live_uses_api_key_only_in_request_params_and_sanitizes_http_errors(monkeypatch):
    monkeypatch.setenv("THE_ODDS_API_KEY", "private-key")
    session = FakeSession(FakeResponse(live_payload()))
    rows = fetch_league_odds("epl", session=session)
    assert rows
    url, params, timeout = session.calls[0]
    assert "/v4/sports/soccer_epl/odds/" in url
    assert params["apiKey"] == "private-key"
    assert params["oddsFormat"] == "decimal"
    assert timeout > 0

    bad_session = FakeSession(FakeResponse({"message": "nope"}, status_code=401))
    with pytest.raises(OddsProviderError) as error:
        fetch_league_odds("epl", session=bad_session)
    assert "private-key" not in str(error.value)


def test_fetch_requires_key_and_historical_timestamp_must_be_timezone_aware(monkeypatch):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    with pytest.raises(OddsProviderError, match="Missing THE_ODDS_API_KEY"):
        fetch_league_odds("epl", session=FakeSession(FakeResponse([])))
    monkeypatch.setenv("THE_ODDS_API_KEY", "test-key")
    with pytest.raises(ValueError, match="timezone"):
        fetch_league_odds("epl", historical_date="2026-11-01T12:00:00", session=FakeSession(FakeResponse({})))


def test_snapshot_upsert_keeps_distinct_times_and_skips_exact_duplicates(tmp_path):
    rows = parse_odds_payload(live_payload(), sport_key="soccer_epl", competition_name="Premier League",
                              retrieved_at_utc="2026-11-01T14:02:00Z")
    path = tmp_path / "odds.csv"
    assert upsert_odds_snapshots(path, rows) == len(rows)
    assert upsert_odds_snapshots(path, rows) == len(rows)
    newer = parse_odds_payload(live_payload(), sport_key="soccer_epl", competition_name="Premier League",
                               retrieved_at_utc="2026-11-01T14:03:00Z")
    assert upsert_odds_snapshots(path, newer) == len(rows) * 2
    with path.open("r", newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == len(rows) * 2
    assert tuple(loaded[0].keys()) == tuple(ODDS_COLUMNS)


def test_market_filter_and_region_validation_happens_before_http():
    with pytest.raises(ValueError, match="regions"):
        fetch_league_odds("epl", regions="somewhere", api_key="x", session=FakeSession(FakeResponse([])))
    with pytest.raises(ValueError, match="markets"):
        fetch_league_odds("epl", markets="player_props", api_key="x", session=FakeSession(FakeResponse([])))
    with pytest.raises(ValueError, match="Unsupported"):
        fetch_league_odds("ucl", api_key="x", session=FakeSession(FakeResponse([])))



def pick(market_key="home_win", probability=0.55):
    return {
        "league": "Premier League",
        "competition_name": "Premier League",
        "competition_id": "openfootball:england:premierleague",
        "match_id": "event-1",
        "match_date": "2026-11-01",
        "kickoff_time_local": "16:30",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "market": "Home win" if market_key == "home_win" else "Over 2.5",
        "market_key": market_key,
        "market_line": 2.5 if market_key in {"over", "under"} else None,
        "model_probability": probability,
        "rank_score": 0.52,
    }


def test_attach_market_odds_requires_fresh_exact_pre_kickoff_and_calculates_experimental_ev():
    odds_rows = parse_odds_payload(
        live_payload(),
        sport_key="soccer_epl",
        competition_name="Premier League",
        retrieved_at_utc="2026-11-01T14:02:00Z",
    )
    rows = attach_market_odds(
        [pick()], odds_rows,
        now_utc=datetime(2026, 11, 1, 14, 3, tzinfo=timezone.utc),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["odds_available"] is True
    assert row["best_odds_decimal"] == pytest.approx(2.0)
    assert row["market_probability"] == pytest.approx(0.4827586, abs=1e-5)
    assert row["market_edge_probability"] == pytest.approx(0.55 - 0.4827586, abs=1e-5)
    assert row["expected_value_per_unit"] == pytest.approx(0.1)
    assert row["odds_status"] == "EXPERIMENTAL POSITIVE EV"
    assert row["best_odds_bookmaker"] == "Book A"


def test_attach_market_odds_does_not_match_wrong_totals_line_or_in_play_quotes():
    odds_rows = parse_odds_payload(
        live_payload(),
        sport_key="soccer_epl",
        competition_name="Premier League",
        retrieved_at_utc="2026-11-01T14:02:00Z",
    )
    totals_pick = pick(market_key="over")
    totals_pick["market_line"] = 3.5
    rows = attach_market_odds(
        [totals_pick], odds_rows,
        now_utc=datetime(2026, 11, 1, 14, 3, tzinfo=timezone.utc),
    )
    assert rows[0]["odds_available"] is False
    assert rows[0]["odds_status"] == "NO EXACT MARKET / LINE"

    in_play = attach_market_odds(
        [pick()], odds_rows,
        now_utc=datetime(2026, 11, 1, 17, 0, tzinfo=timezone.utc),
    )
    assert in_play[0]["odds_available"] is False
    assert in_play[0]["odds_status"] == "STALE OR NOT PRE-KICKOFF"


def test_attach_market_odds_returns_no_edge_without_odds_and_validates_age_gates():
    rows = attach_market_odds([pick()], [], now_utc=datetime(2026, 11, 1, 14, 3, tzinfo=timezone.utc))
    assert rows[0]["market_edge_probability"] is None
    assert rows[0]["expected_value_per_unit"] is None
    with pytest.raises(ValueError):
        attach_market_odds([pick()], [], max_quote_age_minutes=0)
