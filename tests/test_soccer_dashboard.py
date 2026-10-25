from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi import HTTPException

from src import dashboard


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def soccer_row(provider, competition_id, competition_name, match_id, day, home, away, hg, ag, state):
    return {
        "provider": provider,
        "competition_id": competition_id,
        "competition_name": competition_name,
        "season_id": "2026-27",
        "season_name": "2026/27",
        "match_id": match_id,
        "match_date": day,
        "kickoff_time_local": "15:00",
        "kickoff_timezone": None,
        "kickoff_utc": None,
        "home_team": home,
        "away_team": away,
        "home_score": hg,
        "away_score": ag,
        "match_state": state,
        "home_xg": None,
        "away_xg": None,
        "xg_status": "not_provided",
        "source_cache_status": "fresh",
    }


def patch_soccer_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "SOCCER_DATA", tmp_path / "data" / "soccer")
    monkeypatch.setattr(dashboard, "SOCCER_MATCHES", tmp_path / "data" / "soccer" / "matches.csv")
    monkeypatch.setattr(dashboard, "SOCCER_UPCOMING", tmp_path / "predictions" / "soccer" / "upcoming.csv")
    monkeypatch.setattr(dashboard, "SOCCER_BACKTEST", tmp_path / "predictions" / "soccer" / "backtest.csv")
    monkeypatch.setattr(dashboard, "SOCCER_METRICS", tmp_path / "predictions" / "soccer" / "backtest_metrics.json")


def test_soccer_state_reports_empty_state_without_creating_files(tmp_path, monkeypatch):
    patch_soccer_paths(monkeypatch, tmp_path)
    result = dashboard.api_soccer_state()
    assert result["available"] is False
    assert result["match_rows"] == 0
    assert result["scored_matches"] == 0
    assert result["forecast_rows"] == 0
    assert result["backtest_rows"] == 0
    assert result["competitions"] == []


def test_soccer_state_counts_matches_competitions_and_files(tmp_path, monkeypatch):
    patch_soccer_paths(monkeypatch, tmp_path)
    write_csv(dashboard.SOCCER_MATCHES, [
        soccer_row("openfootball", "openfootball:england:premierleague", "Premier League", "a", "2026-10-20", "Arsenal", "Chelsea", 2, 0, "finished"),
        soccer_row("openfootball", "openfootball:england:premierleague", "Premier League", "b", "2026-10-25", "Arsenal", "Liverpool", None, None, "scheduled"),
        soccer_row("statsbomb", "2", "Premier League", "c", "2026-10-21", "Liverpool", "Chelsea", 1, 1, "finished"),
    ])
    write_csv(dashboard.SOCCER_UPCOMING, [{"match_date": "2026-10-25", "competition_id": "openfootball:england:premierleague"}])
    write_csv(dashboard.SOCCER_BACKTEST, [{"match_date": "2026-01-01"}])
    dashboard.SOCCER_METRICS.parent.mkdir(parents=True, exist_ok=True)
    dashboard.SOCCER_METRICS.write_text(json.dumps({"backtested_matches": 12, "models": []}), encoding="utf-8")

    result = dashboard.api_soccer_state()
    assert result["available"] is True
    assert result["match_rows"] == 3
    assert result["scored_matches"] == 2
    assert result["forecast_rows"] == 1
    assert result["backtest_rows"] == 1
    assert result["metrics_available"] is True
    assert len(result["competitions"]) == 2
    assert result["matches_updated"] is not None


def test_soccer_match_endpoint_filters_upcoming_finished_dates_and_competitions(tmp_path, monkeypatch):
    patch_soccer_paths(monkeypatch, tmp_path)
    write_csv(dashboard.SOCCER_MATCHES, [
        soccer_row("openfootball", "epl", "Premier League", "past", "2026-10-20", "Arsenal", "Chelsea", 2, 0, "finished"),
        soccer_row("openfootball", "epl", "Premier League", "next", "2026-10-25", "Arsenal", "Liverpool", None, None, "scheduled"),
        soccer_row("openfootball", "epl", "Premier League", "later", "2026-11-05", "Spurs", "Newcastle", None, None, "scheduled"),
        soccer_row("statsbomb", "ucl", "Champions League", "other", "2026-10-26", "Barcelona", "Milan", 1, 1, "finished"),
    ])
    upcoming = dashboard.api_soccer_matches(view="upcoming", date_from="2026-10-24", date_to="2026-10-31", competition_id="epl", limit=100)
    assert upcoming["count"] == 1
    assert upcoming["rows"][0]["match_id"] == "next"
    finished = dashboard.api_soccer_matches(view="finished", date_from=None, date_to=None, competition_id="epl", limit=100)
    assert finished["count"] == 1
    assert finished["rows"][0]["match_id"] == "past"
    all_rows = dashboard.api_soccer_matches(view="all", date_from=None, date_to=None, competition_id=None, limit=100)
    assert all_rows["count"] == 4


@pytest.mark.parametrize(
    "start,end",
    [("not-a-date", None), ("2026-11-01", "2026-10-01")],
)
def test_soccer_match_endpoint_returns_http_error_for_invalid_date_filter(tmp_path, monkeypatch, start, end):
    patch_soccer_paths(monkeypatch, tmp_path)
    write_csv(dashboard.SOCCER_MATCHES, [
        soccer_row("openfootball", "epl", "Premier League", "a", "2026-10-20", "Arsenal", "Chelsea", 2, 0, "finished")
    ])
    with pytest.raises(HTTPException) as error:
        dashboard.api_soccer_matches(view="all", date_from=start, date_to=end, competition_id=None, limit=100)
    assert error.value.status_code == 400


def test_soccer_forecast_endpoint_filters_saved_predictions(tmp_path, monkeypatch):
    patch_soccer_paths(monkeypatch, tmp_path)
    write_csv(dashboard.SOCCER_UPCOMING, [
        {"match_date": "2026-10-25", "competition_id": "epl", "home_team": "Arsenal", "away_team": "Liverpool", "dixon_coles_home_win": 0.5},
        {"match_date": "2026-11-02", "competition_id": "ucl", "home_team": "Barcelona", "away_team": "Milan", "dixon_coles_home_win": 0.4},
    ])
    result = dashboard.api_soccer_forecasts(date_from="2026-10-24", date_to="2026-10-31", competition_id="epl", limit=50)
    assert result["available"] is True
    assert result["count"] == 1
    assert result["rows"][0]["home_team"] == "Arsenal"


def test_soccer_metrics_endpoint_handles_available_missing_and_bad_json(tmp_path, monkeypatch):
    patch_soccer_paths(monkeypatch, tmp_path)
    assert dashboard.api_soccer_metrics()["available"] is False
    dashboard.SOCCER_METRICS.parent.mkdir(parents=True, exist_ok=True)
    dashboard.SOCCER_METRICS.write_text(json.dumps({"models": [{"model": "poisson", "matches": 10}]}), encoding="utf-8")
    assert dashboard.api_soccer_metrics()["report"]["models"][0]["model"] == "poisson"
    dashboard.SOCCER_METRICS.write_text("{", encoding="utf-8")
    assert dashboard.api_soccer_metrics()["available"] is False


def test_soccer_catalog_parses_statsbomb_ids_from_fixed_cli(monkeypatch):
    captured = {}
    def fake_run_step(module, args, timeout_seconds=600):
        captured["module"] = module
        captured["args"] = args
        captured["timeout"] = timeout_seconds
        return {"ok": True, "output": "competition_id\tseason_id\tcompetition\tseason\tcountry\tcache\n2\t27\tPremier League\t2015/16\tEngland\tfresh\n1 entries"}
    monkeypatch.setattr(dashboard, "_run_step", fake_run_step)
    result = dashboard.api_soccer_catalog(search="Premier League")
    assert captured["module"] == "scripts.import_soccer_data"
    assert captured["args"] == ["--provider", "statsbomb", "--list-competitions", "--search=Premier League"]
    assert captured["timeout"] == 120
    assert result["count"] == 1
    assert result["rows"][0]["competition_id"] == "2"
    assert result["rows"][0]["season_id"] == "27"


def test_soccer_import_uses_allowlisted_cli_arguments(monkeypatch):
    captured = {}
    monkeypatch.setattr(dashboard, "_soccer_state", lambda: {"available": False, "match_rows": 0})
    def fake_run_steps(plan):
        captured["plan"] = plan
        return {"ok": True, "steps": [], "message": "imported"}
    monkeypatch.setattr(dashboard, "_run_steps", fake_run_steps)
    request = dashboard.SoccerImportRequest(provider="openfootball", league="england", season="2026-27", limit=3)
    result = dashboard.api_soccer_import(request)
    assert result["ok"] is True
    assert captured["plan"] == [("scripts.import_soccer_data", ["--provider", "openfootball", "--league", "england", "--season", "2026-27", "--limit", "3"])]
    assert result["soccer_state"]["match_rows"] == 0


def test_soccer_dashboard_contains_navigation_and_data_controls():
    html = dashboard.HTML_FILE.read_text(encoding="utf-8")
    for text in (
        'data-tab="soccer"',
        'id="section-soccer"',
        'id="soccer-import-btn"',
        'id="soccer-forecast-btn"',
        'id="soccer-backtest-btn"',
        'id="soccer-metrics-table"',
        'data-soccer-view="forecasts"',
        'id="soccer-picks-table"',
        'id="soccer-calibration-table"',
        'id="soccer-reliability-table"',
        'Best picks · Top 5 European leagues',
        'async function loadSoccerPicks',
    ):
        assert text in html

@pytest.mark.parametrize(
    "start,end",
    [("bad-date", None), ("2026-11-01", "2026-10-01")],
)
def test_soccer_forecast_endpoint_returns_http_error_for_invalid_date_filter(tmp_path, monkeypatch, start, end):
    patch_soccer_paths(monkeypatch, tmp_path)
    write_csv(dashboard.SOCCER_UPCOMING, [
        {"match_date": "2026-10-25", "competition_id": "epl", "home_team": "Arsenal", "away_team": "Liverpool"}
    ])
    with pytest.raises(HTTPException) as error:
        dashboard.api_soccer_forecasts(date_from=start, date_to=end, competition_id=None, limit=50)
    assert error.value.status_code == 400



def test_soccer_picks_endpoint_exposes_rejection_diagnostics_for_empty_results(tmp_path, monkeypatch):
    patch_soccer_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(dashboard, "SOCCER_ODDS", tmp_path / "data" / "soccer" / "odds_snapshots.csv")
    write_csv(dashboard.SOCCER_UPCOMING, [{
        "competition_id": "openfootball:england:premierleague",
        "competition_name": "Premier League",
        "match_id": "thin-history",
        "match_date": "2026-11-01",
        "match_state": "scheduled",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "history_matches": 2,
        "total_goals_line": 2.5,
        "poisson_home_win": 0.55,
        "poisson_draw": 0.25,
        "poisson_away_win": 0.20,
        "poisson_over": 0.52,
        "poisson_under": 0.48,
    }])

    result = dashboard.api_soccer_picks()
    assert result["available"] is False
    assert result["diagnostics"]["input_rows"] == 1
    assert result["diagnostics"]["rejection_counts"]["INSUFFICIENT_HISTORY"] == 1
    assert "INSUFFICIENT_HISTORY=1" in result["message"]
    assert "Evidence gates were not relaxed" in result["message"]
