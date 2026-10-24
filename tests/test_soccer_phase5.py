from __future__ import annotations

import pandas as pd
import pytest
from fastapi import HTTPException

from src import dashboard


def set_soccer_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "SOCCER_UPCOMING", tmp_path / "predictions" / "soccer" / "upcoming.csv")
    monkeypatch.setattr(dashboard, "SOCCER_MATCHES", tmp_path / "data" / "soccer" / "matches.csv")
    monkeypatch.setattr(dashboard, "SOCCER_BACKTEST", tmp_path / "predictions" / "soccer" / "backtest.csv")
    monkeypatch.setattr(dashboard, "SOCCER_ODDS", tmp_path / "data" / "soccer" / "odds_snapshots.csv")


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def fixture(**overrides):
    row = {
        "competition_id": "openfootball:england:premierleague",
        "competition_name": "English Premier League",
        "match_id": "premier-1",
        "match_date": "2026-11-01",
        "match_state": "scheduled",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_score": None,
        "away_score": None,
        "history_matches": 24,
        "home_team_home_matches": 10,
        "away_team_away_matches": 10,
        "expected_home_goals": 1.8,
        "expected_away_goals": 1.1,
        "total_goals_line": 2.5,
        "poisson_home_win": 0.52,
        "poisson_draw": 0.25,
        "poisson_away_win": 0.23,
        "poisson_over": 0.54,
        "poisson_under": 0.46,
    }
    row.update(overrides)
    return row


def test_picks_api_filters_to_top_five_and_returns_ranked_markets(tmp_path, monkeypatch):
    set_soccer_paths(monkeypatch, tmp_path)
    write(dashboard.SOCCER_UPCOMING, [
        fixture(),
        fixture(match_id="ucl", competition_id="uefa:ucl", competition_name="Champions League"),
        fixture(match_id="thin", history_matches=2),
        fixture(match_id="la-liga", competition_id="openfootball:spain:laliga",
                competition_name="Spanish La Liga", home_team="Barcelona", away_team="Real Madrid",
                poisson_under=0.72, poisson_over=0.28),
    ])
    response = dashboard.api_soccer_picks(
        date_from="2026-10-24", date_to="2026-11-07",
        min_history_matches=8, min_probability=0.52, limit=25,
    )
    assert response["available"] is True
    assert response["count"] == 2
    assert {row["league"] for row in response["rows"]} == {"Premier League", "La Liga"}
    assert response["rows"][0]["market"] == "Under 2.5"
    assert response["rows"][0]["odds_available"] is False
    assert "not a market edge" in response["warning"] or "not an odds-based" in response["warning"]


@pytest.mark.parametrize(
    "start,end",
    [("bad", None), ("2026-11-10", "2026-11-01")],
)
def test_picks_api_rejects_invalid_dates(tmp_path, monkeypatch, start, end):
    set_soccer_paths(monkeypatch, tmp_path)
    write(dashboard.SOCCER_UPCOMING, [fixture()])
    with pytest.raises(HTTPException) as error:
        dashboard.api_soccer_picks(
            date_from=start, date_to=end, min_history_matches=8,
            min_probability=0.52, limit=25,
        )
    assert error.value.status_code == 400


def test_calibration_api_is_scoped_to_recognized_top_five_competitions(tmp_path, monkeypatch):
    set_soccer_paths(monkeypatch, tmp_path)
    write(dashboard.SOCCER_MATCHES, [
        {"competition_id": "openfootball:england:premierleague", "competition_name": "English Premier League"},
        {"competition_id": "openfootball:france:ligue1", "competition_name": "French Ligue 1"},
        {"competition_id": "uefa:ucl", "competition_name": "Champions League"},
    ])
    rows = []
    for idx, actual in enumerate(("H", "H", "D", "A")):
        shared = {
            "match_id": f"m{idx}", "match_date": f"2026-01-0{idx+1}",
            "actual_outcome": actual, "actual_home_score": 1, "actual_away_score": 0,
            "poisson_home_win": .7, "poisson_draw": .2, "poisson_away_win": .1,
            "dixon_coles_home_win": .65, "dixon_coles_draw": .25, "dixon_coles_away_win": .1,
        }
        rows.append({**shared, "competition_id": "openfootball:england:premierleague"})
        rows.append({**shared, "match_id": f"ucl{idx}", "competition_id": "uefa:ucl"})
    write(dashboard.SOCCER_BACKTEST, rows)
    report = dashboard.api_soccer_calibration(bins=5, min_samples=5)
    assert report["available"] is True
    assert report["eligible_competitions"] == 2
    assert report["model_reports"][0]["matches"] == 4
    assert report["model_reports"][0]["sample_sufficient"] is False
    assert report["matches_evaluated"] == 4
