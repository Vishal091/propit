from __future__ import annotations

import pytest

from src.soccer.insights import build_best_picks, build_calibration_report, canonical_top_five_league


def forecast(**overrides):
    row = {
        "competition_id": "openfootball:england:premierleague",
        "competition_name": "Premier League",
        "match_id": "fx-1",
        "match_date": "2026-11-01",
        "match_state": "scheduled",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
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


def test_top_five_league_normalization_only_accepts_selected_domestic_leagues():
    assert canonical_top_five_league({"competition_name": "Premier League"}) == "Premier League"
    assert canonical_top_five_league({"competition_name": "La Liga"}) == "La Liga"
    assert canonical_top_five_league({"competition_id": "openfootball:germany:1-bundesliga"}) == "Bundesliga"
    assert canonical_top_five_league({"competition_name": "Serie A"}) == "Serie A"
    assert canonical_top_five_league({"competition_id": "openfootball:france:ligue1"}) == "Ligue 1"
    assert canonical_top_five_league({"competition_name": "Champions League"}) is None
    assert canonical_top_five_league({"competition_name": "Ligue 2"}) is None
    assert canonical_top_five_league({"competition_name": "2. Bundesliga"}) is None


def test_best_picks_rank_top_five_fixtures_one_market_each_and_report_odds_missing():
    rows = [
        forecast(),
        forecast(match_id="champions", competition_id="ucl", competition_name="Champions League"),
        forecast(match_id="thin", history_matches=3),
        forecast(match_id="played", home_score=2, away_score=0),
        forecast(match_id="other", competition_id="openfootball:italy:seriea", competition_name="Serie A",
                 home_team="Inter", away_team="Milan", history_matches=18, poisson_under=0.72,
                 poisson_over=0.28),
    ]
    picks = build_best_picks(rows, min_history_matches=8, min_probability=0.52, limit=20)
    assert len(picks) == 2
    assert picks[0]["match_id"] == "other"
    assert picks[0]["market"] == "Under 2.5"
    assert picks[0]["model_probability"] == pytest.approx(0.72)
    assert picks[0]["support_status"] == "LIMITED HISTORY"
    assert picks[0]["odds_available"] is False
    assert "Unavailable" in picks[0]["value_assessment"]
    assert len({item["match_id"] for item in picks}) == len(picks)
    assert [item["rank"] for item in picks] == [1, 2]


def test_best_picks_validate_configuration_and_probability_gate():
    with pytest.raises(ValueError):
        build_best_picks([], min_history_matches=-1)
    with pytest.raises(ValueError):
        build_best_picks([], min_probability=1.5)
    with pytest.raises(ValueError):
        build_best_picks([], limit=0)
    assert build_best_picks([forecast()], min_probability=0.9) == []


def test_calibration_report_builds_reliability_bins_and_sample_guard():
    rows = []
    for index, actual in enumerate(["H", "H", "D", "A"]):
        rows.append({
            "match_id": f"m{index}",
            "competition_id": "epl",
            "match_date": f"2026-01-0{index + 1}",
            "actual_outcome": actual,
            "poisson_home_win": 0.70,
            "poisson_draw": 0.20,
            "poisson_away_win": 0.10,
            "dixon_coles_home_win": 0.65,
            "dixon_coles_draw": 0.25,
            "dixon_coles_away_win": 0.10,
        })
    report = build_calibration_report(rows, competition_ids={"epl"}, bins=5, min_samples=5)
    assert report["available"] is True
    assert report["model_reports"][0]["matches"] == 4
    assert report["model_reports"][0]["sample_sufficient"] is False
    assert report["model_reports"][0]["expected_calibration_error"] is not None
    assert sum(bucket["count"] for bucket in report["model_reports"][0]["reliability"]) == 4
    assert build_calibration_report(rows, competition_ids={"other"})["available"] is False


def test_calibration_rejects_invalid_bin_settings():
    with pytest.raises(ValueError):
        build_calibration_report([], bins=1)
    with pytest.raises(ValueError):
        build_calibration_report([], min_samples=0)
