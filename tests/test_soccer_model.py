from __future__ import annotations

import math

import pytest

from src.soccer.backtest import run_rolling_backtest
from src.soccer.model import estimate_expected_goals, forecast_fixture
from src.soccer.poisson import predict_match


def row(comp, day, home, away, hg, ag, match_id=""):
    return {
        "competition_id": comp,
        "match_date": day,
        "home_team": home,
        "away_team": away,
        "home_score": hg,
        "away_score": ag,
        "match_state": "finished" if hg is not None and ag is not None else "scheduled",
        "match_id": match_id or f"{day}:{home}:{away}",
    }


def test_cold_start_uses_documented_prior_and_produces_valid_forecast():
    rates = estimate_expected_goals([], "league-a", "2026-10-24", "Arsenal", "Chelsea")
    assert rates["expected_home_goals"] == pytest.approx(1.45)
    assert rates["expected_away_goals"] == pytest.approx(1.15)
    assert rates["history_matches"] == 0
    forecast = forecast_fixture([], "league-a", "2026-10-24", "Arsenal", "Chelsea")
    assert forecast["home_win"] + forecast["draw"] + forecast["away_win"] == pytest.approx(1)
    assert forecast["over"] + forecast["under"] == pytest.approx(1)


def test_same_day_and_future_results_cannot_change_fixture_forecast():
    past = [row("L", "2026-10-20", "Arsenal", "Chelsea", 3, 0, "past")]
    same_day_extreme = row("L", "2026-10-24", "Arsenal", "Spurs", 9, 0, "same-day")
    future_extreme = row("L", "2026-10-25", "Arsenal", "Liverpool", 10, 0, "future")
    expected = estimate_expected_goals(past, "L", "2026-10-24", "Arsenal", "Chelsea")
    actual = estimate_expected_goals(
        past + [same_day_extreme, future_extreme], "L", "2026-10-24", "Arsenal", "Chelsea"
    )
    assert actual["expected_home_goals"] == pytest.approx(expected["expected_home_goals"])
    assert actual["expected_away_goals"] == pytest.approx(expected["expected_away_goals"])
    assert actual["history_matches"] == expected["history_matches"] == 1


def test_estimator_isolates_competitions_and_limits_lookback():
    history = [
        row("L1", "2024-01-01", "Arsenal", "Chelsea", 5, 0, "old"),
        row("L2", "2026-10-20", "Arsenal", "Chelsea", 8, 0, "other-league"),
        row("L1", "2026-10-20", "Liverpool", "Everton", 1, 1, "recent"),
    ]
    expected = estimate_expected_goals(history, "L1", "2026-10-24", "Arsenal", "Chelsea", lookback_days=730)
    assert expected["history_matches"] == 1
    assert expected["expected_home_goals"] < 2.0


def test_home_attack_and_away_defence_strengthen_home_goal_rate():
    history = [
        row("L", "2026-10-01", "Arsenal", "Wolves", 4, 0, "a1"),
        row("L", "2026-10-02", "Arsenal", "Everton", 3, 0, "a2"),
        row("L", "2026-10-03", "Man City", "Chelsea", 1, 1, "a3"),
        row("L", "2026-10-04", "Liverpool", "Spurs", 1, 0, "a4"),
        row("L", "2026-10-05", "Wolves", "Bournemouth", 0, 1, "a5"),
        row("L", "2026-10-06", "Everton", "Fulham", 0, 0, "a6"),
        row("L", "2026-10-07", "Chelsea", "Man United", 1, 1, "a7"),
    ]
    rates = estimate_expected_goals(history, "L", "2026-10-10", "Arsenal", "Wolves")
    assert rates["home_team_home_matches"] == 2
    assert rates["expected_home_goals"] > rates["league_home_goals_prior"]


def test_model_rejects_bad_inputs():
    with pytest.raises(ValueError):
        estimate_expected_goals([], "L", "not-a-date", "A", "B")
    with pytest.raises(ValueError):
        estimate_expected_goals([], "L", "2026-10-24", "A", "A")
    with pytest.raises(ValueError):
        estimate_expected_goals([], "L", "2026-10-24", "A", "B", prior_games=0)


def test_backtest_uses_strictly_prior_dates_and_reports_both_models():
    matches = [
        row("L", "2026-01-01", "A", "B", 1, 0, "d1a"),
        row("L", "2026-01-01", "C", "D", 0, 0, "d1b"),
        row("L", "2026-01-02", "A", "C", 2, 1, "d2a"),
        row("L", "2026-01-02", "B", "D", 0, 2, "d2b"),
        row("L", "2026-01-03", "A", "D", 1, 1, "d3a"),
        row("L", "2026-01-04", "B", "C", 3, 0, "d4a"),
    ]
    predictions, report = run_rolling_backtest(matches, min_history_matches=2, dixon_coles_rho=-0.05)
    assert len(predictions) == 4
    assert report["source_scored_matches"] == 6
    assert report["backtested_matches"] == 4
    assert report["skipped_warmup_matches"] == 2
    day_two = [p for p in predictions if p["match_date"] == "2026-01-02"]
    assert len(day_two) == 2
    assert {p["history_matches"] for p in day_two} == {2}
    assert all(p["poisson_home_win"] + p["poisson_draw"] + p["poisson_away_win"] == pytest.approx(1) for p in predictions)
    assert {m["model"] for m in report["models"]} == {"poisson", "dixon_coles"}
    assert all(m["matches"] == 4 for m in report["models"])
    assert all(m["log_loss_1x2"] is not None and m["brier_1x2"] is not None for m in report["models"])


def test_backtest_ignores_unfinished_matches_and_validates_options():
    matches = [
        row("L", "2026-01-01", "A", "B", 1, 0, "d1"),
        row("L", "2026-01-02", "B", "C", None, None, "future"),
        row("L", "2026-01-03", "A", "C", 0, 0, "d3"),
    ]
    predictions, report = run_rolling_backtest(matches, min_history_matches=1)
    assert report["source_scored_matches"] == 2
    assert len(predictions) == 1
    with pytest.raises(ValueError):
        run_rolling_backtest(matches, dixon_coles_rho=2)


def test_explicit_non_finished_state_is_never_used_as_training_history():
    history = [
        row("L", "2026-10-20", "Arsenal", "Chelsea", 3, 0, "finished"),
        {**row("L", "2026-10-21", "Arsenal", "Spurs", 9, 0, "not-finished"), "match_state": "postponed"},
    ]
    rates = estimate_expected_goals(history, "L", "2026-10-24", "Arsenal", "Chelsea")
    assert rates["history_matches"] == 1
