from __future__ import annotations

import pytest

from scripts.predict_soccer import build_upcoming_forecasts
from src.soccer.model import estimate_expected_goals


def match(comp, day, home, away, hg, ag, status, ident):
    return {
        "competition_id": comp, "match_date": day, "home_team": home, "away_team": away,
        "home_score": hg, "away_score": ag, "match_state": status, "match_id": ident,
        "provider": "test", "competition_name": "Test League", "season_id": "2026-27",
        "season_name": "2026/27", "kickoff_time_local": "15:00", "kickoff_timezone": None,
    }


def test_upcoming_forecasts_emit_both_models_and_use_only_past_history():
    rows = [
        match("L", "2026-10-15", "Arsenal", "Chelsea", "2", "0", "finished", "past"),
        match("L", "2026-10-25", "Arsenal", "Liverpool", "", "", "scheduled", "future"),
        match("L", "2026-10-25", "Arsenal", "Spurs", "9", "0", "finished", "same-day-result"),
        match("OTHER", "2026-10-25", "Arsenal", "Liverpool", "", "", "scheduled", "other-league"),
    ]
    forecasts = build_upcoming_forecasts(
        rows, date_from="2026-10-24", date_to="2026-10-26",
        competition_id="L", generated_at_utc="2026-10-24T00:00:00Z",
    )
    assert len(forecasts) == 1
    forecast = forecasts[0]
    assert forecast["match_id"] == "future"
    assert forecast["history_matches"] == 1
    assert forecast["forecast_generated_at_utc"] == "2026-10-24T00:00:00Z"
    assert forecast["poisson_home_win"] + forecast["poisson_draw"] + forecast["poisson_away_win"] == pytest.approx(1)
    assert forecast["dixon_coles_home_win"] + forecast["dixon_coles_draw"] + forecast["dixon_coles_away_win"] == pytest.approx(1)
    assert forecast["poisson_over"] + forecast["poisson_under"] == pytest.approx(1)
    assert "odds" not in forecast


def test_same_day_result_is_not_used_even_if_supplied_as_history():
    past = [match("L", "2026-10-15", "Arsenal", "Chelsea", "2", "0", "finished", "past")]
    same_day = match("L", "2026-10-25", "Arsenal", "Spurs", "9", "0", "finished", "same")
    first = estimate_expected_goals(past, "L", "2026-10-25", "Arsenal", "Liverpool")
    second = estimate_expected_goals(past + [same_day], "L", "2026-10-25", "Arsenal", "Liverpool")
    assert first["expected_home_goals"] == pytest.approx(second["expected_home_goals"])
    assert first["expected_away_goals"] == pytest.approx(second["expected_away_goals"])


@pytest.mark.parametrize(
    "start,end",
    [("bad-date", None), ("2026-10-30", "2026-10-24")],
)
def test_upcoming_forecasts_validate_date_filters(start, end):
    with pytest.raises(ValueError):
        build_upcoming_forecasts([], date_from=start, date_to=end)
