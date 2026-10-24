"""Rolling, point-in-time soccer goal-rate estimates.

Uses only completed matches in the same competition strictly before the target date.
Venue-specific team rates are shrunk towards a competition baseline with exponential
time decay. These are transparent baseline estimates, not calibrated forecasts.
"""
from __future__ import annotations
import math
from datetime import date
from typing import Any, Iterable, Mapping
from src.soccer.poisson import MatchProbabilities, predict_match

DEFAULT_HOME_GOALS = 1.45
DEFAULT_AWAY_GOALS = 1.15
MIN_EXPECTED_GOALS = 0.05
MAX_EXPECTED_GOALS = 5.0
_FINISHED_STATES = {"finished", "complete", "completed", "played", "full time", "full_time", "ft"}

def _get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)

def _iso_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None

def _goal(value: Any) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)

def completed_matches(rows: Iterable[Any], competition_id: str | None = None,
                      before_date: str | None = None) -> list[dict[str, Any]]:
    """Normalize completed scored matches, applying a strict optional date cutoff."""
    cutoff = _iso_date(before_date) if before_date else None
    selected: list[dict[str, Any]] = []
    for row in rows:
        comp = str(_get(row, "competition_id", "") or "")
        if competition_id is not None and comp != str(competition_id):
            continue
        match_date = _iso_date(_get(row, "match_date"))
        if match_date is None or (cutoff is not None and match_date >= cutoff):
            continue
        state = str(_get(row, "match_state", "") or "").strip().casefold()
        if state and state not in _FINISHED_STATES:
            continue
        home_score, away_score = _goal(_get(row, "home_score")), _goal(_get(row, "away_score"))
        home = str(_get(row, "home_team", "") or "").strip()
        away = str(_get(row, "away_team", "") or "").strip()
        if home_score is None or away_score is None or not home or not away:
            continue
        selected.append({
            "competition_id": comp, "match_date": match_date.isoformat(),
            "home_team": home, "away_team": away,
            "home_score": home_score, "away_score": away_score,
            "match_id": str(_get(row, "match_id", "") or ""),
        })
    return sorted(selected, key=lambda x: (x["match_date"], x["match_id"], x["home_team"], x["away_team"]))

def estimate_expected_goals(history: Iterable[Any], competition_id: str, match_date: str,
                            home_team: str, away_team: str, prior_games: float = 5.0,
                            half_life_days: float = 365.0, lookback_days: int = 730) -> dict[str, Any]:
    """Estimate expected goals from prior finished matches only.

    Same-day matches are excluded. prior_games shrinks samples towards a weak baseline;
    weights halve every half_life_days. Old observations are limited by lookback_days.
    """
    cutoff = _iso_date(match_date)
    if cutoff is None:
        raise ValueError("match_date must be a real ISO date YYYY-MM-DD")
    if not math.isfinite(prior_games) or prior_games <= 0:
        raise ValueError("prior_games must be finite and greater than zero")
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be finite and greater than zero")
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days < 1:
        raise ValueError("lookback_days must be an integer >= 1")
    if not home_team.strip() or not away_team.strip():
        raise ValueError("home_team and away_team are required")
    if home_team.casefold() == away_team.casefold():
        raise ValueError("home_team and away_team must be different")

    prior = completed_matches(history, competition_id=str(competition_id), before_date=cutoff.isoformat())
    weighted = []
    for row in prior:
        age = (cutoff - date.fromisoformat(row["match_date"])).days
        if age < 1 or age > lookback_days:
            continue
        weighted.append((row, 2.0 ** (-age / half_life_days)))

    home_sum = sum(row["home_score"] * weight for row, weight in weighted)
    away_sum = sum(row["away_score"] * weight for row, weight in weighted)
    total_weight = sum(weight for _row, weight in weighted)
    league_home = (home_sum + prior_games * DEFAULT_HOME_GOALS) / (total_weight + prior_games)
    league_away = (away_sum + prior_games * DEFAULT_AWAY_GOALS) / (total_weight + prior_games)

    home_home = [(row, weight) for row, weight in weighted if row["home_team"].casefold() == home_team.casefold()]
    away_away = [(row, weight) for row, weight in weighted if row["away_team"].casefold() == away_team.casefold()]

    def smoothed_ratio(sample, score_key: str, baseline: float) -> float:
        sample_weight = sum(weight for _row, weight in sample)
        score_sum = sum(row[score_key] * weight for row, weight in sample)
        smoothed_rate = (score_sum + prior_games * baseline) / (sample_weight + prior_games)
        return smoothed_rate / baseline if baseline > 0 else 1.0

    home_attack = smoothed_ratio(home_home, "home_score", league_home)
    home_defence_weakness = smoothed_ratio(home_home, "away_score", league_away)
    away_attack = smoothed_ratio(away_away, "away_score", league_away)
    away_defence_weakness = smoothed_ratio(away_away, "home_score", league_home)
    expected_home = min(MAX_EXPECTED_GOALS, max(MIN_EXPECTED_GOALS, league_home * home_attack * away_defence_weakness))
    expected_away = min(MAX_EXPECTED_GOALS, max(MIN_EXPECTED_GOALS, league_away * away_attack * home_defence_weakness))
    return {
        "expected_home_goals": expected_home, "expected_away_goals": expected_away,
        "league_home_goals_prior": league_home, "league_away_goals_prior": league_away,
        "history_matches": len(weighted), "home_team_home_matches": len(home_home),
        "away_team_away_matches": len(away_away), "match_date_cutoff": cutoff.isoformat(),
        "lookback_days": lookback_days, "half_life_days": half_life_days, "prior_games": prior_games,
    }

def forecast_fixture(history: Iterable[Any], competition_id: str, match_date: str,
                     home_team: str, away_team: str, total_goals_line: float = 2.5,
                     prior_games: float = 5.0, half_life_days: float = 365.0,
                     lookback_days: int = 730, rho: float = 0.0) -> dict[str, Any]:
    """Return expected goals and 1X2/total-goal probabilities for a fixture."""
    rates = estimate_expected_goals(history, competition_id, match_date, home_team, away_team,
                                    prior_games, half_life_days, lookback_days)
    probs: MatchProbabilities = predict_match(
        rates["expected_home_goals"], rates["expected_away_goals"],
        total_goals_line=total_goals_line, rho=rho,
    )
    return {
        **rates, "home_win": probs.home_win, "draw": probs.draw, "away_win": probs.away_win,
        "over": probs.over, "under": probs.under, "push": probs.push, "rho": rho,
    }
