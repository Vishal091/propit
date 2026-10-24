"""Transparent baseline soccer probabilities using independent Poisson goal counts.

These are research baselines, not calibrated forecasts. Expected goals must come from
historical, pre-match data; do not fit or evaluate using information recorded after kickoff.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class MatchProbabilities:
    home_win: float
    draw: float
    away_win: float
    over: float
    under: float
    push: float
    expected_home_goals: float
    expected_away_goals: float

def _validate_rate(rate: float, name: str) -> float:
    value = float(rate)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value

def poisson_pmf(goals: int, rate: float) -> float:
    """Return P(X=goals) for a Poisson variable with mean rate."""
    if isinstance(goals, bool) or not isinstance(goals, int) or goals < 0:
        raise ValueError("goals must be a non-negative integer")
    mean = _validate_rate(rate, "rate")
    if mean == 0:
        return 1.0 if goals == 0 else 0.0
    return math.exp(-mean + goals * math.log(mean) - math.lgamma(goals + 1))

def _goal_distribution(rate: float, max_goals: int) -> list[float]:
    return [poisson_pmf(g, rate) for g in range(max_goals + 1)]

def predict_match(expected_home_goals: float, expected_away_goals: float,
                  total_goals_line: float = 2.5, max_goals: int = 12) -> MatchProbabilities:
    """Estimate 1X2 and total-goal probabilities from independent goal rates.

    For an integer total-goals line, probability mass exactly on the line is reported
    as push; over and under are unconditional probabilities and are not renormalized.
    The finite score grid is normalized to account for its tiny omitted upper tail.
    """
    home_rate = _validate_rate(expected_home_goals, "expected_home_goals")
    away_rate = _validate_rate(expected_away_goals, "expected_away_goals")
    line = float(total_goals_line)
    if not math.isfinite(line) or line < 0:
        raise ValueError("total_goals_line must be finite and non-negative")
    if isinstance(max_goals, bool) or not isinstance(max_goals, int) or max_goals < 5:
        raise ValueError("max_goals must be an integer >= 5")
    home = _goal_distribution(home_rate, max_goals)
    away = _goal_distribution(away_rate, max_goals)
    matrix = [[ph * pa for pa in away] for ph in home]
    mass = sum(map(sum, matrix))
    if mass <= 0:
        raise ValueError("goal distribution has no probability mass")
    matrix = [[p / mass for p in row] for row in matrix]
    p_home = p_draw = p_away = p_over = p_under = p_push = 0.0
    for h, row in enumerate(matrix):
        for a, probability in enumerate(row):
            if h > a: p_home += probability
            elif h == a: p_draw += probability
            else: p_away += probability
            total = h + a
            if math.isclose(line, round(line), abs_tol=1e-9) and total == round(line):
                p_push += probability
            elif total > line: p_over += probability
            else: p_under += probability
    return MatchProbabilities(p_home, p_draw, p_away, p_over, p_under, p_push,
                              home_rate, away_rate)
