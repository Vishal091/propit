"""Inference utilities for soccer match goal probabilities using a Poisson score grid."""
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


def _dixon_coles_factor(home_goals: int, away_goals: int,
                        expected_home: float, expected_away: float, rho: float) -> float:
    """Low-score correction used by the Dixon-Coles goal model."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - expected_home * expected_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + expected_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + expected_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def predict_match(
    expected_home_goals: float,
    expected_away_goals: float,
    total_goals_line: float = 2.5,
    max_goals: int = 12,
    rho: float = 0.0,
) -> MatchProbabilities:
    """Estimate 1X2 and total-goal probabilities from a Poisson score grid.

    rho=0 is the independent-Poisson baseline. Nonzero rho applies the classic
    Dixon-Coles low-score adjustment and then renormalizes the truncated score grid.
    This function does not estimate rho; tune/compare it only on chronological holdouts.

    For integer total-goals lines, exact-line probability is reported as push. Over and
    under remain unconditional probabilities and are not renormalized around pushes.
    """
    home_rate = _validate_rate(expected_home_goals, "expected_home_goals")
    away_rate = _validate_rate(expected_away_goals, "expected_away_goals")
    line = float(total_goals_line)
    if not math.isfinite(line) or line < 0:
        raise ValueError("total_goals_line must be finite and non-negative")
    if isinstance(max_goals, bool) or not isinstance(max_goals, int) or max_goals < 5:
        raise ValueError("max_goals must be an integer >= 5")
    rho_value = float(rho)
    if not math.isfinite(rho_value) or not -0.5 <= rho_value <= 0.5:
        raise ValueError("rho must be finite and within [-0.5, 0.5]")

    home = _goal_distribution(home_rate, max_goals)
    away = _goal_distribution(away_rate, max_goals)
    matrix: list[list[float]] = []
    for h, ph in enumerate(home):
        row = []
        for a, pa in enumerate(away):
            factor = _dixon_coles_factor(h, a, home_rate, away_rate, rho_value)
            if not math.isfinite(factor) or factor < 0:
                raise ValueError("rho produces an invalid Dixon-Coles low-score adjustment")
            row.append(ph * pa * factor)
        matrix.append(row)
    mass = sum(map(sum, matrix))
    if mass <= 0:
        raise ValueError("goal distribution has no probability mass")
    matrix = [[p / mass for p in row] for row in matrix]

    p_home = p_draw = p_away = p_over = p_under = p_push = 0.0
    integer_line = math.isclose(line, round(line), abs_tol=1e-9)
    for h, row in enumerate(matrix):
        for a, probability in enumerate(row):
            if h > a:
                p_home += probability
            elif h == a:
                p_draw += probability
            else:
                p_away += probability
            total = h + a
            if integer_line and total == round(line):
                p_push += probability
            elif total > line:
                p_over += probability
            else:
                p_under += probability

    return MatchProbabilities(
        home_win=p_home, draw=p_draw, away_win=p_away,
        over=p_over, under=p_under, push=p_push,
        expected_home_goals=home_rate, expected_away_goals=away_rate,
    )
