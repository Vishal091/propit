"""Chronological backtesting helpers for the soccer goal model.

For every test match, training history contains completed matches with an earlier calendar
date only. Matches played on the same date are excluded from one another's training set.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from typing import Any, Iterable, Mapping

from src.soccer.model import completed_matches, estimate_expected_goals
from src.soccer.poisson import predict_match


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-15, max(1e-15, float(value)))


def _actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _metrics(predictions: list[dict[str, Any]], variant: str, total_goals_line: float) -> dict[str, Any]:
    n = len(predictions)
    if n == 0:
        return {
            "model": variant, "matches": 0, "log_loss_1x2": None, "brier_1x2": None,
            "accuracy_1x2": None, "expected_goals_mae": None, "total_matches": 0,
            "total_log_loss": None, "total_brier": None, "total_accuracy": None,
        }
    keys = {"H": f"{variant}_home_win", "D": f"{variant}_draw", "A": f"{variant}_away_win"}
    log_loss = brier = correct = 0.0
    goals_abs_error: list[float] = []
    total_log_losses: list[float] = []
    total_briers: list[float] = []
    total_correct = 0
    is_integer_line = math.isclose(total_goals_line, round(total_goals_line), abs_tol=1e-9)
    for row in predictions:
        outcome = _actual_outcome(row["actual_home_score"], row["actual_away_score"])
        probs = {
            "H": _clip_probability(row[keys["H"]]),
            "D": _clip_probability(row[keys["D"]]),
            "A": _clip_probability(row[keys["A"]]),
        }
        log_loss -= math.log(probs[outcome])
        brier += sum((probs[key] - (1.0 if key == outcome else 0.0)) ** 2 for key in ("H", "D", "A"))
        correct += int(max(probs, key=probs.get) == outcome)
        goals_abs_error.extend([
            abs(row["expected_home_goals"] - row["actual_home_score"]),
            abs(row["expected_away_goals"] - row["actual_away_score"]),
        ])
        actual_total = row["actual_home_score"] + row["actual_away_score"]
        if is_integer_line and actual_total == round(total_goals_line):
            continue
        over_raw = row[f"{variant}_over"]
        under_raw = row[f"{variant}_under"]
        denominator = over_raw + under_raw
        if denominator <= 0:
            continue
        p_over = _clip_probability(over_raw / denominator)
        actual_over = actual_total > total_goals_line
        total_log_losses.append(-math.log(p_over if actual_over else 1.0 - p_over))
        total_briers.append((p_over - (1.0 if actual_over else 0.0)) ** 2)
        total_correct += int((p_over >= 0.5) == actual_over)
    total_n = len(total_log_losses)
    return {
        "model": variant,
        "matches": n,
        "log_loss_1x2": log_loss / n,
        "brier_1x2": brier / n,
        "accuracy_1x2": correct / n,
        "expected_goals_mae": sum(goals_abs_error) / len(goals_abs_error),
        "total_matches": total_n,
        "total_log_loss": sum(total_log_losses) / total_n if total_n else None,
        "total_brier": sum(total_briers) / total_n if total_n else None,
        "total_accuracy": total_correct / total_n if total_n else None,
    }


def run_rolling_backtest(
    rows: Iterable[Any],
    min_history_matches: int = 30,
    prior_games: float = 5.0,
    half_life_days: float = 365.0,
    lookback_days: int = 730,
    total_goals_line: float = 2.5,
    dixon_coles_rho: float = -0.05,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a date-grouped rolling backtest for independent Poisson and Dixon-Coles."""
    if isinstance(min_history_matches, bool) or min_history_matches < 0:
        raise ValueError("min_history_matches must be a non-negative integer")
    if not math.isfinite(total_goals_line) or total_goals_line < 0:
        raise ValueError("total_goals_line must be finite and non-negative")
    if not math.isfinite(dixon_coles_rho) or not -0.5 <= dixon_coles_rho <= 0.5:
        raise ValueError("dixon_coles_rho must be finite and within [-0.5, 0.5]")

    completed = completed_matches(rows)
    by_competition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_competition[row["competition_id"]].append(row)

    predictions: list[dict[str, Any]] = []
    skipped = 0
    competition_summaries = []
    for competition_id in sorted(by_competition):
        comp_rows = sorted(by_competition[competition_id], key=lambda item: (
            item["match_date"], item["match_id"], item["home_team"], item["away_team"]
        ))
        dates = sorted({row["match_date"] for row in comp_rows})
        comp_predictions: list[dict[str, Any]] = []
        for match_date in dates:
            training = [row for row in comp_rows if row["match_date"] < match_date]
            day_matches = [row for row in comp_rows if row["match_date"] == match_date]
            if len(training) < min_history_matches:
                skipped += len(day_matches)
                continue
            for match in day_matches:
                rates = estimate_expected_goals(
                    history=training,
                    competition_id=competition_id,
                    match_date=match_date,
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    prior_games=prior_games,
                    half_life_days=half_life_days,
                    lookback_days=lookback_days,
                )
                baseline = predict_match(
                    rates["expected_home_goals"], rates["expected_away_goals"],
                    total_goals_line=total_goals_line, rho=0.0,
                )
                adjusted = predict_match(
                    rates["expected_home_goals"], rates["expected_away_goals"],
                    total_goals_line=total_goals_line, rho=dixon_coles_rho,
                )
                league_average = predict_match(
                    rates["league_home_goals_prior"], rates["league_away_goals_prior"],
                    total_goals_line=total_goals_line, rho=0.0,
                )
                row = {
                    "provider": match.get("provider", ""),
                    "competition_id": competition_id,
                    "competition_name": match.get("competition_name", ""),
                    "season_id": match.get("season_id", ""),
                    "season_name": match.get("season_name", ""),
                    "match_id": match["match_id"],
                    "match_date": match_date,
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "actual_home_score": match["home_score"],
                    "actual_away_score": match["away_score"],
                    "actual_outcome": _actual_outcome(match["home_score"], match["away_score"]),
                    "actual_total_goals": match["home_score"] + match["away_score"],
                    "history_matches": rates["history_matches"],
                    "home_team_home_matches": rates["home_team_home_matches"],
                    "away_team_away_matches": rates["away_team_away_matches"],
                    "expected_home_goals": rates["expected_home_goals"],
                    "expected_away_goals": rates["expected_away_goals"],
                    "league_home_goals_prior": rates["league_home_goals_prior"],
                    "league_away_goals_prior": rates["league_away_goals_prior"],
                    "total_goals_line": total_goals_line,
                    "dixon_coles_rho": dixon_coles_rho,
                }
                for variant, probs in (
                    ("poisson", baseline),
                    ("dixon_coles", adjusted),
                    ("league_average", league_average),
                ):
                    row.update({
                        f"{variant}_home_win": probs.home_win,
                        f"{variant}_draw": probs.draw,
                        f"{variant}_away_win": probs.away_win,
                        f"{variant}_over": probs.over,
                        f"{variant}_under": probs.under,
                        f"{variant}_push": probs.push,
                    })
                predictions.append(row)
                comp_predictions.append(row)
        competition_summaries.append({
            "competition_id": competition_id,
            "competition_name": next((row.get("competition_name", "") for row in comp_rows if row.get("competition_name")), ""),
            "scored_matches_in_source": len(comp_rows),
            "backtested_matches": len(comp_predictions),
            "metrics": [
                _metrics(comp_predictions, "poisson", total_goals_line),
                _metrics(comp_predictions, "dixon_coles", total_goals_line),
                _metrics(comp_predictions, "league_average", total_goals_line),
            ],
        })
    season_summaries = []
    season_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        season_groups[(
            str(prediction.get("competition_id", "")),
            str(prediction.get("season_id", "")),
            str(prediction.get("season_name", "")),
        )].append(prediction)
    completed_by_season: dict[tuple[str, str, str], int] = defaultdict(int)
    for match in completed:
        completed_by_season[(
            str(match.get("competition_id", "")),
            str(match.get("season_id", "")),
            str(match.get("season_name", "")),
        )] += 1
    for (competition_id, season_id, season_name), group_rows in sorted(season_groups.items()):
        season_summaries.append({
            "competition_id": competition_id,
            "season_id": season_id,
            "season_name": season_name,
            "scored_matches_in_source": completed_by_season[(competition_id, season_id, season_name)],
            "backtested_matches": len(group_rows),
            "metrics": [
                _metrics(group_rows, "poisson", total_goals_line),
                _metrics(group_rows, "dixon_coles", total_goals_line),
                _metrics(group_rows, "league_average", total_goals_line),
            ],
        })
    report = {
        "method": "strict-date rolling-origin backtest; same-day results are excluded from training",
        "source_scored_matches": len(completed),
        "backtested_matches": len(predictions),
        "skipped_warmup_matches": skipped,
        "minimum_prior_matches": min_history_matches,
        "prior_games": prior_games,
        "half_life_days": half_life_days,
        "lookback_days": lookback_days,
        "total_goals_line": total_goals_line,
        "dixon_coles_rho": dixon_coles_rho,
        "models": [
            _metrics(predictions, "poisson", total_goals_line),
            _metrics(predictions, "dixon_coles", total_goals_line),
            _metrics(predictions, "league_average", total_goals_line),
        ],
        "competitions": competition_summaries,
        "seasons": season_summaries,
        "warning": "Metrics are meaningful only for the imported coverage. The Dixon-Coles rho is a supplied experimental value, not fitted or assumed optimal.",
    }
    return predictions, report
