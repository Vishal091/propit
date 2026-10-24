"""Soccer best-pick ranking and out-of-sample calibration diagnostics.

Only the five major European domestic leagues are eligible for pick ranking.
The score ranks model conviction with an explicit sample-support adjustment; it is not
an odds-based edge, expected value, or guarantee of profitability.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping

TOP_FIVE_LEAGUES = (
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
)
_FINISHED = {"finished", "complete", "completed", "played", "full time", "full_time", "ft"}


def _get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _clean(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def canonical_top_five_league(row: Any) -> str | None:
    """Return a canonical top-five league name from provider names/IDs, otherwise None."""
    name = _clean(_get(row, "competition_name", ""))
    comp_id = _clean(_get(row, "competition_id", ""))
    joined = f"{name} {comp_id}"

    if ("premier league" in name or "england premierleague" in joined
            or "england premier league" in joined or "1 premierleague" in joined):
        return "Premier League"
    if (name in {"la liga", "laliga", "primera division", "spanish primera division"}
            or "spain laliga" in joined or "espana laliga" in joined
            or "spain 1 liga" in joined or "espana 1 liga" in joined):
        return "La Liga"
    if ("bundesliga" in name or "germany bundesliga" in joined or "deutschland bundesliga" in joined):
        if "2 bundesliga" not in name and "zweite" not in name:
            return "Bundesliga"
    if (name in {"serie a", "italian serie a"} or "italy seriea" in joined
            or "italy serie a" in joined or "italia serie a" in joined):
        return "Serie A"
    if (name in {"ligue 1", "ligue un", "french ligue 1"} or "france ligue1" in joined
            or "france ligue 1" in joined):
        return "Ligue 1"
    return None


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value)
    if number is None or number < 0:
        return default
    return int(number)


def build_best_picks(
    forecasts: Iterable[Any],
    *,
    min_history_matches: int = 8,
    min_probability: float = 0.52,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Rank the strongest modeled market for each eligible upcoming fixture.

    Picks use the independent-Poisson probability columns by default; Dixon-Coles rho is
    experimental and not fitted. The confidence rank score discounts forecasts with thin
    competition history. No market odds are assumed, so no value/EV claim is produced.
    """
    if isinstance(min_history_matches, bool) or min_history_matches < 0:
        raise ValueError("min_history_matches must be a non-negative integer")
    if not math.isfinite(min_probability) or not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must be between 0 and 1")
    if isinstance(limit, bool) or limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")

    ranked: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in forecasts:
        league = canonical_top_five_league(row)
        if league is None:
            continue
        match_date = str(_get(row, "match_date", "") or "")[:10]
        home = str(_get(row, "home_team", "") or "").strip()
        away = str(_get(row, "away_team", "") or "").strip()
        comp_id = str(_get(row, "competition_id", "") or "")
        if not (match_date and home and away and comp_id):
            continue
        state = str(_get(row, "match_state", "") or "").strip().casefold()
        if state in _FINISHED:
            continue
        if _number(_get(row, "home_score")) is not None or _number(_get(row, "away_score")) is not None:
            continue
        key = (comp_id, match_date, home.casefold(), away.casefold())
        if key in seen:
            continue
        seen.add(key)

        history_n = _integer(_get(row, "history_matches", 0))
        if history_n < min_history_matches:
            continue
        markets = []
        for suffix, label in (("home_win", "Home win"), ("draw", "Draw"), ("away_win", "Away win"),
                              ("over", f"Over {_get(row, 'total_goals_line', 2.5)}"),
                              ("under", f"Under {_get(row, 'total_goals_line', 2.5)}")):
            probability = _number(_get(row, f"poisson_{suffix}"))
            if probability is None:
                # Backwards compatibility with fixtures that store unprefixed probabilities.
                probability = _number(_get(row, suffix))
            if probability is not None and 0.0 <= probability <= 1.0:
                markets.append((probability, label, suffix))
        if not markets:
            continue
        probability, market, market_key = max(markets, key=lambda item: (item[0], item[2]))
        if probability < min_probability:
            continue

        # Score combines probability and history support without pretending to be EV.
        support = min(1.0, history_n / 20.0)
        rank_score = probability * (0.5 + 0.5 * support)
        expected_home = _number(_get(row, "expected_home_goals"))
        expected_away = _number(_get(row, "expected_away_goals"))
        ranked.append({
            "competition_id": comp_id,
            "competition_name": str(_get(row, "competition_name", "") or league),
            "league": league,
            "match_id": str(_get(row, "match_id", "") or ""),
            "match_date": match_date,
            "kickoff_time_local": str(_get(row, "kickoff_time_local", "") or ""),
            "home_team": home,
            "away_team": away,
            "market": market,
            "market_key": market_key,
            "model": "independent_poisson",
            "model_probability": probability,
            "history_matches": history_n,
            "home_team_home_matches": _integer(_get(row, "home_team_home_matches", 0)),
            "away_team_away_matches": _integer(_get(row, "away_team_away_matches", 0)),
            "expected_home_goals": expected_home,
            "expected_away_goals": expected_away,
            "rank_score": rank_score,
            "support_status": "SUPPORTED" if history_n >= 20 else "LIMITED HISTORY",
            "odds_available": False,
            "value_assessment": "Unavailable: bookmaker odds not imported",
        })

    ranked.sort(key=lambda item: (
        -item["rank_score"], -item["model_probability"], -item["history_matches"],
        item["match_date"], item["league"], item["home_team"],
    ))
    for index, pick in enumerate(ranked[:limit], start=1):
        pick["rank"] = index
    return ranked[:limit]


def build_calibration_report(
    backtest_rows: Iterable[Any],
    *,
    competition_ids: set[str] | None = None,
    bins: int = 10,
    min_samples: int = 30,
) -> dict[str, Any]:
    """Calculate top-outcome reliability/ECE on chronological backtest predictions."""
    if isinstance(bins, bool) or bins < 2 or bins > 20:
        raise ValueError("bins must be between 2 and 20")
    if isinstance(min_samples, bool) or min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    selected = []
    for row in backtest_rows:
        comp_id = str(_get(row, "competition_id", "") or "")
        if competition_ids is not None and comp_id not in competition_ids:
            continue
        actual = str(_get(row, "actual_outcome", "") or "").strip().upper()
        if actual not in {"H", "D", "A"}:
            home, away = _number(_get(row, "actual_home_score")), _number(_get(row, "actual_away_score"))
            if home is None or away is None:
                continue
            actual = "H" if home > away else "A" if away > home else "D"
        for model in ("poisson", "dixon_coles"):
            probabilities = {
                "H": _number(_get(row, f"{model}_home_win")),
                "D": _number(_get(row, f"{model}_draw")),
                "A": _number(_get(row, f"{model}_away_win")),
            }
            if any(p is None or not 0 <= p <= 1 for p in probabilities.values()):
                continue
            predicted = max(probabilities, key=lambda key: probabilities[key])
            confidence = float(probabilities[predicted])
            selected.append({
                "model": model, "match_date": str(_get(row, "match_date", "") or ""),
                "actual": actual, "predicted": predicted, "confidence": confidence,
                "correct": int(predicted == actual),
            })

    model_reports = []
    for model in ("poisson", "dixon_coles"):
        model_rows = [item for item in selected if item["model"] == model]
        n = len(model_rows)
        reliability = []
        weighted_gap = 0.0
        for index in range(bins):
            lower, upper = index / bins, (index + 1) / bins
            bucket = [item for item in model_rows if lower <= item["confidence"] < upper
                      or (index == bins - 1 and item["confidence"] == 1.0)]
            if not bucket:
                continue
            mean_predicted = sum(item["confidence"] for item in bucket) / len(bucket)
            observed = sum(item["correct"] for item in bucket) / len(bucket)
            gap = observed - mean_predicted
            weighted_gap += len(bucket) / n * abs(gap) if n else 0.0
            reliability.append({
                "lower": lower, "upper": upper, "count": len(bucket),
                "mean_predicted_confidence": mean_predicted,
                "observed_accuracy": observed, "observed_minus_predicted": gap,
            })
        model_reports.append({
            "model": model, "matches": n,
            "expected_calibration_error": weighted_gap if n else None,
            "reliability": reliability,
            "sample_sufficient": n >= min_samples,
            "interpretation": ("Sample meets the configured minimum; calibration is still descriptive, not a guarantee."
                               if n >= min_samples else
                               f"Insufficient sample: {n} matches; at least {min_samples} are required for a calibration claim."),
        })

    return {
        "available": bool(selected),
        "method": "Top-class confidence reliability over rolling-origin backtest predictions; no calibration transform is fitted.",
        "bins_requested": bins,
        "minimum_samples": min_samples,
        "matches_evaluated": len({(item["match_date"], item["model"]) for item in selected}),
        "model_reports": model_reports,
        "warning": "Calibration describes historical out-of-sample predictions only. It does not establish profitability or future accuracy.",
    }
