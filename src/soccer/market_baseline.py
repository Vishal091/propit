"""Point-in-time market-implied benchmark for chronological soccer forecasts.

Only exact team names, league keys, event IDs and snapshots observed before kickoff are used.
Market probabilities are the equal-weighted mean of complete, same-book no-vig markets.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from src.soccer.insights import canonical_top_five_league
from src.soccer.odds import TOP_FIVE_SPORTS


def _get(row: Any, key: str, default: Any = None) -> Any:
    return row.get(key, default) if isinstance(row, Mapping) else getattr(row, key, default)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _utc(value: Any) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value or "").strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        return None
    return result.astimezone(timezone.utc)


def _team(value: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    for suffix in ("footballclub", "soccerclub", "fc", "afc"):
        if key.endswith(suffix) and len(key) > len(suffix) + 3:
            return key[:-len(suffix)]
    return key


def _outcomes(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(str(row.get("outcomes_json") or "[]"))
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, list):
        return {}
    return {str(item.get("key") or ""): item for item in data if isinstance(item, Mapping)}


def build_market_implied_comparisons(
    predictions: Iterable[Mapping[str, Any]],
    odds_rows: Iterable[Mapping[str, Any]],
    *,
    max_hours_before_kickoff: float = 168.0,
    max_quote_age_hours: float = 24.0,
    total_goals_line: float = 2.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach historic market-implied 1X2/totals probabilities for exact matched fixtures."""
    if not math.isfinite(max_hours_before_kickoff) or max_hours_before_kickoff <= 0:
        raise ValueError("max_hours_before_kickoff must be finite and positive")
    if not math.isfinite(max_quote_age_hours) or max_quote_age_hours <= 0:
        raise ValueError("max_quote_age_hours must be finite and positive")
    if not math.isfinite(total_goals_line) or total_goals_line < 0:
        raise ValueError("total_goals_line must be finite and non-negative")

    prediction_list = [dict(item) for item in predictions]
    sport_by_league = {value[1]: value[0] for value in TOP_FIVE_SPORTS.values()}
    odds = [dict(item) for item in odds_rows]
    comparisons = []
    no_match = 0
    matches_h2h = 0
    matches_totals = 0

    for raw in prediction_list:
        prediction = dict(raw)
        league = canonical_top_five_league(prediction)
        sport_key = sport_by_league.get(league or "")
        match_id = str(prediction.get("match_id") or "")
        home, away = _team(prediction.get("home_team")), _team(prediction.get("away_team"))
        day = str(prediction.get("match_date") or "")[:10]
        try:
            match_day = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError:
            no_match += 1
            continue
        event_matches = []
        for quote in odds:
            if not sport_key or str(quote.get("sport_key") or "") != sport_key:
                continue
            if _team(quote.get("home_team")) != home or _team(quote.get("away_team")) != away:
                continue
            kickoff = _utc(quote.get("event_commence_time_utc"))
            snapshot = _utc(quote.get("snapshot_at_utc"))
            market_update = _utc(quote.get("market_last_update_utc"))
            if kickoff is None or snapshot is None or market_update is None:
                continue
            if abs((kickoff.date() - match_day).days) > 1:
                continue
            hours_before = (kickoff - snapshot).total_seconds() / 3600.0
            quote_age_at_snapshot = (snapshot - market_update).total_seconds() / 3600.0
            if (hours_before < 0 or hours_before > max_hours_before_kickoff
                    or quote_age_at_snapshot < 0 or quote_age_at_snapshot > max_quote_age_hours
                    or market_update > kickoff):
                continue
            event_matches.append((quote, kickoff, snapshot))
        if not event_matches:
            no_match += 1
            continue

        # Choose the last available snapshot per event/bookmaker/market/line before kickoff.
        latest: dict[tuple[str, str, str, str], tuple[Mapping[str, Any], datetime, datetime]] = {}
        for quote, kickoff, snapshot in event_matches:
            market = str(quote.get("market_key") or "")
            point = str(quote.get("market_point") or "")
            if market not in {"h2h", "totals"}:
                continue
            key = (
                str(quote.get("event_id") or ""),
                str(quote.get("bookmaker_key") or ""),
                market, point,
            )
            if key not in latest or snapshot > latest[key][2]:
                latest[key] = (quote, kickoff, snapshot)

        h2h_books: dict[str, dict[str, float]] = {}
        totals_books: dict[str, dict[str, float]] = {}
        for (event_id, bookmaker, market, point), (quote, _kickoff, _snapshot) in latest.items():
            if market == "totals":
                quoted_point = _number(point)
                if quoted_point is None or not math.isclose(quoted_point, total_goals_line, abs_tol=1e-9):
                    continue
            outcome_map = _outcomes(quote)
            required = {"home", "draw", "away"} if market == "h2h" else {"over", "under"}
            if not required.issubset(outcome_map):
                continue
            probability_values = {
                item: _number(outcome_map[item].get("no_vig_probability")) for item in required
            }
            if any(value is None or value < 0 or value > 1 for value in probability_values.values()):
                continue
            if market == "h2h":
                h2h_books[bookmaker] = {key: float(value) for key, value in probability_values.items()}
            else:
                totals_books[bookmaker] = {key: float(value) for key, value in probability_values.items()}

        comparison = dict(prediction)
        if h2h_books:
            for key in ("home", "draw", "away"):
                comparison[f"market_implied_{'home_win' if key == 'home' else 'away_win' if key == 'away' else 'draw'}"] = (
                    sum(book[key] for book in h2h_books.values()) / len(h2h_books)
                )
            comparison["market_implied_bookmakers_1x2"] = len(h2h_books)
            matches_h2h += 1
        if totals_books:
            comparison["market_implied_over"] = sum(book["over"] for book in totals_books.values()) / len(totals_books)
            comparison["market_implied_under"] = sum(book["under"] for book in totals_books.values()) / len(totals_books)
            comparison["market_implied_bookmakers_totals"] = len(totals_books)
            matches_totals += 1
        if h2h_books:
            comparison["market_match_status"] = "MATCHED_PRE_KICKOFF"
            comparisons.append(comparison)
        else:
            no_match += 1

    return comparisons, {
        "available": bool(matches_h2h),
        "input_odds_rows": len(odds),
        "prediction_rows": len(prediction_list),
        "matched_1x2_predictions": matches_h2h,
        "matched_totals_predictions": matches_totals,
        "unmatched_predictions": no_match,
        "max_hours_before_kickoff": max_hours_before_kickoff,
        "max_quote_age_hours": max_quote_age_hours,
        "total_goals_line": total_goals_line,
        "market_method": "Equal-weighted average of complete same-book de-vigged probabilities; exact team names/league key, exact totals line, and snapshots no later than kickoff.",
        "warning": "Market benchmarks are only valid when historical snapshots were licensed, imported, and captured before kickoff. Unmatched fixtures are excluded; no current price is backfilled into past matches.",
    }
