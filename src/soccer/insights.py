"""Soccer best-pick ranking and out-of-sample calibration diagnostics.

Only the five major European domestic leagues are eligible for pick ranking.
The score ranks model conviction with an explicit sample-support adjustment; it is not
an odds-based edge, expected value, or guarantee of profitability.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from src.soccer.odds import TOP_FIVE_SPORTS

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
    if ("bundesliga" in name or "germany bundesliga" in joined or "deutschland bundesliga" in joined
            or "germany 1 bundesliga" in joined or "deutschland 1 bundesliga" in joined):
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
            "market_line": _number(_get(row, "total_goals_line", 2.5)) if market_key in {"over", "under"} else None,
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
                "model": model, "match_id": str(_get(row, "match_id", "") or (comp_id + ":" + str(_get(row, "match_date", "") or ""))),
                "match_date": str(_get(row, "match_date", "") or ""),
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
        "matches_evaluated": len({item["match_id"] for item in selected}),
        "model_reports": model_reports,
        "warning": "Calibration describes historical out-of-sample predictions only. It does not establish profitability or future accuracy.",
    }



def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _team_match_key(value: Any) -> str:
    """Conservative exact-name normalization; no fuzzy team-name matching."""
    cleaned = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    for suffix in ("footballclub", "soccerclub", "fc", "afc"):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 3:
            cleaned = cleaned[:-len(suffix)]
            break
    return cleaned


def attach_market_odds(
    picks: Iterable[Mapping[str, Any]],
    odds_rows: Iterable[Any],
    *,
    now_utc: datetime | None = None,
    max_quote_age_minutes: float = 120.0,
    max_snapshot_age_minutes: float = 120.0,
    max_hours_before_kickoff: float = 168.0,
) -> list[dict[str, Any]]:
    """Attach only exact-team, same-book, complete and fresh pre-kickoff odds.

    Expected market probability is de-vigged using every outcome in that exact bookmaker
    market. Expected value per unit is experimental (model_probability * decimal_price - 1);
    it must not be treated as calibrated or guaranteed.
    """
    for value, label in (
        (max_quote_age_minutes, "max_quote_age_minutes"),
        (max_snapshot_age_minutes, "max_snapshot_age_minutes"),
        (max_hours_before_kickoff, "max_hours_before_kickoff"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    odds = [dict(row) if isinstance(row, Mapping) else vars(row) for row in odds_rows]
    results: list[dict[str, Any]] = []
    sport_key_by_league = {name: config[0] for _selector, config in TOP_FIVE_SPORTS.items()
                           for name in [config[1]]}
    for raw_pick in picks:
        pick = dict(raw_pick)
        pick["market_probability"] = None
        pick["best_odds_decimal"] = None
        pick["best_odds_bookmaker"] = None
        pick["odds_quote_time_utc"] = None
        pick["odds_snapshot_time_utc"] = None
        pick["odds_age_minutes"] = None
        pick["market_edge_probability"] = None
        pick["expected_value_per_unit"] = None
        pick["odds_status"] = "NOT IMPORTED" if not odds else "NO EXACT MATCH"
        pick["odds_available"] = False

        league_key = sport_key_by_league.get(str(pick.get("league") or ""))
        home_key = _team_match_key(pick.get("home_team"))
        away_key = _team_match_key(pick.get("away_team"))
        pick_date = str(pick.get("match_date") or "")[:10]
        event_rows: list[dict[str, Any]] = []
        pre_kickoff_failed = False
        for item in odds:
            if league_key and str(item.get("sport_key") or "") != league_key:
                continue
            if _team_match_key(item.get("home_team")) != home_key or _team_match_key(item.get("away_team")) != away_key:
                continue
            event_time = _parse_utc(item.get("event_commence_time_utc"))
            if event_time is None:
                continue
            if event_time <= now:
                pre_kickoff_failed = True
                continue
            try:
                odds_date = event_time.date()
                target_date = datetime.strptime(pick_date, "%Y-%m-%d").date()
            except (TypeError, ValueError):
                continue
            if abs((odds_date - target_date).days) > 1:
                continue
            event_rows.append(item)
        if not event_rows and pre_kickoff_failed:
            pick["odds_status"] = "STALE OR NOT PRE-KICKOFF"

        candidates = []
        stale_match_seen = False
        for item in event_rows:
            market_key = str(item.get("market_key") or "")
            pick_market_key = str(pick.get("market_key") or "")
            if pick_market_key in {"home_win", "draw", "away_win"}:
                if market_key != "h2h":
                    continue
                target_key = pick_market_key.replace("_win", "")
                if target_key == "home":
                    target_key = "home"
            elif pick_market_key in {"over", "under"}:
                if market_key != "totals":
                    continue
                item_point = _number(item.get("market_point"))
                pick_point = _number(pick.get("market_line"))
                if item_point is None or pick_point is None or not math.isclose(item_point, pick_point, abs_tol=1e-9):
                    continue
                target_key = pick_market_key
            else:
                continue

            snapshot_time = _parse_utc(item.get("snapshot_at_utc"))
            quote_time = _parse_utc(item.get("market_last_update_utc"))
            kickoff = _parse_utc(item.get("event_commence_time_utc"))
            if snapshot_time is None or quote_time is None or kickoff is None:
                stale_match_seen = True
                continue
            quote_age = (now - quote_time).total_seconds() / 60.0
            snapshot_age = (now - snapshot_time).total_seconds() / 60.0
            hours_to_kickoff = (kickoff - now).total_seconds() / 3600.0
            if (quote_age < 0 or snapshot_age < 0 or hours_to_kickoff <= 0
                    or hours_to_kickoff > max_hours_before_kickoff
                    or quote_age > max_quote_age_minutes or snapshot_age > max_snapshot_age_minutes
                    or quote_time > snapshot_time or quote_time > kickoff or snapshot_time > kickoff):
                stale_match_seen = True
                continue
            try:
                outcome_rows = json.loads(str(item.get("outcomes_json") or "[]"))
            except (ValueError, TypeError):
                stale_match_seen = True
                continue
            by_key = {str(outcome.get("key") or ""): outcome for outcome in outcome_rows if isinstance(outcome, Mapping)}
            expected_keys = {"home", "draw", "away"} if market_key == "h2h" else {"over", "under"}
            if not expected_keys.issubset(by_key):
                continue
            target = by_key.get(target_key)
            if not target:
                continue
            price = _number(target.get("price_decimal"))
            fair = _number(target.get("no_vig_probability"))
            model_probability = _number(pick.get("model_probability"))
            if price is None or price <= 1 or fair is None or model_probability is None:
                continue
            candidates.append({
                "price": price,
                "fair_probability": fair,
                "edge": model_probability - fair,
                "ev": model_probability * price - 1.0,
                "bookmaker": str(item.get("bookmaker_title") or item.get("bookmaker_key") or "Unknown"),
                "quote_time": quote_time,
                "snapshot_time": snapshot_time,
                "quote_age": quote_age,
            })
        if candidates:
            best = max(candidates, key=lambda candidate: (candidate["ev"], candidate["edge"], candidate["price"]))
            pick.update({
                "market_probability": best["fair_probability"],
                "best_odds_decimal": best["price"],
                "best_odds_bookmaker": best["bookmaker"],
                "odds_quote_time_utc": best["quote_time"].isoformat(timespec="seconds").replace("+00:00", "Z"),
                "odds_snapshot_time_utc": best["snapshot_time"].isoformat(timespec="seconds").replace("+00:00", "Z"),
                "odds_age_minutes": best["quote_age"],
                "market_edge_probability": best["edge"],
                "expected_value_per_unit": best["ev"],
                "odds_status": "EXPERIMENTAL POSITIVE EV" if best["ev"] > 0 and best["edge"] > 0 else "MATCHED · NO POSITIVE SIGNAL",
                "odds_available": True,
            })
        elif stale_match_seen:
            pick["odds_status"] = "STALE OR NOT PRE-KICKOFF"
        elif event_rows:
            pick["odds_status"] = "NO EXACT MARKET / LINE"
        results.append(pick)

    # When verified time-aligned quotes exist, surface positive model signals first;
    # otherwise preserve Phase 5's history-weighted model rank.
    results.sort(key=lambda pick: (
        0 if _number(pick.get("expected_value_per_unit")) is not None and _number(pick.get("market_edge_probability")) is not None
             and _number(pick.get("expected_value_per_unit")) > 0 and _number(pick.get("market_edge_probability")) > 0 else 1,
        -(_number(pick.get("expected_value_per_unit")) if _number(pick.get("expected_value_per_unit")) is not None else -999.0),
        -(_number(pick.get("rank_score")) if _number(pick.get("rank_score")) is not None else 0.0),
        str(pick.get("match_date") or ""),
        str(pick.get("league") or ""),
        str(pick.get("home_team") or ""),
    ))
    for index, pick in enumerate(results, start=1):
        pick["rank"] = index
    return results
