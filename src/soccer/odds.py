"""Timestamped football odds snapshots and same-book no-vig normalization.

Provider: The Odds API v4. API keys are read from THE_ODDS_API_KEY or the local ignored
.env file; the key is never written to data files or included in logged error messages.
Historical snapshots are a separately billed/plan-gated API feature.
"""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

PROVIDER = "the_odds_api"
BASE_URL = "https://api.the-odds-api.com/v4"
TOP_FIVE_SPORTS = {
    "epl": ("soccer_epl", "Premier League"),
    "la_liga": ("soccer_spain_la_liga", "La Liga"),
    "bundesliga": ("soccer_germany_bundesliga", "Bundesliga"),
    "serie_a": ("soccer_italy_serie_a", "Serie A"),
    "ligue_1": ("soccer_france_ligue_one", "Ligue 1"),
}
ODDS_COLUMNS = [
    "provider", "sport_key", "competition_name", "event_id", "home_team", "away_team",
    "event_commence_time_utc", "bookmaker_key", "bookmaker_title", "market_key",
    "market_point", "market_last_update_utc", "snapshot_at_utc", "requested_snapshot_at_utc",
    "snapshot_mode", "outcomes_json", "implied_probability_sum", "overround",
    "source_url", "retrieved_at_utc",
]


class OddsProviderError(RuntimeError):
    """Sanitized odds-provider failure; never includes the request URL or API key."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _aware_iso(value: str, field: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp with a timezone, e.g. 2026-10-24T12:00:00Z") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone (UTC recommended)")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_local_env_key() -> str | None:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return None
    try:
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "THE_ODDS_API_KEY":
                value = value.strip().strip("\"'").strip()
                return value or None
    except OSError:
        return None
    return None


def get_api_key(api_key: str | None = None) -> str:
    value = (api_key or os.getenv("THE_ODDS_API_KEY") or _read_local_env_key() or "").strip()
    if not value:
        raise OddsProviderError(
            "Missing THE_ODDS_API_KEY. Set it in the current PowerShell session or in an ignored local .env file."
        )
    return value


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def decimal_implied_probability(decimal_odds: float) -> float:
    if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
        raise ValueError("decimal odds must be finite and greater than 1.0")
    return 1.0 / decimal_odds


def _canonical_outcome(name: str, home: str, away: str, market_key: str) -> str:
    norm = " ".join(str(name or "").casefold().replace(".", " ").split())
    home_norm = " ".join(home.casefold().replace(".", " ").split())
    away_norm = " ".join(away.casefold().replace(".", " ").split())
    if market_key == "h2h":
        if norm == home_norm:
            return "home"
        if norm == away_norm:
            return "away"
        if norm in {"draw", "tie", "x"}:
            return "draw"
    if market_key == "totals":
        if norm == "over":
            return "over"
        if norm == "under":
            return "under"
    return norm


def _normalize_outcomes(
    outcomes: list[Mapping[str, Any]], home: str, away: str, market_key: str, point: float | None = None
) -> tuple[list[dict[str, Any]], float, float] | None:
    parsed: list[dict[str, Any]] = []
    for outcome in outcomes:
        name = str(outcome.get("name") or "").strip()
        price = _float(outcome.get("price"))
        if not name or price is None or price <= 1:
            continue
        outcome_point = _float(outcome.get("point"))
        if market_key == "totals":
            if outcome_point is None or point is None or not math.isclose(outcome_point, point, abs_tol=1e-9):
                continue
            key = _canonical_outcome(name, home, away, market_key)
        else:
            key = _canonical_outcome(name, home, away, market_key)
        parsed.append({"name": name, "key": key, "point": outcome_point, "price_decimal": price,
                       "implied_probability": decimal_implied_probability(price)})
    required = {"home", "draw", "away"} if market_key == "h2h" else {"over", "under"}
    keys = [row["key"] for row in parsed]
    if set(keys) != required or len(keys) != len(required):
        return None
    total = sum(row["implied_probability"] for row in parsed)
    if not math.isfinite(total) or total <= 0:
        return None
    for row in parsed:
        row["no_vig_probability"] = row["implied_probability"] / total
    return parsed, total, total - 1.0


def _market_records(
    event: Mapping[str, Any],
    snapshot_at_utc: str,
    requested_snapshot_at_utc: str,
    snapshot_mode: str,
    sport_key: str,
    competition_name: str,
    source_url: str,
    retrieved_at_utc: str,
) -> list[dict[str, Any]]:
    event_id = str(event.get("id") or "").strip()
    home = str(event.get("home_team") or "").strip()
    away = str(event.get("away_team") or "").strip()
    commence = event.get("commence_time")
    if not (event_id and home and away and commence):
        return []
    try:
        commence = _aware_iso(str(commence), "commence_time")
    except ValueError:
        return []
    records: list[dict[str, Any]] = []
    for bookmaker in event.get("bookmakers") or []:
        book_key = str(bookmaker.get("key") or "").strip()
        book_title = str(bookmaker.get("title") or book_key).strip()
        for market in bookmaker.get("markets") or []:
            market_key = str(market.get("key") or "").strip()
            updated = market.get("last_update") or bookmaker.get("last_update")
            if market_key not in {"h2h", "totals"} or not updated:
                continue
            try:
                updated = _aware_iso(str(updated), "market last_update")
            except ValueError:
                continue
            outcomes = market.get("outcomes") or []
            point_values = sorted({p for p in (_float(o.get("point")) for o in outcomes) if p is not None})
            points: list[float | None] = [None] if market_key == "h2h" else point_values
            for point in points:
                normalized = _normalize_outcomes(outcomes, home, away, market_key, point)
                if normalized is None:
                    continue
                parsed, total, overround = normalized
                records.append({
                    "provider": PROVIDER,
                    "sport_key": sport_key,
                    "competition_name": competition_name,
                    "event_id": event_id,
                    "home_team": home,
                    "away_team": away,
                    "event_commence_time_utc": commence,
                    "bookmaker_key": book_key,
                    "bookmaker_title": book_title,
                    "market_key": market_key,
                    "market_point": "" if point is None else f"{point:g}",
                    "market_last_update_utc": updated,
                    "snapshot_at_utc": snapshot_at_utc,
                    "requested_snapshot_at_utc": requested_snapshot_at_utc,
                    "snapshot_mode": snapshot_mode,
                    "outcomes_json": json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
                    "implied_probability_sum": f"{total:.10f}",
                    "overround": f"{overround:.10f}",
                    "source_url": source_url,
                    "retrieved_at_utc": retrieved_at_utc,
                })
    return records


def parse_odds_payload(
    payload: Any,
    *,
    sport_key: str,
    competition_name: str,
    snapshot_mode: str = "live",
    requested_snapshot_at_utc: str = "",
    retrieved_at_utc: str | None = None,
    source_url: str | None = None,
) -> list[dict[str, Any]]:
    """Convert live or historical The Odds API response payload to one row per book-market-line."""
    if snapshot_mode not in {"live", "historical"}:
        raise ValueError("snapshot_mode must be live or historical")
    retrieved = retrieved_at_utc or _utc_now()
    requested = _aware_iso(requested_snapshot_at_utc, "requested_snapshot_at_utc") if requested_snapshot_at_utc else ""
    if snapshot_mode == "historical":
        if not isinstance(payload, Mapping):
            raise OddsProviderError("Historical odds response was not a JSON object.")
        actual_snapshot = payload.get("timestamp")
        events = payload.get("data")
        if not actual_snapshot or not isinstance(events, list):
            raise OddsProviderError("Historical odds response is missing its snapshot timestamp or data list.")
        snapshot_at = _aware_iso(str(actual_snapshot), "historical response timestamp")
    else:
        events = payload
        if not isinstance(events, list):
            raise OddsProviderError("Live odds response was not a JSON list.")
        snapshot_at = retrieved
    url = source_url or f"{BASE_URL}/sports/{sport_key}/odds/"
    rows: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, Mapping):
            rows.extend(_market_records(
                event, snapshot_at, requested, snapshot_mode, sport_key, competition_name, url, retrieved
            ))
    return rows


def fetch_league_odds(
    league: str,
    *,
    regions: str = "uk,eu",
    markets: str = "h2h,totals",
    historical_date: str | None = None,
    api_key: str | None = None,
    session: Any = requests,
    timeout_seconds: int = 25,
) -> list[dict[str, Any]]:
    """Fetch one top-five league from the live or paid historical odds endpoint."""
    if league not in TOP_FIVE_SPORTS:
        raise ValueError(f"Unsupported top-five league selector: {league}")
    valid_regions = {"us", "us2", "uk", "eu", "au"}
    region_items = [item.strip() for item in regions.split(",") if item.strip()]
    if not region_items or any(item not in valid_regions for item in region_items):
        raise ValueError("regions must be a comma-separated selection of us, us2, uk, eu, au")
    market_items = [item.strip() for item in markets.split(",") if item.strip()]
    if not market_items or any(item not in {"h2h", "totals"} for item in market_items):
        raise ValueError("markets currently supports h2h and totals only")
    sport_key, competition_name = TOP_FIVE_SPORTS[league]
    key = get_api_key(api_key)
    retrieved = _utc_now()
    historical_date_utc = _aware_iso(historical_date, "historical_date") if historical_date else None
    if historical_date_utc:
        endpoint = f"{BASE_URL}/historical/sports/{sport_key}/odds/"
        params = {
            "apiKey": key, "regions": ",".join(region_items), "markets": ",".join(market_items),
            "oddsFormat": "decimal", "dateFormat": "iso", "date": historical_date_utc,
        }
        mode = "historical"
    else:
        endpoint = f"{BASE_URL}/sports/{sport_key}/odds/"
        params = {
            "apiKey": key, "regions": ",".join(region_items), "markets": ",".join(market_items),
            "oddsFormat": "decimal", "dateFormat": "iso",
        }
        mode = "live"
    try:
        response = session.get(endpoint, params=params, timeout=timeout_seconds)
        if response.status_code >= 400:
            status = response.status_code
            if status == 401:
                message = "The Odds API rejected the key (HTTP 401). Check THE_ODDS_API_KEY."
            elif status == 403:
                message = "The Odds API denied this sport/market for the current subscription (HTTP 403)."
            elif status == 429:
                message = "The Odds API quota or rate limit was reached (HTTP 429)."
            else:
                message = f"The Odds API request failed with HTTP {status}."
            raise OddsProviderError(message)
        payload = response.json()
    except OddsProviderError:
        raise
    except requests.RequestException as exc:
        raise OddsProviderError(f"The Odds API request failed ({type(exc).__name__}); URL and key were hidden.") from None
    except (ValueError, json.JSONDecodeError):
        raise OddsProviderError("The Odds API returned invalid JSON.") from None
    if mode == "historical":
        return parse_odds_payload(
            payload, sport_key=sport_key, competition_name=competition_name,
            snapshot_mode="historical", requested_snapshot_at_utc=historical_date_utc,
            retrieved_at_utc=retrieved, source_url=endpoint,
        )
    return parse_odds_payload(
        payload, sport_key=sport_key, competition_name=competition_name,
        snapshot_mode="live", retrieved_at_utc=retrieved, source_url=endpoint,
    )


def upsert_odds_snapshots(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append immutable snapshots, skipping only exact repeated provider/source records."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    if output.exists() and output.stat().st_size:
        with output.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                normalized = {column: row.get(column, "") for column in ODDS_COLUMNS}
                key = tuple(str(normalized.get(name, "")) for name in (
                    "provider", "sport_key", "event_id", "bookmaker_key", "market_key",
                    "market_point", "market_last_update_utc", "snapshot_at_utc",
                ))
                merged[key] = normalized
    for row in rows:
        normalized = {column: row.get(column, "") for column in ODDS_COLUMNS}
        key = tuple(str(normalized.get(name, "")) for name in (
            "provider", "sport_key", "event_id", "bookmaker_key", "market_key",
            "market_point", "market_last_update_utc", "snapshot_at_utc",
        ))
        merged[key] = normalized
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ODDS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(merged):
            writer.writerow(merged[key])
    return len(merged)


def read_odds_snapshots(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return []
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
