"""Public NBA schedule helpers for player-team and next-game metadata.

Uses the NBA CDN schedule JSON (no API key). Missing or unreachable schedule data
must not break projections; callers can show an unavailable match date instead.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from src.data.seasons import current_season

log = logging.getLogger(__name__)

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
SCHEDULE_URLS = (
    SCHEDULE_URL,
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json",
)
CACHE_PATH = Path(".cache/nba/scheduleLeagueV2.json")
CACHE_TTL_SECONDS = 6 * 60 * 60
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
}



# ESPN's public scoreboard is the primary fallback when the NBA CDN is blocked
# by a network, region, or upstream change. Query individual days; range queries
# have not been reliable. Only a short rolling window is needed to find each
# team's next game.
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_CACHE_PATH = Path(".cache/nba/espn_scoreboards.json")
ESPN_CACHE_TTL_SECONDS = 6 * 60 * 60
ESPN_LOOKAHEAD_DAYS = 21
ESPN_HEADERS = {
    "User-Agent": HEADERS["User-Agent"],
    "Accept": "application/json, text/plain, */*",
}
ESPN_ABBREVIATION_ALIASES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "WSH": "WAS",
    "BRK": "BKN",
}


def _espn_abbreviation(team: dict[str, Any]) -> str:
    raw = str(team.get("abbreviation") or team.get("teamTricode") or "").strip().upper()
    return ESPN_ABBREVIATION_ALIASES.get(raw, raw)


def _espn_team_name(team: dict[str, Any]) -> str:
    display = str(team.get("displayName") or "").strip()
    if display:
        return display
    city = str(team.get("location") or team.get("teamCity") or "").strip()
    name = str(team.get("name") or team.get("teamName") or "").strip()
    return " ".join(part for part in (city, name) if part) or _espn_abbreviation(team)


def _espn_event_day(value: Any) -> date | None:
    """Convert ESPN's UTC start timestamp to the NBA-local calendar date."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo("America/New_York")).date()
    except (ValueError, ZoneInfoNotFoundError, OSError):
        return _as_date(raw)


def build_espn_next_game_map(
    payloads: list[dict[str, Any]],
    as_of: date | None = None,
) -> dict[str, dict[str, str]]:
    """Normalize a set of ESPN daily scoreboard payloads to the same team map."""
    as_of = as_of or _current_nba_date()
    candidates: dict[str, list[dict[str, str]]] = {}

    for payload in payloads:
        events = payload.get("events", []) if isinstance(payload, dict) else []
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            game_day = _espn_event_day(event.get("date"))
            if game_day is None or game_day < as_of:
                continue

            status = event.get("status") or {}
            status_type = status.get("type") or {}
            state = str(status_type.get("state") or "").strip().casefold()
            completed = status_type.get("completed") is True or str(
                status_type.get("completed") or ""
            ).strip().casefold() == "true"
            if completed or state in {"in", "post"}:
                continue

            competitions = event.get("competitions") or []
            competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
            competitors = competition.get("competitors") or []
            if not isinstance(competitors, list):
                continue
            home = next(
                (item.get("team", {}) for item in competitors
                 if isinstance(item, dict) and str(item.get("homeAway", "")).casefold() == "home"),
                {},
            )
            away = next(
                (item.get("team", {}) for item in competitors
                 if isinstance(item, dict) and str(item.get("homeAway", "")).casefold() == "away"),
                {},
            )
            if not isinstance(home, dict) or not isinstance(away, dict):
                continue

            home_abbr = _espn_abbreviation(home)
            away_abbr = _espn_abbreviation(away)
            if not home_abbr or not away_abbr:
                continue
            home_name = _espn_team_name(home)
            away_name = _espn_team_name(away)
            day = game_day.isoformat()

            candidates.setdefault(home_abbr, []).append({
                "team": home_name,
                "team_abbreviation": home_abbr,
                "match_date": day,
                "opponent_team": away_name,
                "opponent_abbreviation": away_abbr,
                "matchup": f"{home_abbr} vs {away_abbr}",
                "home_away": "HOME",
            })
            candidates.setdefault(away_abbr, []).append({
                "team": away_name,
                "team_abbreviation": away_abbr,
                "match_date": day,
                "opponent_team": home_name,
                "opponent_abbreviation": home_abbr,
                "matchup": f"{away_abbr} @ {home_abbr}",
                "home_away": "AWAY",
            })

    result: dict[str, dict[str, str]] = {}
    for team, games in candidates.items():
        games.sort(key=lambda game: (game["match_date"], game["matchup"]))
        result[team] = games[0]
    return result


def _read_espn_cache() -> list[dict[str, Any]] | None:
    try:
        if ESPN_CACHE_PATH.exists():
            with ESPN_CACHE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload.get("payloads") if isinstance(payload, dict) else None
            if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                return rows
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read ESPN scoreboard cache: %s", exc)
    return None


def _fetch_espn_scoreboards(as_of: date) -> tuple[list[dict[str, Any]], int]:
    """Fetch individual daily scoreboards concurrently and return valid payloads."""
    days = [as_of.fromordinal(as_of.toordinal() + offset) for offset in range(ESPN_LOOKAHEAD_DAYS)]
    payloads_by_day: dict[date, dict[str, Any]] = {}
    errors: list[str] = []

    def fetch_day(day: date) -> tuple[date, dict[str, Any]]:
        response = requests.get(
            ESPN_SCOREBOARD_URL,
            params={"dates": day.strftime("%Y%m%d"), "limit": "500"},
            headers=ESPN_HEADERS,
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
            raise ValueError("ESPN response does not contain an events array")
        return day, payload

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_day, day): day for day in days}
        for future in as_completed(futures):
            day = futures[future]
            try:
                returned_day, payload = future.result()
                payloads_by_day[returned_day] = payload
            except Exception as exc:
                errors.append(f"{day.isoformat()}: {exc}")

    payloads = [payloads_by_day[day] for day in days if day in payloads_by_day]
    log.info(
        "ESPN NBA scoreboard responses: %s/%s days successful",
        len(payloads), len(days),
    )
    if not payloads:
        summary = "; ".join(errors[:3]) or "no valid scoreboard responses"
        raise RuntimeError(f"ESPN scoreboard unavailable: {summary}")
    if errors:
        log.warning("some ESPN schedule dates failed: %s", "; ".join(errors[:3]))
    return payloads, len(payloads)


def _write_espn_cache(payloads: list[dict[str, Any]]) -> None:
    ESPN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ESPN_CACHE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps({"payloads": payloads}), encoding="utf-8")
    temp_path.replace(ESPN_CACHE_PATH)


def _current_nba_date() -> date:
    """Return today's date in US Eastern time, where NBA game dates are anchored."""
    try:
        return datetime.now(ZoneInfo("America/New_York")).date()
    except (ZoneInfoNotFoundError, OSError):
        return date.today()


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    # The NBA CDN may use ISO dates, ISO datetimes, or MM/DD/YYYY strings.
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        pass
    for fmt in (
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _team_abbreviation(team: dict[str, Any]) -> str:
    return str(team.get("teamTricode") or team.get("teamAbbreviation") or team.get("teamCode") or "").strip().upper()


def _team_display_name(team: dict[str, Any]) -> str:
    city = str(team.get("teamCity") or team.get("city") or "").strip()
    name = str(team.get("teamName") or team.get("name") or "").strip()
    full_name = " ".join(part for part in (city, name) if part)
    return full_name or _team_abbreviation(team)


def build_next_game_map(payload: dict[str, Any], as_of: date | None = None) -> dict[str, dict[str, str]]:
    """Map each team tricode to its next scheduled game on/after as_of."""
    as_of = as_of or _current_nba_date()
    league = payload.get("leagueSchedule", payload)
    date_blocks = league.get("gameDates", []) if isinstance(league, dict) else []
    candidates: dict[str, list[dict[str, str]]] = {}

    for block in date_blocks:
        if not isinstance(block, dict):
            continue
        block_date = _as_date(block.get("gameDate") or block.get("gameDateEst"))
        games = block.get("games", [])
        if not isinstance(games, list):
            continue
        for game in games:
            if not isinstance(game, dict):
                continue
            home = game.get("homeTeam") or {}
            away = game.get("awayTeam") or {}
            if not isinstance(home, dict) or not isinstance(away, dict):
                continue

            game_day = (
                _as_date(game.get("gameDateEst"))
                or _as_date(game.get("gameDate"))
                or _as_date(block.get("gameDateEst"))
                or block_date
            )
            if game_day is None or game_day < as_of:
                continue

            status = game.get("gameStatus")
            status_text = str(game.get("gameStatusText") or "").strip().casefold()
            try:
                status_num = int(status) if status is not None else None
            except (TypeError, ValueError):
                status_num = None
            if status_num == 3 or status_text in {"final", "final/ot", "final/2ot"} or status_text.startswith("final "):
                continue

            home_abbr = _team_abbreviation(home)
            away_abbr = _team_abbreviation(away)
            if not home_abbr or not away_abbr:
                continue

            home_name = _team_display_name(home)
            away_name = _team_display_name(away)
            day = game_day.isoformat()
            candidates.setdefault(home_abbr, []).append({
                "team": home_name,
                "team_abbreviation": home_abbr,
                "match_date": day,
                "opponent_team": away_name,
                "opponent_abbreviation": away_abbr,
                "matchup": f"{home_abbr} vs {away_abbr}",
                "home_away": "HOME",
            })
            candidates.setdefault(away_abbr, []).append({
                "team": away_name,
                "team_abbreviation": away_abbr,
                "match_date": day,
                "opponent_team": home_name,
                "opponent_abbreviation": home_abbr,
                "matchup": f"{away_abbr} @ {home_abbr}",
                "home_away": "AWAY",
            })

    result: dict[str, dict[str, str]] = {}
    for team, games in candidates.items():
        games.sort(key=lambda game: (game["match_date"], game["matchup"]))
        result[team] = games[0]
    return result


def _matches_current_season(payload: dict[str, Any]) -> bool:
    league = payload.get("leagueSchedule", {})
    season_value = str(league.get("seasonYear") or "").strip()
    if not season_value:
        return True
    actual_start = season_value[:4]
    expected_start = current_season().split("-", 1)[0]
    return actual_start == expected_start


def _read_cached_schedule() -> dict[str, Any] | None:
    try:
        if CACHE_PATH.exists():
            with CACHE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and isinstance(payload.get("leagueSchedule"), dict):
                return payload
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read NBA schedule cache: %s", exc)
    return None


def get_next_game_map(
    as_of: date | None = None,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Get next matchups using ESPN's scoreboard, then the NBA CDN as backup."""
    as_of = as_of or _current_nba_date()
    partial_espn_map: dict[str, dict[str, str]] = {}
    espn_payloads: list[dict[str, Any]] | None = None
    cache_fresh = (
        not force_refresh
        and ESPN_CACHE_PATH.exists()
        and (datetime.now().timestamp() - ESPN_CACHE_PATH.stat().st_mtime) < ESPN_CACHE_TTL_SECONDS
    )
    if cache_fresh:
        espn_payloads = _read_espn_cache()
        if espn_payloads is not None:
            cached_map = build_espn_next_game_map(espn_payloads, as_of=as_of)
            if len(cached_map) >= 20:
                log.info("loaded next scheduled games for %s NBA teams from ESPN cache", len(cached_map))
                return cached_map
            partial_espn_map = cached_map
            log.warning(
                "fresh ESPN cache only maps %s teams; trying a fresh schedule lookup",
                len(cached_map),
            )

    if not cache_fresh or force_refresh or not partial_espn_map:
        try:
            fresh_payloads, successful_days = _fetch_espn_scoreboards(as_of)
            fresh_map = build_espn_next_game_map(fresh_payloads, as_of=as_of)
            if len(fresh_map) >= 20:
                if successful_days >= 18:
                    _write_espn_cache(fresh_payloads)
                log.info("loaded next scheduled games for %s NBA teams from ESPN", len(fresh_map))
                return fresh_map
            if len(fresh_map) > len(partial_espn_map):
                partial_espn_map = fresh_map
            log.warning(
                "ESPN scoreboard only mapped %s NBA teams across the next %s days",
                len(fresh_map), ESPN_LOOKAHEAD_DAYS,
            )
            if successful_days >= 18 and fresh_map:
                _write_espn_cache(fresh_payloads)
        except Exception as exc:
            log.warning("could not load ESPN NBA schedule fallback: %s", exc)

    stale_payload = _read_cached_schedule()
    if stale_payload is not None and not _matches_current_season(stale_payload):
        log.info("cached NBA schedule is for a previous season; refreshing it")
        stale_payload = None
    nba_cache_fresh = (
        not force_refresh
        and stale_payload is not None
        and CACHE_PATH.exists()
        and (datetime.now().timestamp() - CACHE_PATH.stat().st_mtime) < CACHE_TTL_SECONDS
    )
    payload = stale_payload if nba_cache_fresh else None
    last_error: Exception | None = None

    if payload is None:
        for url in SCHEDULE_URLS:
            try:
                response = requests.get(url, headers=HEADERS, timeout=12)
                response.raise_for_status()
                fresh = response.json()
                if (
                    not isinstance(fresh, dict)
                    or not isinstance(fresh.get("leagueSchedule"), dict)
                    or not isinstance(fresh["leagueSchedule"].get("gameDates"), list)
                    or not fresh["leagueSchedule"].get("gameDates")
                ):
                    raise ValueError("NBA CDN response did not contain non-empty leagueSchedule.gameDates")
                if not _matches_current_season(fresh):
                    raise ValueError(
                        f"{url} returned schedule season {fresh.get('leagueSchedule', {}).get('seasonYear')!r}, "
                        f"not current season {current_season()}"
                    )
                payload = fresh
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                temp_path = CACHE_PATH.with_suffix(".tmp")
                temp_path.write_text(json.dumps(fresh), encoding="utf-8")
                temp_path.replace(CACHE_PATH)
                break
            except (requests.RequestException, ValueError, OSError) as exc:
                last_error = exc
                log.warning("could not use NBA schedule endpoint %s: %s", url, exc)

    if payload is not None:
        nba_map = build_next_game_map(payload, as_of=as_of)
        if nba_map:
            combined = dict(partial_espn_map)
            combined.update(nba_map)
            log.info("loaded next scheduled games for %s NBA teams from NBA CDN", len(nba_map))
            return combined

    if partial_espn_map:
        log.warning(
            "using partial ESPN matchup map for %s teams; some teams may not have upcoming schedule rows",
            len(partial_espn_map),
        )
        return partial_espn_map

    if payload is None:
        if stale_payload is not None:
            log.warning("could not refresh NBA schedule; reusing cached schedule: %s", last_error)
            stale_payload = _read_cached_schedule()
            if stale_payload is not None:
                fallback = build_next_game_map(stale_payload, as_of=as_of)
                if fallback:
                    return fallback
        raise RuntimeError(
            "No upcoming NBA schedule could be loaded from ESPN or the NBA CDN. "
            f"Last NBA CDN error: {last_error}"
        )

    log.warning("schedule feeds returned no upcoming NBA matchups on or after %s", as_of.isoformat())
    return {}

