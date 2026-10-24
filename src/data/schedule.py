"""Public NBA schedule helpers for player-team and next-game metadata.

Uses the NBA CDN schedule JSON (no API key). Missing or unreachable schedule data
must not break projections; callers can show an unavailable match date instead.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from src.data.seasons import current_season

log = logging.getLogger(__name__)

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
SCHEDULE_URLS = (
    SCHEDULE_URL,
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
)
CACHE_PATH = Path(".cache/nba/scheduleLeagueV2_1.json")
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
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
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
    """Fetch/cache the NBA CDN schedule and return the next matchup for each team."""
    stale_payload = _read_cached_schedule()
    if stale_payload is not None and not _matches_current_season(stale_payload):
        log.info("cached NBA schedule is for a previous season; refreshing it")
        stale_payload = None
    cache_fresh = (
        not force_refresh
        and stale_payload is not None
        and CACHE_PATH.exists()
        and (datetime.now().timestamp() - CACHE_PATH.stat().st_mtime) < CACHE_TTL_SECONDS
    )
    payload = stale_payload if cache_fresh else None
    last_error: Exception | None = None

    if payload is None:
        for url in SCHEDULE_URLS:
            try:
                response = requests.get(url, headers=HEADERS, timeout=25)
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

        if payload is None:
            if stale_payload is None:
                raise RuntimeError(f"Could not retrieve the current public NBA schedule: {last_error}")
            log.warning("could not refresh public NBA schedule; using stale cache: %s", last_error)
            payload = stale_payload

    return build_next_game_map(payload, as_of=as_of)
