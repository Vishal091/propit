"""OpenFootball public-domain fixture/result importer.

OpenFootball provides plain-text fixtures and results, not xG, odds, or guaranteed
UTC kickoff times. Keep those fields null rather than infer them.
Source: https://github.com/openfootball/england (public-domain football.db datasets).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from src.soccer.providers.http_cache import CachedHttpClient
from src.soccer.schemas import SoccerMatch, utc_now_iso


OPENFOOTBALL_REPOS = {
    "england": {
        "repo": "england", "path_template": "{season}/1-premierleague.txt",
        "competition_id": "openfootball:england:premierleague",
        "competition_name": "English Premier League", "country": "England",
    },
    "germany": {
        "repo": "deutschland", "path_template": "{season}/1-bundesliga.txt",
        "competition_id": "openfootball:germany:bundesliga",
        "competition_name": "German Bundesliga", "country": "Germany",
    },
    "spain": {
        "repo": "espana", "path_template": "{season}/1-liga.txt",
        "competition_id": "openfootball:spain:laliga",
        "competition_name": "Spanish La Liga", "country": "Spain",
    },
    "italy": {
        "repo": "italy", "path_template": "{season}/1-seriea.txt",
        "competition_id": "openfootball:italy:seriea",
        "competition_name": "Italian Serie A", "country": "Italy",
    },
    "france": {
        "repo": "europe", "path_template": "france/{season}_fr1.txt",
        "competition_id": "openfootball:france:ligue1",
        "competition_name": "French Ligue 1", "country": "France",
    },
}
_DATE_RE = re.compile(r"^\s*([A-Za-z]{3})\s+([A-Za-z]{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
_TIME_RE = re.compile(r"^(?P<time>\d{1,2}:\d{2})\s+(?P<body>.+)$")
_SCORE_RE = re.compile(r"\s+(?P<hg>\d+)-(?P<ag>\d+)(?:\s+\((?P<hthg>\d+)-(?P<htag>\d+)\))?\s+")
_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_match_line(line: str) -> tuple[str, str, str | None, int | None, int | None] | None:
    """Handle both upcoming 'Home v Away' and completed 'Home 2-1 Away' formats."""
    body = line.strip()
    if not body:
        return None
    time_match = _TIME_RE.match(body)
    kickoff = time_match.group("time") if time_match else None
    if time_match:
        body = time_match.group("body")

    if re.search(r"\s+v\s+", body):
        home, away_part = re.split(r"\s+v\s+", body, maxsplit=1)
        score = re.match(
            r"^(?P<away>.+?)\s+(?P<hg>\d+)-(?P<ag>\d+)"
            r"(?:\s+\((?P<hthg>\d+)-(?P<htag>\d+)\))?\s*$",
            away_part,
        )
        if score:
            away = score.group("away").strip()
            home_score, away_score = int(score.group("hg")), int(score.group("ag"))
        else:
            home, away, home_score, away_score = home.strip(), away_part.strip(), None, None
    else:
        score = _SCORE_RE.search(body)
        if not score:
            return None
        home = body[:score.start()].strip()
        away = body[score.end():].strip()
        if not home or not away:
            return None
        home_score, away_score = int(score.group("hg")), int(score.group("ag"))
    home, away = home.strip(), away.strip()
    if not home or not away:
        return None
    return home, away, kickoff, home_score, away_score


def parse_openfootball_fixtures(
    text: str,
    league: str,
    season: str,
    source_url: str,
    source_updated_at: str | None = None,
    source_retrieved_at_utc: str | None = None,
    source_cache_status: str = "fresh",
) -> list[SoccerMatch]:
    """Parse OpenFootball matchday text into normalized match records."""
    league_key = league.strip().lower()
    if league_key not in OPENFOOTBALL_REPOS:
        raise ValueError(f"Unsupported OpenFootball league: {league}")
    config = OPENFOOTBALL_REPOS[league_key]
    season_match = re.match(r"^(\d{4})[-/](\d{2,4})$", season.strip())
    if not season_match:
        raise ValueError("season must look like 2026-27 or 2026/27")
    season_start_year = int(season_match.group(1))
    season_name = f"{season_start_year}/{season_match.group(2)[-2:]}"
    retrieved_at = source_retrieved_at_utc or utc_now_iso()
    observed_at = utc_now_iso()
    current_date = None
    current_year = season_start_year
    previous_month = None
    match_week = None
    records: list[SoccerMatch] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("▪"):
            group = re.search(r"(?:matchday|regular season|round)\s*[-:]?\s*(\d+)", stripped, re.IGNORECASE)
            match_week = int(group.group(1)) if group else match_week
            continue
        date_match = _DATE_RE.match(line)
        if date_match:
            _weekday, month_text, day_text, explicit_year = date_match.groups()
            if month_text not in _MONTHS:
                continue
            month = _MONTHS[month_text]
            if explicit_year:
                current_year = int(explicit_year)
            elif previous_month is not None and previous_month >= 10 and month <= 3:
                current_year += 1
            try:
                from datetime import date
                current_date = date(current_year, month, int(day_text)).isoformat()
            except ValueError:
                current_date = None
                continue
            previous_month = month
            continue
        if current_date is None or stripped.startswith("="):
            continue
        parsed = _parse_match_line(line)
        if parsed is None:
            continue
        home, away, kickoff, home_score, away_score = parsed
        identity = "|".join((
            league_key, season.strip(), current_date, str(match_week or ""),
            kickoff or "", home.casefold(), away.casefold(),
        ))
        provider_match_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        records.append(SoccerMatch(
            provider="openfootball",
            match_id=f"openfootball:{provider_match_id}",
            provider_match_id=provider_match_id,
            competition_id=config["competition_id"],
            competition_name=config["competition_name"],
            season_id=season.strip(),
            season_name=season_name,
            match_date=current_date,
            kickoff_time_local=kickoff,
            kickoff_timezone=None,
            kickoff_utc=None,
            match_week=match_week,
            home_team_id=None,
            home_team=home,
            away_team_id=None,
            away_team=away,
            home_score=home_score,
            away_score=away_score,
            match_state="finished" if home_score is not None and away_score is not None else "scheduled",
            home_xg=None,
            away_xg=None,
            home_shots=None,
            away_shots=None,
            home_shots_on_target=None,
            away_shots_on_target=None,
            xg_status="not_provided",
            source_url=source_url,
            source_updated_at=source_updated_at,
            source_retrieved_at_utc=retrieved_at,
            source_cache_status=source_cache_status,
            observed_at_utc=observed_at,
        ))
    return records


class OpenFootballProvider:
    def __init__(
        self,
        cache_dir: str | Path = ".cache/soccer/openfootball",
        timeout_seconds: float = 20.0,
        session: Any = None,
        allow_stale_cache: bool = True,
    ) -> None:
        self.http = CachedHttpClient(cache_dir, timeout_seconds, session, allow_stale_cache)

    def list_matches(self, league: str, season: str, limit: int | None = None) -> list[SoccerMatch]:
        league_key = league.strip().lower()
        if league_key not in OPENFOOTBALL_REPOS:
            raise ValueError(f"Unsupported OpenFootball league: {league}")
        if not re.match(r"^\d{4}-\d{2,4}$", season.strip()):
            raise ValueError("season must look like 2026-27")
        config = OPENFOOTBALL_REPOS[league_key]
        relative_path = config["path_template"].format(season=season.strip())
        url = f"https://raw.githubusercontent.com/openfootball/{config['repo']}/master/{relative_path}"
        text, meta = self.http.fetch_text(url)
        matches = parse_openfootball_fixtures(
            text=text,
            league=league_key,
            season=season,
            source_url=url,
            source_updated_at=meta.get("source_updated_at"),
            source_retrieved_at_utc=meta.get("source_retrieved_at_utc"),
            source_cache_status=meta.get("cache_status", "unknown"),
        )
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            matches = matches[:limit]
        return matches
