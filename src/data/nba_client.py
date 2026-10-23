"""Free, API-key-free wrappers around public NBA stats endpoints.

Historical league game logs are cached indefinitely; the in-progress season and
current roster use short cache TTLs so repeat runs can pick up new games and
roster changes without hammering stats.nba.com.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.seasons import current_season

log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/nba")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_SEASON_CACHE_TTL = 6 * 60 * 60
ROSTER_CACHE_TTL = 24 * 60 * 60


@dataclass(frozen=True)
class CacheKey:
    """Deterministic cache key for a request."""
    endpoint: str
    params: tuple[tuple[str, str], ...]

    def to_path(self) -> Path:
        raw = self.endpoint + json.dumps(self.params, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return CACHE_DIR / f"{self.endpoint}_{digest}.parquet"


def _cached_call(endpoint_cls: Any, *, cache_ttl_seconds: int | None = None, **params: Any) -> pd.DataFrame:
    """Call an nba_api endpoint and cache its first DataFrame."""
    key = CacheKey(
        endpoint=endpoint_cls.__name__,
        params=tuple(sorted((k, str(v)) for k, v in params.items())),
    )
    path = key.to_path()
    if path.exists():
        age = max(0.0, time.time() - path.stat().st_mtime)
        if cache_ttl_seconds is None or age < cache_ttl_seconds:
            return pd.read_parquet(path)
        log.info("refreshing expired %s cache (%0.1fh old)", key.endpoint, age / 3600)

    last_err: Exception | None = None
    for attempt in range(4):
        try:
            response = endpoint_cls(**params)
            frames = response.get_data_frames()
            if not frames or frames[0].empty:
                raise RuntimeError("empty payload from NBA stats endpoint")
            df = frames[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            return df
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt
            log.warning("nba_api call failed (attempt %s): %s. retry in %ss", attempt, exc, wait)
            if attempt < 3:
                time.sleep(wait)
    raise RuntimeError(f"nba_api call failed after retries: {last_err}")


def _opponent_ids_from_league_log(df: pd.DataFrame) -> pd.Series:
    """Map each (game, team) pair to the opposing team's ID where possible."""
    required = {"GAME_ID", "TEAM_ID"}
    if not required.issubset(df.columns):
        return pd.Series(0, index=df.index, dtype="int64")
    team_rows = df[["GAME_ID", "TEAM_ID"]].dropna().drop_duplicates()
    opponents: dict[tuple[Any, Any], Any] = {}
    for game_id, group in team_rows.groupby("GAME_ID", sort=False):
        team_ids = group["TEAM_ID"].drop_duplicates().tolist()
        if len(team_ids) == 2:
            opponents[(game_id, team_ids[0])] = team_ids[1]
            opponents[(game_id, team_ids[1])] = team_ids[0]
    mapped = [opponents.get((game_id, team_id), 0) for game_id, team_id in zip(df["GAME_ID"], df["TEAM_ID"])]
    return pd.Series(mapped, index=df.index, dtype="int64")


def get_league_game_log(season: str) -> pd.DataFrame:
    """Fetch every NBA player game log for one regular season in one request."""
    from nba_api.stats.endpoints import playergamelogs
    ttl = CURRENT_SEASON_CACHE_TTL if season == current_season() else None
    df = _cached_call(
        playergamelogs.PlayerGameLogs,
        cache_ttl_seconds=ttl,
        league_id_nullable="00",
        season_nullable=season,
        season_type_nullable="Regular Season",
    ).copy()
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    if "PLAYER_ID" not in df.columns or "PLAYER_NAME" not in df.columns:
        raise RuntimeError("PlayerGameLogs response is missing PLAYER_ID or PLAYER_NAME")
    df["OPPONENT_TEAM_ID"] = _opponent_ids_from_league_log(df)
    sort_columns = [c for c in ("GAME_DATE", "GAME_ID", "PLAYER_ID") if c in df.columns]
    return df.dropna(subset=["GAME_DATE"]).sort_values(sort_columns, kind="stable").reset_index(drop=True)


def get_active_players() -> pd.DataFrame:
    """Return players marked active on NBA's public all-players endpoint."""
    from nba_api.stats.endpoints import commonallplayers
    df = _cached_call(
        commonallplayers.CommonAllPlayers,
        cache_ttl_seconds=ROSTER_CACHE_TTL,
        is_only_current_season=0,
        league_id="00",
        season=current_season(),
    ).copy()
    required = {"PERSON_ID", "DISPLAY_FIRST_LAST", "ROSTERSTATUS"}
    if not required.issubset(df.columns):
        raise RuntimeError("CommonAllPlayers response is missing roster fields")
    status = pd.to_numeric(df["ROSTERSTATUS"], errors="coerce")
    active = df.loc[status.eq(1), ["PERSON_ID", "DISPLAY_FIRST_LAST"]].copy()
    active = active.rename(columns={"PERSON_ID": "PLAYER_ID", "DISPLAY_FIRST_LAST": "PLAYER_NAME"})
    active["PLAYER_ID"] = pd.to_numeric(active["PLAYER_ID"], errors="coerce")
    return active.dropna(subset=["PLAYER_ID"]).drop_duplicates("PLAYER_ID").reset_index(drop=True)


def get_player_game_log(player_id: int, season: str | None = None) -> pd.DataFrame:
    """Per-game log for one player in one season (legacy helper)."""
    from nba_api.stats.endpoints import playergamelog
    season = season or current_season()
    ttl = CURRENT_SEASON_CACHE_TTL if season == current_season() else None
    df = _cached_call(playergamelog.PlayerGameLog, cache_ttl_seconds=ttl, player_id=player_id, season=season).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE").reset_index(drop=True)


def get_team_advanced(team_id: int, season: str | None = None) -> pd.DataFrame:
    """Team advanced metrics, used for pace and defensive rating context."""
    from nba_api.stats.endpoints import teamdashboardbygeneralsplits
    season = season or current_season()
    ttl = CURRENT_SEASON_CACHE_TTL if season == current_season() else None
    return _cached_call(
        teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits,
        cache_ttl_seconds=ttl,
        team_id=team_id,
        season=season,
        measure_type_detailed_defense="Advanced",
    )


def get_scoreboard(date: str) -> pd.DataFrame:
    """Tonight's games. Date format: YYYY-MM-DD."""
    from nba_api.stats.endpoints import scoreboardv2
    return _cached_call(scoreboardv2.ScoreboardV2, game_date=date)
