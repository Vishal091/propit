"""Provider-neutral contracts for soccer fixture and team-match data.

Do not infer UTC kickoff timestamps from provider-local times unless the source defines
the timezone. kickoff_utc therefore remains null when timezone provenance is unknown.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class CompetitionSeason:
    provider: str
    competition_id: str
    competition_name: str
    country_name: str | None
    season_id: str
    season_name: str
    gender: str | None
    source_url: str
    source_updated_at: str | None
    source_retrieved_at_utc: str
    observed_at_utc: str
    source_cache_status: str = "fresh"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SoccerMatch:
    provider: str
    match_id: str
    provider_match_id: str
    competition_id: str
    competition_name: str
    season_id: str
    season_name: str
    match_date: str | None
    kickoff_time_local: str | None
    kickoff_timezone: str | None
    kickoff_utc: str | None
    match_week: int | None
    home_team_id: str | None
    home_team: str
    away_team_id: str | None
    away_team: str
    home_score: int | None
    away_score: int | None
    match_state: str
    home_xg: float | None
    away_xg: float | None
    home_shots: int | None
    away_shots: int | None
    home_shots_on_target: int | None
    away_shots_on_target: int | None
    xg_status: str
    source_url: str
    source_updated_at: str | None
    source_retrieved_at_utc: str
    source_cache_status: str
    observed_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MATCH_COLUMNS = [field.name for field in fields(SoccerMatch)]
COMPETITION_COLUMNS = [field.name for field in fields(CompetitionSeason)]
