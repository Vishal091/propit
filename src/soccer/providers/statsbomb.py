"""StatsBomb Open Data adapter for historical match and shot-level data.

StatsBomb Open Data exposes a curated subset of competitions and seasons. It is not a
complete live fixture or odds API. Attribute StatsBomb in any published analysis based
on this data and follow the repository terms: https://github.com/statsbomb/open-data
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.soccer.providers.http_cache import CachedHttpClient, DataNotAvailableError, DataProviderError
from src.soccer.schemas import CompetitionSeason, SoccerMatch, utc_now_iso


DEFAULT_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_name_id(value: Any, id_key: str, name_key: str) -> tuple[str | None, str]:
    if not isinstance(value, dict):
        return None, ""
    ident = value.get(id_key)
    return (str(ident) if ident is not None else None, str(value.get(name_key) or ""))


def summarize_shot_events(events: list[dict[str, Any]], home_team_id: int, away_team_id: int) -> dict[str, Any]:
    """Aggregate xG, shots and shots-on-target from StatsBomb Shot events.

    If any recorded shot lacks a numeric xG value, shot counts remain available but xG
    is withheld to avoid returning a silently partial team total.
    """
    stats: dict[int, dict[str, Any]] = {
        home_team_id: {"shots": 0, "shots_on_target": 0, "xg": 0.0},
        away_team_id: {"shots": 0, "shots_on_target": 0, "xg": 0.0},
    }
    xg_complete = True
    on_target_outcomes = {"Goal", "Saved", "Saved to Post"}
    for event in events:
        if not isinstance(event, dict) or (event.get("type") or {}).get("name") != "Shot":
            continue
        team_id = _as_int((event.get("team") or {}).get("id"))
        if team_id not in stats:
            continue
        shot = event.get("shot") or {}
        row = stats[team_id]
        row["shots"] += 1
        if (shot.get("outcome") or {}).get("name") in on_target_outcomes:
            row["shots_on_target"] += 1
        raw_xg = shot.get("statsbomb_xg")
        try:
            xg = float(raw_xg)
            if not math.isfinite(xg) or xg < 0:
                raise ValueError("invalid xG")
            row["xg"] += xg
        except (TypeError, ValueError):
            xg_complete = False

    return {
        "home_shots": stats[home_team_id]["shots"],
        "away_shots": stats[away_team_id]["shots"],
        "home_shots_on_target": stats[home_team_id]["shots_on_target"],
        "away_shots_on_target": stats[away_team_id]["shots_on_target"],
        "home_xg": round(stats[home_team_id]["xg"], 4) if xg_complete else None,
        "away_xg": round(stats[away_team_id]["xg"], 4) if xg_complete else None,
        "xg_status": "complete" if xg_complete else "partial",
    }


class StatsBombOpenDataProvider:
    def __init__(
        self,
        cache_dir: str | Path = ".cache/soccer/statsbomb",
        timeout_seconds: float = 20.0,
        base_url: str = DEFAULT_BASE_URL,
        session: Any = None,
        allow_stale_cache: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = CachedHttpClient(cache_dir, timeout_seconds, session, allow_stale_cache)

    def _get_json(self, relative_path: str) -> tuple[Any, dict[str, Any], str]:
        if relative_path.startswith("/") or ".." in Path(relative_path).parts:
            raise ValueError("Invalid StatsBomb data path")
        url = f"{self.base_url}/{relative_path}"
        text, metadata = self.http.fetch_text(url)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataProviderError(f"StatsBomb returned invalid JSON for {relative_path}") from exc
        return payload, metadata, url

    def list_competitions(self) -> list[CompetitionSeason]:
        payload, meta, url = self._get_json("competitions.json")
        if not isinstance(payload, list):
            raise DataProviderError("StatsBomb competitions resource must be a JSON list")
        observed_at = utc_now_iso()
        results: list[CompetitionSeason] = []
        for item in payload:
            if not isinstance(item, dict) or item.get("competition_id") is None or item.get("season_id") is None:
                continue
            results.append(CompetitionSeason(
                provider="statsbomb",
                competition_id=str(item["competition_id"]),
                competition_name=str(item.get("competition_name") or ""),
                country_name=item.get("country_name"),
                season_id=str(item["season_id"]),
                season_name=str(item.get("season_name") or ""),
                gender=item.get("competition_gender"),
                source_url=url,
                source_updated_at=item.get("match_updated") or meta.get("source_updated_at"),
                source_retrieved_at_utc=meta.get("source_retrieved_at_utc") or observed_at,
                observed_at_utc=observed_at,
                source_cache_status=meta.get("cache_status", "unknown"),
            ))
        return results

    def list_matches(
        self, competition_id: int | str, season_id: int | str,
        include_xg: bool = False, limit: int | None = None,
    ) -> list[SoccerMatch]:
        comp_id, season_key = str(competition_id), str(season_id)
        competitions = self.list_competitions()
        comp = next((c for c in competitions if c.competition_id == comp_id and c.season_id == season_key), None)
        if comp is None:
            raise ValueError(f"Competition/season {comp_id}/{season_key} is not listed by StatsBomb Open Data")
        payload, meta, url = self._get_json(f"matches/{comp_id}/{season_key}.json")
        if not isinstance(payload, list):
            raise DataProviderError(f"StatsBomb match resource for {comp_id}/{season_key} must be a JSON list")
        observed_at = utc_now_iso()
        valid_items = [item for item in payload if isinstance(item, dict) and item.get("match_id") is not None]
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            valid_items = valid_items[:limit]
        results: list[SoccerMatch] = []
        for item in valid_items:
            home_id, home_name = _nested_name_id(item.get("home_team"), "home_team_id", "home_team_name")
            away_id, away_name = _nested_name_id(item.get("away_team"), "away_team_id", "away_team_name")
            if home_id is None or away_id is None or not home_name or not away_name:
                continue
            raw_date = item.get("match_date")
            raw_kickoff = item.get("kick_off")
            kickoff_time = str(raw_kickoff).split(".")[0] if raw_kickoff else None
            home_score, away_score = _as_int(item.get("home_score")), _as_int(item.get("away_score"))
            match_state = "finished" if home_score is not None and away_score is not None else "scheduled"
            xg = {
                "home_xg": None, "away_xg": None, "home_shots": None, "away_shots": None,
                "home_shots_on_target": None, "away_shots_on_target": None, "xg_status": "not_requested",
            }
            match_source_url = url
            source_updated_at = item.get("last_updated") or comp.source_updated_at
            source_retrieved_at = meta.get("source_retrieved_at_utc") or observed_at
            cache_status = meta.get("cache_status", "unknown")
            if include_xg:
                try:
                    event_payload, event_meta, event_url = self._get_json(f"events/{item['match_id']}.json")
                    if not isinstance(event_payload, list):
                        raise DataProviderError(f"StatsBomb event resource for match {item['match_id']} is not a list")
                    xg = summarize_shot_events(event_payload, int(home_id), int(away_id))
                    match_source_url = event_url
                    source_updated_at = event_meta.get("source_updated_at") or source_updated_at
                    source_retrieved_at = event_meta.get("source_retrieved_at_utc") or source_retrieved_at
                    cache_status = event_meta.get("cache_status", cache_status)
                except DataNotAvailableError:
                    xg["xg_status"] = "unavailable"
            results.append(SoccerMatch(
                provider="statsbomb",
                match_id=f"statsbomb:{item['match_id']}",
                provider_match_id=str(item["match_id"]),
                competition_id=comp_id,
                competition_name=comp.competition_name,
                season_id=season_key,
                season_name=comp.season_name,
                match_date=str(raw_date) if raw_date else None,
                kickoff_time_local=kickoff_time,
                kickoff_timezone=None,
                kickoff_utc=None,
                match_week=_as_int(item.get("match_week")),
                home_team_id=home_id,
                home_team=home_name,
                away_team_id=away_id,
                away_team=away_name,
                home_score=home_score,
                away_score=away_score,
                match_state=match_state,
                home_xg=xg["home_xg"],
                away_xg=xg["away_xg"],
                home_shots=xg["home_shots"],
                away_shots=xg["away_shots"],
                home_shots_on_target=xg["home_shots_on_target"],
                away_shots_on_target=xg["away_shots_on_target"],
                xg_status=xg["xg_status"],
                source_url=match_source_url,
                source_updated_at=source_updated_at,
                source_retrieved_at_utc=source_retrieved_at,
                source_cache_status=cache_status,
                observed_at_utc=observed_at,
            ))
        return results
