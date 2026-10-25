"""Point-in-time starter/bench snapshots and conservative matching."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.player_identity import normalize_player_name

DEFAULT_LINEUP_PATH = Path("data/player_lineups.csv")
DEFAULT_LINEUP_MAX_AGE_HOURS = 2.0
_OUTPUT_COLUMNS = (
    "lineup_status", "lineup_gate", "lineup_reason", "lineup_source",
    "lineup_reported_at_utc", "lineup_age_hours", "minutes_limit",
)


def normalize_lineup_status(value: Any) -> str:
    raw = " ".join(str(value or "").strip().upper().replace("_", " ").split())
    aliases = {
        "STARTER": "STARTER", "STARTING": "STARTER", "CONFIRMED STARTER": "STARTER",
        "BENCH": "BENCH", "RESERVE": "BENCH", "CONFIRMED BENCH": "BENCH",
        "UNKNOWN": "UNKNOWN", "": "UNKNOWN",
    }
    return aliases.get(raw, "UNKNOWN")


def attach_lineups(
    rows: pd.DataFrame,
    path: str | Path = DEFAULT_LINEUP_PATH,
    *,
    now_utc: str | pd.Timestamp | None = None,
    max_age_hours: float = DEFAULT_LINEUP_MAX_AGE_HOURS,
) -> pd.DataFrame:
    """Attach only exact player/team/match-date lineups with a fresh source time."""
    if max_age_hours < 0:
        raise ValueError("max_age_hours must be non-negative")
    result = rows.copy()
    if result.empty:
        for column in _OUTPUT_COLUMNS:
            result[column] = pd.Series(dtype="object")
        return result
    observed = pd.to_datetime(now_utc, utc=True, errors="coerce") if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if pd.isna(observed):
        raise ValueError("now_utc must be a valid UTC timestamp")

    source_rows: list[dict[str, Any]] = []
    file_path = Path(path)
    if file_path.exists():
        frame = pd.read_csv(file_path, dtype={"player_id": "string"})
        frame.columns = [str(col).strip().lower() for col in frame.columns]
        required = {"match_date", "starting_status", "reported_at_utc"}
        if "player_id" not in frame and "player_name" not in frame:
            raise ValueError(f"{file_path} needs player_id or player_name")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{file_path} missing required columns: {', '.join(sorted(missing))}")
        for optional in ("player_id", "player_name", "team_abbreviation", "source", "minutes_limit", "reason"):
            if optional not in frame:
                frame[optional] = ""
        frame["_reported"] = pd.to_datetime(frame["reported_at_utc"], utc=True, errors="coerce")
        frame["_date"] = pd.to_datetime(frame["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["_name"] = frame["player_name"].map(normalize_player_name)
        frame["_team"] = frame["team_abbreviation"].fillna("").astype(str).str.strip().str.upper()
        frame = frame.loc[frame["_date"].notna() & (frame["_name"].ne("") | frame["player_id"].notna())]
        frame = frame.sort_values("_reported", na_position="first", kind="stable")
        source_rows = frame.to_dict(orient="records")

    statuses: list[str] = []
    gates: list[str] = []
    reasons: list[str] = []
    sources: list[str] = []
    times: list[str] = []
    ages: list[float | None] = []
    limits: list[float | None] = []
    for row in result.to_dict(orient="records"):
        date_text = pd.to_datetime(row.get("match_date"), errors="coerce")
        date_key = date_text.strftime("%Y-%m-%d") if pd.notna(date_text) else ""
        pid = str(row.get("player_id") or "").strip().split(".")[0]
        name = normalize_player_name(row.get("player_name"))
        team = str(row.get("team_abbreviation") or "").strip().upper()
        matching = [
            candidate for candidate in source_rows
            if candidate.get("_date") == date_key
            and (not team or not candidate.get("_team") or candidate.get("_team") == team)
            and (
                (pid and str(candidate.get("player_id") or "").strip().split(".")[0] == pid)
                or (name and candidate.get("_name") == name)
            )
        ]
        candidate = matching[-1] if matching else None
        if candidate is None:
            statuses.append("UNKNOWN")
            gates.append("UNVERIFIED")
            reasons.append("No starter/bench snapshot matched this player, team, and scheduled match date.")
            sources.append("")
            times.append("")
            ages.append(None)
            limits.append(None)
            continue
        stamp = candidate.get("_reported")
        stamp_text = stamp.isoformat().replace("+00:00", "Z") if pd.notna(stamp) else ""
        age = (observed - stamp).total_seconds() / 3600 if pd.notna(stamp) else None
        status = normalize_lineup_status(candidate.get("starting_status"))
        if age is None or age < -5 / 60 or age > max_age_hours:
            gate, reason = "STALE", f"Lineup snapshot age is {age if age is not None else 'unknown'} hours; limit is {max_age_hours:.1f}."
        elif status == "UNKNOWN":
            gate, reason = "UNVERIFIED", "Lineup status is blank or unrecognized."
        else:
            gate = "CLEAR"
            reason = str(candidate.get("reason") or f"Latest matched lineup status: {status}.")
        limit_value = pd.to_numeric(pd.Series([candidate.get("minutes_limit")]), errors="coerce").iloc[0]
        limit = float(limit_value) if pd.notna(limit_value) and 0 < float(limit_value) <= 48 else None
        statuses.append(status)
        gates.append(gate)
        reasons.append(reason)
        sources.append(str(candidate.get("source") or ""))
        times.append(stamp_text)
        ages.append(round(max(0.0, age), 2) if age is not None else None)
        limits.append(limit)

    result["lineup_status"] = statuses
    result["lineup_gate"] = gates
    result["lineup_reason"] = reasons
    result["lineup_source"] = sources
    result["lineup_reported_at_utc"] = times
    result["lineup_age_hours"] = ages
    result["minutes_limit"] = limits
    return result
