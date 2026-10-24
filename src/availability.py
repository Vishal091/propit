"""Conservative, point-in-time player-availability checks for NBA prop rows.

PropIt does not scrape or infer injuries. Feed this module a status snapshot
that was checked against the official NBA injury report or a trusted source.
Every row is matched to a specific player and scheduled match date.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_AVAILABILITY_PATH = Path("data/player_availability.csv")
DEFAULT_MAX_AGE_HOURS = 6.0

OUTPUT_COLUMNS = (
    "availability_status",
    "availability_gate",
    "availability_reason",
    "availability_source",
    "availability_reported_at_utc",
    "availability_age_hours",
)

_STATUS_ALIASES = {
    "AVAILABLE": "AVAILABLE",
    "ACTIVE": "AVAILABLE",
    "PROBABLE": "PROBABLE",
    "LIKELY TO PLAY": "PROBABLE",
    "QUESTIONABLE": "QUESTIONABLE",
    "GTD": "QUESTIONABLE",
    "GAME TIME DECISION": "QUESTIONABLE",
    "GAME-TIME DECISION": "QUESTIONABLE",
    "DOUBTFUL": "DOUBTFUL",
    "OUT": "OUT",
    "INACTIVE": "INACTIVE",
    "SUSPENDED": "SUSPENDED",
    "REST": "OUT",
    "WILL NOT PLAY": "OUT",
    "NOT PLAYING": "OUT",
    "UNKNOWN": "UNKNOWN",
    "": "UNKNOWN",
}
_BLOCKED_STATUSES = {"OUT", "INACTIVE", "SUSPENDED"}
_REVIEW_STATUSES = {"QUESTIONABLE", "DOUBTFUL"}


def normalize_status(value: Any) -> str:
    """Return a canonical report status without guessing unknown values."""
    if value is None or pd.isna(value):
        return "UNKNOWN"
    raw = " ".join(str(value).strip().upper().replace("_", " ").split())
    return _STATUS_ALIASES.get(raw, "UNKNOWN")


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _player_id(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not pd.notna(number) or number <= 0 or not number.is_integer():
        return ""
    return str(int(number))


def _date_key(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def _gate_for_status(status: str) -> str:
    if status in _BLOCKED_STATUSES:
        return "BLOCKED"
    if status in _REVIEW_STATUSES:
        return "REVIEW"
    if status in {"AVAILABLE", "PROBABLE"}:
        return "CLEAR"
    return "UNVERIFIED"


def attach_availability(
    rows: pd.DataFrame,
    path: str | Path = DEFAULT_AVAILABILITY_PATH,
    *,
    now_utc: str | pd.Timestamp | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> pd.DataFrame:
    """Attach a fresh report for each exact player/match date, or mark unverified.

    Expected CSV fields: player_id, match_date, status, reported_at_utc.
    Optional fields: team_abbreviation, reason, source. Stale or undated reports
    are never treated as current availability. Missing files produce explicit
    UNVERIFIED rows so the caller can show the gap instead of assuming healthy.
    """
    if max_age_hours < 0:
        raise ValueError("max_age_hours must be non-negative")
    result = rows.copy()
    if result.empty:
        for column in OUTPUT_COLUMNS:
            if column not in result:
                result[column] = pd.Series(dtype="object")
        return result

    observed_at = pd.to_datetime(now_utc, utc=True, errors="coerce") if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if pd.isna(observed_at):
        raise ValueError("now_utc must be a valid UTC timestamp")

    file_path = Path(path)
    snapshots: dict[tuple[str, str, str], dict[str, Any]] = {}
    if file_path.exists():
        try:
            source = pd.read_csv(file_path, dtype={"player_id": "string"})
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ValueError(f"could not read availability CSV {file_path}: {exc}") from exc
        source.columns = [str(column).strip().lower() for column in source.columns]
        required = {"player_id", "match_date", "status", "reported_at_utc"}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(
                f"{file_path} is missing required availability columns: {', '.join(sorted(missing))}"
            )
        for optional in ("team_abbreviation", "reason", "source"):
            if optional not in source:
                source[optional] = ""
        source["_player_key"] = source["player_id"].map(_player_id)
        source["_date_key"] = source["match_date"].map(_date_key)
        source["_team_key"] = source["team_abbreviation"].map(lambda value: _text(value).upper())
        source["_reported"] = pd.to_datetime(source["reported_at_utc"], utc=True, errors="coerce")
        source = source.loc[source["_player_key"].ne("") & source["_date_key"].ne("")].copy()
        source = source.sort_values("_reported", na_position="first", kind="stable")
        for record in source.to_dict(orient="records"):
            key = (record["_player_key"], record["_date_key"], record["_team_key"])
            snapshots[key] = record

    statuses: list[str] = []
    gates: list[str] = []
    reasons: list[str] = []
    sources: list[str] = []
    timestamps: list[str] = []
    ages: list[float | None] = []

    for row in result.to_dict(orient="records"):
        pid = _player_id(row.get("player_id"))
        game_date = _date_key(row.get("match_date"))
        team = _text(row.get("team_abbreviation")).upper()
        exact = snapshots.get((pid, game_date, team)) if pid and game_date and team else None
        if exact is None and pid and game_date:
            # Team is optional in manual files; never match a conflicting team.
            possible = [
                value for (source_pid, source_date, source_team), value in snapshots.items()
                if source_pid == pid and source_date == game_date and (not source_team or not team or source_team == team)
            ]
            if possible:
                exact = max(
                    possible,
                    key=lambda value: value.get("_reported") if pd.notna(value.get("_reported")) else pd.Timestamp.min.tz_localize("UTC"),
                )

        if exact is None:
            statuses.append("UNKNOWN")
            gates.append("UNVERIFIED")
            reasons.append("No availability record matched this player and scheduled match date.")
            sources.append("")
            timestamps.append("")
            ages.append(None)
            continue

        status = normalize_status(exact.get("status"))
        reported = exact.get("_reported")
        reported_text = reported.isoformat().replace("+00:00", "Z") if pd.notna(reported) else ""
        age: float | None = None
        if not reported_text:
            gate = "STALE"
            reason = "Availability record has no valid report timestamp; refresh before relying on it."
        else:
            age = max(0.0, (observed_at - reported).total_seconds() / 3600.0)
            if reported > observed_at + pd.Timedelta(minutes=5) or age > max_age_hours:
                gate = "STALE"
                reason = f"Availability report is stale ({age:.1f} hours old; limit {max_age_hours:.1f} hours)."
            else:
                gate = _gate_for_status(status)
                reason = _text(exact.get("reason")) or f"Latest matched report status: {status}."
                if gate == "UNVERIFIED":
                    reason = f"Unrecognized or missing status in the report: {_text(exact.get('status')) or '(blank)'}."
        statuses.append(status)
        gates.append(gate)
        reasons.append(reason)
        sources.append(_text(exact.get("source")))
        timestamps.append(reported_text)
        ages.append(round(age, 2) if age is not None else None)

    result["availability_status"] = statuses
    result["availability_gate"] = gates
    result["availability_reason"] = reasons
    result["availability_source"] = sources
    result["availability_reported_at_utc"] = timestamps
    result["availability_age_hours"] = ages
    return result
