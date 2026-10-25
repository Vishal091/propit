"""Conservative, point-in-time availability matching for NBA prop rows.

Availability is matched by exact scheduled date and team, using player ID first
and a normalized player name as a fallback for official injury reports whose
name format differs from the NBA roster. A missing row is never treated as active.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.player_identity import normalize_player_name

DEFAULT_AVAILABILITY_PATH = Path("data/player_availability.csv")
DEFAULT_MAX_AGE_HOURS = 2.0
OUTPUT_COLUMNS = (
    "availability_status", "availability_gate", "availability_reason",
    "availability_source", "availability_reported_at_utc", "availability_age_hours",
    "availability_retrieved_at_utc", "availability_minutes_limit",
)
_STATUS_ALIASES = {
    "AVAILABLE": "AVAILABLE", "ACTIVE": "AVAILABLE", "PROBABLE": "PROBABLE",
    "LIKELY TO PLAY": "PROBABLE", "QUESTIONABLE": "QUESTIONABLE", "GTD": "QUESTIONABLE",
    "GAME TIME DECISION": "QUESTIONABLE", "GAME-TIME DECISION": "QUESTIONABLE",
    "DOUBTFUL": "DOUBTFUL", "OUT": "OUT", "INACTIVE": "INACTIVE",
    "SUSPENDED": "SUSPENDED", "REST": "OUT", "WILL NOT PLAY": "OUT",
    "NOT PLAYING": "OUT", "NOT WITH TEAM": "OUT", "UNKNOWN": "UNKNOWN", "": "UNKNOWN",
}
_BLOCKED_STATUSES = {"OUT", "INACTIVE", "SUSPENDED"}
_REVIEW_STATUSES = {"QUESTIONABLE", "DOUBTFUL"}


def normalize_status(value: Any) -> str:
    """Return a canonical status without guessing an unfamiliar label."""
    if value is None or pd.isna(value):
        return "UNKNOWN"
    raw = " ".join(str(value).strip().upper().replace("_", " ").split())
    return _STATUS_ALIASES.get(raw, "UNKNOWN")


def _text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


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
    return "" if pd.isna(parsed) else parsed.date().isoformat()


def _gate_for_status(status: str) -> str:
    if status in _BLOCKED_STATUSES:
        return "BLOCKED"
    if status in _REVIEW_STATUSES:
        return "REVIEW"
    if status in {"AVAILABLE", "PROBABLE"}:
        return "CLEAR"
    return "UNVERIFIED"


def _read_team_report_coverage(
    imports_path: Path,
    game_dates: set[str],
    observed_at: pd.Timestamp,
    max_age_hours: float,
) -> dict[str, dict[str, Any]]:
    """Read fresh report-level coverage metadata, without inventing player statuses."""
    if not imports_path.exists() or not game_dates:
        return {}
    try:
        imports = pd.read_csv(imports_path, dtype=str).fillna("")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return {}
    required = {
        "game_date", "reported_at_utc", "team_report_coverage_parsed",
        "team_submission_markers_resolved", "teams_not_yet_submitted_json",
    }
    if not required.issubset(imports.columns):
        return {}

    imports["_game_date"] = imports["game_date"].map(_date_key)
    imports = imports.loc[imports["_game_date"].isin(game_dates)].copy()
    if imports.empty:
        return {}
    imports["_retrieved"] = pd.to_datetime(
        imports.get("retrieved_at_utc", imports["reported_at_utc"]),
        utc=True, errors="coerce",
    )
    imports = imports.sort_values("_retrieved", kind="stable")

    coverage: dict[str, dict[str, Any]] = {}
    for game_date, group in imports.groupby("_game_date", sort=False):
        row = group.iloc[-1]
        parsed_flag = str(row.get("team_report_coverage_parsed", "")).strip().casefold()
        markers_resolved = str(row.get("team_submission_markers_resolved", "")).strip().casefold()
        if parsed_flag not in {"true", "1", "yes"} or markers_resolved not in {"true", "1", "yes"}:
            continue
        reported = pd.to_datetime(row.get("reported_at_utc"), utc=True, errors="coerce")
        if pd.isna(reported):
            continue
        age = (observed_at - reported).total_seconds() / 3600.0
        if age < -5 / 60 or age > max_age_hours:
            continue
        try:
            not_submitted_raw = json.loads(row.get("teams_not_yet_submitted_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(not_submitted_raw, list):
            continue
        not_submitted = {
            str(team).strip().upper() for team in not_submitted_raw if str(team).strip()
        }
        coverage[str(game_date)] = {
            "not_yet_submitted_teams": not_submitted,
            "reported_at_utc": reported.isoformat().replace("+00:00", "Z"),
            "retrieved_at_utc": str(row.get("retrieved_at_utc") or ""),
            "report_url": str(row.get("report_url") or ""),
            "age_hours": max(0.0, age),
        }
    return coverage


def attach_availability(
    rows: pd.DataFrame,
    path: str | Path = DEFAULT_AVAILABILITY_PATH,
    *,
    now_utc: str | pd.Timestamp | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> pd.DataFrame:
    """Attach the freshest exact player/date/team availability snapshot."""
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

    source_rows: list[dict[str, Any]] = []
    file_path = Path(path)
    if file_path.exists():
        try:
            source = pd.read_csv(file_path, dtype={"player_id": "string"})
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ValueError(f"could not read availability CSV {file_path}: {exc}") from exc
        source.columns = [str(column).strip().lower() for column in source.columns]
        required = {"match_date", "status", "reported_at_utc"}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"{file_path} is missing required availability columns: {', '.join(sorted(missing))}")
        if "player_id" not in source and "player_name" not in source:
            raise ValueError(f"{file_path} needs player_id or player_name")
        for optional in ("player_id", "player_name", "team_abbreviation", "reason", "source", "minutes_limit", "retrieved_at_utc"):
            if optional not in source:
                source[optional] = ""
        source["_player_key"] = source["player_id"].map(_player_id) if "player_id" in source else ""
        source["_name_key"] = source["player_name"].map(normalize_player_name) if "player_name" in source else ""
        source["_date_key"] = source["match_date"].map(_date_key)
        source["_team_key"] = source["team_abbreviation"].map(lambda value: _text(value).upper())
        source["_reported"] = pd.to_datetime(source["reported_at_utc"], utc=True, errors="coerce")
        source = source.loc[
            source["_date_key"].ne("") & (source["_player_key"].ne("") | source["_name_key"].ne(""))
        ].copy()
        source = source.sort_values("_reported", na_position="first", kind="stable")
        source_rows = source.to_dict(orient="records")

    target_dates = {
        _date_key(value) for value in result.get("match_date", pd.Series(dtype="object")).tolist()
    }
    target_dates.discard("")
    team_coverage = _read_team_report_coverage(
        file_path.parent / "injury_report_imports.csv",
        target_dates,
        observed_at,
        max_age_hours,
    )

    statuses: list[str] = []
    gates: list[str] = []
    reasons: list[str] = []
    sources: list[str] = []
    timestamps: list[str] = []
    ages: list[float | None] = []
    retrieved_values: list[str] = []
    limits: list[float | None] = []

    for row in result.to_dict(orient="records"):
        pid = _player_id(row.get("player_id"))
        game_date = _date_key(row.get("match_date"))
        name = normalize_player_name(row.get("player_name"))
        team = _text(row.get("team_abbreviation")).upper()
        matches = [
            candidate for candidate in source_rows
            if candidate.get("_date_key") == game_date
            and (not team or not candidate.get("_team_key") or candidate.get("_team_key") == team)
            and (
                (pid and candidate.get("_player_key") == pid)
                or (name and candidate.get("_name_key") == name)
            )
        ]
        # Prefer a player-ID match over a name-only match; within that group use
        # the newest report timestamp. Conflicting teams never match.
        if pid and any(item.get("_player_key") == pid for item in matches):
            matches = [item for item in matches if item.get("_player_key") == pid]
        exact = matches[-1] if matches else None

        if exact is None:
            coverage = team_coverage.get(game_date) if team else None
            if coverage is not None:
                report_source = str(coverage.get("report_url") or "")
                report_stamp = str(coverage.get("reported_at_utc") or "")
                report_age = float(coverage.get("age_hours", 0.0))
                if team in coverage["not_yet_submitted_teams"]:
                    statuses.append("NOT_YET_SUBMITTED")
                    gates.append("UNVERIFIED")
                    reasons.append(
                        "The latest official injury report explicitly marks this team's report as NOT YET SUBMITTED."
                    )
                else:
                    statuses.append("NOT_LISTED")
                    gates.append("CLEAR")
                    reasons.append(
                        "No player-specific issue is listed in the fresh official report, and this team is not marked "
                        "NOT YET SUBMITTED. This is not an explicit Available designation; verify again near tip-off."
                    )
                sources.append(report_source)
                timestamps.append(report_stamp)
                ages.append(round(report_age, 2))
                retrieved_values.append(str(coverage.get("retrieved_at_utc") or ""))
                limits.append(None)
                continue
            statuses.append("UNKNOWN")
            gates.append("UNVERIFIED")
            reasons.append("No availability record matched this player, team, and scheduled match date; report coverage is unavailable.")
            sources.append("")
            timestamps.append("")
            ages.append(None)
            retrieved_values.append("")
            limits.append(None)
            continue

        status = normalize_status(exact.get("status"))
        reported = exact.get("_reported")
        reported_text = reported.isoformat().replace("+00:00", "Z") if pd.notna(reported) else ""
        age: float | None = None
        if not reported_text:
            gate = "STALE"
            reason = "Availability record has no valid report timestamp; refresh before relying on it."
        else:
            age = (observed_at - reported).total_seconds() / 3600.0
            if age < -5 / 60 or age > max_age_hours:
                gate = "STALE"
                reason = f"Availability report is {age:.1f} hours old; limit is {max_age_hours:.1f} hours."
            else:
                gate = _gate_for_status(status)
                reason = _text(exact.get("reason")) or f"Latest matched report status: {status}."
                if gate == "UNVERIFIED":
                    reason = f"Unrecognized or missing status in report: {_text(exact.get('status')) or '(blank)'}."
        limit_value = pd.to_numeric(pd.Series([exact.get("minutes_limit")]), errors="coerce").iloc[0]
        limit = float(limit_value) if pd.notna(limit_value) and 0 < float(limit_value) <= 48 else None
        statuses.append(status)
        gates.append(gate)
        reasons.append(reason)
        sources.append(_text(exact.get("source")))
        timestamps.append(reported_text)
        ages.append(round(max(0.0, age), 2) if age is not None else None)
        retrieved_values.append(_text(exact.get("retrieved_at_utc")))
        limits.append(limit)

    result["availability_status"] = statuses
    result["availability_gate"] = gates
    result["availability_reason"] = reasons
    result["availability_source"] = sources
    result["availability_reported_at_utc"] = timestamps
    result["availability_age_hours"] = ages
    result["availability_retrieved_at_utc"] = retrieved_values
    result["availability_minutes_limit"] = limits
    return result
