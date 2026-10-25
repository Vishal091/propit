"""Point-in-time starter/bench snapshots with projected/confirmed provenance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.player_identity import normalize_player_name

DEFAULT_LINEUP_PATH = Path("data/player_lineups.csv")
DEFAULT_AUTO_LINEUP_PATH = Path("data/player_lineups_auto.csv")
DEFAULT_LINEUP_MAX_AGE_HOURS = 2.0
_OUTPUT_COLUMNS = (
    "lineup_status", "lineup_gate", "lineup_reason", "lineup_source",
    "lineup_reported_at_utc", "lineup_age_hours", "minutes_limit",
    "lineup_confirmation_status", "lineup_confidence_label", "lineup_source_status",
)


def normalize_lineup_status(value: Any) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    raw = " ".join(str(value).strip().upper().replace("_", " ").split())
    aliases = {
        "STARTER": "STARTER", "STARTING": "STARTER", "CONFIRMED STARTER": "STARTER",
        "BENCH": "BENCH", "RESERVE": "BENCH", "CONFIRMED BENCH": "BENCH",
        "UNKNOWN": "UNKNOWN", "": "UNKNOWN",
    }
    return aliases.get(raw, "UNKNOWN")


def _normalize_confirmation(value: Any, *, default: str) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return default
    raw = " ".join(str(value).strip().upper().replace("_", " ").split())
    if raw in {"CONFIRMED", "SOURCE CONFIRMED", "MANUAL VERIFIED", "VERIFIED", "OFFICIAL"}:
        return "CONFIRMED"
    if raw in {"PROJECTED", "EXPECTED", "UNCONFIRMED", "SOURCE PROJECTED"}:
        return "PROJECTED"
    return "UNKNOWN"


def _read_lineup_source(path: Path, *, default_confirmation: str) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path, dtype={"player_id": "string"}).fillna("")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not read lineup CSV {path}: {exc}") from exc
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    required = {"match_date", "starting_status", "reported_at_utc"}
    if "player_id" not in frame and "player_name" not in frame:
        raise ValueError(f"{path} needs player_id or player_name")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")
    for optional in (
        "player_id", "player_name", "team_abbreviation", "source", "minutes_limit",
        "reason", "confirmation_status", "confidence_label", "source_status",
    ):
        if optional not in frame:
            frame[optional] = ""
    frame["_reported"] = pd.to_datetime(frame["reported_at_utc"], utc=True, errors="coerce")
    frame["_date"] = pd.to_datetime(frame["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["_name"] = frame["player_name"].map(normalize_player_name)
    frame["_team"] = frame["team_abbreviation"].astype(str).str.strip().str.upper()
    frame["_role"] = frame["starting_status"].map(normalize_lineup_status)
    frame["_confirmation"] = frame["confirmation_status"].map(
        lambda value: _normalize_confirmation(value, default=default_confirmation)
    )
    frame["_priority"] = (
        frame["_confirmation"].eq("CONFIRMED") & frame["_role"].isin(["STARTER", "BENCH"])
    ).astype(int)
    frame = frame.loc[
        frame["_date"].notna()
        & (frame["_name"].ne("") | frame["player_id"].astype(str).str.strip().ne(""))
    ]
    # Confirmed entries are preferred over later projections. Within a class,
    # the latest capture wins. A projection can never shadow a confirmed role.
    frame = frame.sort_values(["_priority", "_reported"], na_position="first", kind="stable")
    return frame.to_dict(orient="records")


def attach_lineups(
    rows: pd.DataFrame,
    path: str | Path = DEFAULT_LINEUP_PATH,
    *,
    auto_path: str | Path | None = None,
    now_utc: str | pd.Timestamp | None = None,
    max_age_hours: float = DEFAULT_LINEUP_MAX_AGE_HOURS,
) -> pd.DataFrame:
    """Attach exact player/team/date roles; only fresh CONFIRMED rows clear the gate.

    The operator-reviewed ledger is backward compatible: rows written by the
    earlier capture utility with no confirmation_status were explicitly entered
    as verified STARTER/BENCH rows. Automated source rows live in a separate
    ledger and must carry confirmation_status explicitly; missing labels default
    to PROJECTED there, never confirmed.
    """
    if max_age_hours < 0:
        raise ValueError("max_age_hours must be non-negative")
    result = rows.copy()
    if result.empty:
        for column in _OUTPUT_COLUMNS:
            if column not in result:
                result[column] = pd.Series(dtype="object")
        return result
    observed = (
        pd.to_datetime(now_utc, utc=True, errors="coerce")
        if now_utc is not None else pd.Timestamp.now(tz="UTC")
    )
    if pd.isna(observed):
        raise ValueError("now_utc must be a valid UTC timestamp")

    manual_path = Path(path)
    automated_path = Path(auto_path) if auto_path is not None else manual_path.parent / DEFAULT_AUTO_LINEUP_PATH.name
    source_rows = _read_lineup_source(manual_path, default_confirmation="CONFIRMED")
    source_rows.extend(_read_lineup_source(automated_path, default_confirmation="PROJECTED"))

    statuses: list[str] = []
    gates: list[str] = []
    reasons: list[str] = []
    sources: list[str] = []
    times: list[str] = []
    ages: list[float | None] = []
    limits: list[float | None] = []
    confirmations: list[str] = []
    confidence_labels: list[str] = []
    source_statuses: list[str] = []
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
            confirmations.append("UNKNOWN")
            confidence_labels.append("LOW")
            source_statuses.append("")
            continue

        stamp = candidate.get("_reported")
        stamp_text = stamp.isoformat().replace("+00:00", "Z") if pd.notna(stamp) else ""
        age = (observed - stamp).total_seconds() / 3600 if pd.notna(stamp) else None
        status = normalize_lineup_status(candidate.get("starting_status"))
        confirmation = _normalize_confirmation(candidate.get("_confirmation"), default="UNKNOWN")
        source_status = str(candidate.get("source_status") or "").strip()
        confidence = str(candidate.get("confidence_label") or "").strip()
        reason_text = str(candidate.get("reason") or "").strip()
        if age is None or age < -5 / 60 or age > max_age_hours:
            gate = "STALE"
            reason = f"Lineup snapshot age is {age if age is not None else 'unknown'} hours; limit is {max_age_hours:.1f}."
        elif status == "UNKNOWN":
            gate, reason = "UNVERIFIED", "Lineup role is blank or unrecognized."
        elif confirmation != "CONFIRMED":
            gate = "UNVERIFIED"
            reason = reason_text or (
                f"Source status is {source_status or confirmation}; a projected lineup cannot satisfy the strict gate."
            )
        else:
            gate = "CLEAR"
            reason = reason_text or f"Latest matched confirmed lineup status: {status}."

        limit_value = pd.to_numeric(pd.Series([candidate.get("minutes_limit")]), errors="coerce").iloc[0]
        limit = float(limit_value) if pd.notna(limit_value) and 0 < float(limit_value) <= 48 else None
        statuses.append(status)
        gates.append(gate)
        reasons.append(reason)
        sources.append(str(candidate.get("source") or ""))
        times.append(stamp_text)
        ages.append(round(max(0.0, age), 2) if age is not None else None)
        limits.append(limit)
        confirmations.append(confirmation)
        confidence_labels.append(confidence or ("VERIFIED" if confirmation == "CONFIRMED" else "PROJECTED"))
        source_statuses.append(source_status)

    result["lineup_status"] = statuses
    result["lineup_gate"] = gates
    result["lineup_reason"] = reasons
    result["lineup_source"] = sources
    result["lineup_reported_at_utc"] = times
    result["lineup_age_hours"] = ages
    result["minutes_limit"] = limits
    result["lineup_confirmation_status"] = confirmations
    result["lineup_confidence_label"] = confidence_labels
    result["lineup_source_status"] = source_statuses
    return result
