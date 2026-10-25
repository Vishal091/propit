"""Import the latest official NBA injury-report PDF into timestamped local snapshots.

Usage: python -m scripts.import_nba_status --game-date YYYY-MM-DD

Only player rows explicitly listed in the selected official report are saved.
Players missing from the report remain unverified in the slate; they are not
assumed available. The script keeps a per-date current snapshot and an
append-only import history for audit.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.availability import DEFAULT_AVAILABILITY_PATH, normalize_status
from src.data.injury_report import fetch_official_injury_report
from src.data.nba_client import get_active_players
from src.player_identity import normalize_player_name

log = logging.getLogger("import_nba_status")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_HISTORY = Path("data/player_availability_history.csv")
DEFAULT_IMPORTS = Path("data/injury_report_imports.csv")
OUTPUT_COLUMNS = [
    "player_id", "player_name", "team_abbreviation", "match_date", "status",
    "reason", "source", "reported_at_utc", "retrieved_at_utc", "minutes_limit",
]


def _resolve_player_ids(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = frame.copy()
    result["player_id"] = ""
    try:
        roster = get_active_players()
    except Exception as exc:
        log.warning("Could not load current roster IDs; availability will match by normalized name/team: %s", exc)
        return result, 0
    if roster.empty or not {"PLAYER_ID", "PLAYER_NAME"}.issubset(roster.columns):
        log.warning("Current roster data has no usable player IDs; name-based matching will be used.")
        return result, 0

    roster = roster.copy()
    roster["_name_key"] = roster["PLAYER_NAME"].map(normalize_player_name)
    roster["_team_key"] = roster.get("TEAM_ABBREVIATION", pd.Series("", index=roster.index)).fillna("").astype(str).str.upper().str.strip()
    exact: dict[tuple[str, str], str] = {}
    by_name: dict[str, set[str]] = {}
    for row in roster.to_dict(orient="records"):
        key = str(row.get("_name_key") or "")
        team = str(row.get("_team_key") or "")
        pid = str(int(float(row["PLAYER_ID"]))) if pd.notna(row.get("PLAYER_ID")) else ""
        if not key or not pid:
            continue
        exact[(key, team)] = pid
        by_name.setdefault(key, set()).add(pid)

    resolved = 0
    for index, row in result.iterrows():
        key = normalize_player_name(row.get("player_name"))
        team = str(row.get("team_abbreviation") or "").upper().strip()
        player_id = exact.get((key, team), "") if team else ""
        if not player_id and len(by_name.get(key, set())) == 1:
            player_id = next(iter(by_name[key]))
        if player_id:
            result.at[index, "player_id"] = player_id
            resolved += 1
    return result, resolved


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"player_id": "string"})
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    old = _read_csv(path)
    combined = pd.concat([old, frame], ignore_index=True, sort=False) if not old.empty else frame.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def import_report(
    game_date: str,
    *,
    output_path: Path = DEFAULT_AVAILABILITY_PATH,
    history_path: Path = DEFAULT_HISTORY,
    imports_path: Path = DEFAULT_IMPORTS,
) -> dict[str, Any]:
    selected_date = date.fromisoformat(game_date).isoformat()
    report, metadata = fetch_official_injury_report(selected_date)
    report, resolved_count = _resolve_player_ids(report)
    retrieved_at = str(metadata["retrieved_at_utc"])
    records = pd.DataFrame({
        "player_id": report.get("player_id", pd.Series("", index=report.index)).fillna("").astype(str),
        "player_name": report["player_name"].astype(str),
        "team_abbreviation": report["team_abbreviation"].fillna("").astype(str),
        "match_date": report["scheduled_date"].astype(str),
        "status": report["status"].map(normalize_status),
        "reason": report["reason"].fillna("").astype(str),
        "source": report["report_url"].astype(str),
        "reported_at_utc": report["reported_at_utc"].astype(str),
        "retrieved_at_utc": retrieved_at,
        "minutes_limit": "",
    }, columns=OUTPUT_COLUMNS)
    if records.empty:
        raise RuntimeError("Official report produced no player status rows. No output was changed.")

    existing = _read_csv(output_path)
    if not existing.empty and "match_date" in existing.columns:
        existing = existing.loc[
            pd.to_datetime(existing["match_date"], errors="coerce").dt.strftime("%Y-%m-%d") != selected_date
        ]
    current = pd.concat([existing, records], ignore_index=True, sort=False) if not existing.empty else records
    _write_csv_atomic(current, output_path)

    # Append-only history and import metadata preserve which report was actually
    # used at prediction time; a later PDF does not rewrite this history.
    _append_csv(records, history_path)
    meta = {
        **metadata,
        "teams_not_yet_submitted_json": json.dumps(
            metadata.get("teams_not_yet_submitted", []), ensure_ascii=False
        ),
        "team_report_coverage_parsed": bool(metadata.get("team_report_coverage_parsed", False)),
        "output_path": str(output_path),
        "history_path": str(history_path),
        "resolved_player_ids": int(resolved_count),
        "unresolved_player_ids": int(len(records) - resolved_count),
        "status_counts": records["status"].value_counts(dropna=False).to_dict(),
    }
    # Store team lists as JSON text in the CSV audit row, not Python repr strings.
    meta.pop("teams_not_yet_submitted", None)
    _append_csv(pd.DataFrame([meta]), imports_path)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-date", required=True, help="NBA-local scheduled match date, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=DEFAULT_AVAILABILITY_PATH)
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--imports-output", type=Path, default=DEFAULT_IMPORTS)
    args = parser.parse_args()
    try:
        date.fromisoformat(args.game_date)
    except ValueError:
        parser.error("--game-date must be a real YYYY-MM-DD date")
    result = import_report(
        args.game_date,
        output_path=args.output,
        history_path=args.history_output,
        imports_path=args.imports_output,
    )
    print(
        "Official NBA injury report imported: "
        f"game_date={result['game_date']} players={result['players']} rows={result['rows_parsed']} "
        f"resolved_ids={result['resolved_player_ids']} unresolved_ids={result['unresolved_player_ids']}\n"
        f"Report timestamp (UTC): {result['reported_at_utc']}\n"
        f"Source: {result['report_url']}\n"
        f"Current snapshot: {args.output}\nHistory: {args.history_output}"
    )
    if result["unresolved_player_ids"]:
        print("Warning: unmatched roster IDs will use normalized player-name/team matching; review unresolved rows.")


if __name__ == "__main__":
    main()
