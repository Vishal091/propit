"""Stamp a user-verified starting lineup into the point-in-time snapshot ledger.

The operator must enter STARTER/BENCH only after checking a trusted, current source.
The tool does not scrape, infer, or auto-promote projected lineups into confirmed ones.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.lineups import DEFAULT_LINEUP_PATH, normalize_lineup_status
from src.player_identity import normalize_player_name

DEFAULT_HISTORY = Path("data/player_lineups_history.csv")
INPUT_COLUMNS = {"player_id", "player_name", "team_abbreviation", "starting_status"}
OUTPUT_COLUMNS = [
    "player_id", "player_name", "team_abbreviation", "match_date",
    "starting_status", "reported_at_utc", "source", "reason", "minutes_limit",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"player_id": "string"})
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def build_snapshot_rows(
    input_frame: pd.DataFrame,
    *,
    game_date: str,
    source: str,
    reported_at_utc: str | None = None,
) -> pd.DataFrame:
    """Validate and stamp a manually verified starting-lineup input table."""
    try:
        selected_date = date.fromisoformat(game_date).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("game_date must be a valid YYYY-MM-DD date") from exc
    source_text = str(source or "").strip()
    if not source_text:
        raise ValueError("A source URL or recognizable source label is required.")
    frame = input_frame.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    missing = INPUT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(sorted(missing))}")
    stamp = pd.to_datetime(
        reported_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        utc=True, errors="coerce",
    )
    if pd.isna(stamp):
        raise ValueError("reported_at_utc must be a valid ISO-8601 timestamp")
    stamped = stamp.isoformat().replace("+00:00", "Z")

    frame["player_id"] = frame["player_id"].fillna("").astype(str).str.strip()
    frame["player_name"] = frame["player_name"].fillna("").astype(str).str.strip()
    frame["team_abbreviation"] = frame["team_abbreviation"].fillna("").astype(str).str.strip().str.upper()
    frame["starting_status"] = frame["starting_status"].map(normalize_lineup_status)
    frame["reason"] = frame["reason"].fillna("").astype(str) if "reason" in frame else ""
    if "minutes_limit" not in frame:
        frame["minutes_limit"] = ""
    frame["minutes_limit"] = pd.to_numeric(frame["minutes_limit"], errors="coerce")
    frame["minutes_limit"] = frame["minutes_limit"].where(frame["minutes_limit"].between(1, 48), None)

    invalid_identity = frame["player_name"].eq("") & frame["player_id"].eq("")
    if invalid_identity.any():
        raise ValueError("Every input row needs player_id or player_name.")
    if frame["team_abbreviation"].eq("").any():
        raise ValueError("Every lineup row requires a team_abbreviation.")
    if frame["starting_status"].eq("UNKNOWN").any():
        raise ValueError("Every lineup status must be an explicit STARTER or BENCH after verification.")
    normalized_keys = frame.apply(
        lambda row: (
            row["player_id"] if row["player_id"] else row["player_name"].casefold(),
            row["team_abbreviation"],
        ),
        axis=1,
    )
    if normalized_keys.duplicated().any():
        raise ValueError("Duplicate player/team rows found in the input file.")

    frame["match_date"] = selected_date
    frame["reported_at_utc"] = stamped
    frame["source"] = source_text
    return frame[OUTPUT_COLUMNS].reset_index(drop=True)


def write_snapshot(
    input_path: Path,
    *,
    game_date: str,
    source: str,
    output_path: Path = DEFAULT_LINEUP_PATH,
    history_path: Path = DEFAULT_HISTORY,
    reported_at_utc: str | None = None,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Lineup input file not found: {input_path}")
    input_frame = pd.read_csv(input_path, dtype={"player_id": "string"})
    rows = build_snapshot_rows(
        input_frame,
        game_date=game_date,
        source=source,
        reported_at_utc=reported_at_utc,
    )
    old = _read(output_path)
    if not old.empty and "match_date" in old.columns:
        old_dates = pd.to_datetime(old["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        new_ids = set(rows["player_id"].fillna("").astype(str).str.strip().loc[lambda value: value.ne("")])
        new_name_team = {
            (normalize_player_name(row.get("player_name")), str(row.get("team_abbreviation") or "").upper().strip())
            for row in rows.to_dict(orient="records")
        }
        old_ids = old.get("player_id", pd.Series("", index=old.index)).fillna("").astype(str).str.strip()
        old_names = old.get("player_name", pd.Series("", index=old.index)).map(normalize_player_name)
        old_teams = old.get("team_abbreviation", pd.Series("", index=old.index)).fillna("").astype(str).str.upper().str.strip()
        replace_row = old_dates.eq(game_date) & (
            old_ids.isin(new_ids)
            | pd.Series(
                [(name, team) in new_name_team for name, team in zip(old_names, old_teams)],
                index=old.index,
            )
        )
        # Preserve other players already captured for the same slate date. This
        # lets the operator update one starter or return limit without erasing
        # still-fresh rows for teammates captured earlier.
        old = old.loc[~replace_row]
    current = pd.concat([old, rows], ignore_index=True, sort=False) if not old.empty else rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    current.to_csv(temp, index=False)
    temp.replace(output_path)

    hist = _read(history_path)
    all_rows = pd.concat([hist, rows], ignore_index=True, sort=False) if not hist.empty else rows
    history_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(history_path, index=False)
    return {
        "game_date": game_date,
        "rows": len(rows),
        "reported_at_utc": rows["reported_at_utc"].iloc[0],
        "source": source,
        "output_path": str(output_path),
        "history_path": str(history_path),
        "status_counts": rows["starting_status"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-date", required=True, help="NBA-local game date, YYYY-MM-DD")
    parser.add_argument("--input", type=Path, default=Path("data/player_lineups_input.csv"))
    parser.add_argument("--source", required=True, help="Source URL or human-readable source label")
    parser.add_argument("--reported-at-utc", default=None, help="Optional ISO-8601 timestamp; defaults to now (UTC)")
    parser.add_argument("--output", type=Path, default=DEFAULT_LINEUP_PATH)
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    try:
        date.fromisoformat(args.game_date)
    except ValueError:
        parser.error("--game-date must be a real YYYY-MM-DD date")
    result = write_snapshot(
        args.input,
        game_date=args.game_date,
        source=args.source,
        output_path=args.output,
        history_path=args.history_output,
        reported_at_utc=args.reported_at_utc,
    )
    print(
        f"Lineup snapshot saved: date={result['game_date']} rows={result['rows']} "
        f"reported_at_utc={result['reported_at_utc']}\n"
        f"Source: {result['source']}\nCurrent: {result['output_path']}\nHistory: {result['history_path']}"
    )


if __name__ == "__main__":
    main()
