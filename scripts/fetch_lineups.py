"""Fetch projected/confirmed NBA lineup slots from RotoWire into a separate ledger.

Projected rows are deliberately kept out of the operator-verified input file.
Only a source-explicit CONFIRMED lineup can satisfy PropIt's strict lineup gate.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.lineup_sources import OUTPUT_COLUMNS, fetch_rotowire_lineups
from src.data.nba_client import get_active_players
from src.lineups import DEFAULT_AUTO_LINEUP_PATH
from src.player_identity import normalize_player_name

DEFAULT_HISTORY = Path("data/player_lineups_auto_history.csv")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, dtype={"player_id": "string"})
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            frame[column] = frame[column].fillna("")
        return frame
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    player_id = str(row.get("player_id") or "").strip().split(".")[0]
    name = normalize_player_name(row.get("player_name"))
    team = str(row.get("team_abbreviation") or "").strip().upper()
    game_date = str(row.get("match_date") or "").strip()[:10]
    return game_date, team, ("id:" + player_id) if player_id else ("name:" + name), team


def write_auto_snapshot(
    rows: pd.DataFrame,
    *,
    output_path: Path = DEFAULT_AUTO_LINEUP_PATH,
    history_path: Path = DEFAULT_HISTORY,
) -> dict[str, Any]:
    """Replace covered teams for one date; preserve other dates and append history."""
    if rows is None or rows.empty:
        raise ValueError("No lineup rows to save; existing current and history files were not changed.")
    frame = rows.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Lineup source rows missing required columns: {', '.join(sorted(missing))}")
    frame = frame[OUTPUT_COLUMNS].copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.loc[frame["match_date"].notna() & frame["team_abbreviation"].astype(str).str.strip().ne("")]
    if frame.empty:
        raise ValueError("Parsed lineup rows have no valid game dates or team abbreviations.")

    game_dates = set(frame["match_date"].astype(str))
    if len(game_dates) != 1:
        raise ValueError("One lineup import must contain exactly one match date.")
    game_date = next(iter(game_dates))
    teams = set(frame["team_abbreviation"].astype(str).str.upper().str.strip())
    old = _read_csv(output_path)
    if not old.empty:
        if "match_date" in old.columns:
            old_dates = pd.to_datetime(old["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            old_dates = pd.Series("", index=old.index)
        old_teams = old.get("team_abbreviation", pd.Series("", index=old.index)).fillna("").astype(str).str.upper().str.strip()
        # A fresh source snapshot replaces all previous rows for teams it
        # actually covers. This prevents removed projected starters surviving
        # a refresh merely because they were absent from the latest source.
        old = old.loc[~(old_dates.eq(game_date) & old_teams.isin(teams))]

    current = pd.concat([old, frame], ignore_index=True, sort=False) if not old.empty else frame.copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    current.to_csv(temporary, index=False)
    temporary.replace(output_path)

    history = _read_csv(history_path)
    accumulated = pd.concat([history, frame], ignore_index=True, sort=False) if not history.empty else frame.copy()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    accumulated.to_csv(history_path, index=False)
    counts = frame["confirmation_status"].fillna("UNKNOWN").astype(str).str.upper().value_counts().to_dict()
    return {
        "game_date": game_date,
        "rows": int(len(frame)),
        "teams": sorted(teams),
        "teams_count": len(teams),
        "status_counts": counts,
        "output_path": str(output_path),
        "history_path": str(history_path),
        "retrieved_at_utc": str(frame["retrieved_at_utc"].iloc[0]),
        "source": str(frame["source"].iloc[0]),
    }


def import_lineups(
    game_date: str,
    *,
    output_path: Path = DEFAULT_AUTO_LINEUP_PATH,
    history_path: Path = DEFAULT_HISTORY,
    url: str | None = None,
) -> dict[str, Any]:
    selected_date = date.fromisoformat(game_date).isoformat()
    roster = None
    try:
        roster = get_active_players()
    except Exception:
        # Name + team matching remains supported; do not fail the source import
        # solely because the separate NBA roster endpoint is unavailable.
        roster = None
    kwargs: dict[str, Any] = {"game_date": selected_date, "roster": roster}
    if url:
        kwargs["url"] = url
    rows, metadata = fetch_rotowire_lineups(**kwargs)
    saved = write_auto_snapshot(rows, output_path=output_path, history_path=history_path)
    return {**metadata, **saved}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-date", required=True, help="NBA-local scheduled game date, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=DEFAULT_AUTO_LINEUP_PATH)
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--url", default=None, help="Override the public lineup page URL (mainly for tests)")
    args = parser.parse_args()
    try:
        date.fromisoformat(args.game_date)
    except ValueError:
        parser.error("--game-date must be a real YYYY-MM-DD date")

    result = import_lineups(
        args.game_date,
        output_path=args.output,
        history_path=args.history_output,
        url=args.url,
    )
    statuses = result.get("status_counts", {})
    print(
        "NBA lineup source imported: "
        f"game_date={result['game_date']} rows={result['rows']} teams={result['teams_count']} "
        f"confirmed={statuses.get('CONFIRMED', 0)} projected={statuses.get('PROJECTED', 0)}\n"
        f"Retrieved at (UTC): {result['retrieved_at_utc']}\n"
        f"Source: {result['source']}\n"
        f"Covered teams: {','.join(result['teams'])}\n"
        f"Auto snapshot: {result['output_path']}\nHistory: {result['history_path']}\n"
        "Safety: projected rows are not promoted to confirmed and do not satisfy the strict shortlist gate."
    )


if __name__ == "__main__":
    main()
