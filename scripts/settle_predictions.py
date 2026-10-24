"""Settle archived PropIt predictions from free NBA public player game logs.

Only players found in the specified game's logs are settled. DNPs, missing games,
and unavailable data remain pending rather than being mislabeled as losses.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.nba_client import get_league_game_log
from src.data.seasons import current_season

DEFAULT_HISTORY = Path("data/prediction_history.csv")
STAT_BY_TARGET = {"PTS": "PTS", "REB": "REB", "AST": "AST"}


def settle_history_rows(history: pd.DataFrame, games: pd.DataFrame, game_date: str) -> tuple[pd.DataFrame, int, int]:
    """Settle matching predictions; preserve pending rows and pushes transparently."""
    game_day = date.fromisoformat(game_date).isoformat()
    out = history.copy()
    for column in ("actual_stat", "actual_over"):
        if column not in out.columns:
            out[column] = pd.NA
    if "settlement_status" not in out.columns:
        out["settlement_status"] = "PENDING"
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype("string")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["actual_stat"] = pd.to_numeric(out["actual_stat"], errors="coerce")
    out["actual_over"] = pd.to_numeric(out["actual_over"], errors="coerce")

    game_rows = games.copy()
    needed = {"GAME_DATE", "PLAYER_ID", "PTS", "REB", "AST"}
    missing = needed - set(game_rows.columns)
    if missing:
        raise ValueError(f"NBA game logs are missing columns: {', '.join(sorted(missing))}")
    game_rows["GAME_DATE"] = pd.to_datetime(game_rows["GAME_DATE"], errors="coerce").dt.date.astype("string")
    game_rows["PLAYER_ID"] = pd.to_numeric(game_rows["PLAYER_ID"], errors="coerce")
    game_rows = game_rows[game_rows["GAME_DATE"].eq(game_day)].dropna(subset=["PLAYER_ID"])
    lookup: dict[tuple[int, str], float] = {}
    for row in game_rows.to_dict("records"):
        player_id = int(row["PLAYER_ID"])
        for target, stat_col in STAT_BY_TARGET.items():
            value = pd.to_numeric(pd.Series([row.get(stat_col)]), errors="coerce").iloc[0]
            if pd.notna(value):
                lookup[(player_id, target)] = float(value)

    date_mask = out["game_date"].eq(game_day)
    pending_mask = out["actual_stat"].isna() & date_mask
    settled_count = push_count = 0
    for index in out.index[pending_mask]:
        player_id = out.at[index, "player_id"]
        target = str(out.at[index, "target"]).upper().strip()
        line = out.at[index, "line"]
        if pd.isna(player_id) or target not in STAT_BY_TARGET or pd.isna(line):
            continue
        actual = lookup.get((int(player_id), target))
        if actual is None:
            continue
        out.at[index, "actual_stat"] = actual
        if actual == float(line):
            out.at[index, "actual_over"] = pd.NA
            out.at[index, "settlement_status"] = "PUSH"
            push_count += 1
        else:
            out.at[index, "actual_over"] = int(actual > float(line))
            out.at[index, "settlement_status"] = "SETTLED"
            settled_count += 1
    out["game_date"] = out["game_date"].astype(str)
    return out, settled_count, push_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-date", required=True, help="NBA-local game date (YYYY-MM-DD)")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    try:
        game_day = date.fromisoformat(args.game_date)
    except ValueError:
        parser.error("--game-date must use YYYY-MM-DD format")
    if not args.history.exists():
        raise SystemExit(f"prediction history not found: {args.history}; archive a pregame slate first.")

    history = pd.read_csv(args.history)
    if history.empty:
        raise SystemExit(f"prediction history is empty: {args.history}")
    season = current_season(game_day)
    try:
        games = get_league_game_log(season)
    except Exception as exc:
        raise SystemExit(f"could not load NBA game logs for {season}: {exc}") from exc
    rows_for_day = games[pd.to_datetime(games["GAME_DATE"], errors="coerce").dt.date.eq(game_day)]
    if rows_for_day.empty:
        raise SystemExit(
            f"No player game logs returned for {game_day}. The public feed may not be updated yet; "
            "retry later. No history rows were changed."
        )
    try:
        updated, settled_count, push_count = settle_history_rows(history, games, game_day.isoformat())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    updated.to_csv(args.history, index=False)
    pending = int(
        updated["game_date"].astype(str).eq(game_day.isoformat()).sum()
        - updated.loc[updated["game_date"].astype(str).eq(game_day.isoformat()), "actual_stat"].notna().sum()
    )
    print(f"Updated {args.history} for {game_day.isoformat()}: settled={settled_count}, pushes={push_count}, still_pending={pending}.")
    print("Players without a matching appearance in the public game log remain pending; verify DNP/postponement cases manually.")


if __name__ == "__main__":
    main()
