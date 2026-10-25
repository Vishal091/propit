from __future__ import annotations

import pandas as pd
import pytest

from scripts.fetch_lineups import write_auto_snapshot
from src.data.lineup_sources import OUTPUT_COLUMNS


def _rows(team="BOS", player="Jayson Tatum", status="PROJECTED", day="2026-10-25"):
    return pd.DataFrame([{
        "player_id": "1628369" if player == "Jayson Tatum" else "",
        "player_name": player,
        "team_abbreviation": team,
        "match_date": day,
        "starting_status": "STARTER",
        "confirmation_status": status,
        "confidence_label": "PROJECTED" if status == "PROJECTED" else "SOURCE_CONFIRMED",
        "source_status": "Expected Lineup" if status == "PROJECTED" else "Confirmed Lineup",
        "reported_at_utc": "2026-10-25T14:30:00Z",
        "source_reported_at_utc": "",
        "retrieved_at_utc": "2026-10-25T14:30:00Z",
        "source": "https://www.rotowire.com/basketball/nba-lineups.php?date=" + day,
        "reason": "Fixture",
        "minutes_limit": "",
    }], columns=OUTPUT_COLUMNS)


def test_auto_snapshot_replaces_only_teams_covered_in_this_fetch(tmp_path):
    current_path = tmp_path / "player_lineups_auto.csv"
    history_path = tmp_path / "player_lineups_auto_history.csv"
    older = pd.concat([
        _rows(team="BOS", player="Old Boston Starter"),
        _rows(team="NYK", player="Jalen Brunson"),
        _rows(team="BOS", player="Boston Future Starter", day="2026-10-26"),
    ], ignore_index=True)
    older.to_csv(current_path, index=False)

    result = write_auto_snapshot(
        _rows(team="BOS", player="Jayson Tatum", status="CONFIRMED"),
        output_path=current_path,
        history_path=history_path,
    )

    current = pd.read_csv(current_path, dtype=str).fillna("")
    assert result["rows"] == 1
    assert result["teams"] == ["BOS"]
    assert set(current["player_name"]) == {"Jayson Tatum", "Jalen Brunson", "Boston Future Starter"}
    assert len(pd.read_csv(history_path)) == 1


def test_auto_snapshot_rejects_empty_rows_without_touching_existing_files(tmp_path):
    current_path = tmp_path / "player_lineups_auto.csv"
    history_path = tmp_path / "player_lineups_auto_history.csv"
    current_path.write_text("sentinel", encoding="utf-8")
    history_path.write_text("history-sentinel", encoding="utf-8")

    with pytest.raises(ValueError, match="No lineup rows"):
        write_auto_snapshot(pd.DataFrame(columns=OUTPUT_COLUMNS), output_path=current_path, history_path=history_path)

    assert current_path.read_text(encoding="utf-8") == "sentinel"
    assert history_path.read_text(encoding="utf-8") == "history-sentinel"
