from __future__ import annotations

from datetime import date, timezone

import pandas as pd
import pytest

from src.data.injury_report import (
    parse_injury_table_rows,
    positioned_words_to_rows,
    report_timestamp_from_url,
    select_report_url,
)
from src.player_identity import normalize_player_name


def test_normalize_official_last_first_player_name():
    assert normalize_player_name("Butler III, Jimmy") == normalize_player_name("Jimmy Butler")
    assert normalize_player_name("Oubre Jr., Kelly") == normalize_player_name("Kelly Oubre")


def test_report_timestamp_is_parsed_in_eastern_time():
    stamp = report_timestamp_from_url(
        "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-25_01_15AM.pdf"
    )
    assert stamp is not None
    assert stamp.isoformat() == "2026-10-25T01:15:00-04:00"
    assert stamp.astimezone(timezone.utc).isoformat() == "2026-10-25T05:15:00+00:00"


def test_select_latest_report_for_target_game_date():
    html = '''
    <a href="https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-25_01_00AM.pdf">1:00</a>
    <a href="https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-25_02_15AM.pdf">2:15</a>
    <a href="https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-24_11_30PM.pdf">prior</a>
    '''
    selected = select_report_url(html, "2026-10-25")
    assert selected.endswith("Injury-Report_2026-10-25_02_15AM.pdf")


def test_table_parser_carries_team_and_ignores_not_yet_submitted():
    tables = [[
        ["Date", "Time (ET)", "Game", "Team", "Player", "Status", "Injury/Illness"],
        ["", "07:00 (ET)", "BOS@NYK", "Boston Celtics", "Tatum, Jayson", "Questionable", "Injury/Illness - Wrist"],
        ["", "", "", "", "Brown, Jaylen", "Available", ""],
        ["", "", "", "New York Knicks", "", "", "NOT YET SUBMITTED"],
        ["10/26/2026", "07:00 (ET)", "MIA@BOS", "Miami Heat", "Player X", "Out", "Foot"],
    ]]
    parsed = parse_injury_table_rows(
        tables, report_date="2026-10-25", target_date="2026-10-25",
        report_url="https://example.test/report.pdf", reported_at_utc="2026-10-25T05:00:00Z",
    )
    assert len(parsed) == 2
    assert set(parsed["player_name"]) == {"Tatum, Jayson", "Brown, Jaylen"}
    assert set(parsed["team_abbreviation"]) == {"BOS"}
    assert parsed.iloc[0]["matchup"] == "BOS@NYK"


def test_parser_rejects_non_player_metadata_and_unknown_status_rows():
    parsed = parse_injury_table_rows(
        [[
            ["Date", "Time", "Game", "Team", "Player", "Status", "Injury/Illness"],
            ["", "", "", "", "10/25/26 06:45", "UNKNOWN", ""],
            ["", "", "", "", "Example Player", "UNKNOWN", "Report metadata"],
        ]],
        report_date="2026-10-25",
        target_date="2026-10-25",
        report_url="https://example.test/official.pdf",
        reported_at_utc="2026-10-25T10:45:00Z",
    )
    assert parsed.empty


def test_parser_does_not_backfill_absent_players_as_available():
    parsed = parse_injury_table_rows(
        [[["Date", "Time", "Game", "Team", "Player", "Status", "Injury/Illness"],
          ["", "07:00 (ET)", "BOS@NYK", "Boston Celtics", "", "", "NOT YET SUBMITTED"]]],
        report_date=date(2026, 10, 25), target_date=date(2026, 10, 25),
        report_url="x", reported_at_utc="2026-10-25T05:00:00Z",
    )
    assert parsed.empty



def test_positioned_word_fallback_reconstructs_official_seven_columns():
    words = []
    def add(top, x0, text):
        words.append({"top": top, "x0": x0, "text": text})

    # Use the NBA report's post-2024 column layout scaled to an 842-point page.
    for top, values in [
        (100, [
            (20, "10/25/2026"), (120, "07:00 (ET)"), (195, "BOS@NYK"),
            (275, "Boston Celtics"), (390, "Tatum, Jayson"),
            (565, "Questionable"), (675, "Injury/Illness - Wrist"),
        ]),
        (120, [(390, "Brown, Jaylen"), (565, "Available"), (675, "G League - Two-Way")]),
        (140, [(275, "New York Knicks"), (390, "Brunson, Jalen"), (565, "Out"), (675, "Injury/Illness - Ankle")]),
    ]:
        for x0, text in values:
            add(top, x0, text)

    rows = positioned_words_to_rows(words, 842.0)
    assert len(rows) == 3
    parsed = parse_injury_table_rows(
        [rows],
        report_date="2026-10-25",
        target_date="2026-10-25",
        report_url="https://example.test/official.pdf",
        reported_at_utc="2026-10-25T05:00:00Z",
    )
    assert len(parsed) == 3
    assert parsed.iloc[0]["player_name"] == "Tatum, Jayson"
    assert parsed.iloc[0]["team_abbreviation"] == "BOS"
    assert parsed.iloc[1]["team_abbreviation"] == "BOS"
    assert parsed.iloc[2]["team_abbreviation"] == "NYK"
    assert parsed.iloc[2]["status"] == "Out"
