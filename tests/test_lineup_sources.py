from __future__ import annotations

import pandas as pd
import pytest

from src.data.lineup_sources import parse_rotowire_lineups


def _side(side: str, team: str, status: str, players: list[tuple[str, str]]) -> str:
    items = "".join(
        f'<li><div class="lineup__pos">{position}</div><a title="{name}">{name}</a></li>'
        for position, name in players
    )
    return f"""
    <div class="lineup__side is-{side}">
      <a class="lineup__mteam is-{side} white"><span class="lineup__abbr">{team}</span></a>
      <div class="lineup__status">{status}</div>
      <ul class="lineup__list is-{side}">{items}</ul>
    </div>
    """


@pytest.fixture
def sample_lineup_html() -> str:
    away = _side("visit", "BOS", "Expected Lineup", [
        ("PG", "Jayson Tatum"), ("SG", "Derrick White"), ("SF", "Jaylen Brown"),
        ("PF", "Al Horford"), ("C", "Kristaps Porzingis"),
    ])
    home = _side("home", "NYK", "Confirmed Lineup", [
        ("PG", "Jalen Brunson"), ("SG", "Josh Hart"), ("SF", "Mikal Bridges"),
        ("PF", "OG Anunoby"), ("C", "Karl-Anthony Towns"),
    ])
    return f'<h1>NBA Daily Starting Lineups</h1><p>Starting lineups for October 25, 2026</p><div class="lineup is-nba">{away}{home}</div>'


def test_rotiwire_parser_separates_expected_and_explicitly_confirmed(sample_lineup_html):
    frame, metadata = parse_rotowire_lineups(
        sample_lineup_html,
        game_date="2026-10-25",
        source_url="https://www.rotowire.com/basketball/nba-lineups.php?date=2026-10-25",
        retrieved_at_utc="2026-10-25T14:00:00Z",
    )

    assert len(frame) == 10
    assert metadata["teams"] == ["BOS", "NYK"]
    assert frame.loc[frame["team_abbreviation"] == "BOS", "confirmation_status"].unique().tolist() == ["PROJECTED"]
    assert frame.loc[frame["team_abbreviation"] == "NYK", "confirmation_status"].unique().tolist() == ["CONFIRMED"]
    assert set(frame["starting_status"]) == {"STARTER"}
    assert set(frame.loc[frame["confirmation_status"] == "PROJECTED", "confidence_label"]) == {"PROJECTED"}
    assert set(frame.loc[frame["confirmation_status"] == "CONFIRMED", "confidence_label"]) == {"SOURCE_CONFIRMED"}


def test_rotowire_parser_resolves_player_ids_by_full_name_and_team(sample_lineup_html):
    roster = pd.DataFrame([
        {"PLAYER_ID": 1628369, "PLAYER_NAME": "Jayson Tatum", "TEAM_ABBREVIATION": "BOS"},
        {"PLAYER_ID": 1628973, "PLAYER_NAME": "Jalen Brunson", "TEAM_ABBREVIATION": "NYK"},
    ])
    frame, _ = parse_rotowire_lineups(
        sample_lineup_html, game_date="2026-10-25", roster=roster,
    )
    assert frame.loc[frame["player_name"] == "Jayson Tatum", "player_id"].iloc[0] == "1628369"
    assert frame.loc[frame["player_name"] == "Jalen Brunson", "player_id"].iloc[0] == "1628973"


def test_rotowire_parser_rejects_wrong_date_and_missing_date():
    html = '<h1>NBA Daily Starting Lineups</h1><p>Starting lineups for October 25, 2026</p>'
    with pytest.raises(ValueError, match="not requested"):
        parse_rotowire_lineups(html, game_date="2026-10-26")
    with pytest.raises(ValueError, match="date could not be verified"):
        parse_rotowire_lineups("<html><body>lineup</body></html>", game_date="2026-10-25")


def test_rotowire_parser_fails_when_markup_has_no_lineup_cards():
    html = '<h1>NBA Daily Starting Lineups</h1><p>Starting lineups for October 25, 2026</p>'
    with pytest.raises(ValueError, match="markup was not found"):
        parse_rotowire_lineups(html, game_date="2026-10-25")
