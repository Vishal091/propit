from __future__ import annotations

from urllib.parse import urlsplit

import pandas as pd
import pytest
import requests

from src.data.lineup_sources import (
    LineupSourceUnavailableError,
    fetch_rotowire_lineups,
    parse_rotowire_lineups,
)


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



class _FakeResponse:
    def __init__(self, text: str, url: str, status_code: int = 200):
        self.text = text
        self.url = url
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code} for {self.url}")


class _FakeSession:
    def __init__(self, *, html: str, fail_hosts: set[str]):
        self.html = html
        self.fail_hosts = fail_hosts
        self.calls: list[str] = []

    def get(self, url: str, **kwargs):
        host = urlsplit(url).netloc
        self.calls.append(host)
        if host in self.fail_hosts:
            raise requests.ConnectionError(f"Failed to resolve {host}")
        return _FakeResponse(self.html, url)


def test_fetch_rotowire_uses_public_host_fallback_after_dns_failure(sample_lineup_html):
    session = _FakeSession(
        html=sample_lineup_html,
        fail_hosts={"www.rotowire.com", "rotowire.com"},
    )

    frame, metadata = fetch_rotowire_lineups(
        "2026-10-25",
        session=session,
    )

    assert len(frame) == 10
    assert session.calls[:3] == [
        "www.rotowire.com", "rotowire.com", "aws-prod-web9.rotowire.com",
    ]
    assert metadata["fallback_host_used"] is True
    assert metadata["source"].startswith("https://aws-prod-web9.rotowire.com/")
    assert metadata["attempted_hosts"] == session.calls[:3]


def test_fetch_rotowire_does_not_fail_over_to_unverified_wrong_date(sample_lineup_html):
    class StaleFirstHostSession(_FakeSession):
        def get(self, url: str, **kwargs):
            host = urlsplit(url).netloc
            self.calls.append(host)
            if host == "www.rotowire.com":
                return _FakeResponse(
                    '<h1>NBA Daily Starting Lineups</h1><p>Starting lineups for October 24, 2026</p>',
                    url,
                )
            return _FakeResponse(self.html, url)

    session = StaleFirstHostSession(html=sample_lineup_html, fail_hosts=set())
    frame, metadata = fetch_rotowire_lineups("2026-10-25", session=session)
    assert len(frame) == 10
    assert metadata["fallback_host_used"] is True
    assert metadata["source"].startswith("https://rotowire.com/")


def test_fetch_rotowire_reports_all_dns_failures_clearly_without_traceback():
    session = _FakeSession(html="", fail_hosts={
        "www.rotowire.com", "rotowire.com", "aws-prod-web9.rotowire.com", "web7.rotowire.com",
    })

    with pytest.raises(LineupSourceUnavailableError) as exc:
        fetch_rotowire_lineups("2026-10-25", session=session)

    message = str(exc.value)
    assert "Hosts tried:" in message
    assert "Resolve-DnsName www.rotowire.com" in message
    assert "Existing lineup snapshots were left unchanged" in message
    assert len(session.calls) == 4
