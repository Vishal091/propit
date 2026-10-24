from __future__ import annotations

from scripts.fetch_public_props import normalize_name, parse_odds_and_book, parse_props_html


def test_normalize_name_accent_and_punctuation():
    assert normalize_name("Nikola Jokić") == normalize_name("Nikola Jokic")
    assert normalize_name("Kel'el Ware") == normalize_name("Kelel-Ware")


def test_parse_american_odds_unicode_minus():
    assert parse_odds_and_book("DK−128 (opens DraftKings in a new tab)") == (-128.0, "DK")
    assert parse_odds_and_book("FD+104") == (104.0, "FD")


def test_parse_public_table_and_skip_integer_lines():
    html = '''
    <html><body><table>
      <thead><tr><th>Save</th><th>Player</th><th>Game</th><th>Line</th><th>Over</th><th>Under</th></tr></thead>
      <tbody>
       <tr><td>☆</td><td><a href="/player/test">Test Player</a></td><td>AAA@BBB</td><td>20.5</td><td><a>DK−110</a></td><td><a>FD+100</a></td></tr>
       <tr><td>☆</td><td><a href="/player/another">Another Player</a></td><td>AAA@BBB</td><td>20</td><td><a>DK−110</a></td><td><a>FD+100</a></td></tr>
      </tbody>
    </table></body></html>'''
    rows = parse_props_html(html, "PTS", {"testplayer": (12, "Test Player")})
    assert len(rows) == 1
    assert rows[0]["player_id"] == 12
    assert rows[0]["line"] == 20.5
    assert rows[0]["odds_over"] == -110
    assert rows[0]["odds_under"] == 100
    assert rows[0]["target"] == "PTS"

def test_parse_card_layout_with_div_rows_when_no_html_table_exists():
    html = """
    <html><body>
      <div class="prop-row">
        <span>☆</span><span>Aaron Nesmith</span><span>2/ 11.5</span>
        <span>11.5</span><span>DK−113 (opens DraftKings in a new tab)</span>
        <span>FD−102 (opens FanDuel in a new tab)</span>
      </div>
      <div class="prop-row">
        <span>☆</span><span>Test Guard</span><span>8/ 7.5</span>
        <span>7.5</span><span>MGM+105 (opens BetMGM in a new tab)</span>
        <span>DK−115 (opens DraftKings in a new tab)</span>
      </div>
    </body></html>
    """
    players = {
        normalize_name("Aaron Nesmith"): (100, "Aaron Nesmith"),
        normalize_name("Test Guard"): (200, "Test Guard"),
    }
    rows = parse_props_html(html, "PTS", players)
    by_name = {row["player_name"]: row for row in rows}
    assert set(by_name) == {"Aaron Nesmith", "Test Guard"}
    assert by_name["Aaron Nesmith"]["line"] == 11.5
    assert by_name["Aaron Nesmith"]["odds_over"] == -113
    assert by_name["Aaron Nesmith"]["odds_under"] == -102
    assert by_name["Test Guard"]["line"] == 7.5
    assert by_name["Test Guard"]["odds_over"] == 105
    assert by_name["Test Guard"]["odds_under"] == -115


def test_parse_card_layout_still_skips_integer_lines():
    html = """
    <html><body><div class="prop-row">
      <span>Test Guard</span><span>5/ 8.0</span><span>8.0</span>
      <span>DK−110</span><span>FD−110</span>
    </div></body></html>
    """
    rows = parse_props_html(html, "PTS", {normalize_name("Test Guard"): (200, "Test Guard")})
    assert rows == []

def test_parse_bare_positive_american_odds_as_underdog():
    assert parse_odds_and_book("165") == (165.0, "Underdog")
    assert parse_odds_and_book("+104") == (104.0, "Underdog")
    assert parse_odds_and_book("—") == (None, "")


def test_parse_flash_odds_public_html_table_with_bare_odds():
    html = """
    <html><body><table>
      <thead><tr><th>Player</th><th>Open</th><th>Line</th><th>Move</th><th>Over</th><th>Under</th><th>Match</th></tr></thead>
      <tbody>
        <tr><td>Aaron Wiggins</td><td>9.5</td><td>9.5</td><td>—</td><td>165</td><td>-204</td><td>game</td></tr>
        <tr><td>Alex Sarr</td><td>13.5</td><td>14.5</td><td>+1</td><td>104</td><td>-128</td><td>game</td></tr>
        <tr><td>Test Integer</td><td>8.0</td><td>8.0</td><td>—</td><td>105</td><td>-125</td><td>game</td></tr>
      </tbody>
    </table></body></html>
    """
    players = {
        normalize_name("Aaron Wiggins"): (11, "Aaron Wiggins"),
        normalize_name("Alex Sarr"): (22, "Alex Sarr"),
        normalize_name("Test Integer"): (33, "Test Integer"),
    }
    rows = parse_props_html(html, "PTS", players)
    by_name = {row["player_name"]: row for row in rows}
    assert set(by_name) == {"Aaron Wiggins", "Alex Sarr"}
    assert by_name["Aaron Wiggins"]["line"] == 9.5
    assert by_name["Aaron Wiggins"]["odds_over"] == 165
    assert by_name["Aaron Wiggins"]["odds_under"] == -204
    assert by_name["Aaron Wiggins"]["bookmaker"] == "Flash Odds public board (over Underdog; under Underdog)"
    assert by_name["Alex Sarr"]["line"] == 14.5
    assert by_name["Alex Sarr"]["odds_over"] == 104

