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
