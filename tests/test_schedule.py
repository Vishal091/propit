from __future__ import annotations

from datetime import date

from src.data.schedule import _as_date, build_next_game_map


def _game(day, status, home, away):
    return {
        "gameDateEst": f"{day}T00:00:00",
        "gameStatus": status,
        "gameStatusText": "Final" if status == 3 else "Scheduled",
        "homeTeam": home,
        "awayTeam": away,
    }


def _team(abbreviation, city, name):
    return {"teamTricode": abbreviation, "teamCity": city, "teamName": name}


def test_as_date_parses_us_date_with_meridiem():
    assert _as_date("10/24/2026 7:30:00 PM") == date(2026, 10, 24)


def test_next_game_map_uses_next_non_final_game_and_opponent():
    payload = {
        "leagueSchedule": {
            "gameDates": [
                {
                    "gameDate": "10/23/2026 00:00:00",
                    "games": [
                        _game(
                            "2026-10-23", 3,
                            _team("NYK", "New York", "Knicks"),
                            _team("BOS", "Boston", "Celtics"),
                        )
                    ],
                },
                {
                    "gameDate": "10/25/2026 00:00:00",
                    "games": [
                        _game(
                            "2026-10-25", 1,
                            _team("NYK", "New York", "Knicks"),
                            _team("TOR", "Toronto", "Raptors"),
                        ),
                        _game(
                            "2026-10-26", 1,
                            _team("BOS", "Boston", "Celtics"),
                            _team("MIA", "Miami", "Heat"),
                        ),
                    ],
                },
            ]
        }
    }

    result = build_next_game_map(payload, as_of=date(2026, 10, 24))

    assert result["NYK"]["match_date"] == "2026-10-25"
    assert result["NYK"]["team"] == "New York Knicks"
    assert result["NYK"]["opponent_team"] == "Toronto Raptors"
    assert result["NYK"]["matchup"] == "NYK vs TOR"
    assert result["TOR"]["home_away"] == "AWAY"
    assert "BOS" in result
    assert result["BOS"]["match_date"] == "2026-10-26"


def test_next_game_map_does_not_invent_dates_for_teams_without_upcoming_games():
    payload = {
        "leagueSchedule": {
            "gameDates": [
                {
                    "gameDate": "10/23/2026 00:00:00",
                    "games": [
                        _game(
                            "2026-10-23", 3,
                            _team("NYK", "New York", "Knicks"),
                            _team("BOS", "Boston", "Celtics"),
                        )
                    ],
                }
            ]
        }
    }
    assert build_next_game_map(payload, as_of=date(2026, 10, 24)) == {}
