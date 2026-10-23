from datetime import date

import pytest

from src.data.seasons import completed_seasons, current_season, recent_seasons


def test_dynamic_seasons_in_october_2026() -> None:
    current = date(2026, 10, 23)
    assert current_season(current) == "2026-27"
    assert completed_seasons(current, count=2) == ["2024-25", "2025-26"]
    assert recent_seasons(current_season(current), count=3) == ["2024-25", "2025-26", "2026-27"]


def test_season_before_october_belongs_to_previous_start_year() -> None:
    assert current_season(date(2026, 9, 1)) == "2025-26"
    assert completed_seasons(date(2026, 9, 1), count=2) == ["2023-24", "2024-25"]


def test_invalid_season_count() -> None:
    with pytest.raises(ValueError):
        completed_seasons(date(2026, 10, 23), count=0)
