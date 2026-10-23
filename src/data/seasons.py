"""NBA season helpers that do not need an API key or external calendar service."""
from __future__ import annotations

from datetime import date


def season_start_year(as_of: date | None = None) -> int:
    """Return the start year of the NBA season in progress on as_of.

    NBA regular seasons begin in October. Between January and September, the
    season that began in the previous calendar year is still the current one.
    """
    as_of = as_of or date.today()
    return as_of.year if as_of.month >= 10 else as_of.year - 1


def season_label(start_year: int) -> str:
    """Convert a season start year such as 2026 to 2026-27."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season(as_of: date | None = None) -> str:
    """Return the current season label, e.g. 2026-27."""
    return season_label(season_start_year(as_of))


def completed_seasons(as_of: date | None = None, count: int = 2) -> list[str]:
    """Return the latest count completed seasons, oldest first."""
    if count < 1:
        raise ValueError("count must be at least 1")
    start = season_start_year(as_of)
    return [season_label(year) for year in range(start - count, start)]


def recent_seasons(latest_season: str, count: int = 3) -> list[str]:
    """Return count seasons ending with latest_season, oldest first."""
    try:
        latest_start = int(latest_season.split("-", maxsplit=1)[0])
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid season label: {latest_season!r}") from exc
    if count < 1:
        raise ValueError("count must be at least 1")
    return [season_label(year) for year in range(latest_start - count + 1, latest_start + 1)]
