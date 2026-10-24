"""Produce forecasts for upcoming soccer fixtures or one manually specified match.

This CLI estimates goal rates from imported, completed historical matches. It does not
fetch sportsbook odds, predict lineups, or place bets.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.soccer.model import estimate_expected_goals
from src.soccer.poisson import predict_match


def _date_value(value: str | None, field: str) -> date | None:
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD") from exc


def read_matches(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _score_is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _forecast_record(
    history: list[dict[str, Any]],
    fixture: dict[str, Any],
    total_goals_line: float,
    prior_games: float,
    half_life_days: float,
    lookback_days: int,
    rho: float,
    generated_at_utc: str,
) -> dict[str, Any]:
    comp_id = str(fixture.get("competition_id", "") or "")
    match_date = str(fixture.get("match_date", "") or "")
    home_team = str(fixture.get("home_team", "") or "").strip()
    away_team = str(fixture.get("away_team", "") or "").strip()
    if not comp_id or not match_date or not home_team or not away_team:
        raise ValueError("Fixture requires competition_id, match_date, home_team, and away_team")
    rates = estimate_expected_goals(
        history, comp_id, match_date, home_team, away_team,
        prior_games=prior_games, half_life_days=half_life_days, lookback_days=lookback_days,
    )
    poisson = predict_match(
        rates["expected_home_goals"], rates["expected_away_goals"],
        total_goals_line=total_goals_line, rho=0.0,
    )
    dixon_coles = predict_match(
        rates["expected_home_goals"], rates["expected_away_goals"],
        total_goals_line=total_goals_line, rho=rho,
    )
    result = {
        "forecast_generated_at_utc": generated_at_utc,
        "match_id": fixture.get("match_id", ""),
        "provider": fixture.get("provider", ""),
        "competition_id": comp_id,
        "competition_name": fixture.get("competition_name", ""),
        "season_id": fixture.get("season_id", ""),
        "season_name": fixture.get("season_name", ""),
        "match_date": match_date,
        "kickoff_time_local": fixture.get("kickoff_time_local", ""),
        "kickoff_timezone": fixture.get("kickoff_timezone", ""),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": fixture.get("home_score", ""),
        "away_score": fixture.get("away_score", ""),
        **rates,
        "total_goals_line": total_goals_line,
        "dixon_coles_rho": rho,
    }
    for name, probabilities in (("poisson", poisson), ("dixon_coles", dixon_coles)):
        result.update({
            f"{name}_home_win": probabilities.home_win,
            f"{name}_draw": probabilities.draw,
            f"{name}_away_win": probabilities.away_win,
            f"{name}_over": probabilities.over,
            f"{name}_under": probabilities.under,
            f"{name}_push": probabilities.push,
        })
    return result


def build_upcoming_forecasts(
    rows: list[dict[str, Any]],
    date_from: str,
    date_to: str | None = None,
    competition_id: str | None = None,
    total_goals_line: float = 2.5,
    prior_games: float = 5.0,
    half_life_days: float = 365.0,
    lookback_days: int = 730,
    dixon_coles_rho: float = -0.05,
    generated_at_utc: str | None = None,
) -> list[dict[str, Any]]:
    """Forecast only scheduled rows in a date range; history is cutoff per fixture."""
    start = _date_value(date_from, "date_from")
    end = _date_value(date_to, "date_to")
    if start is None:
        raise ValueError("date_from is required")
    if end is not None and end < start:
        raise ValueError("date_to must be on or after date_from")
    if not math.isfinite(total_goals_line) or total_goals_line < 0:
        raise ValueError("total_goals_line must be finite and non-negative")
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    forecasts = []
    for row in rows:
        comp_id = str(row.get("competition_id", "") or "")
        if competition_id is not None and comp_id != str(competition_id):
            continue
        if not _score_is_blank(row.get("home_score")) or not _score_is_blank(row.get("away_score")):
            continue
        state = str(row.get("match_state", "") or "").strip().casefold()
        if state and state in {"finished", "complete", "completed", "played", "full time", "full_time", "ft"}:
            continue
        match_date = _date_value(str(row.get("match_date", "") or ""), "match_date")
        if match_date is None or match_date < start or (end is not None and match_date > end):
            continue
        if not row.get("home_team") or not row.get("away_team") or not comp_id:
            continue
        forecasts.append(_forecast_record(
            rows, row, total_goals_line, prior_games, half_life_days, lookback_days,
            dixon_coles_rho, generated,
        ))
    return sorted(forecasts, key=lambda row: (
        row["match_date"], row["competition_id"], row["kickoff_time_local"], row["home_team"], row["away_team"]
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast future soccer fixtures from imported match history")
    parser.add_argument("--input", default="data/soccer/matches.csv", help="normalized soccer matches CSV")
    parser.add_argument("--output", default="predictions/soccer/upcoming.csv", help="forecast CSV output")
    parser.add_argument("--date-from", default=date.today().isoformat(), help="first fixture date, default: local today")
    parser.add_argument("--date-to", help="optional last fixture date, inclusive")
    parser.add_argument("--competition-id", help="restrict forecasts to one provider competition ID")
    parser.add_argument("--total-goals-line", type=float, default=2.5)
    parser.add_argument("--prior-games", type=float, default=5.0)
    parser.add_argument("--half-life-days", type=float, default=365.0)
    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--dixon-coles-rho", type=float, default=-0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = read_matches(args.input)
        forecasts = build_upcoming_forecasts(
            rows,
            date_from=args.date_from,
            date_to=args.date_to,
            competition_id=args.competition_id,
            total_goals_line=args.total_goals_line,
            prior_games=args.prior_games,
            half_life_days=args.half_life_days,
            lookback_days=args.lookback_days,
            dixon_coles_rho=args.dixon_coles_rho,
        )
        if not forecasts:
            print("No scheduled fixtures matched the filter. Import future fixtures and check the date/competition filters.", file=sys.stderr)
            return 2
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(forecasts[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(forecasts)
        print(f"Forecasted {len(forecasts)} scheduled fixtures.")
        print(f"Predictions: {output}")
        print("Probabilities are experimental and do not include sportsbook odds or confirmed lineup information.")
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        print(f"Soccer forecast failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
