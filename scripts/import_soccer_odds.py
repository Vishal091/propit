"""Import timestamped top-five-league odds from The Odds API.

Live fetches are explicit user actions. Historical queries are optional, plan-gated and
cost 10 credits per market per region for each league; the importer prints estimated cost
before sending requests and never calls the historical endpoint implicitly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.soccer.odds import (
    TOP_FIVE_SPORTS,
    OddsProviderError,
    fetch_league_odds,
    upsert_odds_snapshots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import top-five football odds snapshots")
    parser.add_argument("--league", choices=("all", *TOP_FIVE_SPORTS.keys()), default="all")
    parser.add_argument("--regions", default="uk,eu", help="The Odds API regions, default: uk,eu")
    parser.add_argument("--markets", help="h2h,totals or one of them; default live=h2h,totals, historical=h2h")
    parser.add_argument("--historical-date", help="Optional UTC-aware ISO-8601 snapshot timestamp; requires a plan with historical odds")
    parser.add_argument("--output", default="data/soccer/odds_snapshots.csv")
    parser.add_argument("--timeout", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.timeout < 1 or args.timeout > 120:
            raise ValueError("--timeout must be between 1 and 120 seconds")
        regions = [part.strip() for part in args.regions.split(",") if part.strip()]
        if not regions or any(part not in {"us", "us2", "uk", "eu", "au"} for part in regions):
            raise ValueError("--regions must contain only us, us2, uk, eu, au")
        markets = args.markets or ("h2h" if args.historical_date else "h2h,totals")
        market_items = [part.strip() for part in markets.split(",") if part.strip()]
        if not market_items or any(part not in {"h2h", "totals"} for part in market_items):
            raise ValueError("--markets supports h2h and totals only")
        if args.historical_date and len(args.historical_date.strip()) < 16:
            raise ValueError("--historical-date must be a full timezone-aware ISO-8601 timestamp")

        leagues = list(TOP_FIVE_SPORTS) if args.league == "all" else [args.league]
        multiplier = 10 if args.historical_date else 1
        estimated_credits = len(leagues) * len(regions) * len(market_items) * multiplier
        print(f"Mode: {'historical snapshot' if args.historical_date else 'live odds'}")
        print(f"Leagues: {', '.join(leagues)} | regions: {', '.join(regions)} | markets: {', '.join(market_items)}")
        print(f"Estimated maximum API credits: {estimated_credits} (empty responses may cost less).")
        if args.historical_date:
            print("Historical odds require an eligible paid plan. This query is only performed because --historical-date was explicitly supplied.")

        all_rows = []
        errors = []
        for league in leagues:
            try:
                rows = fetch_league_odds(
                    league,
                    regions=",".join(regions),
                    markets=",".join(market_items),
                    historical_date=args.historical_date,
                    timeout_seconds=args.timeout,
                )
                all_rows.extend(rows)
                print(f"{league}: {len(rows)} bookmaker-market-line rows")
            except (OddsProviderError, ValueError) as exc:
                errors.append(f"{league}: {exc}")
                print(f"{league}: FAILED — {exc}", file=sys.stderr)
        if not all_rows and errors:
            return 2
        count = upsert_odds_snapshots(args.output, all_rows)
        print(f"Rows returned this run: {len(all_rows)}")
        print(f"Total unique saved snapshots: {count}")
        print(f"Output: {Path(args.output)}")
        print("Only timestamped, complete h2h markets and matched over/under total lines are retained.")
        print("Odds availability does not establish that a market is legal or available to the user.")
        if errors:
            print(f"Completed with {len(errors)} league error(s); see messages above.", file=sys.stderr)
            return 1
        return 0
    except (ValueError, OSError) as exc:
        print(f"Odds import failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
