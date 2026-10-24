"""Import soccer fixtures/results into PropIt's normalized CSV dataset.

Examples (PowerShell):
  python -m scripts.import_soccer_data --provider statsbomb --list-competitions --search Premier
  python -m scripts.import_soccer_data --provider statsbomb --competition-id 2 --season-id 27 --include-xg
  python -m scripts.import_soccer_data --provider statsbomb --competition-id 9 --season-id 281 --include-xg
  python -m scripts.import_soccer_data --provider openfootball --league england --season 2026-27

StatsBomb IDs are competition/season IDs from its own curated data. OpenFootball is used
for fixture/result coverage, not xG or live odds. All imports retain the provider ID and
retrieval metadata; kickoff UTC is left empty unless the source provides its timezone.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

from src.soccer.providers.http_cache import DataProviderError
from src.soccer.providers.openfootball import OpenFootballProvider
from src.soccer.providers.statsbomb import StatsBombOpenDataProvider
from src.soccer.schemas import MATCH_COLUMNS, SoccerMatch


def upsert_matches(path: str | Path, matches: Iterable[SoccerMatch]) -> int:
    """Atomically merge imported rows by provider + match_id, preserving other feeds."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[tuple[str, str], dict[str, object]] = {}
    if output.exists():
        with output.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                provider, match_id = row.get("provider", ""), row.get("match_id", "")
                if provider and match_id:
                    merged[(provider, match_id)] = {key: row.get(key, "") for key in MATCH_COLUMNS}
    for match in matches:
        row = match.to_dict()
        merged[(match.provider, match.match_id)] = row
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(merged):
            writer.writerow(merged[key])
    temp.replace(output)
    return len(merged)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import open soccer data into data/soccer/matches.csv")
    parser.add_argument("--provider", choices=("statsbomb", "openfootball"), default="statsbomb")
    parser.add_argument("--list-competitions", action="store_true", help="list competition/seasons available in StatsBomb Open Data")
    parser.add_argument("--search", default="", help="filter the StatsBomb competition list by name/country/season")
    parser.add_argument("--competition-id", help="StatsBomb competition ID (required for StatsBomb imports)")
    parser.add_argument("--season-id", help="StatsBomb season ID (required for StatsBomb imports)")
    parser.add_argument("--include-xg", action="store_true", help="download event files and aggregate xG, shots and shots on target (StatsBomb only)")
    parser.add_argument("--league", choices=("england", "germany", "spain", "italy", "france"), help="OpenFootball repository/league key")
    parser.add_argument("--season", help="OpenFootball season folder, e.g. 2026-27")
    parser.add_argument("--limit", type=int, help="import only the first N matches, useful for smoke tests")
    parser.add_argument("--output", default="data/soccer/matches.csv", help="CSV path for imported matches")
    parser.add_argument("--cache-dir", help="optional provider cache directory; defaults to .cache/soccer/<provider>")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.provider == "statsbomb":
            cache_dir = args.cache_dir or ".cache/soccer/statsbomb"
            provider = StatsBombOpenDataProvider(cache_dir=cache_dir)
            if args.list_competitions:
                term = args.search.casefold().strip()
                rows = provider.list_competitions()
                if term:
                    rows = [
                        row for row in rows
                        if term in " ".join((row.competition_name, row.country_name or "", row.season_name)).casefold()
                    ]
                print("competition_id\tseason_id\tcompetition\tseason\tcountry\tcache")
                for row in rows:
                    print(f"{row.competition_id}\t{row.season_id}\t{row.competition_name}\t{row.season_name}\t{row.country_name or ''}\t{row.source_cache_status}")
                print(f"\n{len(rows)} competition-season entries listed.")
                return 0
            if not args.competition_id or not args.season_id:
                raise ValueError("StatsBomb import requires --competition-id and --season-id. Use --list-competitions to find available IDs.")
            if args.league or args.season:
                raise ValueError("--league/--season apply only to OpenFootball.")
            matches = provider.list_matches(
                args.competition_id, args.season_id, include_xg=args.include_xg, limit=args.limit
            )
        else:
            if args.list_competitions or args.search or args.competition_id or args.season_id:
                raise ValueError("--list-competitions/--search/--competition-id/--season-id apply only to StatsBomb.")
            if args.include_xg:
                raise ValueError("OpenFootball does not provide xG; use StatsBomb Open Data with --include-xg.")
            if not args.league or not args.season:
                raise ValueError("OpenFootball import requires --league and --season, for example --league england --season 2026-27.")
            provider = OpenFootballProvider(cache_dir=args.cache_dir or ".cache/soccer/openfootball")
            matches = provider.list_matches(args.league, args.season, limit=args.limit)

        count = upsert_matches(args.output, matches)
        with_xg = sum(row.xg_status == "complete" for row in matches)
        unavailable_xg = sum(row.xg_status in {"unavailable", "partial"} for row in matches)
        print(f"Imported {len(matches)} matches from {args.provider}.")
        print(f"Dataset: {Path(args.output)} (total rows after merge: {count})")
        print(f"Rows with complete xG: {with_xg}; missing/partial xG: {unavailable_xg}.")
        print("Provider kickoff timezone is not assumed; kickoff_utc stays blank unless source timezone is known.")
        return 0
    except (ValueError, DataProviderError, OSError) as exc:
        print(f"Soccer data import failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
