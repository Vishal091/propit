# PropIt Soccer — implementation plan

## Scope and separation

Soccer is a separate sport module. NBA feature engineering, artifacts, availability, market importers,
history and dashboard behavior must remain unchanged unless a sport-neutral interface is explicitly
introduced and regression-tested.

## Phase 1: probability baseline

- Add an independent-Poisson goal baseline.
- Report home/draw/away probabilities and goal-total over/under/push probabilities.
- Expected goals remain explicit inputs until the data pipeline is implemented.
- Input validation and deterministic tests live in tests/test_soccer_poisson.py.

## Phase 2: provider and data contracts (implemented)

### Selected sources

- **StatsBomb Open Data** (https://github.com/statsbomb/open-data): curated competition/season IDs,
  historical match scores, event files, shot counts, shots-on-target counts and shot-level xG where
  available. This is selective historical coverage, not a comprehensive live feed. Attribute StatsBomb
  and follow its data terms when publishing insights.
- **OpenFootball** (https://github.com/openfootball): public-domain text data for fixtures and results
  across supported leagues. Useful for broader schedules, but it does not include xG or odds in this
  adapter and does not guarantee timezone-aware kickoff times.
- **Football-Data.co.uk** was reviewed but is intentionally not selected as an automated ingest source
  for this product: its site terms restrict use to private individuals and prohibit commercial use and
  automated bots/scrapers/AI training products. Revisit only if appropriate written permissions or a
  suitable license is obtained.

### Contracts and import

- Provider-neutral CompetitionSeason and SoccerMatch records are in src/soccer/schemas.py.
- Source ID, source URL, provider update metadata, retrieval timestamp, observed timestamp and cache
  status are retained. Provider-local kickoff times are not converted to UTC without timezone evidence.
- The HTTP cache uses ETag/Last-Modified conditional requests and exposes stale-cache use.
- The importer merges records by (provider, match_id) into data/soccer/matches.csv.
- Generated imports and cache files are local data artifacts; credentials are not needed for these feeds.
- Automated tests cover the independent providers' parsing, xG/shot aggregation, missing xG, date rollover,
  and deterministic dataset upserts.

Commands:

    python -m scripts.import_soccer_data --provider statsbomb --list-competitions --search Premier
    python -m scripts.import_soccer_data --provider statsbomb --competition-id 2 --season-id 27 --include-xg
    python -m scripts.import_soccer_data --provider statsbomb --competition-id 9 --season-id 281 --include-xg
    python -m scripts.import_soccer_data --provider openfootball --league england --season 2026-27

The StatsBomb competition/season IDs must first be present in its downloaded catalog. Use --limit 3
for a small import smoke test before fetching a full season of event files.

## Next phase

### Phase 3: features and expected-goals model

1. Import historical results/team statistics with point-in-time cutoff rules. Ensure every feature used
   for a match was available before kickoff; exclude postponed/cancelled fixtures from training labels.
2. Estimate expected goals using a transparent attack/defence baseline; evaluate a Dixon–Coles low-score
   adjustment only if chronological holdouts justify it.
3. Add supported markets incrementally: 1X2, totals, both-teams-to-score, then player markets only
   when reliable player-level data and lineups are available.
4. Backtest chronologically by competition and season. Report log loss, Brier score, calibration,
   sample sizes and market coverage. ROI is secondary and needs timestamped historical odds, realistic
   price availability and explicit assumptions.
5. Integrate a Soccer dashboard/API view without changing NBA defaults. Show source freshness, missing
   lineups and missing odds, and distinguish model probabilities from market prices.

## Safety and interpretation

Research/educational use only. Probabilities are experimental, not guarantees of accuracy or profit.
Do not automatically place bets. Never present stale or missing odds as current, and never create a
probability edge from mismatched or unavailable prices.
