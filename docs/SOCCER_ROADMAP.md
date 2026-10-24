# PropIt Soccer — implementation plan

## Scope and separation

Soccer is a separate sport module. NBA feature engineering, artifacts, availability, market importers,
history and dashboard behavior must remain unchanged unless a sport-neutral interface is explicitly
introduced and regression-tested.

## Phase 1: probability baseline

- Add an independent-Poisson goal baseline.
- Report home/draw/away probabilities and goal-total over/under/push probabilities.
- Expected goals remain explicit inputs until team-rate estimates were added in Phase 3.
- Input validation and deterministic tests live in tests/test_soccer_poisson.py.

## Phase 2: provider and data contracts

### Selected sources

- StatsBomb Open Data (https://github.com/statsbomb/open-data): curated competition/season IDs,
  historical match scores, events, shot counts, shots-on-target counts and shot-level xG where
  available. This is selective historical coverage, not a comprehensive live feed. Attribute StatsBomb
  and follow its data terms when publishing insights.
- OpenFootball (https://github.com/openfootball): public-domain text data for fixtures and results
  across supported leagues. It does not include xG or odds in this adapter and does not guarantee
  timezone-aware kickoff times.
- Football-Data.co.uk was reviewed but is not wired to automated ingestion because its site terms
  restrict the free data to private use and prohibit commercial/automated scraping or AI-training use.

### Contracts and import

- Provider-neutral CompetitionSeason and SoccerMatch records are in src/soccer/schemas.py.
- Source ID, URL, provider update metadata, retrieval timestamp, observed timestamp and cache
  status are retained. Provider-local kickoff times are not converted to UTC without timezone evidence.
- The HTTP cache uses ETag/Last-Modified conditional requests and exposes stale-cache use.
- The importer merges records by (provider, match_id) into data/soccer/matches.csv.
- Automated tests cover provider parsing, xG/shot aggregation, missing xG, date rollover, source paths,
  and deterministic dataset upserts.

Import commands:

    python -m scripts.import_soccer_data --provider statsbomb --list-competitions --search Premier
    python -m scripts.import_soccer_data --provider statsbomb --competition-id 2 --season-id 27 --include-xg
    python -m scripts.import_soccer_data --provider statsbomb --competition-id 9 --season-id 281 --include-xg
    python -m scripts.import_soccer_data --provider openfootball --league england --season 2026-27

StatsBomb competition/season IDs must first be present in its downloaded catalog. Use --limit 3 for
a small import smoke test before fetching a full season of event files.

## Phase 3: attack/defence model and time-aware evaluation

- Estimate competition home/away scoring baselines and venue-specific team attack and defensive
  weakness factors. Team estimates are shrunk towards a weak prior with exponential time decay.
- Use only completed matches in the same competition strictly before the forecast match date. Same-day
  matches are excluded from one another's training sample, and a configurable lookback limits stale data.
- Generate expected home/away goals and home/draw/away win plus total-goals over/under probabilities.
- Compare independent Poisson with a Dixon-Coles low-score adjustment using the same rolling-origin
  predictions. The rho value is experimental and supplied by CLI argument; it is not fitted/optimal by default.
- Save predictions and machine-readable metrics, including log loss, multiclass Brier score, accuracy,
  expected-goals MAE, totals log loss/Brier/accuracy, source sample size and warm-up exclusions.

Backtest command:

    python -m scripts.backtest_soccer --input data/soccer/matches.csv --min-history 30

The output defaults to predictions/soccer/backtest.csv and predictions/soccer/backtest_metrics.json.
For integer totals, exact-line results are treated as pushes and excluded from binary over/under scoring.
Lower log loss and Brier score are better. The report does not claim profitability or calibration, and
the quality of results depends on the league/season coverage imported.

## Next phase

### Phase 4: dashboard/API integration

1. Add a Soccer tab without changing NBA defaults, filters, or API behavior.
2. Display fixtures, competition, model expected goals, 1X2/totals probabilities, sample size, freshness
   and source availability. Make missing xG/odds/lineups explicit rather than substituting guesses.
3. Add model confidence/calibration summaries only after enough chronological predictions are archived.
4. Continue evaluating player markets only when reliable player-level event data and confirmed lineups exist.

## Safety and interpretation

Research/educational use only. Probabilities are experimental, not guarantees of accuracy or profit.
Do not automatically place bets. Never present stale or missing odds as current, and never create a
probability edge from mismatched or unavailable prices.
### Generate upcoming forecasts

    python -m scripts.predict_soccer --input data/soccer/matches.csv --date-from 2026-10-24 --date-to 2026-11-08

Optional competition filtering uses the provider competition ID exactly as stored by the importer.
The CSV output contains both Poisson and experimental Dixon-Coles probabilities plus a generation
timestamp and historical sample counts. The CLI does not fetch odds, lineups, or injury information.
