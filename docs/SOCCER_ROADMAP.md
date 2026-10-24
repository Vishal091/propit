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

## Phase 4: dashboard/API integration (implemented)

- Add the Soccer analytics sidebar/tab while hiding NBA-specific global workflow actions on the soccer view.
- Display imported/scored/upcoming/backtest counts, competition/source entries and file timestamps.
- Filter saved fixtures/results and forecasts by date, competition and view.
- Explicitly load the StatsBomb catalog, import a selected provider dataset, generate forecasts and run the backtest through allow-listed local commands.
- Display the saved independent-Poisson and Dixon-Coles evaluation metrics and data coverage caveats.
- Add API routes under `/api/soccer/*` and tests for empty/available states, filtering, invalid dates, catalog parsing and controlled CLI invocations.

## Phase 5: top-five picks and calibration diagnostics (implemented)

- Strictly filter Best Picks to Premier League, La Liga, Bundesliga, Serie A and Ligue 1. European cups, lower divisions and unrecognized competitions are ineligible.
- Rank one strongest independent-Poisson market per fixture across home win, draw, away win and the configured total-goals over/under line. Defaults require at least eight prior competition matches and 52% model probability; the ranking score discounts thinner history. Dixon-Coles remains experimental and its rho is unfitted.
- Display expected goals, model probability, historical sample support and ranking score. Odds, confirmed lineups, suspensions and injuries are absent; the model cannot report EV, market edge or profitable bets without reliable time-aligned odds.
- Add top-five calibration diagnostics using chronological rolling-origin backtest predictions, with top-class reliability buckets and expected calibration error (ECE) by model. Require at least 30 evaluated matches per model before passing the sample gate; no calibration transform is fitted.
- Add /api/soccer/picks and /api/soccer/calibration plus tests for league eligibility, ranking, date validation, odds caveats and top-five-only calibration.
- Keep soccer in separate modules and preserve NBA defaults and workflows.

## Phase 6: time-stamped odds and stronger model baselines (implemented)

- Add a local The Odds API v4 adapter for the five domestic leagues, with live and explicitly requested historical snapshots, quote/update/snapshot timestamps, bookmaker identity, exact market line, decimal prices, overround and same-book de-vigged probabilities.
- Keep API keys local in `THE_ODDS_API_KEY` (environment or ignored `.env`); never return the secret from the API, write it to CSV, or log request query strings.
- Add explicit dashboard controls for live imports and paid historical snapshots. The historical action requires a timezone-aware date and an explicit cost acknowledgment. Expected maximum cost is shown before use; empty responses may incur fewer credits. No provider calls happen automatically on dashboard load.
- Attach odds to Best Picks only for exact team and league key, a complete same-book market, the exact totals line where applicable, fresh quote and snapshot timestamps (including rejecting quotes updated after the snapshot or older than 24 hours at snapshot time), and a market observed before kickoff. Display the matched bookmaker, decimal price, no-vig market probability, model-minus-market edge and model EV/unit as experimental diagnostics only. Unmatched/stale/in-play/post-kickoff prices are never guessed or silently substituted.
- Expand rolling-origin backtests to include a league-average baseline and season-level reports in addition to Poisson and experimental Dixon-Coles. Display per-competition and per-season metrics in the dashboard.
- Add a market-implied benchmark to historical backtests only when pre-kickoff, timestamped odds from the local snapshot ledger can be matched. Write matched rows to `predictions/soccer/market_comparison.csv`; unmatched games are excluded and current prices are never backfilled into historical dates.
- Add tests for market completeness, de-vig normalization, exact lines, stale/in-play odds, secret-key sanitization, historical-cost confirmation, market baseline matching, per-season metrics and dashboard controls.

## Next phase

### Phase 7: improve predictive signal only after evidence

1. Import more historical seasons and compare each model and market benchmark on the same match sample where possible. Track sample counts, per-league score drift and calibration over time.
2. Add team-strength ratings and fit Dixon-Coles parameters only if rolling, season-held-out evaluation repeatedly improves log loss/Brier against the league-average and plain Poisson baselines.
3. Add lineup/injury/suspension features only after a trusted provider is selected and it offers reliable timestamps plus historical pre-match snapshots for backtesting. Do not use present-day injury reports as historical inputs.
4. Avoid any claim of profitable picks until time-aligned prices, model probability calibration, transaction/market constraints and a sufficiently large out-of-sample record support it.

## Safety and interpretation

Research/educational use only. Probabilities are experimental, not guarantees of accuracy or profit.
Do not automatically place bets. Never present stale or missing odds as current, and never create a
probability edge from mismatched or unavailable prices.
### Generate upcoming forecasts

    python -m scripts.predict_soccer --input data/soccer/matches.csv --date-from 2026-10-24 --date-to 2026-11-08

Optional competition filtering uses the provider competition ID exactly as stored by the importer.
The CSV output contains both Poisson and experimental Dixon-Coles probabilities plus a generation
timestamp and historical sample counts. The CLI does not fetch odds, lineups, or injury information.
Phase 4 does not modify the existing NBA tab behavior or run data imports on page load. Imports, forecast generation and backtests are user-triggered actions. The UI reads the resulting local data files.
