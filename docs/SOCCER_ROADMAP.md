 # PropIt Soccer — implementation plan

## Scope and separation

Soccer is being added as a separate sport module. Existing NBA feature engineering, model artifacts,
availability rules, market importers, history and dashboard behavior must remain unchanged unless a
sport-neutral interface is explicitly introduced and regression-tested.

## Phase 1: baseline and contracts

- Add a small, dependency-light Poisson baseline for match goals.
- Report home-win/draw/away-win probabilities and total-goals over/under/push probabilities.
- Keep expected goals as explicit inputs. This module does not claim to fetch live data or estimate
  expected goals from teams automatically.
- Add input validation and deterministic unit tests.

## Next phases

1. Define canonical fixture/team/player/competition and market schemas with provider IDs, source,
   observed_at_utc, event kickoff UTC, and freshness metadata.
2. Implement a provider adapter behind a stable interface. Keep credentials in environment variables;
   never commit secrets. Cache responses and handle rate limits, retries, and provider outages.
3. Import historical match/team statistics with point-in-time cutoff rules. Ensure every feature used
   for a match was available before kickoff; exclude postponed/cancelled fixtures from training labels.
4. Estimate expected goals using a transparent attack/defence baseline, then evaluate a Dixon-Coles
   low-score adjustment only if historical validation justifies it.
5. Add supported markets incrementally: 1X2, totals, both-teams-to-score, then player markets only
   when reliable player-level data and lineups are available.
6. Backtest chronologically by competition and season. Report log loss, Brier score, calibration,
   sample sizes and market coverage. ROI is secondary and must use timestamped historical odds,
   realistic price availability, and explicit assumptions.
7. Integrate a Soccer view into the dashboard and API without changing NBA defaults. Add data-source
   freshness, missing-lineup warnings, and a clear distinction between model probabilities and market
   prices.

## Safety and interpretation

Research and educational use only. Probabilities are experimental, not guarantees of accuracy or profit.
Do not automatically place bets. Never present stale or missing odds as current, and never create a
probability edge from mismatched or unavailable prices.
