# PropIt — Free NBA Player Prop Research

PropIt is a local-first NBA player-stat projection and market-comparison project. It trains points, rebounds, and assists models over league-wide player-game data, then can rank supplied markets into a shortlist.

Research/education only. Predictions and probability estimates are experimental, not guarantees of accuracy or profit. Public prop prices can be delayed, unavailable in your region, or tied to US sportsbooks; verify them independently.

## Requirements

- Windows 10/11 (commands below use PowerShell) or another Python 3.11+ system
- Internet access for public NBA stats and optional public-market import
- No NBA API key, paid subscription, or hosted AI API is required

## Install on Windows PowerShell

    git clone https://github.com/Vishal091/propit.git
    Set-Location .\propit
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

If PowerShell blocks venv activation, run Set-ExecutionPolicy -Scope Process Bypass -Force in the same window, then activate again.

## Train models

Train PTS/REB/AST models on the two latest completed regular seasons selected by the season helper:

    .\.venv\Scripts\python.exe -m src.train_model

This uses the public NBA stats endpoint and caches downloaded responses under .cache/nba. It can take time and public NBA servers may rate-limit requests.

## Generate all-player projections

    .\.venv\Scripts\python.exe -m src.predict_slate

The full estimate table is written to predictions/latest.csv. When no prop line is provided, expected stats and intervals are reported, but the program deliberately does not invent an over/under probability.

## Import a public prop board and rank candidates

A best-effort HTML importer tries to read Points, Rebounds, and Assists pages from a public webpage. It is not an official API and can break if the source changes markup or blocks requests.

    .\.venv\Scripts\python.exe -m scripts.fetch_public_props

This writes data/prop_lines_public.csv. Then score available markets and print up to ten candidates:

    .\.venv\Scripts\python.exe -m src.predict_slate --lines data\prop_lines_public.csv --top 10

Output files:
- predictions/latest.csv — full scored market/audit table
- predictions/shortlist.csv — candidates that pass the configured screens; these are not verified bets

If the importer errors or finds zero markets, inspect the logs. Do not interpret missing data as a pick. Prices may be stale, US-market-specific, or unavailable in your region. Check actual player/market IDs, lines and both side prices before using any row.

## Backtest settled historical props (new)

A meaningful probability/ROI backtest needs historical predictions made before the game, the exact line available at that time, the odds available at that time, and the settled result. Current prop-board snapshots cannot substitute for archived odds.

Prepare a CSV with one row per settled player prop and these columns:

    game_date,target,p_over,actual_over,odds_over_american,odds_under_american

- `game_date`: date the game was played.
- `target`: PTS, REB, or AST.
- `p_over`: PropIt's probability saved before tip-off.
- `actual_over`: 1 if the player finished above the line, 0 if below. Pushes must be excluded. Alternatively supply `actual_stat` and `line` instead of `actual_over`; pushes are excluded automatically.
- `odds_over_american` and `odds_under_american`: historical prices available before the game, e.g. -110.

Run the chronological calibration and holdout evaluation:

    .\.venv\Scripts\python.exe -m scripts.calibrate data\settled_props.csv

The script fits isotonic calibration only on the earliest 70% of dates, then evaluates on the later 30%. It reports hit rate, Brier score, log loss, positive-EV simulated bets, net units, and ROI separately for PTS/REB/AST. It writes holdout rows with calibrated probabilities to predictions/backtest_holdout.csv. It refuses to report a target's holdout metrics if the sample is too small.

**Do not train and evaluate calibration on the same rows.** Use predictions stored before outcomes were known; reconstructing predictions from today's model after seeing results creates leakage. Brier/log loss measure probability quality; ROI is meaningful only when the historical odds and line were genuinely available pregame.

## Manually supplied lines

    Copy-Item data\prop_lines.example.csv data\prop_lines.csv
    notepad data\prop_lines.csv

Use one row per player/stat/market with these columns:

    player_id,player_name,target,line,odds_over,odds_under,bookmaker

Targets are PTS, REB, or AST. Enter actual lines and American odds from a market you have personally checked. Test/demo/example labels are intentionally excluded from shortlist candidates. The shortlist ranks the supplied markets; it does not invent line data or place bets.

## Run tests

    .\.venv\Scripts\python.exe -m pytest -q

The tests cover player-isolated rolling features, odds arithmetic, season selection, parser behavior, and same-date chronological holdout boundaries.

## Important limitations

- The over/under probability currently uses each row's predicted 10th–90th percentile spread rather than one global residual standard deviation. It is still a distributional estimate, **not calibrated confidence**, and is capped to avoid extreme 0%/100% claims.
- The probability calibration script requires historical pregame predictions and settled props. The current public importer only supplies a current snapshot, not historical lines/odds, so it cannot by itself establish historical betting ROI.
- A low mean absolute error or high predicted probability is not proof of profitable betting.
- Current roster status is not an official injury or game-availability feed.
- Integer prop lines are skipped by the importer because the current EV model does not account for push probability.
- No automatic bet placement is implemented.
