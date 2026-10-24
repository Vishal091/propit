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

This uses the public NBA stats endpoint and caches downloaded responses under .cache/nba. It can take time and public NBA servers may rate-limit requests. Retrain the models after updates to the feature engineering so historical opponent-context features and next-game prediction features stay aligned.

## Generate all-player projections

    .\.venv\Scripts\python.exe -m src.predict_slate

The full estimate table is written to predictions/latest.csv. When no prop line is provided, expected stats and intervals are reported, but the program deliberately does not invent an over/under probability.

## Import a public prop board and rank candidates

A best-effort HTML importer reads the public Flash Odds pages for Points, Rebounds, and Assists. These are Underdog lines/prices, not an aggregate across books and not an official API. The source can change its markup or block requests.

    .\.venv\Scripts\python.exe -m scripts.fetch_public_props

This writes data/prop_lines_public.csv. The importer first tries the server-rendered public HTML. It falls back to headless Chrome only if a future page version requires JavaScript rendering. Google Chrome must be installed if that fallback is needed; Selenium Manager attempts to obtain a matching driver automatically. Positive prices may appear without a plus sign on the source page (for example, 165 means +165). Then score available markets and print up to ten candidates:

    .\.venv\Scripts\python.exe -m src.predict_slate --lines data\prop_lines_public.csv --top 10

Output files:
- predictions/latest.csv — full scored market/audit table
- predictions/shortlist.csv — candidates that pass the configured screens; these are not verified bets

If the importer errors or finds zero markets, inspect the logs. Do not interpret missing data as a pick. Underdog prices may be stale or unavailable in your region. Check actual player/market IDs, lines and both side prices before using any row.

## Run the local dashboard

PropIt includes a browser-based local dashboard with tabs for Overview, Today's Slate, History & Results, and Calibration. It uses the project's existing FastAPI/Uvicorn dependencies and keeps data files behind the interface.

Start it from PowerShell:

    Set-Location C:\Users\visha\propit
    .\.venv\Scripts\python.exe -m src.dashboard

Then open http://127.0.0.1:8766 in your browser. Keep the PowerShell window open while using the dashboard; press Ctrl+C in that window to stop it. The server binds to 127.0.0.1 and is intended for local use.

Use **Refresh Slate** to import the public prop board and regenerate model predictions. Use **Run Pregame Workflow** to do those steps and archive one snapshot for the selected snapshot date in one action; only run it before tip-off. PropIt uses the current NBA roster when team metadata is available and the public NBA schedule to attach each player's team, next scheduled match date, and opponent. The public prop page's Match column is generic, so these match details come from the NBA schedule rather than the market page. If a player or schedule cannot be matched, the match date remains blank instead of being replaced with the snapshot date. Use **History & Results** to settle a specific match date after the public NBA game logs update, and **Calibration** to run the chronological holdout evaluation when enough settled history exists. The activity log shows command output and errors, so a blocked public endpoint or missing model artifact is visible rather than silently ignored.

The CSV files remain internal storage for the existing pipeline and audit trail; normal viewing and workflow actions happen in the dashboard. No bets are placed automatically. Verify public lines and prices independently.
## Start a persistent pregame history

The current output files are overwritten on each prediction run. To evaluate probabilities later, save a slate to a persistent history file **before tip-off**. This records the exact line, both odds, model probability, player ID, the snapshot date, and the player's match date when schedule mapping succeeds.

After updating to the latest main branch, run the usual importer and prediction commands:

    .\.venv\Scripts\python.exe -m scripts.fetch_public_props
    .\.venv\Scripts\python.exe -m src.predict_slate --lines data\prop_lines_public.csv --top 10

Then archive the slate once, before the games start. The `--game-date` argument is now a **snapshot-date label** used to prevent duplicate snapshots; it is not assigned as the match date for every player when the schedule has a match date:

    .\.venv\Scripts\python.exe -m scripts.archive_predictions --game-date YYYY-MM-DD

This appends rows to data/prediction_history.csv. Each record stores the snapshot date separately from the player's event date. When schedule mapping succeeds, `game_date` and `match_date` reflect that player's next scheduled game, with team, opponent, and matchup metadata preserved. The command refuses a second archive for the same snapshot date by default. Use --append-snapshot only if deliberately preserving another pregame snapshot for that date; duplicate snapshots can overweight repeated markets in a backtest.

After the games for a particular match date finish and the public NBA player-game logs have updated, settle that match date's results:

    .\.venv\Scripts\python.exe -m scripts.settle_predictions --game-date YYYY-MM-DD

The settlement script uses each row's `match_date` (falling back to legacy `game_date`) to settle only matching player appearances. Integer-line pushes are marked PUSH and excluded from probability scoring. A player with no matching appearance in the public logs stays pending; check DNPs, postponements, and data delays rather than assuming they lost. Repeat settlement after the public feed updates if needed.

Once there is enough historical data, point calibration at this persistent history file:

    .\.venv\Scripts\python.exe -m scripts.calibrate data\prediction_history.csv

Do **not** backfill prior days using today's predictions or today's prices. The October 23 output files were not archived with a game date and snapshot timestamp, so they should not be treated as a clean historical pregame record. A useful calibration sample needs multiple dates and enough settled props per target.

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

The tests cover player-isolated rolling features, odds arithmetic, season selection, parser behavior, same-date chronological holdout boundaries, pregame snapshot archival, outcome settlement, and dashboard date/data handling.

## Important limitations

- Opponent defense and possession-volume context is reconstructed from player game logs using estimated possessions (FGA + 0.44 × FTA − OREB + TOV). It is an approximation, not an official NBA pace/defensive-rating feed; each historical feature is joined from games strictly before the target game's date.
- The over/under probability currently uses each row's predicted 10th–90th percentile spread rather than one global residual standard deviation. It is still a distributional estimate, **not calibrated confidence**, and is capped to avoid extreme 0%/100% claims.
- The probability calibration script requires historical pregame predictions and settled props. The current public importer only supplies a current snapshot, not historical lines/odds, so it cannot by itself establish historical betting ROI.
- A low mean absolute error or high predicted probability is not proof of profitable betting.
- Current roster status is not an official injury or game-availability feed.
- Integer prop lines are skipped by the importer because the current EV model does not account for push probability.
- No automatic bet placement is implemented.
