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

This uses the public NBA stats endpoint and caches downloaded responses under .cache/nba. It can take time and public NBA servers may rate-limit requests. Retrain the models after updates to the feature engineering so opponent context, minute/role proxies, usage proxies, cross-stat trends, and next-game prediction features stay aligned. Phase 2 expands the training feature schema, so run training again after pulling updates.

## Generate all-player projections

    .\.venv\Scripts\python.exe -m src.predict_slate

The full estimate table is written to predictions/latest.csv. When no prop line is provided, expected stats and intervals are reported, but the program deliberately does not invent an over/under probability.

## Import a public prop board and rank candidates

A best-effort HTML importer reads the public Flash Odds pages for Points, Rebounds, and Assists. It preserves the sportsbook label shown for the over and under prices independently; bare unlabeled prices are treated as Underdog by the parser. It is not an official API and does not guarantee that every market or every book is represented. The source can change its markup or block requests.

    .\.venv\Scripts\python.exe -m scripts.fetch_public_props

This writes the current board to data/prop_lines_public.csv and appends timestamped observations to data/prop_market_history.csv. Each refresh records the captured line and prices, source opening line/movement when published, and movement since the prior import. The importer first tries the server-rendered public HTML and falls back to headless Chrome only if a future page version requires JavaScript rendering. Google Chrome must be installed if that fallback is needed; Selenium Manager attempts to obtain a matching driver automatically. Positive prices may appear without a plus sign on the source page (for example, 165 means +165). Then score available markets and print up to ten candidates:

    .\.venv\Scripts\python.exe -m src.predict_slate --lines data\prop_lines_public.csv --top 10

Output files:
- predictions/latest.csv — full scored market/audit table, including same-line best prices, market line range, no-vig method, quote age, and line movement
- predictions/shortlist.csv — candidates that pass the configured screens; these are not verified bets
- data/prop_market_history.csv — append-only market observations from each successful public-board import

The slate comparison consolidates same player/stat/line rows and uses the best observed price on each side. When multiple book rows are present, the market table reports distinct books, consensus line range, and best Over/Under prices. A no-vig probability is only computed from a same-book pair (or a row explicitly representing one book); if the over quote and under quote are from different books, PropIt withholds that no-vig edge instead of treating a cross-book pair as fair probability. The default public importer is still only as broad as the Flash Odds page provides; it does not automatically subscribe to every sportsbook.

Market quote age means when PropIt captured the page, not when the provider last changed its quote. Public snapshot rows older than 12 hours are blocked from the shortlist by default; change this intentionally with --max-market-age-hours. The page can still show these rows in All markets for inspection. Movement since the last import requires at least two successful refreshes. Do not interpret missing data as a pick, and verify current line, price, player ID and local availability independently.

## Player availability and shortlist safeguards

PropIt does **not** infer availability from recent box scores and does not automatically scrape an injury feed. Use the current [official NBA injury report](https://official.nba.com/nba-injury-report-2026-27-season/) or another source you independently trust, then maintain a point-in-time CSV for players whose status you checked.

Start from the empty template:

    Copy-Item data\player_availability.example.csv data\player_availability.csv
    notepad data\player_availability.csv

Required columns are `player_id,match_date,status,reported_at_utc`. Optional columns are `player_name,team_abbreviation,reason,source`. Use the NBA player ID, the player's **scheduled match date** (YYYY-MM-DD), the status shown by the source, and the time you checked it as an ISO-8601 UTC timestamp (for example `2026-10-25T15:15:00Z`). Use one row per player/date and refresh it when a report changes. The template is header-only; do not enter example statuses as if they were verified.

The slate shows the status and freshness gate. Fresh `Available`/`Probable` reports are CLEAR; `Out`, `Inactive`, `Suspended`, or `Rest` blocks a row; `Questionable`/`Doubtful` forces review and removes the row from the shortlist. Statuses older than six hours, unrecognized statuses, missing records, or mismatched dates/teams are marked STALE/UNVERIFIED rather than silently treated as healthy. In the normal dashboard flow, unverified rows can still appear as model candidates, but their availability gate and warning are shown explicitly. To fail closed and only shortlist rows with a fresh Available/Probable status, run:

    .\.venv\Scripts\python.exe -m src.predict_slate --lines data\prop_lines_public.csv --require-verified-availability

The flag affects shortlist eligibility, not the all-markets audit output. This is an operator-maintained safeguard, **not** an automatic injury feed; confirm status close to tip-off because the official report can update during the day.

## Run the local dashboard

PropIt includes a browser-based local dashboard with tabs for Overview, Today's Slate, History & Results, and Calibration. It uses the project's existing FastAPI/Uvicorn dependencies and keeps data files behind the interface.

Start it from PowerShell:

    Set-Location C:\Users\visha\propit
    .\.venv\Scripts\python.exe -m src.dashboard

Then open http://127.0.0.1:8766 in your browser. Keep the PowerShell window open while using the dashboard; press Ctrl+C in that window to stop it. The server binds to 127.0.0.1 and is intended for local use.

Use **Refresh Slate** to import the public prop board and regenerate model predictions. Use **Run Pregame Workflow** to do those steps and archive one snapshot for the selected snapshot date in one action; only run it before tip-off. PropIt uses the current NBA roster when team metadata is available and the public NBA schedule to attach each player's team, next scheduled match date, and opponent. The public prop page's Match column is generic, so these match details come from the NBA schedule rather than the market page. If a player or schedule cannot be matched, the match date remains blank instead of being replaced with the snapshot date. Use **History & Results** to settle a specific match date after the public NBA game logs update, and **Calibration** to run the chronological holdout evaluation when enough settled history exists. The activity log shows command output and errors, so a blocked public endpoint or missing model artifact is visible rather than silently ignored.

The market and prediction CSVs remain internal storage for the pipeline and audit trail; availability is the exception and is supplied through data/player_availability.csv when you choose to record a verified report. The dashboard displays availability but does not edit or fetch the injury report for you. No bets are placed automatically. Verify public lines, prices, and player status independently.
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

To start a new calibration baseline deliberately, use:

    .\.venv\Scripts\python.exe -m scripts.calibrate data\settled_props.csv --reset-cutoffs

On the first successful run, the script chooses a chronological split (around 70% of dates for calibration fitting and later dates for holdout), then saves each stat's holdout start in data/calibration_cutoffs.json. Later runs retain those boundaries and add newer dates to the holdout rather than moving the cutoff. Use --reset-cutoffs only to establish a new evaluation baseline. It reports hit rate, Brier score, log loss, and probability reliability buckets separately for PTS/REB/AST. It writes holdout rows with calibrated probabilities to predictions/backtest_holdout.csv and refuses to report a target's holdout metrics if the sample is too small. Historical ROI simulation only counts rows that have both over/under prices and a recorded snapshot_at_utc timestamp from the archived pregame workflow; untimestamped or reconstructed prices are excluded from ROI, although their settled outcomes may still be used to evaluate probability quality.

**Do not train and evaluate calibration on the same rows.** Use predictions stored before outcomes were known; reconstructing predictions from today's model after seeing results creates leakage. Brier/log loss measure probability quality; ROI is meaningful only when the historical odds and line were genuinely available pregame.

## Manually supplied lines

    Copy-Item data\prop_lines.example.csv data\prop_lines.csv
    notepad data\prop_lines.csv

Use one row per player/stat/market with these columns:

    player_id,player_name,target,line,odds_over,odds_under,bookmaker

Targets are PTS, REB, or AST. Enter actual lines and American odds from a market you have personally checked. Test/demo/example labels are intentionally excluded from shortlist candidates. The shortlist ranks the supplied markets; it does not invent line data or place bets.

## Operations health, audit trail, daily report and backup

The **Operations** dashboard tab checks whether the three model artifacts exist, whether public market captures and prediction files are fresh, whether the availability snapshot is current, whether pregame history is settling, and whether recent workflow attempts failed. The status describes local files and timestamps; it cannot guarantee that a remote data source is complete or that predictions are correct.

Every dashboard workflow action (market refresh, pregame workflow, settlement, calibration, daily report and backup) appends a small JSONL record to `data/operations_audit.jsonl`. The audit records start/end timestamps, command module, arguments, return code, duration, success/failure and a short failure excerpt. It does not copy environment variables or credentials. The audit file and generated report/backups are ignored by Git.

Generate a dated JSON and Markdown report from PowerShell:

    .\.venv\Scripts\python.exe -m scripts.ops_report

Reports are written to `reports/operations/operations-YYYY-MM-DD.json` and `.md`. Re-running the report on the same UTC date updates that date's report. When a valid chronological holdout exists, the report also includes per-stat raw/calibrated Brier score, log loss and hit rate from `predictions/backtest_holdout.csv`. These describe that saved holdout only; they do not reconstruct missing prices or guarantee future profitability.

Create a ZIP backup:

    .\.venv\Scripts\python.exe -m scripts.backup_workspace

Backups are stored in `backups/propit-backup-YYYYMMDDTHHMMSSZ.zip`. Each archive includes a `manifest.json` with included paths, byte sizes and SHA-256 checksums. The allow-list covers source code, tests, workflow configuration, README/dependencies, local CSV/JSON/JSONL data, prediction files, and local model artifacts when present. It deliberately excludes `.env`, Python virtual environments, downloaded caches and earlier backup archives. Keep the ZIP somewhere outside this PC for disaster recovery; the tool does not upload it.

Run backup before changing model/data schemas and preserve a known-good archive so you can restore files manually if a later code change causes a regression. Restoring code should be followed by the test suite; restore trained model files and data as a matched set. These utilities do not automatically deploy rollback code or execute wagers.

## Run tests

    .\.venv\Scripts\python.exe -m pytest -q

The tests cover player-isolated rolling features, minutes/usage feature timing, schedule-based next-game rows, stale model-artifact rejection, odds arithmetic, season selection, parser behavior, same-date chronological holdout boundaries, pregame snapshot archival, outcome settlement, and dashboard date/data handling.

## Important limitations

- Opponent defense and possession-volume context is reconstructed from player game logs using estimated possessions (FGA + 0.44 × FTA − OREB + TOV). It is an approximation, not an official NBA pace/defensive-rating feed; each historical feature is joined from games strictly before the target game's date.
- Minutes/role context uses rolling minutes and recent-versus-season minute trends because the current feed does not provide a dependable starter/bench designation for every historical row. The usage proxy (FGA + 0.44 × FTA + TOV) is not the official NBA usage percentage.
- The over/under probability currently uses each row's predicted 10th–90th percentile spread rather than one global residual standard deviation. It is still a distributional estimate, **not calibrated confidence**, and is capped to avoid extreme 0%/100% claims.
- The probability calibration script requires historical pregame predictions and settled props. The public market history file tracks observations only from the moment it is enabled; it cannot reconstruct earlier prices. A timestamped archived price provides traceability, but ROI is still a historical simulation rather than proof that an edge will persist.
- A low mean absolute error or high predicted probability is not proof of profitable betting.
- Current roster status is not an official injury or game-availability feed.
- Integer prop lines are skipped by the importer because the current EV model does not account for push probability.
- No automatic bet placement is implemented.
