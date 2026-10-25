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

Market quote age means when PropIt captured the page, not when the provider last changed its quote. Public snapshot rows older than two hours are blocked from the shortlist by default; relaxing this with --max-market-age-hours is not recommended for game-day decisions. The page can still show these rows in All markets for inspection. Movement since the last import requires at least two successful refreshes. Do not interpret missing data as a pick, and verify current line, price, player ID and local availability independently.

## Player availability, injury reports, minutes distributions, and lineups

PropIt can fetch the latest linked PDF from the official [NBA injury report page](https://official.nba.com/nba-injury-report-2026-27-season/), parse timestamped player rows, and match them by NBA player ID where possible or normalized name plus team and scheduled game date. The official page says reports are updated continually during the day, so a stored snapshot must not be treated as a guarantee of status at tip-off. A listed player status is always authoritative for that player and game date. The importer also preserves team-level **NOT YET SUBMITTED** markers in the import audit. When the report is freshly retrieved and its layout was parsed successfully, a player with no individual row on a team not marked **NOT YET SUBMITTED** is labeled `NOT_LISTED`: this means no player-specific participation concern appears in that report, **not** that the player was explicitly designated Available. Teams explicitly marked **NOT YET SUBMITTED**, stale reports, failed coverage parsing, or missing coverage metadata remain unverified. Refresh near tip-off and inspect the source before making decisions.

Import a report for the player's **NBA-local scheduled match date** (not your computer's snapshot date):

    .\\.venv\\Scripts\\python.exe -m scripts.import_nba_status --game-date YYYY-MM-DD

The command stores the current point-in-time player rows in `data/player_availability.csv`, appends to `data/player_availability_history.csv`, and logs the report URL/timestamp, retrieval time, parser diagnostics, and team submission markers in `data/injury_report_imports.csv`. It resolves IDs against the current NBA roster when that endpoint is available; unmatched rows can still match by normalized player name and team. If the official page does not link a suitable PDF or the table parser cannot parse player rows, the command fails without replacing the current snapshot.

The dashboard's **Official NBA injury report** action does the same thing for the selected game date. The one-click pregame workflow now refreshes the official injury report and lineup feed for the same NBA-local match date before refreshing markets, scoring the slate, and archiving it. Keep the injury-report and lineup date selectors aligned. The pregame snapshot date at the top is separate and labels the local archive, so it may differ from the NBA-local event date when your timezone crosses midnight. A successful refresh does not guarantee a non-empty strict shortlist: projected lineups and unverified/questionable availability remain blocked.

### Confirmed starter/bench and role snapshots

Minutes are built from historical game logs: last 5/10/20, current season and previous season, recent-versus-longer-run trend, robust dispersion, and a role-change check. In the first 14 games of a season, the minutes baseline blends roughly 70% previous-season and 30% current-season minutes when prior-season data is available and a role shift is not detected. A fresh manual return limit can cap the mean; questionable status, missing role confirmation, team changes, and role shifts widen uncertainty. The prop probability is simulated through 10,000 or more minutes scenarios rather than a single projected-minute value.

Starting-five data is kept separate from injury status. PropIt can now collect the public RotoWire starting-five page automatically while preserving its source-provided confidence class. The feed only records the five listed starter slots for each team it can parse; it does not infer bench assignments from missing players.

Run the automated collector for a specific NBA-local game date:

    .\\.venv\\Scripts\\python.exe -m scripts.fetch_lineups --game-date YYYY-MM-DD

The collector requests `https://www.rotowire.com/basketball/nba-lineups.php?date=YYYY-MM-DD`, verifies that the page itself displays the requested date, records source URL and retrieval timestamp, resolves player IDs when the NBA roster endpoint permits, and stores the current feed in `data/player_lineups_auto.csv` with append-only history at `data/player_lineups_auto_history.csv`. It replaces current rows only for teams actually represented in the fetched page. If the page date or expected markup cannot be verified, it stops without overwriting the prior snapshot. The dashboard has a **Fetch lineup updates** action that imports the feed and rebuilds predictions.

Lineup certainty is explicit in `confirmation_status`, `confidence_label`, and `source_status`. A team-side source label of **Confirmed Lineup** is recorded as source-confirmed; **Expected Lineup**, unlabeled lineups, and unknown formats are stored as `PROJECTED`. The collector never converts a projected lineup into a confirmed one. Projected entries are visible in the slate with their status, source, timestamp, and confidence label, but they do not influence the minutes role and cannot satisfy the strict shortlist gate. The public page may not expose an authoritative update timestamp, so `reported_at_utc` records PropIt's retrieval time, while `source_reported_at_utc` remains blank when the publisher does not provide one. Refresh close to tip-off and independently verify source-confirmed roles.

For manual verification or when an official/team announcement is available, the operator can still capture a complete verified starter/bench snapshot:

    Copy-Item data\\player_lineups_input.example.csv data\\player_lineups_input.csv
    notepad data\\player_lineups_input.csv

Enter one row per checked player with `player_id` or `player_name`, `team_abbreviation`, and explicit `starting_status` of `STARTER` or `BENCH`. Use `reason` for a meaningful note, and optional `minutes_limit` from 1 to 48 only when a reported return restriction is known. Do not add a player based only on an unverified projection. Then stamp and import the file:

    .\\.venv\\Scripts\\python.exe -m scripts.capture_lineup_snapshot --game-date YYYY-MM-DD --input data\\player_lineups_input.csv --source "https://www.nba.com/starting-lineups"

The manual command timestamps the operator-verified role in UTC and upserts `data/player_lineups.csv`, with append-only history in `data/player_lineups_history.csv`. Automated projected rows are kept in a separate ledger so a recent projection cannot overwrite or shadow an existing confirmed role. Snapshot freshness defaults to two hours.

### Research view versus qualified shortlist

The dashboard includes a separate **Research view** for inspecting raw or market-shrunk estimates when strict availability or lineup checks prevent a decision-grade probability from being released. With a compatible same-book/no-vig market reference, the exploratory estimate shrinks the raw model toward the market using 15% model weight and 85% market weight. If no compatible no-vig reference exists, the view may show only the normalized raw model probability; edge and EV remain blank. When valid two-sided prices are available, it computes exploratory edge/EV, but never suggests a stake.

Research-view rows are not qualified shortlist picks. Their availability/lineup states and strict screen reason remain available in the row's hover details. The **Shortlist** tab and `predictions/shortlist.csv` remain unchanged: unverified or questionable availability and unconfirmed roles cannot pass strict qualification. Research estimates are for debugging and context only, not betting recommendations.

The slate also includes a separate **Pending confirmation** panel. It only shows rows blocked by lineup/availability evidence that still have a positive exploratory research edge and EV, at least 53% research probability, fresh market/data, paired prices, a same-book no-vig reference, and prior market/movement evidence. Explicitly unavailable players and rows with unrelated market-quality failures are omitted. The panel is a research-only follow-up queue, not a shortlist or bet recommendation; after evidence is verified, refresh and rescore so all strict qualification rules run again.

### Fail-closed shortlist rules

The CLI requires fresh availability and a fresh confirmed STARTER/BENCH snapshot by default. It also requires a usable minutes projection (at least three historical minute records), a current line and both prices, a market no-vig reference, a market snapshot no older than two hours, a calibrated edge of at least 4 percentage points, and positive expected value. It withholds the calibrated probability when availability, lineup role, minimum minutes history, or no-vig market data are missing. Unknown or stale inputs remain visible in the all-markets audit as explicit skip reasons.

For research/debugging only, it is possible to inspect candidates without a verified status:

    .\\.venv\\Scripts\\python.exe -m src.predict_slate --lines data\\prop_lines_public.csv --allow-unverified-availability --allow-unverified-lineup

Those overrides relax shortlist gates; they do **not** turn the missing statuses into verified facts. Never use that mode to treat an unverified candidate as a live bet. The automated RotoWire collector can import expected lineups and any team entries the page explicitly labels Confirmed Lineup, but it may not expose confirmed labels for every matchup and does not infer bench roles, coach rotation changes, or injury-return minute limits. Projected roles remain non-confirmed; use the operator-verified ledger for official/team-confirmed roles that the public feed does not establish. The official injury importer stores the report's own publication time; the model also stores the time it was retrieved, but you still need to refresh close to tip-off.

## Probability and risk controls

For props with a line, the estimator uses a discrete Poisson or negative-binomial count distribution fitted to the model's context-specific quantile spread. Points use a small wider-tail mixture. Whole-number lines carry an explicit push probability; reported over/under probabilities used for calibration are conditional on a non-push result, and expected value accounts for the push separately.

A qualified market probability is shrunk toward a compatible two-sided, same-book no-vig price. Until the model has a verified projected-minutes distribution and role confirmation, the model's weight is limited to 25% and the directional probability cap is 65%. A raw probability breach is flagged for review rather than silently clipped. Missing no-vig market data withholds the calibrated probability. Edges above ten points are red-flagged without a specific logged injury/role/minutes/matchup explanation.

Shortlisting now requires at least a four-point calibrated edge and strictly positive expected value, a market snapshot no older than two hours, verified fresh player availability by default, and prior quote history to check public-board line movement. It also limits exposure to one prop per player-game, two props per game, quarter-Kelly stake sizes capped at 2% of bankroll per prop, and 5% total daily exposure. The minutes model is still a rolling proxy, not a true minutes-distribution model, so current lineup and rotation verification remain essential. These controls improve discipline; they do not establish profitable predictive skill. Validate using chronological Brier score, log loss, calibration buckets, ROI from genuinely pregame prices, and closing-line value before increasing model weight.

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
- Over/under probability now uses a discrete Poisson/negative-binomial outcome distribution, a wider-tail mixture for points, explicit integer-line push probability, and a 10,000-draw minutes Monte Carlo when a valid minutes distribution is available. These assumptions are still experimental and require chronological calibration before they should influence decisions strongly.
- The probability calibration script requires historical pregame predictions and settled props. The public market history file tracks observations only from the moment it is enabled; it cannot reconstruct earlier prices. A timestamped archived price provides traceability, but ROI is still a historical simulation rather than proof that an edge will persist.
- A low mean absolute error or high predicted probability is not proof of profitable betting.
- The official injury report parser depends on the source PDF's current table layout. Parser failure stops the import rather than assuming an all-clear. The official report is a timestamped snapshot that may change again before tip-off.
- The public prop-board importer may not provide complete line/book coverage; integer-line pushes are now modeled, but missing source lines or unmatched same-book prices remain blocked.
- No automatic bet placement is implemented.
