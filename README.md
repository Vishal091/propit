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

This writes data/prop_lines_public.csv. Then score available markets and print up to ten qualified candidates:

    .\.venv\Scripts\python.exe -m src.predict_slate --lines data\prop_lines_public.csv --top 10

Output files:
- predictions/latest.csv — full scored market/audit table
- predictions/shortlist.csv — only candidates that pass edge, expected-value, probability, source-label, and data-freshness screens

If the importer errors or finds zero markets, inspect the logs. Do not interpret missing data as a pick. Prices may be stale, US-market-specific, or unavailable in your region. Check actual player/market IDs, lines and both side prices before using any row.

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

- The model is experimental. A low mean absolute error or a high predicted probability is not proof of profitable betting.
- A public market importer is best effort, not a supported data feed. No API key or subscription is required, but service access/availability isn't guaranteed.
- The model's probability calibration and out-of-sample performance must be validated before relying on estimated edge.
- Current roster status is not an official injury or game-availability feed.
- Integer prop lines are skipped by the importer because the current EV model does not account for push probability.
- No automatic bet placement is implemented.
