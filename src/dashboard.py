"""Local PropIt dashboard with safe, explicit workflow actions.

Run with: python -m src.dashboard
The server binds to loopback only and executes only fixed project commands.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.operations import append_audit_event, build_health_report, read_audit_events
from src.soccer.insights import build_best_picks, build_calibration_report, canonical_top_five_league

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS = ROOT / "predictions"
HISTORY = DATA / "prediction_history.csv"
LATEST = PREDICTIONS / "latest.csv"
SHORTLIST = PREDICTIONS / "shortlist.csv"
HOLDOUT = PREDICTIONS / "backtest_holdout.csv"
PUBLIC_LINES = DATA / "prop_lines_public.csv"
SOCCER_DATA = DATA / "soccer"
SOCCER_MATCHES = SOCCER_DATA / "matches.csv"
SOCCER_UPCOMING = PREDICTIONS / "soccer" / "upcoming.csv"
SOCCER_BACKTEST = PREDICTIONS / "soccer" / "backtest.csv"
SOCCER_METRICS = PREDICTIONS / "soccer" / "backtest_metrics.json"
HTML_FILE = Path(__file__).with_name("dashboard.html")

app = FastAPI(title="PropIt Local Dashboard", version="1.0.0")


class GameDateRequest(BaseModel):
    game_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class SoccerImportRequest(BaseModel):
    provider: str = Field(default="openfootball", pattern=r"^(statsbomb|openfootball)$")
    league: str = Field(default="england", pattern=r"^(england|germany|spain|italy|france)$")
    season: str = Field(default="2026-27", pattern=r"^\d{4}[-/]\d{2,4}$")
    competition_id: str | None = Field(default=None, pattern=r"^\d+$")
    season_id: str | None = Field(default=None, pattern=r"^\d+$")
    include_xg: bool = False
    limit: int | None = Field(default=None, ge=1, le=1000)


class SoccerForecastRequest(BaseModel):
    date_from: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    competition_id: str | None = Field(default=None, max_length=160)
    total_goals_line: float = Field(default=2.5, ge=0, le=15)
    prior_games: float = Field(default=5.0, gt=0, le=1000)
    half_life_days: float = Field(default=365.0, gt=0, le=10000)
    lookback_days: int = Field(default=730, ge=1, le=10000)
    dixon_coles_rho: float = Field(default=-0.05, ge=-0.5, le=0.5)


class SoccerBacktestRequest(BaseModel):
    competition_id: str | None = Field(default=None, max_length=160)
    min_history: int = Field(default=30, ge=0, le=100000)
    prior_games: float = Field(default=5.0, gt=0, le=1000)
    half_life_days: float = Field(default=365.0, gt=0, le=10000)
    lookback_days: int = Field(default=730, ge=1, le=10000)
    total_goals_line: float = Field(default=2.5, ge=0, le=15)
    dixon_coles_rho: float = Field(default=-0.05, ge=-0.5, le=0.5)


def _validate_game_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("Game date must be a real date in YYYY-MM-DD format.") from exc


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                clean[key] = None
                continue
            try:
                if pd.isna(value):
                    clean[key] = None
                    continue
            except (TypeError, ValueError):
                pass
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except (ValueError, AttributeError):
                    pass
            clean[key] = value
        records.append(clean)
    return records


def _file_stamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _date_already_archived(game_date: str) -> bool:
    """Check snapshot date, not per-player match dates, to avoid duplicate snapshots."""
    history = _read_csv(HISTORY)
    if history.empty:
        return False
    if "snapshot_date" in history.columns:
        dates = history["snapshot_date"].copy()
        if "game_date" in history.columns:
            missing = dates.isna() | dates.astype(str).str.strip().isin(["", "nan", "NaT", "<NA>"])
            dates = dates.where(~missing, history["game_date"])
    elif "game_date" in history.columns:
        dates = history["game_date"]
    else:
        return False
    normalized = pd.to_datetime(dates, errors="coerce").dt.strftime("%Y-%m-%d")
    return bool(normalized.eq(game_date).any())


def _run_step(module: str, args: list[str] | None = None, timeout_seconds: int = 600) -> dict[str, Any]:
    command = [sys.executable, "-m", module, *(args or [])]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        result = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        return {
            "step": module, "args": args or [], "ok": result.returncode == 0,
            "returncode": result.returncode, "output": output or "(no output)",
            "started_at_utc": started_at, "duration_seconds": round(time.perf_counter() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
        return {
            "step": module, "args": args or [], "ok": False, "returncode": None,
            "output": (output + "\n" if output else "") + f"Timed out after {timeout_seconds} seconds.",
            "started_at_utc": started_at, "duration_seconds": round(time.perf_counter() - started, 3),
        }
    except OSError as exc:
        return {
            "step": module, "args": args or [], "ok": False, "returncode": None, "output": str(exc),
            "started_at_utc": started_at, "duration_seconds": round(time.perf_counter() - started, 3),
        }


def _run_steps(plan: list[tuple[str, list[str]]]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for module, args in plan:
        step = _run_step(module, args)
        steps.append(step)
        if not step["ok"]:
            break
    ok = bool(steps) and all(step["ok"] for step in steps) and len(steps) == len(plan)
    result = {
        "ok": ok,
        "steps": steps,
        "message": "All requested steps completed." if ok
                   else "Workflow stopped at the first failed step. Review the logs below.",
    }
    event = {
        "event_type": "workflow",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "started_at_utc": started_at,
        "ok": ok,
        "message": result["message"],
        "steps": [
            {
                "step": step["step"],
                "ok": bool(step["ok"]),
                "returncode": step.get("returncode"),
                "duration_seconds": step.get("duration_seconds"),
                "args": step.get("args", []),
                "error_excerpt": str(step.get("output", ""))[-1000:] if not step["ok"] else "",
            }
            for step in steps
        ],
    }
    try:
        append_audit_event(event, ROOT / "data" / "operations_audit.jsonl")
        result["audit_recorded"] = True
    except OSError as exc:
        result["audit_recorded"] = False
        result["audit_error"] = str(exc)
    return result


def _state() -> dict[str, Any]:
    latest = _read_csv(LATEST)
    shortlist = _read_csv(SHORTLIST)
    history = _read_csv(HISTORY)
    lines = _read_csv(PUBLIC_LINES)
    settled = pushes = pending = 0
    if not history.empty and "settlement_status" in history.columns:
        statuses = history["settlement_status"].fillna("PENDING").astype(str).str.upper()
        settled = int(statuses.eq("SETTLED").sum())
        pushes = int(statuses.eq("PUSH").sum())
        pending = int(statuses.isin(["", "PENDING", "UNSETTLED"]).sum())
    return {
        "latest_rows": int(len(latest)),
        "shortlist_rows": int(len(shortlist)),
        "market_rows": int(len(lines)),
        "history_rows": int(len(history)),
        "settled_rows": settled,
        "push_rows": pushes,
        "pending_rows": pending,
        "latest_updated": _file_stamp(LATEST),
        "lines_updated": _file_stamp(PUBLIC_LINES),
        "history_updated": _file_stamp(HISTORY),
        "holdout_available": HOLDOUT.exists() and HOLDOUT.stat().st_size > 0,
    }


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(HTML_FILE)


def _soccer_competitions(matches: pd.DataFrame) -> list[dict[str, Any]]:
    if matches.empty or "competition_id" not in matches.columns:
        return []
    columns = [column for column in ("provider", "competition_id", "competition_name", "season_id", "season_name") if column in matches.columns]
    frame = matches[columns].fillna("").drop_duplicates()
    rows = _records(frame)
    counts = matches.groupby("competition_id", dropna=False).size().to_dict()
    for row in rows:
        row["match_count"] = int(counts.get(row.get("competition_id", ""), 0))
    return sorted(rows, key=lambda row: (
        str(row.get("competition_name", "")).casefold(),
        str(row.get("season_name", "")).casefold(),
        str(row.get("provider", "")).casefold(),
    ))


def _soccer_state() -> dict[str, Any]:
    matches = _read_csv(SOCCER_MATCHES)
    upcoming = _read_csv(SOCCER_UPCOMING)
    backtest = _read_csv(SOCCER_BACKTEST)
    scored = 0
    if not matches.empty and {"home_score", "away_score"}.issubset(matches.columns):
        scored = int((pd.to_numeric(matches["home_score"], errors="coerce").notna()
                      & pd.to_numeric(matches["away_score"], errors="coerce").notna()).sum())
    forecast_metrics = {}
    if SOCCER_METRICS.exists():
        try:
            forecast_metrics = json.loads(SOCCER_METRICS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            forecast_metrics = {}
    return {
        "available": not matches.empty,
        "matches_path": str(SOCCER_MATCHES.relative_to(ROOT)),
        "match_rows": int(len(matches)),
        "scored_matches": scored,
        "forecast_rows": int(len(upcoming)),
        "backtest_rows": int(len(backtest)),
        "matches_updated": _file_stamp(SOCCER_MATCHES),
        "forecasts_updated": _file_stamp(SOCCER_UPCOMING),
        "backtest_updated": _file_stamp(SOCCER_BACKTEST),
        "metrics_updated": _file_stamp(SOCCER_METRICS),
        "competitions": _soccer_competitions(matches),
        "metrics_available": bool(forecast_metrics),
    }


@app.get("/api/soccer/state")
def api_soccer_state() -> dict[str, Any]:
    """Return soccer data counts, competition choices and local file timestamps."""
    return _soccer_state()


@app.get("/api/soccer/matches")
def api_soccer_matches(
    view: str = Query(default="all", pattern="^(all|upcoming|finished)$"),
    date_from: str | None = Query(default=None, max_length=10),
    date_to: str | None = Query(default=None, max_length=10),
    competition_id: str | None = Query(default=None, max_length=160),
    limit: int = Query(default=500, ge=1, le=3000),
) -> dict[str, Any]:
    """Filter imported soccer fixtures/results without making a network request."""
    frame = _read_csv(SOCCER_MATCHES)
    if frame.empty:
        return {"available": False, "view": view, "count": 0, "rows": [], "message": "No soccer dataset yet. Import fixtures/results first."}
    if competition_id:
        if "competition_id" not in frame.columns:
            frame = frame.iloc[0:0]
        else:
            frame = frame[frame["competition_id"].fillna("").astype(str).eq(competition_id)]
    try:
        start = _validate_game_date(date_from) if date_from else None
        end = _validate_game_date(date_to) if date_to else None
        if start and end and end < start:
            raise ValueError("date_to must be on or after date_from")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if start and "match_date" in frame.columns:
        dates = pd.to_datetime(frame["match_date"], errors="coerce")
        frame = frame[dates >= pd.Timestamp(start)]
    if end and "match_date" in frame.columns:
        dates = pd.to_datetime(frame["match_date"], errors="coerce")
        frame = frame[dates <= pd.Timestamp(end)]
    if view != "all" and not frame.empty:
        home_score = pd.to_numeric(frame.get("home_score", pd.Series(index=frame.index, dtype="object")), errors="coerce")
        away_score = pd.to_numeric(frame.get("away_score", pd.Series(index=frame.index, dtype="object")), errors="coerce")
        scored = home_score.notna() & away_score.notna()
        if view == "finished":
            frame = frame[scored]
        else:
            frame = frame[~scored]
    sort_cols = [column for column in ("match_date", "kickoff_time_local", "competition_name", "home_team") if column in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=True, kind="stable", na_position="last")
    count = int(len(frame))
    return {"available": True, "view": view, "count": count, "rows": _records(frame.head(limit))}


@app.get("/api/soccer/forecasts")
def api_soccer_forecasts(
    date_from: str | None = Query(default=None, max_length=10),
    date_to: str | None = Query(default=None, max_length=10),
    competition_id: str | None = Query(default=None, max_length=160),
    limit: int = Query(default=500, ge=1, le=3000),
) -> dict[str, Any]:
    """Read the latest local soccer forecast file; this endpoint does not trigger a model run."""
    frame = _read_csv(SOCCER_UPCOMING)
    if frame.empty:
        return {"available": False, "count": 0, "rows": [], "message": "No soccer forecasts yet. Generate forecasts for a date range."}
    if competition_id and "competition_id" in frame.columns:
        frame = frame[frame["competition_id"].fillna("").astype(str).eq(competition_id)]
    try:
        start = _validate_game_date(date_from) if date_from else None
        end = _validate_game_date(date_to) if date_to else None
        if start and end and end < start:
            raise ValueError("date_to must be on or after date_from")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if start:
        dates = pd.to_datetime(frame["match_date"], errors="coerce") if "match_date" in frame.columns else pd.Series(index=frame.index, dtype="datetime64[ns]")
        frame = frame[dates >= pd.Timestamp(start)]
    if end:
        dates = pd.to_datetime(frame["match_date"], errors="coerce") if "match_date" in frame.columns else pd.Series(index=frame.index, dtype="datetime64[ns]")
        frame = frame[dates <= pd.Timestamp(end)]
    sort_cols = [column for column in ("match_date", "kickoff_time_local", "competition_name", "home_team") if column in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=True, kind="stable", na_position="last")
    return {"available": True, "count": int(len(frame)), "rows": _records(frame.head(limit))}


@app.get("/api/soccer/picks")
def api_soccer_picks(
    date_from: str | None = Query(default=None, max_length=10),
    date_to: str | None = Query(default=None, max_length=10),
    min_history_matches: int = Query(default=8, ge=0, le=10000),
    min_probability: float = Query(default=0.52, ge=0, le=1),
    limit: int = Query(default=25, ge=1, le=500),
) -> dict[str, Any]:
    """Return ranked model-only picks for the top five European domestic leagues."""
    frame = _read_csv(SOCCER_UPCOMING)
    if frame.empty:
        return {
            "available": False, "count": 0, "rows": [],
            "message": "No saved soccer forecasts yet. Import fixtures/results, then generate forecasts.",
            "eligible_leagues": ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
        }
    try:
        start = _validate_game_date(date_from) if date_from else None
        end = _validate_game_date(date_to) if date_to else None
        if start and end and end < start:
            raise ValueError("date_to must be on or after date_from")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if start and "match_date" in frame.columns:
        dates = pd.to_datetime(frame["match_date"], errors="coerce")
        frame = frame[dates >= pd.Timestamp(start)]
    if end and "match_date" in frame.columns:
        dates = pd.to_datetime(frame["match_date"], errors="coerce")
        frame = frame[dates <= pd.Timestamp(end)]
    picks = build_best_picks(
        _records(frame), min_history_matches=min_history_matches,
        min_probability=min_probability, limit=limit,
    )
    return {
        "available": bool(picks), "count": len(picks), "rows": picks,
        "eligible_leagues": ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
        "min_history_matches": min_history_matches, "min_probability": min_probability,
        "method": "One strongest independent-Poisson market per fixture; ranked with a historical-sample support adjustment.",
        "message": "No pick met the top-five-league, sample-history, and probability filters." if not picks else "",
        "warning": "No sportsbook odds, lineups, or injury information are included. This is a model-confidence ranking, not an odds-based value/EV recommendation.",
    }


@app.get("/api/soccer/calibration")
def api_soccer_calibration(
    bins: int = Query(default=10, ge=2, le=20),
    min_samples: int = Query(default=30, ge=1, le=100000),
) -> dict[str, Any]:
    """Summarize top-five-league reliability from saved chronological backtest predictions."""
    matches = _read_csv(SOCCER_MATCHES)
    backtest = _read_csv(SOCCER_BACKTEST)
    eligible_ids = {
        str(row.get("competition_id", ""))
        for row in _records(matches)
        if row.get("competition_id") and canonical_top_five_league(row)
    }
    report = build_calibration_report(
        _records(backtest), competition_ids=eligible_ids,
        bins=bins, min_samples=min_samples,
    )
    report["eligible_leagues"] = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    report["eligible_competitions"] = len(eligible_ids)
    if not eligible_ids:
        report["message"] = "Import recognized top-five league results first."
    elif backtest.empty:
        report["message"] = "Run the chronological soccer backtest to generate out-of-sample predictions."
    return report


@app.get("/api/soccer/metrics")
def api_soccer_metrics() -> dict[str, Any]:
    """Read the last chronological soccer backtest report, if present."""
    if not SOCCER_METRICS.exists():
        return {"available": False, "report": None, "message": "No soccer backtest metrics yet. Run a backtest after importing historical results."}
    try:
        report = json.loads(SOCCER_METRICS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "report": None, "message": "Soccer metrics file is unreadable. Run the backtest again."}
    return {"available": bool(report), "report": report}


@app.get("/api/soccer/catalog")
def api_soccer_catalog(search: str = Query(default="", max_length=80)) -> dict[str, Any]:
    """Explicitly fetch the StatsBomb Open Data catalog; never called by normal dashboard loading."""
    clean_search = search.strip()
    if any(ord(char) < 32 for char in clean_search):
        raise ValueError("Search cannot contain control characters")
    args = ["--provider", "statsbomb", "--list-competitions"]
    if clean_search:
        args.append("--search=" + clean_search)
    step = _run_step("scripts.import_soccer_data", args, timeout_seconds=120)
    rows = []
    if step["ok"]:
        for line in str(step.get("output", "")).splitlines():
            values = line.split("\t")
            if len(values) >= 6 and values[0] != "competition_id" and values[0].isdigit() and values[1].isdigit():
                rows.append({
                    "competition_id": values[0], "season_id": values[1],
                    "competition_name": values[2], "season_name": values[3],
                    "country_name": values[4], "cache_status": values[5],
                })
    return {"ok": bool(step["ok"]), "rows": rows, "count": len(rows), "output": step.get("output", ""), "message": "Catalog loaded." if step["ok"] else "Could not load StatsBomb catalog; inspect the output."}


@app.post("/api/soccer/import")
def api_soccer_import(request: SoccerImportRequest) -> dict[str, Any]:
    """Run one allow-listed provider import and expose its operation log."""
    if request.provider == "statsbomb":
        if not request.competition_id or not request.season_id:
            return {"ok": False, "steps": [], "message": "Load the StatsBomb catalog and select a competition and season first."}
        args = ["--provider", "statsbomb", "--competition-id", request.competition_id, "--season-id", request.season_id]
        if request.include_xg:
            args.append("--include-xg")
    else:
        args = ["--provider", "openfootball", "--league", request.league, "--season", request.season]
    if request.limit is not None:
        args.extend(["--limit", str(request.limit)])
    result = _run_steps([("scripts.import_soccer_data", args)])
    result["soccer_state"] = _soccer_state()
    return result


@app.post("/api/soccer/forecast")
def api_soccer_forecast(request: SoccerForecastRequest) -> dict[str, Any]:
    """Run forecast generation for the selected date range and optional competition."""
    try:
        start = _validate_game_date(request.date_from)
        end = _validate_game_date(request.date_to) if request.date_to else None
        if end is not None and end < start:
            raise ValueError("date_to must be on or after date_from")
    except ValueError as exc:
        return {"ok": False, "steps": [], "message": str(exc)}
    args = [
        "--input", "data/soccer/matches.csv",
        "--date-from", start,
        "--total-goals-line", str(request.total_goals_line),
        "--prior-games", str(request.prior_games),
        "--half-life-days", str(request.half_life_days),
        "--lookback-days", str(request.lookback_days),
        "--dixon-coles-rho", str(request.dixon_coles_rho),
    ]
    if end is not None:
        args.extend(["--date-to", end])
    if request.competition_id:
        args.extend(["--competition-id", request.competition_id])
    result = _run_steps([("scripts.predict_soccer", args)])
    result["soccer_state"] = _soccer_state()
    return result


@app.post("/api/soccer/backtest")
def api_soccer_backtest(request: SoccerBacktestRequest) -> dict[str, Any]:
    """Run a historical soccer backtest with explicit model parameters."""
    args = [
        "--input", "data/soccer/matches.csv",
        "--min-history", str(request.min_history),
        "--prior-games", str(request.prior_games),
        "--half-life-days", str(request.half_life_days),
        "--lookback-days", str(request.lookback_days),
        "--total-goals-line", str(request.total_goals_line),
        "--dixon-coles-rho", str(request.dixon_coles_rho),
    ]
    if request.competition_id:
        args.extend(["--competition-id", request.competition_id])
    result = _run_steps([("scripts.backtest_soccer", args)])
    result["soccer_state"] = _soccer_state()
    return result



@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return _state()


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """Report local data freshness, model artifacts, history and recent failures."""
    return build_health_report(ROOT)


@app.get("/api/audit")
def api_audit(limit: int = Query(default=40, ge=1, le=200)) -> dict[str, Any]:
    """Read the most recent local workflow audit events."""
    rows = read_audit_events(ROOT / "data" / "operations_audit.jsonl", limit=limit)
    return {"count": len(rows), "rows": rows}


@app.get("/api/slate")
def api_slate(view: str = Query(default="shortlist", pattern="^(shortlist|all)$")) -> dict[str, Any]:
    path = SHORTLIST if view == "shortlist" else LATEST
    frame = _read_csv(path)
    return {"view": view, "count": int(len(frame)), "rows": _records(frame.head(500))}


@app.get("/api/history")
def api_history(
    limit: int = Query(default=1000, ge=1, le=5000),
    status: str = Query(default="all", pattern="^(all|pending|settled|push)$"),
) -> dict[str, Any]:
    frame = _read_csv(HISTORY)
    if frame.empty:
        return {"count": 0, "rows": []}
    if "game_date" in frame.columns:
        frame = frame.assign(_sort_date=pd.to_datetime(frame["game_date"], errors="coerce"))
        frame = frame.sort_values(["_sort_date"], ascending=False, kind="stable").drop(columns=["_sort_date"])
    if status != "all" and "settlement_status" in frame.columns:
        values = frame["settlement_status"].fillna("PENDING").astype(str).str.upper()
        wanted = {"pending": ["PENDING", "", "UNSETTLED"], "settled": ["SETTLED"], "push": ["PUSH"]}[status]
        frame = frame[values.isin([v.upper() for v in wanted])]
    return {"count": int(len(frame)), "rows": _records(frame.head(limit))}


@app.get("/api/calibration")
def api_calibration() -> dict[str, Any]:
    from scripts.calibrate import _metrics, reliability

    holdout = _read_csv(HOLDOUT)
    if holdout.empty or not {"target", "actual_over", "p_over", "calibrated_p_over"}.issubset(holdout.columns):
        return {"available": False, "summary": [], "reliability": [], "rows": [], "count": 0}

    summaries: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for target, group in holdout.groupby("target", sort=True):
        actual = pd.to_numeric(group["actual_over"], errors="coerce")
        raw = pd.to_numeric(group["p_over"], errors="coerce")
        calibrated = pd.to_numeric(group["calibrated_p_over"], errors="coerce")
        valid = actual.isin([0, 1]) & raw.notna() & calibrated.notna()
        valid_group = group.loc[valid].copy()
        if valid_group.empty:
            continue
        valid_group["actual_over"] = pd.to_numeric(valid_group["actual_over"], errors="coerce").astype(int)
        valid_group["p_over"] = pd.to_numeric(valid_group["p_over"], errors="coerce")
        valid_group["calibrated_p_over"] = pd.to_numeric(valid_group["calibrated_p_over"], errors="coerce")
        raw_metrics = _metrics(valid_group, "p_over")
        cal_metrics = _metrics(valid_group, "calibrated_p_over")
        date_values = pd.to_datetime(valid_group["game_date"], errors="coerce") if "game_date" in valid_group else pd.Series(dtype="datetime64[ns]")
        saved_holdout_start = None
        if "calibration_holdout_start" in valid_group:
            saved_values = valid_group["calibration_holdout_start"].dropna().astype(str).str.strip()
            saved_values = saved_values[saved_values.ne("")]
            if not saved_values.empty:
                saved_holdout_start = saved_values.iloc[0][:10]
        saved_train_end = None
        if "calibration_train_end" in valid_group:
            train_values = valid_group["calibration_train_end"].dropna().astype(str).str.strip()
            train_values = train_values[train_values.ne("")]
            if not train_values.empty:
                saved_train_end = train_values.iloc[0][:10]
        summaries.append({
            "target": str(target),
            "n": int(len(valid_group)),
            "raw_brier": round(float(raw_metrics["brier"]), 4),
            "calibrated_brier": round(float(cal_metrics["brier"]), 4),
            "raw_log_loss": round(float(raw_metrics["log_loss"]), 4),
            "calibrated_log_loss": round(float(cal_metrics["log_loss"]), 4),
            "raw_hit_rate": round(float(raw_metrics["hit_rate"]), 4),
            "calibrated_hit_rate": round(float(cal_metrics["hit_rate"]), 4),
            "raw_roi": raw_metrics["roi"] if pd.notna(raw_metrics["roi"]) else None,
            "calibrated_roi": cal_metrics["roi"] if pd.notna(cal_metrics["roi"]) else None,
            "raw_bets": int(raw_metrics["bets"]),
            "calibrated_bets": int(cal_metrics["bets"]),
            "timestamped_price_rows": int(cal_metrics["timestamped_rows"]),
            "eligible_price_rows": int(cal_metrics["priced_rows"]),
            "raw_net_units": round(float(raw_metrics["net_units"]), 3),
            "calibrated_net_units": round(float(cal_metrics["net_units"]), 3),
            "calibration_train_end": saved_train_end,
            "holdout_start": saved_holdout_start or (
                str(date_values.min().date()) if not date_values.empty and date_values.notna().any() else None
            ),
        })
        for version, probability_col in (("Raw", "p_over"), ("Calibrated", "calibrated_p_over")):
            buckets = reliability(valid_group, probability_col, n_bins=10)
            for bucket in buckets.to_dict(orient="records"):
                reliability_rows.append({"target": str(target), "version": version, **bucket})

    cols = [c for c in [
        "game_date", "snapshot_date", "snapshot_at_utc", "player_name", "target", "line",
        "p_over", "calibrated_p_over", "actual_over", "actual_stat",
        "odds_over_american", "odds_under_american", "raw_brier", "calibrated_brier",
    ] if c in holdout.columns]
    return {
        "available": bool(summaries),
        "summary": summaries,
        "reliability": reliability_rows,
        "rows": _records(holdout[cols].tail(300).iloc[::-1]),
        "count": int(len(holdout)),
        "roi_policy": "ROI only counts positive-EV bets with both odds and snapshot_at_utc. It is a historical simulation, not proof of future profitability.",
    }


@app.post("/api/refresh")
def api_refresh() -> dict[str, Any]:
    """Refresh the public line snapshot, then regenerate the ranked slate."""
    return _run_steps([
        ("scripts.fetch_public_props", []),
        ("src.predict_slate", ["--lines", "data/prop_lines_public.csv", "--top", "10"]),
    ])


@app.post("/api/pregame-workflow")
def api_pregame_workflow(request: GameDateRequest) -> dict[str, Any]:
    """Refresh lines, predict, and archive one snapshot for the selected snapshot date."""
    try:
        game_date = _validate_game_date(request.game_date)
    except ValueError as exc:
        return {"ok": False, "steps": [], "message": str(exc)}
    if _date_already_archived(game_date):
        return {
            "ok": False, "steps": [],
            "message": f"A snapshot dated {game_date} is already in data/prediction_history.csv. No files were changed. Use Refresh Slate for a new view, or intentionally archive another snapshot from the command line.",
        }
    plan = [
        ("scripts.fetch_public_props", []),
        ("src.predict_slate", ["--lines", "data/prop_lines_public.csv", "--top", "10"]),
        ("scripts.archive_predictions", ["--game-date", game_date]),
    ]
    result = _run_steps(plan)
    if result["ok"]:
        result["message"] = f"Slate refreshed and pregame snapshot archived for snapshot date {game_date}. Confirm this was before tip-off."
    return result


@app.post("/api/settle")
def api_settle(request: GameDateRequest) -> dict[str, Any]:
    try:
        game_date = _validate_game_date(request.game_date)
    except ValueError as exc:
        return {"ok": False, "steps": [], "message": str(exc)}
    if not HISTORY.exists():
        return {"ok": False, "steps": [], "message": "No prediction history exists yet. Run the pregame workflow before settling results."}
    return _run_steps([("scripts.settle_predictions", ["--game-date", game_date])])


@app.post("/api/calibrate")
def api_calibrate() -> dict[str, Any]:
    if not HISTORY.exists() or _read_csv(HISTORY).empty:
        return {"ok": False, "steps": [], "message": "No prediction history exists. Archive pregame slates and settle them first."}
    return _run_steps([("scripts.calibrate", ["data/prediction_history.csv"])])


@app.post("/api/ops-report")
def api_ops_report() -> dict[str, Any]:
    """Write a dated JSON and Markdown daily operations report."""
    return _run_steps([("scripts.ops_report", [])])


@app.post("/api/backup")
def api_backup() -> dict[str, Any]:
    """Create a ZIP backup of tracked source and essential local data."""
    return _run_steps([("scripts.backup_workspace", [])])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8766, reload=False)
