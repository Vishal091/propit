"""Local PropIt dashboard with safe, explicit workflow actions.

Run with: python -m src.dashboard
The server binds to loopback only and executes only fixed project commands.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREDICTIONS = ROOT / "predictions"
HISTORY = DATA / "prediction_history.csv"
LATEST = PREDICTIONS / "latest.csv"
SHORTLIST = PREDICTIONS / "shortlist.csv"
HOLDOUT = PREDICTIONS / "backtest_holdout.csv"
PUBLIC_LINES = DATA / "prop_lines_public.csv"
HTML_FILE = Path(__file__).with_name("dashboard.html")

app = FastAPI(title="PropIt Local Dashboard", version="1.0.0")


class GameDateRequest(BaseModel):
    game_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


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
    try:
        result = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        return {
            "step": module, "ok": result.returncode == 0,
            "returncode": result.returncode, "output": output or "(no output)",
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
        return {
            "step": module, "ok": False, "returncode": None,
            "output": (output + "\n" if output else "") + f"Timed out after {timeout_seconds} seconds.",
        }
    except OSError as exc:
        return {"step": module, "ok": False, "returncode": None, "output": str(exc)}


def _run_steps(plan: list[tuple[str, list[str]]]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for module, args in plan:
        step = _run_step(module, args)
        steps.append(step)
        if not step["ok"]:
            break
    return {
        "ok": bool(steps) and all(step["ok"] for step in steps) and len(steps) == len(plan),
        "steps": steps,
        "message": "All requested steps completed." if steps and all(step["ok"] for step in steps) and len(steps) == len(plan)
                   else "Workflow stopped at the first failed step. Review the logs below.",
    }


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


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return _state()


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
            "holdout_start": str(date_values.min().date()) if not date_values.empty and date_values.notna().any() else None,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8766, reload=False)
