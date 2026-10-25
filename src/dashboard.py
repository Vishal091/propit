"""Local PropIt dashboard with safe, explicit workflow actions.

Run with: python -m src.dashboard
The server binds to loopback only and executes only fixed project commands.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.operations import append_audit_event, build_health_report, read_audit_events

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


class PregameWorkflowRequest(GameDateRequest):
    """Keep the archive snapshot date separate from the NBA-local event date."""
    nba_game_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


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


def _shortlist_diagnostics(frame: pd.DataFrame, *, limit: int = 5) -> dict[str, Any]:
    """Summarize scored-row rejection reasons for the empty-shortlist UI."""
    if frame.empty:
        return {"scored_rows": 0, "status_counts": []}
    statuses = frame.get(
        "shortlist_status", pd.Series("UNCLASSIFIED", index=frame.index)
    ).fillna("UNCLASSIFIED").astype(str).str.strip().replace("", "UNCLASSIFIED").str.upper()
    reasons = frame.get("shortlist_reason", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    grouped: list[dict[str, Any]] = []
    counts = statuses.value_counts(dropna=False)
    for status, count in counts.head(limit).items():
        matching_reasons = reasons.loc[statuses.eq(status)]
        matching_reasons = matching_reasons.loc[matching_reasons.ne("")]
        reason = matching_reasons.value_counts().index[0] if not matching_reasons.empty else ""
        grouped.append({"status": str(status), "count": int(count), "reason": reason})
    return {"scored_rows": int(len(frame)), "status_counts": grouped}


_PENDING_CONFIRMATION_STATUSES = {
    "INSUFFICIENT_LINEUP_DATA",
    "LINEUP_UNVERIFIED",
    "AVAILABILITY_UNVERIFIED",
    "AVAILABILITY_STALE",
    "AVAILABILITY_REVIEW",
    "INSUFFICIENT_AVAILABILITY_DATA",
}


def _pending_confirmation_candidates(
    frame: pd.DataFrame,
    *,
    limit: int = 50,
    min_probability: float = 0.53,
    max_market_age_hours: float = 2.0,
    max_data_age_days: int = 14,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a research-only queue for evidence-blocked candidates.

    Positive research signals sort first. Watch-only rows are also surfaced when
    they pass data, minutes, market integrity, price-history, and safety checks,
    but fail one of the positive edge/EV/probability tests. This never changes
    shortlist eligibility or represents a qualified bet.
    """
    if isinstance(limit, bool) or limit < 1 or limit > 200:
        raise ValueError("limit must be between 1 and 200")
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must be between 0 and 1")

    if frame.empty:
        return frame.copy(), {
            "scored_rows": 0,
            "evidence_blocked_rows": 0,
            "research_estimate_rows": 0,
            "positive_research_signal_rows": 0,
            "watch_only_rows": 0,
            "pending_rows": 0,
            "blocker_counts": {},
            "filter_stage_counts": {},
            "filter_rejections": {},
        }

    statuses = frame.get(
        "shortlist_status", pd.Series("", index=frame.index)
    ).fillna("").astype(str).str.strip().str.upper()
    evidence_mask = statuses.isin(_PENDING_CONFIRMATION_STATUSES)
    evidence = frame.loc[evidence_mask].copy()
    evidence_statuses = statuses.loc[evidence_mask]
    blocker_counts = {
        str(status): int(count)
        for status, count in evidence_statuses.value_counts().items()
    }

    def numeric(rows: pd.DataFrame, column: str) -> pd.Series:
        if column not in rows:
            return pd.Series(float("nan"), index=rows.index, dtype="float64")
        return pd.to_numeric(rows[column], errors="coerce")

    diag: dict[str, Any] = {
        "scored_rows": int(len(frame)),
        "evidence_blocked_rows": int(len(evidence)),
        "research_estimate_rows": 0,
        "positive_research_signal_rows": 0,
        "watch_only_rows": 0,
        "pending_rows": 0,
        "blocker_counts": blocker_counts,
        "min_research_probability": min_probability,
        "max_market_age_hours": max_market_age_hours,
        "max_data_age_days": max_data_age_days,
        "filter_stage_counts": {},
        "filter_rejections": {},
    }
    if evidence.empty:
        return evidence, diag

    research_status = evidence.get(
        "research_status", pd.Series("", index=evidence.index)
    ).fillna("").astype(str).str.upper()
    side = evidence.get(
        "research_recommended_side", pd.Series("", index=evidence.index)
    ).fillna("").astype(str).str.upper()
    edge = numeric(evidence, "research_selected_edge")
    ev = numeric(evidence, "research_selected_ev")
    probability = numeric(evidence, "research_selected_probability")
    market_age = numeric(evidence, "market_age_hours")
    data_age = numeric(evidence, "data_age_days")
    line = numeric(evidence, "line")
    over_odds = numeric(evidence, "odds_over_american")
    under_odds = numeric(evidence, "odds_under_american")
    market_over = numeric(evidence, "market_no_vig_p_over")
    market_under = numeric(evidence, "market_no_vig_p_under")

    research_ready = (
        research_status.eq("RESEARCH_ESTIMATE_AVAILABLE")
        & side.isin({"OVER", "UNDER"})
        & probability.notna()
        & edge.notna()
        & ev.notna()
    )
    diag["research_estimate_rows"] = int(research_ready.sum())

    market_method = evidence.get(
        "market_no_vig_method", pd.Series("", index=evidence.index)
    ).fillna("").astype(str).str.strip().str.casefold()
    bookmaker = evidence.get(
        "bookmaker", pd.Series("", index=evidence.index)
    ).fillna("").astype(str).str.strip().str.casefold()
    placeholder = bookmaker.eq("") | bookmaker.apply(
        lambda value: any(marker in value for marker in ("test", "example", "sample", "demo", "not_live", "not live"))
    )
    line_source = evidence.get(
        "line_source", pd.Series("", index=evidence.index)
    ).fillna("").astype(str).str.strip().str.casefold()
    market_source = evidence.get(
        "market_source", pd.Series("", index=evidence.index)
    ).fillna("").astype(str).str.strip().str.casefold()
    snapshot_stamp = evidence.get(
        "market_snapshot_at_utc", pd.Series("", index=evidence.index)
    ).fillna("").astype(str).str.strip()
    is_public = (
        line_source.eq("public_web_import")
        | market_source.str.contains(r"public board|flash odds", regex=True, na=False)
    )

    minutes_mean = numeric(evidence, "minutes_projection_mean")
    minutes_std = numeric(evidence, "minutes_projection_std")
    minutes_sample = numeric(evidence, "minutes_sample_size")
    probability_gate = evidence.get(
        "probability_gate_status", pd.Series("", index=evidence.index)
    ).fillna("").astype(str).str.strip().str.upper()

    prior_count = numeric(evidence, "market_prior_snapshot_count")
    movement_open = numeric(evidence, "line_move_from_open")
    movement_previous = numeric(evidence, "line_move_from_previous")
    movement = movement_open.where(movement_open.notna(), movement_previous)
    movement_explanation = evidence.get(
        "movement_explanation", pd.Series("", index=evidence.index)
    ).astype("string").fillna("").str.strip()
    edge_explanation = evidence.get(
        "edge_explanation", pd.Series("", index=evidence.index)
    ).astype("string").fillna("").str.strip()

    movement_present = prior_count.ge(1) | movement_explanation.str.len().ge(15)
    movement_against = (
        (side.eq("OVER") & movement.gt(0.5))
        | (side.eq("UNDER") & movement.lt(-0.5))
    )
    movement_documented = movement_explanation.str.len().ge(15)
    edge_documented = edge_explanation.str.lower().apply(
        lambda value: len(value) >= 20 and any(
            term in value for term in ("injury", "role", "minutes", "matchup", "starter", "return", "out")
        )
    )

    stages: list[tuple[str, pd.Series]] = [
        ("research_estimate", research_ready),
        ("fresh_market", market_age.ge(0) & market_age.le(max_market_age_hours)),
        ("fresh_player_data", data_age.ge(0) & data_age.le(max_data_age_days)),
        ("line_and_both_prices", line.notna() & line.ge(0) & over_odds.notna() & over_odds.ne(0) & under_odds.notna() & under_odds.ne(0)),
        ("same_book_no_vig", market_over.ge(0) & market_over.le(1) & market_under.ge(0) & market_under.le(1) & market_method.ne("") & ~market_method.eq("cross_book_pair_unavailable")),
        ("real_source_and_timestamp", ~placeholder & (~is_public | snapshot_stamp.ne(""))),
        ("usable_minutes_and_model", minutes_mean.notna() & minutes_std.notna() & minutes_sample.ge(3) & ~probability_gate.isin({"INSUFFICIENT_DATA", "INSUFFICIENT_MINUTES_DATA", "RAW_MODEL_CAP_RED_FLAG", "CALIBRATED_CAP_RED_FLAG"})),
        ("prior_market_movement", ~is_public | movement_present),
        ("documented_adverse_movement", ~movement_against | movement_documented),
        ("documented_extreme_edge", ~edge.abs().gt(0.10) | edge_documented),
    ]

    cumulative = research_ready.copy()
    stage_counts: dict[str, int] = {"evidence_blocked": int(len(evidence))}
    rejections: dict[str, int] = {}
    previous_count = int(cumulative.sum())
    stage_counts["research_estimate"] = previous_count
    rejections["missing_research_estimate_or_edge_ev"] = int(len(evidence) - previous_count)
    for index, (name, condition) in enumerate(stages[1:], start=1):
        cumulative = cumulative & condition.fillna(False)
        count = int(cumulative.sum())
        stage_counts[name] = count
        rejections["failed_" + name] = max(0, previous_count - count)
        previous_count = count

    quality_pass = cumulative
    signal_mask = (
        edge.gt(0)
        & ev.gt(0)
        & probability.ge(min_probability)
    )
    priority_mask = quality_pass & signal_mask
    diag["positive_research_signal_rows"] = int(priority_mask.sum())
    diag["watch_only_rows"] = int((quality_pass & ~signal_mask).sum())
    diag["filter_stage_counts"] = stage_counts
    diag["filter_rejections"] = rejections

    result = evidence.loc[quality_pass].copy()
    if result.empty:
        diag["pending_rows"] = 0
        return result, diag

    # Explain the actual research-signal miss instead of silently hiding a row.
    selected_edge = edge.loc[quality_pass]
    selected_ev = ev.loc[quality_pass]
    selected_probability = probability.loc[quality_pass]
    labels: list[str] = []
    notes: list[str] = []
    for idx in result.index:
        if bool(priority_mask.loc[idx]):
            labels.append("POSITIVE RESEARCH SIGNAL")
            notes.append("Positive exploratory edge and EV; research probability meets the monitoring threshold. Still pending verified evidence.")
        elif pd.isna(selected_edge.loc[idx]) or selected_edge.loc[idx] <= 0:
            labels.append("WATCH ONLY · EDGE NOT POSITIVE")
            notes.append("Research estimate is available, but its selected-side edge is not positive.")
        elif pd.isna(selected_ev.loc[idx]) or selected_ev.loc[idx] <= 0:
            labels.append("WATCH ONLY · EV NOT POSITIVE")
            notes.append("Research estimate is available, but its selected-side expected value is not positive.")
        elif pd.isna(selected_probability.loc[idx]) or selected_probability.loc[idx] < min_probability:
            labels.append("WATCH ONLY · BELOW PROBABILITY THRESHOLD")
            notes.append(f"Research probability is below the {min_probability:.0%} monitoring threshold.")
        else:
            labels.append("WATCH ONLY")
            notes.append("Research signal does not meet every priority-monitoring condition.")
    result["pending_label"] = labels
    result["pending_note"] = notes
    result["_priority_sort"] = (~priority_mask.loc[quality_pass]).astype(int)
    result["_pending_ev_sort"] = ev.loc[quality_pass]
    result["_pending_edge_sort"] = edge.loc[quality_pass]
    result["_pending_probability_sort"] = probability.loc[quality_pass]
    result = result.sort_values(
        ["_priority_sort", "_pending_ev_sort", "_pending_edge_sort", "_pending_probability_sort"],
        ascending=[True, False, False, False],
        na_position="last",
        kind="stable",
    ).head(limit).drop(
        columns=["_priority_sort", "_pending_ev_sort", "_pending_edge_sort", "_pending_probability_sort"]
    )
    result.insert(0, "pending_rank", range(1, len(result) + 1))
    diag["pending_rows"] = int(len(result))
    diag["returned_positive_signals"] = int(result["pending_label"].eq("POSITIVE RESEARCH SIGNAL").sum())
    diag["returned_watch_only"] = int(len(result) - diag["returned_positive_signals"])
    return result, diag


@app.get("/api/slate")
def api_slate(view: str = Query(default="shortlist", pattern="^(shortlist|all|research)$")) -> dict[str, Any]:
    path = SHORTLIST if view == "shortlist" else LATEST
    frame = _read_csv(path)
    diagnostics: dict[str, Any] = {}
    if view == "research" and not frame.empty:
        if "research_selected_ev" in frame.columns:
            ev_values = pd.to_numeric(frame["research_selected_ev"], errors="coerce")
            edge_values = (
                pd.to_numeric(frame["research_selected_edge"], errors="coerce")
                if "research_selected_edge" in frame.columns
                else pd.Series([float("nan")] * len(frame), index=frame.index, dtype="float64")
            )
            frame["_research_ev_sort"] = ev_values
            frame["_research_edge_sort"] = edge_values
            frame = frame.sort_values(
                ["_research_ev_sort", "_research_edge_sort"],
                ascending=[False, False], na_position="last", kind="stable",
            ).drop(columns=["_research_ev_sort", "_research_edge_sort"])
            research_probabilities = (
                pd.to_numeric(frame["research_model_p_over"], errors="coerce")
                if "research_model_p_over" in frame.columns
                else pd.Series([float("nan")] * len(frame), index=frame.index, dtype="float64")
            )
            research_ev_values = (
                pd.to_numeric(frame["research_selected_ev"], errors="coerce")
                if "research_selected_ev" in frame.columns
                else pd.Series([float("nan")] * len(frame), index=frame.index, dtype="float64")
            )
            available = int(research_probabilities.notna().sum())
            with_ev = int(research_ev_values.notna().sum())
            diagnostics = {
                "scored_rows": int(len(frame)),
                "rows_with_research_estimates": available,
                "rows_without_research_estimates": int(len(frame) - available),
                "rows_with_research_ev": with_ev,
                "notice": "Research probabilities are raw when no compatible no-vig reference exists; otherwise they use 15% raw-model weight and 85% market weight. Edge/EV are withheld without a valid no-vig market and both side prices. Research rows never satisfy strict shortlist gates.",
            }
        else:
            diagnostics = {
                "scored_rows": int(len(frame)),
                "rows_with_research_estimates": 0,
                "rows_without_research_estimates": int(len(frame)),
                "notice": "Research columns are not present in this saved snapshot. Click Refresh & score to regenerate predictions with the research view enabled.",
            }
    result = {"view": view, "count": int(len(frame)), "rows": _records(frame.head(500))}
    if view == "shortlist" and frame.empty:
        result["diagnostics"] = _shortlist_diagnostics(_read_csv(LATEST))
    if diagnostics:
        result["diagnostics"] = diagnostics
    return result


@app.get("/api/pending-confirmation")
def api_pending_confirmation(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    """Return research-only leads held out of the strict shortlist by evidence gates."""
    latest = _read_csv(LATEST)
    rows, diagnostics = _pending_confirmation_candidates(latest, limit=limit)
    if rows.empty:
        stages = diagnostics.get("filter_stage_counts", {})
        reject_counts = diagnostics.get("filter_rejections", {})
        message = (
            "No pending rows passed the research-data and market-integrity screens. "
            f"Scored rows: {diagnostics.get('scored_rows', 0)}; "
            f"evidence-blocked rows: {diagnostics.get('evidence_blocked_rows', 0)}; "
            f"rows with usable research edge/EV: {diagnostics.get('research_estimate_rows', 0)}. "
            f"Filter stage counts: {stages}. Filter rejections: {reject_counts}. "
            "No shortlist gates were relaxed."
        )
    else:
        positive = int(diagnostics.get("returned_positive_signals", 0))
        watch = int(diagnostics.get("returned_watch_only", 0))
        message = (
            f"Returned {len(rows)} research-only leads: {positive} positive research signal(s), "
            f"{watch} watch-only row(s). Positive signals sort first; watch-only rows explicitly "
            "show which edge/EV/probability check they missed. All remain blocked from the qualified "
            "shortlist until current lineup/availability evidence is verified and the full slate is rescored."
        )
    return {
        "count": int(len(rows)),
        "rows": _records(rows),
        "diagnostics": diagnostics,
        "message": message,
        "warning": (
            "This is a research and follow-up queue, not an approved bet list. Watch-only rows may fail "
            "edge, EV, or research-probability thresholds. Pending status never relaxes the qualified "
            "shortlist's strict availability, confirmed-role, market, minutes, and exposure controls."
        ),
    }


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


@app.post("/api/injury-report")
def api_injury_report(request: GameDateRequest) -> dict[str, Any]:
    """Fetch the latest official NBA injury-report PDF for the selected match date."""
    try:
        game_date = _validate_game_date(request.game_date)
    except ValueError as exc:
        return {"ok": False, "steps": [], "message": str(exc)}
    return _run_steps([("scripts.import_nba_status", ["--game-date", game_date])])


@app.post("/api/lineups")
def api_lineups(request: GameDateRequest) -> dict[str, Any]:
    """Fetch lineup projections/explicit confirmations, then rebuild the current slate."""
    try:
        game_date = _validate_game_date(request.game_date)
    except ValueError as exc:
        return {"ok": False, "steps": [], "message": str(exc)}
    return _run_steps([
        ("scripts.fetch_lineups", ["--game-date", game_date]),
        ("src.predict_slate", ["--lines", "data/prop_lines_public.csv", "--top", "10"]),
    ])


@app.post("/api/pregame-workflow")
def api_pregame_workflow(request: PregameWorkflowRequest) -> dict[str, Any]:
    """Refresh date-specific availability and lineup evidence before scoring and archiving."""
    try:
        snapshot_date = _validate_game_date(request.game_date)
        nba_game_date = _validate_game_date(request.nba_game_date)
    except ValueError as exc:
        return {"ok": False, "steps": [], "message": str(exc)}
    snapshot_already_archived = _date_already_archived(snapshot_date)
    # A duplicate archive must prevent only the append-only history write. It must
    # not prevent refreshing current injuries, lineups, odds, and the slate itself.
    plan = [
        ("scripts.import_nba_status", ["--game-date", nba_game_date]),
        ("scripts.fetch_lineups", ["--game-date", nba_game_date]),
        ("scripts.fetch_public_props", []),
        ("src.predict_slate", ["--lines", "data/prop_lines_public.csv", "--top", "10"]),
    ]
    if not snapshot_already_archived:
        plan.append(("scripts.archive_predictions", ["--game-date", snapshot_date]))
    result = _run_steps(plan)
    if result["ok"]:
        if snapshot_already_archived:
            result["message"] = (
                f"Availability and lineup sources refreshed for NBA-local game date {nba_game_date}; "
                f"public markets scored. The snapshot dated {snapshot_date} already exists, so archive "
                "creation was skipped; no duplicate history row was added. "
                "Expected/projected lineups remain unverified and cannot pass the strict shortlist gate; "
                "check the row-level screening diagnostics before relying on any candidate."
            )
        else:
            result["message"] = (
                f"Availability and lineup sources refreshed for NBA-local game date {nba_game_date}; "
                f"public markets scored and pregame snapshot archived for snapshot date {snapshot_date}. "
                "Expected/projected lineups remain unverified and cannot pass the strict shortlist gate; "
                "check the row-level screening diagnostics before relying on any candidate."
            )
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
