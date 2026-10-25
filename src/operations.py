"""Operational health checks and auditable workflow records for PropIt."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_RELATIVE_PATH = Path("data/operations_audit.jsonl")
MAX_MARKET_AGE_HOURS = 2.0
MAX_AVAILABILITY_AGE_HOURS = 2.0
MAX_PREDICTION_FILE_AGE_HOURS = 24.0


def _utc_now(value: str | datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("now must be a valid timestamp")
    return parsed


def _file_age_hours(path: Path, now: pd.Timestamp) -> float | None:
    try:
        modified = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    except OSError:
        return None
    return max(0.0, (now - modified).total_seconds() / 3600.0)


def _csv_read(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, "missing"
    if path.stat().st_size == 0:
        return None, "empty"
    try:
        return pd.read_csv(path), None
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        return None, f"unreadable CSV: {exc}"


def _check(
    name: str,
    status: str,
    detail: str,
    *,
    path: Path | None = None,
    age_hours: float | None = None,
    rows: int | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "path": str(path) if path is not None else "",
        "age_hours": round(age_hours, 2) if age_hours is not None else None,
        "rows": rows,
    }


def _read_audit(path: Path, limit: int = 250) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    events.append({
                        "event_type": "audit_parse_error",
                        "timestamp_utc": "",
                        "ok": False,
                        "message": "An invalid JSON line was found in the local audit log.",
                    })
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        return [{"event_type": "audit_read_error", "ok": False, "message": "Could not read local audit log."}]
    return events[-limit:][::-1]


def read_audit_events(path: str | Path, limit: int = 40) -> list[dict[str, Any]]:
    """Public wrapper for retrieving recent audit entries, newest first."""
    if limit < 1:
        return []
    return _read_audit(Path(path), limit=limit)


def append_audit_event(event: dict[str, Any], path: Path | None = None) -> None:
    """Append one workflow result as JSONL. Never store environment variables or credentials."""
    destination = path or (DEFAULT_ROOT / AUDIT_RELATIVE_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _timestamp_age(frame: pd.DataFrame, candidates: tuple[str, ...], now: pd.Timestamp) -> tuple[float | None, int]:
    if frame.empty:
        return None, 0
    values = pd.Series(dtype="datetime64[ns, UTC]")
    for column in candidates:
        if column not in frame.columns:
            continue
        candidate_values = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
        if not candidate_values.empty:
            values = candidate_values
            break
    if values.empty:
        return None, 0
    age = (now - values.max()).total_seconds() / 3600.0
    if age < -5 / 60:
        return age, int(values.notna().sum())
    return max(0.0, age), int(values.notna().sum())


def _holdout_quality(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Summarize raw/calibrated probabilities from the existing chronological holdout only."""
    required = {"target", "actual_over", "p_over", "calibrated_p_over"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        return {
            "available": False,
            "sample_basis": "predictions/backtest_holdout.csv only",
            "by_target": [],
            "note": "No valid calibration holdout is available. This is not a live or in-sample performance metric.",
        }
    data = frame.copy()
    data["actual_over"] = pd.to_numeric(data["actual_over"], errors="coerce")
    data["p_over"] = pd.to_numeric(data["p_over"], errors="coerce")
    data["calibrated_p_over"] = pd.to_numeric(data["calibrated_p_over"], errors="coerce")
    valid = data["actual_over"].isin([0, 1]) & data["p_over"].notna() & data["calibrated_p_over"].notna()
    data = data.loc[valid].copy()
    if data.empty:
        return {
            "available": False,
            "sample_basis": "predictions/backtest_holdout.csv only",
            "by_target": [],
            "note": "No valid settled holdout rows. Probability metrics are withheld.",
        }
    output = []
    for target, group in data.groupby("target", sort=True):
        y = group["actual_over"].astype(int).to_numpy()
        entry: dict[str, Any] = {"target": str(target), "n": int(len(group))}
        for key, column in (("raw", "p_over"), ("calibrated", "calibrated_p_over")):
            p = group[column].clip(1e-6, 1 - 1e-6).to_numpy(dtype=float)
            brier = float(((p - y) ** 2).mean())
            log_loss = float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean())
            hit_rate = float(((p >= 0.5) == (y == 1)).mean())
            entry[f"{key}_brier"] = round(brier, 4)
            entry[f"{key}_log_loss"] = round(log_loss, 4)
            entry[f"{key}_hit_rate"] = round(hit_rate, 4)
        output.append(entry)
    return {
        "available": True,
        "sample_basis": "predictions/backtest_holdout.csv only",
        "by_target": output,
        "note": "Descriptive metrics from the saved chronological holdout. Lower Brier/log loss is better; these metrics do not guarantee future performance or profitability.",
    }


def build_health_report(root: str | Path = DEFAULT_ROOT, now: str | datetime | pd.Timestamp | None = None) -> dict[str, Any]:
    """Summarize local data readiness, freshness, history and recent workflow failures."""
    base = Path(root).resolve()
    current = _utc_now(now)
    checks: list[dict[str, Any]] = []
    data = base / "data"
    predictions = base / "predictions"
    artifacts = base / "artifacts"

    model_names = ("PTS", "REB", "AST")
    missing_models = [target for target in model_names if not (artifacts / f"model_{target}.joblib").is_file()]
    if missing_models:
        checks.append(_check("Model artifacts", "ERROR", "Missing model artifact(s): " + ", ".join(missing_models), path=artifacts))
    else:
        checks.append(_check("Model artifacts", "OK", "PTS, REB and AST model artifacts are present.", path=artifacts))

    latest_path = predictions / "latest.csv"
    latest, latest_error = _csv_read(latest_path)
    latest_age = _file_age_hours(latest_path, current) if latest_path.exists() else None
    if latest_error == "missing":
        checks.append(_check("Latest predictions", "WARN", "No prediction snapshot exists yet.", path=latest_path))
    elif latest_error:
        checks.append(_check("Latest predictions", "ERROR", latest_error, path=latest_path, age_hours=latest_age))
    elif latest is not None:
        required = {"player_id", "player_name", "target"}
        missing = sorted(required - set(latest.columns))
        if missing:
            checks.append(_check("Latest predictions", "ERROR", "Missing columns: " + ", ".join(missing), path=latest_path, age_hours=latest_age, rows=len(latest)))
        elif latest.empty:
            checks.append(_check("Latest predictions", "WARN", "Prediction file has no rows.", path=latest_path, age_hours=latest_age, rows=0))
        elif latest_age is not None and latest_age > MAX_PREDICTION_FILE_AGE_HOURS:
            checks.append(_check("Latest predictions", "WARN", f"Snapshot file is {latest_age:.1f} hours old.", path=latest_path, age_hours=latest_age, rows=len(latest)))
        else:
            checks.append(_check("Latest predictions", "OK", "Prediction file exists and has the required identifiers.", path=latest_path, age_hours=latest_age, rows=len(latest)))

    lines_path = data / "prop_lines_public.csv"
    lines, lines_error = _csv_read(lines_path)
    lines_age = _file_age_hours(lines_path, current) if lines_path.exists() else None
    if lines_error == "missing":
        checks.append(_check("Public market board", "WARN", "No imported market board yet.", path=lines_path))
    elif lines_error:
        checks.append(_check("Public market board", "ERROR", lines_error, path=lines_path, age_hours=lines_age))
    elif lines is not None:
        market_age, dated_rows = _timestamp_age(lines, ("market_snapshot_at_utc", "snapshot_at_utc", "captured_at_utc"), current)
        age = market_age if market_age is not None else lines_age
        if lines.empty:
            checks.append(_check("Public market board", "WARN", "Market file is empty.", path=lines_path, age_hours=age, rows=0))
        elif market_age is None:
            checks.append(_check("Public market board", "WARN", "Could not find a valid market capture timestamp; refresh and inspect source data.", path=lines_path, age_hours=lines_age, rows=len(lines)))
        elif market_age is not None and market_age < -5 / 60:
            checks.append(_check("Public market board", "WARN", f"Market capture timestamp is {-market_age:.1f} hours in the future; check local clock or source data.", path=lines_path, age_hours=market_age, rows=len(lines)))
        elif market_age > MAX_MARKET_AGE_HOURS:
            checks.append(_check("Public market board", "WARN", f"Latest captured market quote is {market_age:.1f} hours old; limit is {MAX_MARKET_AGE_HOURS:.0f} hours.", path=lines_path, age_hours=market_age, rows=len(lines)))
        else:
            checks.append(_check("Public market board", "OK", f"Latest market capture is within the {MAX_MARKET_AGE_HOURS:.0f}-hour freshness window ({dated_rows} dated rows).", path=lines_path, age_hours=market_age, rows=len(lines)))

    shortlist_path = predictions / "shortlist.csv"
    shortlist, short_error = _csv_read(shortlist_path)
    short_age = _file_age_hours(shortlist_path, current) if shortlist_path.exists() else None
    if short_error == "missing":
        checks.append(_check("Shortlist", "WARN", "No scored shortlist exists yet.", path=shortlist_path))
    elif short_error:
        checks.append(_check("Shortlist", "ERROR", short_error, path=shortlist_path, age_hours=short_age))
    elif shortlist is not None:
        detail = f"{len(shortlist)} candidates passed current research screens." if len(shortlist) else "No candidates passed current research screens; this is a valid outcome."
        status = "OK" if short_age is None or short_age <= MAX_PREDICTION_FILE_AGE_HOURS else "WARN"
        if status == "WARN":
            detail = f"Shortlist file is {short_age:.1f} hours old. " + detail
        checks.append(_check("Shortlist", status, detail, path=shortlist_path, age_hours=short_age, rows=len(shortlist)))

    history_path = data / "prediction_history.csv"
    history, history_error = _csv_read(history_path)
    history_age = _file_age_hours(history_path, current) if history_path.exists() else None
    settlement_counts = {"settled": 0, "pending": 0, "push": 0}
    if history_error == "missing":
        checks.append(_check("Prediction history", "WARN", "No pregame snapshots archived yet.", path=history_path))
    elif history_error:
        checks.append(_check("Prediction history", "ERROR", history_error, path=history_path, age_hours=history_age))
    elif history is not None:
        statuses = history.get("settlement_status", pd.Series("PENDING", index=history.index)).fillna("PENDING").astype(str).str.upper()
        settlement_counts = {
            "settled": int(statuses.eq("SETTLED").sum()),
            "pending": int(statuses.isin(["", "PENDING", "UNSETTLED"]).sum()),
            "push": int(statuses.eq("PUSH").sum()),
        }
        checks.append(_check("Prediction history", "OK", f"{len(history)} archived rows; {settlement_counts['settled']} settled, {settlement_counts['pending']} pending, {settlement_counts['push']} pushes.", path=history_path, age_hours=history_age, rows=len(history)))

    availability_path = data / "player_availability.csv"
    availability, availability_error = _csv_read(availability_path)
    availability_age = _file_age_hours(availability_path, current) if availability_path.exists() else None
    if availability_error == "missing":
        checks.append(_check("Player availability", "WARN", "No manually verified player-status file. Reports remain unverified.", path=availability_path))
    elif availability_error:
        checks.append(_check("Player availability", "ERROR", availability_error, path=availability_path, age_hours=availability_age))
    elif availability is not None:
        row_age = None
        fresh_count = 0
        if "reported_at_utc" in availability.columns:
            stamps = pd.to_datetime(availability["reported_at_utc"], utc=True, errors="coerce").dropna()
            if not stamps.empty:
                row_age = (current - stamps.max()).total_seconds() / 3600.0
                age_values = (current - stamps).dt.total_seconds() / 3600.0
                fresh_count = int(((age_values >= -5 / 60) & (age_values <= MAX_AVAILABILITY_AGE_HOURS)).sum())
        if availability.empty:
            checks.append(_check("Player availability", "WARN", "Availability file has headers only; no statuses are verified.", path=availability_path, age_hours=availability_age, rows=0))
        elif row_age is None:
            checks.append(_check("Player availability", "WARN", "No valid reported_at_utc timestamps; statuses are not current.", path=availability_path, age_hours=availability_age, rows=len(availability)))
        elif row_age < -5 / 60:
            checks.append(_check("Player availability", "WARN", f"Newest report timestamp is {-row_age:.1f} hours in the future; check the local clock and source record.", path=availability_path, age_hours=row_age, rows=len(availability)))
        elif row_age > MAX_AVAILABILITY_AGE_HOURS:
            checks.append(_check("Player availability", "WARN", f"Newest report is {row_age:.1f} hours old; refresh after checking the official report.", path=availability_path, age_hours=row_age, rows=len(availability)))
        else:
            checks.append(_check("Player availability", "OK", f"{fresh_count}/{len(availability)} rows have timestamps within the {MAX_AVAILABILITY_AGE_HOURS:.0f}-hour window. Verify by player and match date.", path=availability_path, age_hours=row_age, rows=len(availability)))

    # Minutes distribution readiness is tracked separately from general model-file health.
    if latest is not None and not latest.empty:
        minute_cols = {"minutes_projection_mean", "minutes_projection_std", "minutes_sample_size"}
        if not minute_cols.issubset(latest.columns):
            checks.append(_check(
                "Minutes projections", "WARN",
                "Latest predictions predate the minutes-distribution fields; regenerate the slate.",
                path=latest_path, rows=len(latest),
            ))
        else:
            minute_mean = pd.to_numeric(latest["minutes_projection_mean"], errors="coerce")
            minute_std = pd.to_numeric(latest["minutes_projection_std"], errors="coerce")
            minute_n = pd.to_numeric(latest["minutes_sample_size"], errors="coerce")
            usable_minutes = minute_mean.notna() & minute_std.notna() & minute_n.ge(3)
            if "line" in latest.columns:
                line_rows = pd.to_numeric(latest["line"], errors="coerce").notna()
                count = int((usable_minutes & line_rows).sum())
                denominator = int(line_rows.sum())
            else:
                count, denominator = int(usable_minutes.sum()), len(latest)
            status = "OK" if denominator > 0 and count == denominator else "WARN"
            checks.append(_check(
                "Minutes projections", status,
                f"{count}/{denominator} scored rows have a usable minutes distribution (at least 3 historical minute rows)."
                if denominator else "No actual prop lines were scored.",
                path=latest_path, rows=count,
            ))

    lineup_path = data / "player_lineups.csv"
    lineups, lineups_error = _csv_read(lineup_path)
    if lineups_error == "missing":
        checks.append(_check("Starter/bench snapshots", "WARN", "No verified player_lineups.csv snapshot. Shortlist picks require fresh explicit STARTER/BENCH statuses.", path=lineup_path))
    elif lineups_error:
        checks.append(_check("Starter/bench snapshots", "ERROR", lineups_error, path=lineup_path))
    elif lineups is not None:
        lineup_age, dated_lineup_rows = _timestamp_age(lineups, ("reported_at_utc",), current)
        statuses = lineups.get("starting_status", pd.Series("", index=lineups.index)).fillna("").astype(str).str.upper()
        valid_status = statuses.isin(["STARTER", "BENCH"])
        fresh_status_count = 0
        if "reported_at_utc" in lineups.columns:
            stamps = pd.to_datetime(lineups["reported_at_utc"], utc=True, errors="coerce")
            fresh = stamps.notna() & ((current - stamps).dt.total_seconds() >= -(5 / 60) * 3600) & ((current - stamps).dt.total_seconds() <= MAX_AVAILABILITY_AGE_HOURS * 3600)
            fresh_status_count = int((fresh & valid_status).sum())
        if lineups.empty:
            status, detail = "WARN", "Lineup snapshot contains headers only."
        elif lineup_age is None:
            status, detail = "WARN", "No valid lineup report timestamp; starter/bench data is not current."
        elif lineup_age < -5 / 60:
            status, detail = "WARN", "Lineup timestamp is in the future; check the source time and system clock."
        elif lineup_age > MAX_AVAILABILITY_AGE_HOURS:
            status, detail = "WARN", f"Newest lineup snapshot is {lineup_age:.1f} hours old; limit is {MAX_AVAILABILITY_AGE_HOURS:.0f} hours."
        else:
            status = "OK" if fresh_status_count == len(lineups) and len(lineups) > 0 else "WARN"
            detail = f"{fresh_status_count}/{len(lineups)} rows have a fresh timestamp and explicit STARTER/BENCH status."
        checks.append(_check("Starter/bench snapshots", status, detail, path=lineup_path, age_hours=lineup_age, rows=len(lineups)))

    imports_path = data / "injury_report_imports.csv"
    imports, imports_error = _csv_read(imports_path)
    if imports_error == "missing":
        checks.append(_check("Official injury report import", "WARN", "No official NBA injury report has been imported yet.", path=imports_path))
    elif imports_error:
        checks.append(_check("Official injury report import", "ERROR", imports_error, path=imports_path))
    elif imports is not None:
        import_age, import_rows = _timestamp_age(imports, ("reported_at_utc",), current)
        if imports.empty:
            status, detail = "WARN", "Injury report import ledger is empty."
        elif import_age is None:
            status, detail = "WARN", "No valid official report publication timestamp was recorded."
        elif import_age < -5 / 60:
            status, detail = "WARN", "Official injury report publication time is in the future; check report metadata."
        elif import_age > MAX_AVAILABILITY_AGE_HOURS:
            status, detail = "WARN", f"Newest official report was published {import_age:.1f} hours ago; a newer PDF may be available."
        else:
            status, detail = "OK", f"Latest official report publication timestamp is within {MAX_AVAILABILITY_AGE_HOURS:.0f} hours."
        checks.append(_check("Official injury report import", status, detail, path=imports_path, age_hours=import_age, rows=import_rows))

    holdout_path = predictions / "backtest_holdout.csv"
    holdout, holdout_error = _csv_read(holdout_path)
    performance = _holdout_quality(holdout)
    if holdout_error == "missing":
        checks.append(_check("Calibration holdout", "WARN", "No out-of-sample holdout report yet."))
    elif holdout_error:
        checks.append(_check("Calibration holdout", "ERROR", holdout_error, path=holdout_path))
    elif holdout is not None:
        valid = 0
        if {"actual_over", "p_over", "calibrated_p_over"}.issubset(holdout.columns):
            actual = pd.to_numeric(holdout["actual_over"], errors="coerce")
            raw = pd.to_numeric(holdout["p_over"], errors="coerce")
            calibrated = pd.to_numeric(holdout["calibrated_p_over"], errors="coerce")
            valid = int((actual.isin([0, 1]) & raw.notna() & calibrated.notna()).sum())
        status = "OK" if valid > 0 else "WARN"
        detail = f"{valid} rows have valid settled outcomes and both raw/calibrated probabilities." if valid else "Holdout exists but has no valid scored outcomes."
        checks.append(_check("Calibration holdout", status, detail, path=holdout_path, rows=len(holdout)))

    market_history_path = data / "prop_market_history.csv"
    market_history, market_history_error = _csv_read(market_history_path)
    if market_history_error == "missing":
        checks.append(_check("Market movement history", "WARN", "No persisted market snapshots yet.", path=market_history_path))
    elif market_history_error:
        checks.append(_check("Market movement history", "ERROR", market_history_error, path=market_history_path))
    elif market_history is not None:
        checks.append(_check("Market movement history", "OK", f"{len(market_history)} market observations stored.", path=market_history_path, age_hours=_file_age_hours(market_history_path, current), rows=len(market_history)))

    audit_path = base / AUDIT_RELATIVE_PATH
    events = _read_audit(audit_path)
    cutoff = current - pd.Timedelta(hours=24)
    recent_failures = 0
    recent_events = 0
    for event in events:
        stamp = pd.to_datetime(event.get("timestamp_utc"), utc=True, errors="coerce")
        if pd.notna(stamp) and cutoff <= stamp <= current + pd.Timedelta(minutes=5):
            recent_events += 1
            if event.get("ok") is False:
                recent_failures += 1
    malformed_audit = any(event.get("event_type") in {"audit_parse_error", "audit_read_error"} for event in events)
    audit_status = "WARN" if recent_failures or malformed_audit or not audit_path.exists() else "OK"
    audit_detail = f"{recent_events} workflow events in the past 24 hours; {recent_failures} failed." if audit_path.exists() else "No workflow audit events yet; events are written after workflow actions."
    if malformed_audit:
        audit_detail += " The audit log also has a read/parse issue."
    checks.append(_check("Workflow audit", audit_status, audit_detail, path=audit_path, rows=len(events)))

    statuses = [check["status"] for check in checks]
    overall = "ERROR" if "ERROR" in statuses else "WARN" if "WARN" in statuses else "OK"
    return {
        "generated_at_utc": current.isoformat().replace("+00:00", "Z"),
        "overall": overall,
        "root": str(base),
        "checks": checks,
        "performance": performance,
        "summary": {
            "check_count": len(checks),
            "ok": statuses.count("OK"),
            "warning": statuses.count("WARN"),
            "errors": statuses.count("ERROR"),
            "history": settlement_counts,
            "recent_workflow_events_24h": recent_events,
            "recent_workflow_failures_24h": recent_failures,
        },
        "recent_audit": events[:30],
        "limitations": [
            "Local file age and capture timestamps cannot prove the upstream source is complete or correct.",
            "Availability checks cover only statuses manually entered into the local CSV; no injury feed is fetched automatically.",
            "This tool is for local research operations and does not submit bets.",
        ],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a concise, auditable daily operations report."""
    lines = [
        "# PropIt Operations Report",
        "",
        f"- Generated (UTC): {report.get('generated_at_utc', 'unknown')}",
        f"- Overall status: **{report.get('overall', 'UNKNOWN')}**",
        f"- Workspace: `{report.get('root', '')}`",
        "",
        "## Health checks",
        "",
        "| Check | Status | Details | Age (hours) | Rows |",
        "|---|---|---|---:|---:|",
    ]
    for check in report.get("checks", []):
        detail = str(check.get("detail", "")).replace("|", "\\|").replace("\n", " ")
        age = "—" if check.get("age_hours") is None else f"{check['age_hours']:.2f}"
        rows = "—" if check.get("rows") is None else str(check["rows"])
        lines.append(f"| {check.get('name', '')} | {check.get('status', '')} | {detail} | {age} | {rows} |")
    summary = report.get("summary", {})
    lines.extend([
        "",
        "## Holdout probability quality",
        "",
    ])
    performance = report.get("performance", {})
    quality_rows = performance.get("by_target", []) if performance.get("available") else []
    if quality_rows:
        lines.extend([
            "| Stat | N | Raw Brier | Calibrated Brier | Raw log loss | Calibrated log loss | Raw hit rate | Calibrated hit rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in quality_rows:
            lines.append(
                f"| {item.get('target', '')} | {item.get('n', 0)} | {item.get('raw_brier', '—')} | "
                f"{item.get('calibrated_brier', '—')} | {item.get('raw_log_loss', '—')} | "
                f"{item.get('calibrated_log_loss', '—')} | {item.get('raw_hit_rate', '—')} | "
                f"{item.get('calibrated_hit_rate', '—')} |"
            )
    else:
        lines.append("No valid chronological holdout metrics are available yet.")
    lines.extend([str(performance.get("note", "Metrics are unavailable.")), "", "## Activity and ledger", "",
        "",
        f"- Archived props: {summary.get('history', {}).get('settled', 0) + summary.get('history', {}).get('pending', 0) + summary.get('history', {}).get('push', 0)}",
        f"- Settled: {summary.get('history', {}).get('settled', 0)}; pending: {summary.get('history', {}).get('pending', 0)}; pushes: {summary.get('history', {}).get('push', 0)}",
        f"- Workflow events in the last 24 hours: {summary.get('recent_workflow_events_24h', 0)}",
        f"- Failed workflows in the last 24 hours: {summary.get('recent_workflow_failures_24h', 0)}",
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    lines.extend(["", "This report describes local file and workflow state. It is not a prediction of betting outcomes.", ""])
    return "\n".join(lines)
