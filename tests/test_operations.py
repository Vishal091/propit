from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.operations import append_audit_event, build_health_report, render_markdown_report
from scripts.backup_workspace import create_backup, collect_backup_files


def test_health_report_flags_missing_models_as_error(tmp_path):
    report = build_health_report(tmp_path, now="2026-10-24T12:00:00Z")
    assert report["overall"] == "ERROR"
    model = next(check for check in report["checks"] if check["name"] == "Model artifacts")
    assert model["status"] == "ERROR"
    assert "PTS" in model["detail"]


def test_health_report_checks_market_capture_timestamp(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "predictions").mkdir()
    (tmp_path / "artifacts").mkdir()
    for target in ("PTS", "REB", "AST"):
        (tmp_path / "artifacts" / f"model_{target}.joblib").write_bytes(b"placeholder")
    (tmp_path / "data" / "prop_lines_public.csv").write_text(
        "player_id,target,line,market_snapshot_at_utc\n12,PTS,20.5,2026-10-24T11:00:00Z\n",
        encoding="utf-8",
    )
    report = build_health_report(tmp_path, now="2026-10-24T12:00:00Z")
    market = next(check for check in report["checks"] if check["name"] == "Public market board")
    assert market["status"] == "OK"
    assert market["age_hours"] == 1.0
    assert market["rows"] == 1


def test_health_report_records_recent_failed_workflow(tmp_path):
    now = pd.Timestamp.now(tz="UTC").isoformat()
    audit_path = tmp_path / "data" / "operations_audit.jsonl"
    append_audit_event({
        "event_type": "workflow",
        "timestamp_utc": now,
        "ok": False,
        "steps": [{"step": "src.predict_slate", "ok": False}],
    }, audit_path)
    report = build_health_report(tmp_path)
    assert report["summary"]["recent_workflow_events_24h"] == 1
    assert report["summary"]["recent_workflow_failures_24h"] == 1
    audit = next(check for check in report["checks"] if check["name"] == "Workflow audit")
    assert audit["status"] == "WARN"


def test_markdown_report_contains_health_and_activity(tmp_path):
    report = build_health_report(tmp_path, now="2026-10-24T12:00:00Z")
    rendered = render_markdown_report(report)
    assert "# PropIt Operations Report" in rendered
    assert "## Health checks" in rendered
    assert "## Holdout probability quality" in rendered
    assert "Failed workflows in the last 24 hours" in rendered


def test_backup_uses_allowlist_and_manifest_checksums(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "predictions").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "src" / "dashboard.py").write_text("print('safe')\n", encoding="utf-8")
    (tmp_path / "data" / "prediction_history.csv").write_text("id,value\n1,2\n", encoding="utf-8")
    (tmp_path / "artifacts" / "model_PTS.joblib").write_bytes(b"model")
    (tmp_path / ".env").write_text("SECRET=never-backup-this", encoding="utf-8")
    (tmp_path / ".cache" / "ignored.bin").write_bytes(b"cache")
    (tmp_path / ".venv" / "ignored.py").write_text("ignored", encoding="utf-8")

    selected = collect_backup_files(tmp_path)
    assert (tmp_path / "src" / "dashboard.py") in selected
    assert (tmp_path / "data" / "prediction_history.csv") in selected
    assert all(path.name != ".env" for path in selected)
    assert all(".cache" not in path.parts and ".venv" not in path.parts for path in selected)

    result = create_backup(tmp_path / "backups" / "test.zip", root=tmp_path)
    with ZipFile(result["path"]) as archive:
        names = set(archive.namelist())
        assert "src/dashboard.py" in names
        assert "data/prediction_history.csv" in names
        assert "artifacts/model_PTS.joblib" in names
        assert "manifest.json" in names
        assert ".env" not in names
        assert ".cache/ignored.bin" not in names
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["files_count"] == result["files_count"]
        assert any(item["path"] == "data/prediction_history.csv" and item["sha256"] for item in manifest["files"])



def test_holdout_quality_reports_raw_and_calibrated_metrics_by_stat(tmp_path):
    from src.operations import _holdout_quality

    frame = pd.DataFrame({
        "target": ["PTS", "PTS", "REB", "REB"],
        "actual_over": [1, 0, 1, 0],
        "p_over": [0.8, 0.3, 0.55, 0.7],
        "calibrated_p_over": [0.75, 0.25, 0.6, 0.4],
    })
    result = _holdout_quality(frame)
    assert result["available"] is True
    points = next(row for row in result["by_target"] if row["target"] == "PTS")
    assert points["n"] == 2
    assert points["raw_brier"] == 0.065
    assert points["calibrated_brier"] == 0.0625
    assert points["raw_hit_rate"] == 1.0
    assert "chronological holdout" in result["note"]


def test_holdout_quality_withholds_metrics_without_valid_outcomes():
    from src.operations import _holdout_quality

    frame = pd.DataFrame({
        "target": ["PTS"],
        "actual_over": [None],
        "p_over": [0.7],
        "calibrated_p_over": [0.65],
    })
    result = _holdout_quality(frame)
    assert result["available"] is False
    assert result["by_target"] == []



def test_future_market_timestamp_is_not_reported_as_fresh(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "predictions").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "data" / "prop_lines_public.csv").write_text(
        "player_id,target,line,market_snapshot_at_utc,snapshot_at_utc\n"
        "12,PTS,20.5,,2026-10-24T12:30:00Z\n",
        encoding="utf-8",
    )
    report = build_health_report(tmp_path, now="2026-10-24T12:00:00Z")
    market = next(check for check in report["checks"] if check["name"] == "Public market board")
    assert market["status"] == "WARN"
    assert "future" in market["detail"].lower()
    assert market["age_hours"] == -0.5


def test_health_report_includes_minutes_lineup_and_official_report_readiness(tmp_path):
    data = tmp_path / "data"
    predictions = tmp_path / "predictions"
    artifacts = tmp_path / "artifacts"
    data.mkdir()
    predictions.mkdir()
    artifacts.mkdir()
    for target in ("PTS", "REB", "AST"):
        (artifacts / f"model_{target}.joblib").write_bytes(b"placeholder")
    (predictions / "latest.csv").write_text(
        "player_id,player_name,target,line,minutes_projection_mean,minutes_projection_std,minutes_sample_size\n"
        "1,Player One,PTS,20.5,30,3.5,18\n",
        encoding="utf-8",
    )
    (data / "player_lineups.csv").write_text(
        "player_id,player_name,team_abbreviation,match_date,starting_status,reported_at_utc,source\n"
        "1,Player One,BOS,2026-10-25,STARTER,2026-10-25T11:30:00Z,official source\n",
        encoding="utf-8",
    )
    (data / "injury_report_imports.csv").write_text(
        "game_date,reported_at_utc,retrieved_at_utc,rows_parsed\n"
        "2026-10-25,2026-10-25T11:30:00Z,2026-10-25T11:35:00Z,10\n",
        encoding="utf-8",
    )
    report = build_health_report(tmp_path, now="2026-10-25T12:00:00Z")
    by_name = {check["name"]: check for check in report["checks"]}
    assert by_name["Minutes projections"]["status"] == "OK"
    assert by_name["Starter/bench snapshots"]["status"] == "OK"
    assert by_name["Official injury report import"]["status"] == "OK"
