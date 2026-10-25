from __future__ import annotations

import math

import pandas as pd
import pytest

from src.dashboard import _date_already_archived, _records, _validate_game_date


def test_validate_game_date_accepts_real_iso_date():
    assert _validate_game_date("2026-10-24") == "2026-10-24"


@pytest.mark.parametrize("value", ["", "2026-13-01", "24-10-2026", "2026-02-30"])
def test_validate_game_date_rejects_invalid_date(value):
    with pytest.raises(ValueError):
        _validate_game_date(value)


def test_records_converts_pandas_missing_values_to_json_nulls():
    records = _records(pd.DataFrame({"player": ["A", "B"], "probability": [0.75, float("nan")], "count": [1, 2]}))
    assert records[0]["probability"] == pytest.approx(0.75)
    assert records[1]["probability"] is None
    assert records[0]["count"] == 1

def test_duplicate_snapshot_check_uses_snapshot_date_not_match_date(tmp_path, monkeypatch):
    history_path = tmp_path / "prediction_history.csv"
    pd.DataFrame({
        "snapshot_date": ["2026-10-24", "2026-10-24"],
        "game_date": ["2026-10-25", "2026-10-26"],
        "match_date": ["2026-10-25", "2026-10-26"],
    }).to_csv(history_path, index=False)
    monkeypatch.setattr("src.dashboard.HISTORY", history_path)

    assert _date_already_archived("2026-10-24") is True
    assert _date_already_archived("2026-10-25") is False




def test_calibration_api_returns_log_loss_reliability_and_timestamped_price_counts(tmp_path, monkeypatch):
    from src.dashboard import api_calibration

    holdout_path = tmp_path / "backtest_holdout.csv"
    pd.DataFrame({
        "game_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "player_name": ["A", "B", "C", "D"],
        "target": ["PTS", "PTS", "REB", "REB"],
        "line": [20.5, 15.5, 7.5, 5.5],
        "p_over": [0.8, 0.2, 0.7, 0.3],
        "calibrated_p_over": [0.75, 0.25, 0.65, 0.35],
        "actual_over": [1, 0, 1, 0],
        "actual_stat": [22, 14, 8, 5],
        "odds_over_american": [-110, -110, -110, -110],
        "odds_under_american": [-110, -110, -110, -110],
        "snapshot_at_utc": [
            "2026-01-01T16:00:00Z", "2026-01-02T16:00:00Z",
            "2026-01-03T16:00:00Z", "2026-01-04T16:00:00Z"
        ],
    }).to_csv(holdout_path, index=False)
    monkeypatch.setattr("src.dashboard.HOLDOUT", holdout_path)

    result = api_calibration()

    assert result["available"] is True
    assert {row["target"] for row in result["summary"]} == {"PTS", "REB"}
    assert all("raw_log_loss" in row and "calibrated_log_loss" in row for row in result["summary"])
    assert all("calibrated_roi" in row and row["timestamped_price_rows"] == 2 for row in result["summary"])
    assert result["reliability"]
    assert all("bin_center" in row and "calibration_gap" in row for row in result["reliability"])



def test_workflow_runner_writes_audit_without_capturing_secrets(tmp_path, monkeypatch):
    import json
    from src import dashboard

    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_run_step", lambda module, args=None: {
        "step": module,
        "args": args or [],
        "ok": False,
        "returncode": 2,
        "output": "a sample failure message",
        "duration_seconds": 0.25,
        "started_at_utc": "2026-10-24T12:00:00Z",
    })
    result = dashboard._run_steps([("src.predict_slate", ["--top", "10"]), ("scripts.archive_predictions", [])])
    assert result["ok"] is False
    assert result["audit_recorded"] is True
    assert len(result["steps"]) == 1

    lines = (tmp_path / "data" / "operations_audit.jsonl").read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[-1])
    assert event["event_type"] == "workflow"
    assert event["ok"] is False
    assert event["steps"][0]["step"] == "src.predict_slate"
    assert event["steps"][0]["error_excerpt"] == "a sample failure message"
    assert "environment" not in json.dumps(event).lower()


def test_dashboard_health_endpoint_returns_checklist(tmp_path, monkeypatch):
    from src import dashboard

    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    result = dashboard.api_health()
    assert result["overall"] == "ERROR"
    assert result["checks"]
    assert any(check["name"] == "Model artifacts" for check in result["checks"])


def test_dashboard_audit_endpoint_reads_latest_workflow_events(tmp_path, monkeypatch):
    import json
    from src import dashboard

    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    path = tmp_path / "data" / "operations_audit.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "event_type": "workflow",
        "timestamp_utc": "2026-10-24T12:00:00Z",
        "ok": True,
        "steps": [],
    }) + "\n", encoding="utf-8")

    result = dashboard.api_audit(limit=5)
    assert result["count"] == 1
    assert result["rows"][0]["ok"] is True


def test_official_injury_report_dashboard_action_uses_selected_nba_date(monkeypatch):
    from src import dashboard

    captured = {}
    monkeypatch.setattr(
        dashboard, "_run_steps",
        lambda plan: captured.update({"plan": plan}) or {"ok": True, "steps": plan},
    )
    result = dashboard.api_injury_report(dashboard.GameDateRequest(game_date="2026-10-25"))
    assert result["ok"] is True
    assert captured["plan"] == [("scripts.import_nba_status", ["--game-date", "2026-10-25"])]


def test_automated_lineup_dashboard_action_collects_then_rebuilds_predictions(monkeypatch):
    from src import dashboard

    captured = {}
    monkeypatch.setattr(
        dashboard, "_run_steps",
        lambda plan: captured.update({"plan": plan}) or {"ok": True, "steps": plan},
    )
    result = dashboard.api_lineups(dashboard.GameDateRequest(game_date="2026-10-25"))

    assert result["ok"] is True
    assert captured["plan"] == [
        ("scripts.fetch_lineups", ["--game-date", "2026-10-25"]),
        ("src.predict_slate", ["--lines", "data/prop_lines_public.csv", "--top", "10"]),
    ]


def test_automated_lineup_dashboard_action_rejects_invalid_date(monkeypatch):
    from src import dashboard

    called = []
    monkeypatch.setattr(dashboard, "_run_steps", lambda plan: called.append(plan) or {"ok": True})
    result = dashboard.api_lineups(dashboard.GameDateRequest(game_date="2026-02-30"))

    assert result["ok"] is False
    assert called == []


def test_official_injury_report_dashboard_action_rejects_invalid_date(monkeypatch):
    from src import dashboard

    called = []
    monkeypatch.setattr(dashboard, "_run_steps", lambda plan: called.append(plan) or {"ok": True})
    result = dashboard.api_injury_report(dashboard.GameDateRequest(game_date="2026-02-30"))
    assert result["ok"] is False
    assert called == []
