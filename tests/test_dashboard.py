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


def test_research_slate_view_returns_ev_sorted_unverified_rows(tmp_path, monkeypatch):
    from src import dashboard

    latest_path = tmp_path / "latest.csv"
    shortlist_path = tmp_path / "shortlist.csv"
    pd.DataFrame({
        "player_name": ["Lower EV", "Higher EV", "No estimate"],
        "research_status": [
            "RESEARCH_ONLY_UNVERIFIED", "RESEARCH_ONLY_UNVERIFIED", "RESEARCH_INPUTS_MISSING",
        ],
        "research_selected_ev": [0.02, 0.06, None],
        "research_selected_edge": [0.01, 0.03, None],
        "research_model_p_over": [0.54, 0.58, None],
        "research_selected_probability": [0.54, 0.58, None],
        "research_market_probability": [0.53, 0.55, None],
        "research_recommended_side": ["OVER", "UNDER", ""],
        "shortlist_status": ["LINEUP_UNVERIFIED", "AVAILABILITY_UNVERIFIED", "INSUFFICIENT_DATA"],
    }).to_csv(latest_path, index=False)
    pd.DataFrame(columns=["player_name", "shortlist_status"]).to_csv(shortlist_path, index=False)
    monkeypatch.setattr(dashboard, "LATEST", latest_path)
    monkeypatch.setattr(dashboard, "SHORTLIST", shortlist_path)

    result = dashboard.api_slate(view="research")

    assert result["count"] == 3
    assert result["rows"][0]["player_name"] == "Higher EV"
    assert result["rows"][1]["player_name"] == "Lower EV"
    assert result["diagnostics"]["rows_with_research_estimates"] == 2
    assert "never satisfy strict shortlist gates" in result["diagnostics"]["notice"]


def test_empty_shortlist_returns_reason_counts_from_scored_markets(tmp_path, monkeypatch):
    from src import dashboard

    latest_path = tmp_path / "latest.csv"
    shortlist_path = tmp_path / "shortlist.csv"
    pd.DataFrame({
        "player_name": ["Player A", "Player B", "Player C"],
        "shortlist_status": ["LINEUP_UNVERIFIED", "LINEUP_UNVERIFIED", "AVAILABILITY_REVIEW"],
        "shortlist_reason": [
            "Strict mode requires a fresh confirmed lineup.",
            "Strict mode requires a fresh confirmed lineup.",
            "Player is questionable; verify near tip-off.",
        ],
    }).to_csv(latest_path, index=False)
    pd.DataFrame(columns=["player_name", "shortlist_status"]).to_csv(shortlist_path, index=False)
    monkeypatch.setattr(dashboard, "LATEST", latest_path)
    monkeypatch.setattr(dashboard, "SHORTLIST", shortlist_path)

    result = dashboard.api_slate(view="shortlist")

    assert result["count"] == 0
    assert result["rows"] == []
    assert result["diagnostics"]["scored_rows"] == 3
    assert result["diagnostics"]["status_counts"][0] == {
        "status": "LINEUP_UNVERIFIED",
        "count": 2,
        "reason": "Strict mode requires a fresh confirmed lineup.",
    }
    assert result["diagnostics"]["status_counts"][1]["status"] == "AVAILABILITY_REVIEW"


def test_official_injury_report_dashboard_action_rejects_invalid_date(monkeypatch):
    from src import dashboard

    called = []
    monkeypatch.setattr(dashboard, "_run_steps", lambda plan: called.append(plan) or {"ok": True})
    result = dashboard.api_injury_report(dashboard.GameDateRequest(game_date="2026-02-30"))
    assert result["ok"] is False
    assert called == []



def test_pregame_workflow_refreshes_injury_and_lineup_sources_for_nba_date(monkeypatch):
    from src import dashboard

    captured = {}
    monkeypatch.setattr(dashboard, "_date_already_archived", lambda _: False)
    def fake_steps(plan):
        captured["plan"] = plan
        return {"ok": True, "steps": [], "message": "All requested steps completed."}
    monkeypatch.setattr(dashboard, "_run_steps", fake_steps)

    request = dashboard.PregameWorkflowRequest(
        game_date="2026-10-26",
        nba_game_date="2026-10-25",
    )
    result = dashboard.api_pregame_workflow(request)

    assert result["ok"] is True
    assert captured["plan"] == [
        ("scripts.import_nba_status", ["--game-date", "2026-10-25"]),
        ("scripts.fetch_lineups", ["--game-date", "2026-10-25"]),
        ("scripts.fetch_public_props", []),
        ("src.predict_slate", ["--lines", "data/prop_lines_public.csv", "--top", "10"]),
        ("scripts.archive_predictions", ["--game-date", "2026-10-26"]),
    ]
    assert "NBA-local game date 2026-10-25" in result["message"]
    assert "snapshot date 2026-10-26" in result["message"]


def test_pregame_workflow_rejects_impossible_nba_date_before_running_steps(monkeypatch):
    from src import dashboard

    called = []
    monkeypatch.setattr(dashboard, "_run_steps", lambda plan: called.append(plan) or {"ok": True})
    monkeypatch.setattr(dashboard, "_date_already_archived", lambda _: False)
    request = dashboard.PregameWorkflowRequest(
        game_date="2026-10-26",
        nba_game_date="2026-02-30",
    )
    result = dashboard.api_pregame_workflow(request)

    assert result["ok"] is False
    assert "real date" in result["message"]
    assert called == []


def test_pregame_workflow_html_explains_separate_event_and_snapshot_dates():
    from src import dashboard

    html = dashboard.HTML_FILE.read_text(encoding="utf-8")
    assert 'id="injury-game-date"' in html
    assert 'id="lineup-game-date"' in html
    assert "Set the injury-report and lineup dates above to the same NBA-local date" in html
    assert "nba_game_date:injuryDate" in html



def test_pregame_workflow_refreshes_slate_when_snapshot_date_is_already_archived(monkeypatch):
    from src import dashboard

    captured = {}
    monkeypatch.setattr(dashboard, "_date_already_archived", lambda value: value == "2026-10-25")

    def fake_steps(plan):
        captured["plan"] = plan
        return {"ok": True, "steps": [], "message": "All requested steps completed."}

    monkeypatch.setattr(dashboard, "_run_steps", fake_steps)
    request = dashboard.PregameWorkflowRequest(
        game_date="2026-10-25",
        nba_game_date="2026-10-25",
    )

    result = dashboard.api_pregame_workflow(request)

    assert result["ok"] is True
    assert captured["plan"] == [
        ("scripts.import_nba_status", ["--game-date", "2026-10-25"]),
        ("scripts.fetch_lineups", ["--game-date", "2026-10-25"]),
        ("scripts.fetch_public_props", []),
        ("src.predict_slate", ["--lines", "data/prop_lines_public.csv", "--top", "10"]),
    ]
    assert not any(step[0] == "scripts.archive_predictions" for step in captured["plan"])
    assert "public markets scored" in result["message"]
    assert "no duplicate history row was added" in result["message"]



def _pending_confirmation_row(**overrides):
    row = {
        "player_id": 11,
        "player_name": "Test Player",
        "team_abbreviation": "BOS",
        "opponent_abbreviation": "NYK",
        "opponent_team": "New York Knicks",
        "match_date": "2026-10-25",
        "target": "PTS",
        "line": 20.5,
        "bookmaker": "Flash Odds public board (over Underdog; under Underdog)",
        "line_source": "public_web_import",
        "market_snapshot_at_utc": "2026-10-25T14:12:31Z",
        "market_age_hours": 0.3,
        "data_age_days": 2,
        "minutes_projection_mean": 32.0,
        "minutes_projection_std": 5.0,
        "minutes_sample_size": 22,
        "probability_gate_status": "INSUFFICIENT_LINEUP_DATA",
        "market_prior_snapshot_count": 2,
        "line_move_from_open": 0.0,
        "line_move_from_previous": 0.0,
        "movement_explanation": "",
        "edge_explanation": "Rotation role changed after injury news.",
        "market_no_vig_p_over": 0.50,
        "market_no_vig_p_under": 0.50,
        "market_no_vig_method": "same_book_pair",
        "odds_over_american": -110,
        "odds_under_american": -110,
        "research_status": "RESEARCH_ESTIMATE_AVAILABLE",
        "research_recommended_side": "OVER",
        "research_selected_probability": 0.56,
        "research_market_probability": 0.50,
        "research_selected_edge": 0.06,
        "research_selected_ev": 0.07,
        "shortlist_status": "INSUFFICIENT_LINEUP_DATA",
        "shortlist_reason": "No fresh confirmed STARTER/BENCH snapshot matched this player and scheduled game.",
        "lineup_status": "UNKNOWN",
        "lineup_observed_status": "STARTER",
        "lineup_gate": "UNVERIFIED",
        "lineup_confirmation_status": "PROJECTED",
        "lineup_confidence_label": "PROJECTED",
        "lineup_reason": "Projected lineup only.",
        "lineup_source": "https://www.rotowire.com/basketball/nba-lineups.php",
        "lineup_source_status": "Expected Lineup",
        "lineup_reported_at_utc": "2026-10-25T14:12:00Z",
        "lineup_age_hours": 0.3,
        "availability_status": "AVAILABLE",
        "availability_gate": "CLEAR",
        "availability_reason": "Fresh official report checked.",
        "availability_age_hours": 0.2,
        "availability_source": "Official NBA injury report",
    }
    row.update(overrides)
    return row


def test_pending_confirmation_only_surfaces_positive_fresh_evidence_blocked_research_signals(tmp_path, monkeypatch):
    from src import dashboard

    latest_path = tmp_path / "latest.csv"
    pd.DataFrame([
        _pending_confirmation_row(),
        _pending_confirmation_row(
            player_id=12, player_name="Negative EV",
            research_selected_ev=-0.01,
        ),
        _pending_confirmation_row(
            player_id=13, player_name="Stale market",
            market_age_hours=2.1,
        ),
        _pending_confirmation_row(
            player_id=14, player_name="Explicitly out",
            shortlist_status="AVAILABILITY_BLOCKED",
        ),
        _pending_confirmation_row(
            player_id=15, player_name="No evidence blocker",
            shortlist_status="BELOW_THRESHOLDS",
        ),
        _pending_confirmation_row(
            player_id=16, player_name="Low research probability",
            research_selected_probability=0.51,
        ),
    ]).to_csv(latest_path, index=False)
    monkeypatch.setattr(dashboard, "LATEST", latest_path)

    result = dashboard.api_pending_confirmation(limit=50)

    assert result["count"] == 1
    assert result["rows"][0]["player_name"] == "Test Player"
    assert result["rows"][0]["pending_rank"] == 1
    assert result["diagnostics"]["scored_rows"] == 6
    assert result["diagnostics"]["evidence_blocked_rows"] == 4
    assert result["diagnostics"]["positive_research_signal_rows"] == 2
    assert result["diagnostics"]["pending_rows"] == 1
    assert "not qualified picks" in result["message"].lower()


def test_pending_confirmation_requires_market_movement_evidence_and_excludes_placeholder_sources(tmp_path, monkeypatch):
    from src import dashboard

    latest_path = tmp_path / "latest.csv"
    pd.DataFrame([
        _pending_confirmation_row(
            player_id=20, player_name="No price history",
            market_prior_snapshot_count=0,
        ),
        _pending_confirmation_row(
            player_id=21, player_name="Placeholder source",
            bookmaker="Example sportsbook",
        ),
        _pending_confirmation_row(
            player_id=22, player_name="Adverse movement",
            line_move_from_open=1.0,
            movement_explanation="",
        ),
    ]).to_csv(latest_path, index=False)
    monkeypatch.setattr(dashboard, "LATEST", latest_path)

    result = dashboard.api_pending_confirmation(limit=50)

    assert result["count"] == 0
    assert result["diagnostics"]["evidence_blocked_rows"] == 3


def test_pending_confirmation_dashboard_has_separate_research_only_panel():
    from src import dashboard

    html = dashboard.HTML_FILE.read_text(encoding="utf-8")
    assert 'id="pending-confirmation-panel"' in html
    assert 'id="pending-confirmation-table"' in html
    assert 'request("/api/pending-confirmation?limit=50")' in html
    assert "These are leads to recheck, not qualified picks." in html
