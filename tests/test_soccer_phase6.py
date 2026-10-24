from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src import dashboard
from src.soccer.odds import OddsProviderError


def patch_odds_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "SOCCER_DATA", tmp_path / "data" / "soccer")
    monkeypatch.setattr(dashboard, "SOCCER_ODDS", tmp_path / "data" / "soccer" / "odds_snapshots.csv")


def test_historical_odds_import_requires_explicit_cost_confirmation(monkeypatch):
    called = {"count": 0}
    def fail_if_called(plan):
        called["count"] += 1
        return {"ok": True}
    monkeypatch.setattr(dashboard, "_run_steps", fail_if_called)
    request = dashboard.SoccerOddsImportRequest(
        mode="historical", league="all", historical_date="2026-10-23T14:00:00Z",
        confirm_historical_cost=False,
    )
    result = dashboard.api_soccer_odds_import(request)
    assert result["ok"] is False
    assert "Confirm" in result["message"]
    assert called["count"] == 0


def test_historical_odds_import_uses_allowlisted_cli_and_limits_regions_and_market(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        dashboard, "_soccer_odds_state",
        lambda: {"available": False, "rows": 0, "api_key_configured": False},
    )
    def fake_steps(plan):
        captured["plan"] = plan
        return {"ok": True, "steps": [], "message": "imported"}
    monkeypatch.setattr(dashboard, "_run_steps", fake_steps)
    request = dashboard.SoccerOddsImportRequest(
        mode="historical", league="all", historical_date="2026-10-23T14:00:00Z",
        confirm_historical_cost=True,
    )
    result = dashboard.api_soccer_odds_import(request)
    assert result["ok"] is True
    assert captured["plan"] == [(
        "scripts.import_soccer_odds",
        [
            "--league", "all", "--output", "data/soccer/odds_snapshots.csv",
            "--regions", "uk", "--markets", "h2h",
            "--historical-date", "2026-10-23T14:00:00Z",
        ],
    )]
    assert "50 credits" in result["historical_cost_note"]


def test_historical_odds_import_rejects_naive_or_future_timestamps(monkeypatch):
    called = {"count": 0}
    monkeypatch.setattr(dashboard, "_run_steps", lambda plan: called.update(count=called["count"] + 1))
    monkeypatch.setattr(dashboard, "_soccer_odds_state", lambda: {"rows": 0})
    naive = dashboard.SoccerOddsImportRequest(
        mode="historical", historical_date="2026-10-23T14:00:00",
        confirm_historical_cost=True,
    )
    assert dashboard.api_soccer_odds_import(naive)["ok"] is False
    future = datetime.now(timezone.utc).replace(microsecond=0)
    from datetime import timedelta
    future = future + timedelta(days=1)
    request = dashboard.SoccerOddsImportRequest(
        mode="historical", historical_date=future.isoformat().replace("+00:00", "Z"),
        confirm_historical_cost=True,
    )
    assert dashboard.api_soccer_odds_import(request)["ok"] is False
    assert called["count"] == 0


def test_odds_status_never_returns_the_secret_key(monkeypatch, tmp_path):
    patch_odds_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(dashboard, "get_api_key", lambda: "private-secret-key")
    result = dashboard.api_soccer_odds_status()
    assert result["api_key_configured"] is True
    assert "private-secret-key" not in repr(result)
    monkeypatch.setattr(dashboard, "get_api_key", lambda: (_ for _ in ()).throw(OddsProviderError("not configured")))
    assert dashboard.api_soccer_odds_status()["api_key_configured"] is False


def test_dashboard_exposes_market_import_controls_and_segmented_backtest_view():
    html = dashboard.HTML_FILE.read_text(encoding="utf-8")
    for value in (
        'id="soccer-odds-live-btn"',
        'id="soccer-odds-historical-btn"',
        'id="soccer-confirm-historical"',
        'id="soccer-segment-metrics-table"',
        'async function loadSoccerOddsStatus',
        'async function runSoccerOddsImport',
        'Market no-vig P',
        'EV / unit',
    ):
        assert value in html
