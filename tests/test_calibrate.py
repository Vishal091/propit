from __future__ import annotations

import pandas as pd
import pytest

from scripts.calibrate import _metrics, prepare_rows


def test_prepare_rows_excludes_pushes_and_missing_actuals(tmp_path):
    path = tmp_path / "settled.csv"
    pd.DataFrame({
        "game_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "target": ["REB", "REB", "REB"],
        "p_over": [0.6, 0.7, 0.8],
        "actual_stat": [8, 7, None],
        "line": [7.5, 7.0, 7.5],
    }).to_csv(path, index=False)

    rows = prepare_rows(path)
    assert len(rows) == 1
    assert rows.iloc[0]["actual_over"] == 1


def test_metrics_hit_rate_and_brier():
    frame = pd.DataFrame({
        "p_over": [0.8, 0.2],
        "actual_over": [1, 0],
        "odds_over_american": [-110, -110],
        "odds_under_american": [-110, -110],
    })
    metrics = _metrics(frame, "p_over")
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.04)
    assert metrics["log_loss"] > 0
