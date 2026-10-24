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
