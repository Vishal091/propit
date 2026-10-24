from __future__ import annotations

import pandas as pd
import pytest

from scripts.calibrate import _metrics, prepare_rows, reliability


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



def test_roi_is_not_simulated_from_untimestamped_prices():
    frame = pd.DataFrame({
        "p_over": [0.8, 0.2],
        "actual_over": [1, 0],
        "odds_over_american": [-110, -110],
        "odds_under_american": [-110, -110],
    })
    metrics = _metrics(frame, "p_over")
    assert metrics["brier"] == pytest.approx(0.04)
    assert metrics["timestamped_rows"] == 0
    assert metrics["priced_rows"] == 0
    assert metrics["bets"] == 0
    assert pd.isna(metrics["roi"])


def test_roi_requires_a_timestamped_snapshot_and_both_prices():
    frame = pd.DataFrame({
        "p_over": [0.8, 0.2, 0.8],
        "actual_over": [1, 0, 1],
        "odds_over_american": [-110, -110, -110],
        "odds_under_american": [-110, -110, None],
        "snapshot_at_utc": ["2026-01-01T17:00:00Z", "2026-01-02T17:00:00Z", "2026-01-03T17:00:00Z"],
    })
    metrics = _metrics(frame, "p_over")
    assert metrics["timestamped_rows"] == 3
    assert metrics["priced_rows"] == 2
    assert metrics["bets"] == 2
    assert metrics["net_units"] == pytest.approx(200 / 110)
    assert metrics["roi"] == pytest.approx(100 / 110)


def test_reliability_returns_json_friendly_numeric_buckets():
    frame = pd.DataFrame({
        "p_over": [0.05, 0.12, 0.25, 0.55, 0.61, 0.95],
        "actual_over": [0, 0, 1, 1, 0, 1],
    })
    buckets = reliability(frame, "p_over", n_bins=5)
    assert {"bin_lower", "bin_upper", "bin_center", "count", "mean_predicted", "observed_over_rate", "calibration_gap"}.issubset(buckets.columns)
    assert int(buckets["count"].sum()) == len(frame)
    assert buckets["bin_lower"].between(0, 1).all()
    assert buckets["bin_upper"].between(0, 1).all()



def test_evaluate_group_keeps_a_fixed_holdout_start_as_new_dates_arrive():
    from scripts.calibrate import evaluate_group

    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    frame = pd.DataFrame({
        "game_date": [day for day in dates for _ in range(10)],
        "target": ["PTS"] * (len(dates) * 10),
        "p_over": [0.25, 0.75] * (len(dates) * 5),
        "actual_over": [0, 1] * (len(dates) * 5),
    })
    first = evaluate_group(frame, "PTS", holdout_start="2026-01-06")
    assert first["holdout_start"] == "2026-01-06"
    assert first["calibration_train_end"] == "2026-01-05"
    assert first["train_rows"] == 50
    assert first["holdout_rows"] == 30

    extra = pd.DataFrame({
        "game_date": [pd.Timestamp("2026-01-09")] * 10,
        "target": ["PTS"] * 10,
        "p_over": [0.25, 0.75] * 5,
        "actual_over": [0, 1] * 5,
    })
    second = evaluate_group(pd.concat([frame, extra], ignore_index=True), "PTS", holdout_start="2026-01-06")
    assert second["holdout_start"] == "2026-01-06"
    assert second["train_rows"] == 50
    assert second["holdout_rows"] == 40
