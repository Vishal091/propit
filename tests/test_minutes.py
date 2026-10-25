from __future__ import annotations

import pandas as pd
import pytest

from src.model.minutes import parse_minutes, project_minutes


def _history(minutes, season="2025-26", team="BOS"):
    return pd.DataFrame({
        "GAME_DATE": pd.date_range("2025-11-01", periods=len(minutes), freq="2D"),
        "MIN": minutes,
        "SEASON_YEAR": [season] * len(minutes),
        "TEAM_ABBREVIATION": [team] * len(minutes),
    })


def test_parse_minutes_supports_mm_ss_and_invalid():
    got = parse_minutes(pd.Series(["32:30", "18:00", "DNP", 29.5]))
    assert got.iloc[0] == pytest.approx(32.5)
    assert got.iloc[1] == pytest.approx(18.0)
    assert pd.isna(got.iloc[2])
    assert got.iloc[3] == pytest.approx(29.5)


def test_projection_uses_only_games_before_target_date():
    history = _history([20, 22, 24, 26, 28, 30])
    projection = project_minutes(
        history, target_team="BOS", match_date="2025-11-10",
        lineup_status="STARTER", injury_status="AVAILABLE", current_season="2025-26",
    )
    # Excludes the 30-minute game scheduled on Nov 11, after the cutoff.
    assert projection.sample_size == 5
    assert projection.mean is not None
    assert projection.p10 <= projection.mean <= projection.p90


def test_early_season_projection_blends_prior_season_70_30():
    prior = _history([30] * 20, season="2024-25")
    current = _history([20, 22, 24], season="2025-26")
    current["GAME_DATE"] = pd.date_range("2025-10-25", periods=3, freq="2D")
    history = pd.concat([prior, current], ignore_index=True).sort_values("GAME_DATE")
    projection = project_minutes(
        history, target_team="BOS", match_date="2025-11-05",
        lineup_status="BENCH", injury_status="AVAILABLE", current_season="2025-26",
    )
    assert projection.projection_status == "EARLY_SEASON_PRIOR_BLEND"
    assert projection.mean == pytest.approx(0.7 * 30 + 0.3 * 22)


def test_role_change_widens_minutes_uncertainty():
    history = _history([18, 18, 19, 20, 36, 38, 39, 38, 40, 39, 38, 40])
    projection = project_minutes(
        history, target_team="BOS", match_date="2025-12-01",
        lineup_status="STARTER", injury_status="AVAILABLE", current_season="2025-26",
    )
    assert projection.role_change_detected is True
    assert projection.std is not None and projection.std >= 4.5
    assert "shifted" in projection.uncertainty_reason


def test_questionable_status_increases_minutes_uncertainty():
    history = _history([30] * 20)
    available = project_minutes(
        history, target_team="BOS", match_date="2025-12-01",
        lineup_status="STARTER", injury_status="AVAILABLE", current_season="2025-26",
    )
    questionable = project_minutes(
        history, target_team="BOS", match_date="2025-12-01",
        lineup_status="UNKNOWN", injury_status="QUESTIONABLE", current_season="2025-26",
    )
    assert questionable.std > available.std
    assert "questionable" in questionable.uncertainty_reason


def test_manual_return_limit_caps_mean_and_is_audited():
    history = _history([30] * 20)
    projection = project_minutes(
        history, target_team="BOS", match_date="2025-12-01",
        lineup_status="STARTER", injury_status="QUESTIONABLE",
        current_season="2025-26", minutes_limit=22,
    )
    assert projection.mean <= 22
    assert projection.projection_status == "RETURN_LIMIT_APPLIED"
    assert "return limit" in projection.uncertainty_reason
