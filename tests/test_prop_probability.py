from __future__ import annotations

import pytest
import pandas as pd

from src.model.prop_probability import (
    calibrate_toward_market,
    estimate_prop_outcomes,
    exceeds_probability_cap,
    quarter_kelly_fraction,
)
from src.predict_slate import _calibrate_market_probabilities, _decorate_and_rank_candidates


def test_half_point_line_has_no_push_and_probabilities_sum_to_one():
    result = estimate_prop_outcomes(
        mean=24.0, p10=15.0, p90=33.0, line=24.5, target="PTS", fallback_std=6.0
    )
    assert result.p_push == pytest.approx(0.0)
    assert result.p_over + result.p_under == pytest.approx(1.0)
    assert result.distribution == "negative_binomial_mixture"


def test_integer_line_accounts_for_push_probability():
    result = estimate_prop_outcomes(
        mean=8.0, p10=3.0, p90=13.0, line=8.0, target="REB", fallback_std=3.0
    )
    assert result.p_push > 0.0
    assert result.p_over > 0.0
    assert result.p_under > 0.0
    assert result.p_over + result.p_under + result.p_push == pytest.approx(1.0)


def test_low_count_distribution_is_nonnegative_and_selects_family():
    result = estimate_prop_outcomes(
        mean=2.5, p10=0.0, p90=5.0, line=2.5, target="AST", fallback_std=1.5
    )
    assert result.p_over >= 0
    assert result.p_under >= 0
    assert result.p_push == pytest.approx(0.0)
    assert result.distribution in {"poisson", "negative_binomial"}


def test_market_shrinkage_moves_raw_probability_toward_market():
    calibrated = calibrate_toward_market(0.64, 0.50, model_weight=0.25)
    assert calibrated == pytest.approx(0.535)
    assert 0.50 < calibrated < 0.64


def test_probability_cap_flags_uncertain_extremes():
    assert exceeds_probability_cap(0.66, uncertain=True)
    assert exceeds_probability_cap(0.34, uncertain=True)
    assert not exceeds_probability_cap(0.64, uncertain=True)
    assert not exceeds_probability_cap(0.70, uncertain=False)


def test_quarter_kelly_is_bounded_and_rejects_negative_ev():
    assert quarter_kelly_fraction(0.45, -110) == 0.0
    assert 0.0 < quarter_kelly_fraction(0.60, -110) <= 0.02
    assert quarter_kelly_fraction(0.99, 100) == pytest.approx(0.02)



def test_calibration_shrinks_toward_market_and_keeps_push_separate():
    frame = pd.DataFrame([{
        "model_p_over": 0.60,
        "model_p_under": 0.40,
        "model_p_over_raw": 0.60,
        "model_p_under_raw": 0.40,
        "market_no_vig_p_over": 0.50,
        "market_no_vig_p_under": 0.50,
        "p_push": 0.10,
        "availability_gate": "CLEAR",
        "minutes_projection_mean": 34.0,
        "minutes_projection_std": 2.5,
        "minutes_sample_size": 20,
        "minutes_stable": True,
        "lineup_gate": "CLEAR",
        "minutes_role_confirmed": True,
        "odds_over_american": -110,
        "odds_under_american": -110,
    }])
    scored = _calibrate_market_probabilities(frame).iloc[0]
    assert scored["calibration_model_weight"] == pytest.approx(0.50)
    assert scored["calibrated_p_over"] == pytest.approx(0.55)
    assert scored["calibrated_p_under"] == pytest.approx(0.45)
    assert scored["p_push"] == pytest.approx(0.10)
    assert scored["probability_gate_status"] == "CALIBRATED"
    assert scored["ev_over_per_1_unit"] == pytest.approx(0.045)


def test_raw_probability_cap_breach_is_not_silently_clipped():
    frame = pd.DataFrame([{
        "model_p_over": 0.70,
        "model_p_under": 0.30,
        "model_p_over_raw": 0.70,
        "model_p_under_raw": 0.30,
        "market_no_vig_p_over": 0.50,
        "market_no_vig_p_under": 0.50,
        "p_push": 0.0,
        "availability_gate": "CLEAR",
        "minutes_projection_mean": 30.0,
        "minutes_projection_std": 5.0,
        "minutes_sample_size": 20,
        "minutes_stable": False,
        "lineup_gate": "UNVERIFIED",
        "minutes_role_confirmed": False,
        "odds_over_american": -110,
        "odds_under_american": -110,
    }])
    scored = _calibrate_market_probabilities(frame).iloc[0]
    assert scored["probability_cap"] == pytest.approx(0.65)
    assert scored["probability_gate_status"] == "RAW_MODEL_CAP_RED_FLAG"
    assert scored["calibrated_p_over"] == pytest.approx(0.55)


def test_exposure_overlay_limits_game_to_two_distinct_players():
    rows = []
    for player_id in (1, 2, 3, 4):
        rows.append({
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "team": "Boston Celtics",
            "team_abbreviation": "BOS",
            "opponent_team": "New York Knicks",
            "opponent_abbreviation": "NYK",
            "match_date": "2026-10-25",
            "target": "PTS",
            "line": 20.5,
            "last_game_date": "2026-10-24",
            "line_source": "user_supplied",
            "bookmaker": "Book A",
            "model_p_over": 0.56,
            "model_p_under": 0.44,
            "market_no_vig_p_over": 0.50,
            "market_no_vig_p_under": 0.50,
            "model_edge_over_vs_market": 0.06,
            "model_edge_under_vs_market": -0.06,
            "ev_over_per_1_unit": 0.07,
            "ev_under_per_1_unit": -0.20,
            "odds_over_american": -110,
            "odds_under_american": -110,
        })
    ranked, shortlist = _decorate_and_rank_candidates(pd.DataFrame(rows), today=pd.Timestamp("2026-10-25"))
    assert len(shortlist) == 2
    assert shortlist["player_id"].nunique() == 2
    assert shortlist["suggested_stake_percent"].sum() <= 4.0
    assert (ranked["shortlist_status"] == "GAME_EXPOSURE_LIMIT").sum() == 2



def test_minutes_monte_carlo_propagates_distribution_and_respects_pushes():
    from src.model.prop_probability import estimate_prop_outcomes

    tight = estimate_prop_outcomes(
        mean=24.0, p10=16.0, p90=32.0, line=24.0, target="PTS",
        fallback_std=5.0, minutes_mean=32.0, minutes_std=1.5,
        simulations=10_000, seed=55,
    )
    volatile = estimate_prop_outcomes(
        mean=24.0, p10=16.0, p90=32.0, line=24.0, target="PTS",
        fallback_std=5.0, minutes_mean=32.0, minutes_std=8.0,
        simulations=10_000, seed=55,
    )
    assert tight.p_push > 0
    assert tight.p_over + tight.p_under + tight.p_push == pytest.approx(1.0)
    assert volatile.estimated_std > tight.estimated_std
    assert "minutes_mc10000" in volatile.distribution


def test_minutes_monte_carlo_requires_ten_thousand_draws():
    from src.model.prop_probability import estimate_prop_outcomes

    with pytest.raises(ValueError, match="at least 10000"):
        estimate_prop_outcomes(
            mean=10.0, p10=5.0, p90=15.0, line=9.5, target="REB",
            fallback_std=3.0, minutes_mean=25.0, minutes_std=5.0, simulations=9999,
        )


def test_missing_injury_or_minutes_history_withholds_calibrated_probability():
    frame = pd.DataFrame([{
        "model_p_over": 0.60,
        "model_p_under": 0.40,
        "model_p_over_raw": 0.60,
        "model_p_under_raw": 0.40,
        "market_no_vig_p_over": 0.50,
        "market_no_vig_p_under": 0.50,
        "p_push": 0.0,
        "availability_gate": "UNVERIFIED",
        "minutes_projection_mean": 30.0,
        "minutes_projection_std": 5.0,
        "minutes_sample_size": 20,
        "minutes_stable": False,
        "lineup_gate": "UNVERIFIED",
        "minutes_role_confirmed": False,
        "odds_over_american": -110,
        "odds_under_american": -110,
    }])
    scored = _calibrate_market_probabilities(frame).iloc[0]
    assert pd.isna(scored["calibrated_p_over"])
    assert scored["probability_gate_status"] == "INSUFFICIENT_AVAILABILITY_DATA"

    frame.loc[0, "availability_gate"] = "CLEAR"
    frame.loc[0, "minutes_sample_size"] = 1
    scored_minutes = _calibrate_market_probabilities(frame).iloc[0]
    assert pd.isna(scored_minutes["calibrated_p_over"])
    assert scored_minutes["probability_gate_status"] == "INSUFFICIENT_MINUTES_DATA"
