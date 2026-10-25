from __future__ import annotations

import pytest

from src.model.prop_probability import (
    calibrate_toward_market,
    estimate_prop_outcomes,
    exceeds_probability_cap,
    quarter_kelly_fraction,
)


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
