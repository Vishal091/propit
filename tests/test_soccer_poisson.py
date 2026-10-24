import math
import pytest
from src.soccer.poisson import poisson_pmf, predict_match

def test_poisson_pmf_known_values():
    assert poisson_pmf(0, 0) == 1
    assert poisson_pmf(2, 0) == 0
    assert poisson_pmf(0, 1) == pytest.approx(math.exp(-1))
    assert poisson_pmf(1, 1) == pytest.approx(math.exp(-1))

@pytest.mark.parametrize("goals,rate", [(-1, 1), (1.2, 1), (True, 1)])
def test_pmf_rejects_invalid_goals(goals, rate):
    with pytest.raises(ValueError): poisson_pmf(goals, rate)

@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf")])
def test_pmf_rejects_invalid_rate(rate):
    with pytest.raises(ValueError): poisson_pmf(0, rate)

def test_match_probabilities_sum_to_one_for_three_way_market():
    result = predict_match(1.45, 0.95)
    assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)
    assert all(0 <= p <= 1 for p in (result.home_win, result.draw, result.away_win))

def test_equal_teams_have_symmetric_home_and_away_probabilities():
    result = predict_match(1.2, 1.2)
    assert result.home_win == pytest.approx(result.away_win)
    assert result.draw > 0

def test_integer_total_line_reports_push_probability():
    result = predict_match(1.3, 1.1, total_goals_line=2)
    assert result.push > 0
    assert result.over + result.under + result.push == pytest.approx(1.0)

def test_half_goal_line_has_no_push():
    result = predict_match(1.3, 1.1, total_goals_line=2.5)
    assert result.push == 0
    assert result.over + result.under == pytest.approx(1.0)

def test_zero_rates_produce_zero_zero_scoreline():
    result = predict_match(0, 0, total_goals_line=0)
    assert result.draw == pytest.approx(1.0)
    assert result.push == pytest.approx(1.0)

def test_rejects_bad_inputs():
    with pytest.raises(ValueError): predict_match(-0.1, 1)
    with pytest.raises(ValueError): predict_match(1, 1, total_goals_line=-0.5)
    with pytest.raises(ValueError): predict_match(1, 1, max_goals=3)
