from __future__ import annotations

import pandas as pd
import pytest

from src.data.nba_client import _opponent_ids_from_league_log
from src.predict_slate import (
    _decorate_and_rank_candidates,
    _read_line_file,
    american_to_decimal,
    expected_profit_per_unit,
    implied_probability,
)
from src.market_comparison import (
    compare_market_rows,
    consolidate_same_line_markets,
    no_vig_probabilities,
    record_market_snapshot,
)


def test_american_odds_math() -> None:
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert implied_probability(-110) == pytest.approx(0.5238095)
    assert expected_profit_per_unit(0.55, -110) == pytest.approx(0.05)
    assert american_to_decimal(0) is None


def test_opponent_mapping_from_two_teams_per_game() -> None:
    frame = pd.DataFrame({"GAME_ID": ["g1", "g1", "g1", "g1"], "TEAM_ID": [10, 10, 20, 20]})
    assert _opponent_ids_from_league_log(frame).tolist() == [20, 20, 10, 10]


def test_line_file_ignores_comment_example_and_loads_actual_row(tmp_path) -> None:
    path = tmp_path / "lines.csv"
    path.write_text(
        "player_id,player_name,target,line,odds_over,odds_under,bookmaker\n"
        "# 2544,LeBron James,PTS,24.5,-110,-110,EXAMPLE_ONLY\n"
        "2544,LeBron James,PTS,25.5,-105,-115,manual\n",
        encoding="utf-8",
    )
    lines = _read_line_file(path)
    assert len(lines) == 1
    assert lines.iloc[0]["line"] == 25.5
    assert lines.iloc[0]["target"] == "PTS"


def test_predict_without_line_does_not_fabricate_probability() -> None:
    from src.features.build import feature_columns
    from src.model.predict import predict_one

    class StubRegressor:
        feature_importances_ = [0.1] * len(feature_columns("PTS"))

        @staticmethod
        def predict(_X):
            return [25.0]

    class StubBundle:
        q10 = StubRegressor()
        q50 = StubRegressor()
        q90 = StubRegressor()
        residual_std = 5.0
        feature_means = {column: 0.0 for column in feature_columns("PTS")}

    row = pd.Series({column: 1.0 for column in feature_columns("PTS")})
    no_line = predict_one(row, line=None, target="PTS", bundle=StubBundle())
    assert no_line.line is None and no_line.p_over is None
    with_line = predict_one(row, line=24.5, target="PTS", bundle=StubBundle())
    assert with_line.p_over is not None and 0 <= with_line.p_over <= 1


def test_chronological_split_never_splits_same_date() -> None:
    from src.model.train import _chronological_masks
    dates = pd.to_datetime([
        "2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02",
        "2025-01-03", "2025-01-03", "2025-01-04", "2025-01-04",
        "2025-01-05", "2025-01-05",
    ])
    frame = pd.DataFrame({"GAME_DATE": dates})
    train_mask, test_mask = _chronological_masks(frame)
    assert frame.loc[train_mask, "GAME_DATE"].max() < frame.loc[test_mask, "GAME_DATE"].min()



def test_no_vig_requires_a_matched_book_pair():
    over, under, method = no_vig_probabilities(-110, -110, bookmaker_over="DraftKings", bookmaker_under="DraftKings")
    assert over == pytest.approx(0.5)
    assert under == pytest.approx(0.5)
    assert method == "same_book_pair"

    over, under, method = no_vig_probabilities(-110, -110, bookmaker_over="DraftKings", bookmaker_under="FanDuel")
    assert over is None and under is None
    assert method == "cross_book_pair_unavailable"


def test_market_comparison_finds_best_prices_at_the_same_line():
    source = pd.DataFrame({
        "player_id": [12, 12, 12],
        "player_name": ["Test Player"] * 3,
        "target": ["PTS"] * 3,
        "line": [24.5, 24.5, 25.5],
        "odds_over_american": [-105, 100, -110],
        "odds_under_american": [-115, -120, -110],
        "bookmaker": ["DraftKings", "FanDuel", "DraftKings"],
        "market_snapshot_at_utc": ["2026-10-24T12:00:00Z"] * 3,
    })
    compared = compare_market_rows(source, now_utc="2026-10-24T12:15:00Z")
    first = compared.iloc[0]
    assert first["market_books_available"] == 2
    assert first["market_lines_available"] == 2
    assert first["market_consensus_line"] == pytest.approx(25.0)
    assert first["market_line_min"] == pytest.approx(24.5)
    assert first["market_line_max"] == pytest.approx(25.5)
    assert first["best_over_odds_american"] == pytest.approx(100)
    assert first["best_over_bookmaker"] == "FanDuel"
    assert first["best_under_odds_american"] == pytest.approx(-115)
    assert first["best_under_bookmaker"] == "DraftKings"
    assert first["market_no_vig_p_over_consensus"] is not None


def test_consolidate_same_line_markets_uses_best_prices_and_one_row_per_market():
    source = pd.DataFrame({
        "player_id": [12, 12],
        "player_name": ["Test Player", "Test Player"],
        "target": ["PTS", "PTS"],
        "line": [24.5, 24.5],
        "model_p_over": [0.6, 0.6],
        "model_p_under": [0.4, 0.4],
        "odds_over_american": [-105, 100],
        "odds_under_american": [-115, -120],
        "bookmaker": ["DraftKings", "FanDuel"],
        "market_snapshot_at_utc": ["2026-10-24T12:00:00Z"] * 2,
    })
    compared = compare_market_rows(source, now_utc="2026-10-24T12:15:00Z")
    consolidated = consolidate_same_line_markets(compared)
    assert len(consolidated) == 1
    assert consolidated.iloc[0]["odds_over_american"] == pytest.approx(100)
    assert consolidated.iloc[0]["odds_under_american"] == pytest.approx(-115)
    assert consolidated.iloc[0]["market_quoted_rows_count"] == 2


def test_market_snapshot_records_line_and_price_movement(tmp_path):
    history_path = tmp_path / "prop_market_history.csv"
    first = pd.DataFrame({
        "player_id": [12], "player_name": ["Test Player"], "target": ["PTS"],
        "line": [24.5], "odds_over": [-110], "odds_under": [-110],
        "bookmaker_over": ["Underdog"], "bookmaker_under": ["Underdog"],
        "bookmaker": ["Flash Odds public board (over Underdog; under Underdog)"],
        "market_source": ["Flash Odds public board"],
    })
    first_result = record_market_snapshot(
        first, history_path, observed_at_utc="2026-10-24T12:00:00+00:00"
    )
    assert pd.isna(first_result.iloc[0]["previous_line"])
    assert first_result.iloc[0]["market_prior_snapshot_count"] == 0

    second = first.copy()
    second["line"] = 25.5
    second["odds_over"] = [-105]
    second["odds_under"] = [-115]
    second_result = record_market_snapshot(
        second, history_path, observed_at_utc="2026-10-24T13:00:00+00:00"
    )
    row = second_result.iloc[0]
    assert row["previous_line"] == pytest.approx(24.5)
    assert row["line_move_from_previous"] == pytest.approx(1.0)
    assert row["previous_odds_over"] == pytest.approx(-110)
    assert row["market_prior_snapshot_count"] == 1
    assert len(pd.read_csv(history_path)) == 2


def test_public_market_quote_older_than_limit_is_not_shortlisted():
    frame = pd.DataFrame([{
        "player_id": 12, "player_name": "Test Player", "target": "PTS", "line": 24.5,
        "last_game_date": "2026-10-23", "line_source": "public_web_import",
        "bookmaker": "Flash Odds public board (over Underdog; under Underdog)",
        "market_snapshot_at_utc": "2026-10-23T00:00:00Z",
        "market_snapshot_age_hours": 30.0,
        "model_p_over": 0.7, "model_p_under": 0.3,
        "market_no_vig_p_over": 0.5, "market_no_vig_p_under": 0.5,
        "model_edge_over_vs_market": 0.2, "model_edge_under_vs_market": -0.2,
        "ev_over_per_1_unit": 0.1, "ev_under_per_1_unit": -0.2,
        "odds_over_american": -110, "odds_under_american": -110,
    }])
    ranked, shortlist = _decorate_and_rank_candidates(
        frame,
        min_edge=0.0,
        min_ev=0.0,
        min_probability=0.0,
        max_data_age_days=14,
        max_market_age_hours=12,
        today=pd.Timestamp("2026-10-24"),
    )
    assert ranked.iloc[0]["shortlist_status"] == "STALE_MARKET"
    assert shortlist.empty
