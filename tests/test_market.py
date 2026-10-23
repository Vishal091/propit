from __future__ import annotations

import pandas as pd
import pytest

from src.data.nba_client import _opponent_ids_from_league_log
from src.predict_slate import _read_line_file, american_to_decimal, expected_profit_per_unit, implied_probability


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
