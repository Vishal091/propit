from __future__ import annotations

import math

import pandas as pd
import pytest

from src.dashboard import _records, _validate_game_date


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
