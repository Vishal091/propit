"""Tests for training-time feature availability gates."""
from __future__ import annotations

import pandas as pd
import pytest

from src.train_model import REQUIRED_CONTEXT_FEATURES, validate_context_feature_coverage


def test_training_context_gate_accepts_usable_minutes_and_usage_data():
    frame = pd.DataFrame({column: [1.0, 2.0, 3.0, 4.0, 5.0] for column in REQUIRED_CONTEXT_FEATURES})
    validate_context_feature_coverage(frame, "PTS")


def test_training_context_gate_rejects_missing_minute_or_usage_data():
    frame = pd.DataFrame({column: [1.0, 2.0, 3.0, 4.0, 5.0] for column in REQUIRED_CONTEXT_FEATURES})
    frame["last5_MIN_mean"] = float("nan")

    with pytest.raises(RuntimeError, match="lacks Phase 2 minutes/usage context"):
        validate_context_feature_coverage(frame, "REB")


def test_training_context_gate_rejects_missing_columns():
    frame = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in REQUIRED_CONTEXT_FEATURES if column != "last5_FGA_mean"})

    with pytest.raises(RuntimeError, match="last5_FGA_mean=missing column"):
        validate_context_feature_coverage(frame, "AST")
