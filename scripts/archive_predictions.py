"""Archive a pregame PropIt prediction snapshot for later evaluation.

Run once for the slate before tip-off. This stores the exact model probabilities,
lines, odds, and source as they were available at that snapshot; it does not
guess the game date or collect historical sportsbook prices retroactively.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("predictions/latest.csv")
DEFAULT_OUTPUT = Path("data/prediction_history.csv")
COLUMNS = [
    "game_date", "snapshot_date", "snapshot_at_utc",
    "player_id", "player_name", "team", "team_abbreviation",
    "match_date", "opponent_team", "opponent_abbreviation", "matchup",
    "target", "line", "p_over", "model_p_over_raw", "model_p_under_raw",
    "market_shrunk_p_over", "market_shrunk_p_under", "selected_model_probability",
    "selected_market_no_vig_probability", "selected_model_edge", "selected_ev_per_1_unit",
    "recommended_side", "shortlist_status", "shortlist_reason", "probability_gate_status",
    "probability_cap", "calibration_model_weight", "calibration_method", "confidence_grade", "main_risk",
    "p_push", "probability_distribution", "actual_over", "actual_stat",
    "odds_over_american", "odds_under_american", "bookmaker", "bookmaker_over", "bookmaker_under",
    "line_source", "market_source", "market_snapshot_at_utc", "prior_snapshot_at_utc",
    "opening_line", "source_line_move", "previous_line", "previous_odds_over", "previous_odds_under",
    "line_move_from_previous", "line_move_from_open",
    "over_decimal_odds_move", "under_decimal_odds_move", "market_prior_snapshot_count",
    "market_no_vig_method", "market_no_vig_p_over", "market_no_vig_p_under",
    "market_no_vig_p_over_consensus", "market_no_vig_p_under_consensus",
    "market_books_available", "market_lines_available", "market_consensus_line", "market_line_min",
    "market_line_max", "market_line_spread", "market_rows_at_line", "same_book_no_vig_rows",
    "best_over_odds_american", "best_over_bookmaker", "best_under_odds_american", "best_under_bookmaker",
    "market_snapshot_age_hours",
    "availability_status", "availability_gate", "availability_reason", "availability_source",
    "availability_reported_at_utc", "availability_age_hours", "availability_retrieved_at_utc",
    "availability_minutes_limit", "lineup_status", "lineup_gate", "lineup_reason", "lineup_source",
    "lineup_reported_at_utc", "lineup_age_hours",
    "minutes_projection_mean", "minutes_projection_std", "minutes_projection_p10", "minutes_projection_p50",
    "minutes_projection_p90", "minutes_last5_mean", "minutes_last10_mean", "minutes_last20_mean",
    "minutes_season_mean", "minutes_previous_season_mean", "minutes_trend_last5_vs_20",
    "minutes_sample_size", "minutes_current_season_games", "minutes_role_change_detected",
    "minutes_stable", "minutes_projection_status", "minutes_uncertainty_reason", "minutes_role_confirmed",
    "closing_line", "closing_odds_over_american", "closing_odds_under_american", "closing_snapshot_at_utc",
    "clv_line", "clv_price_over", "clv_price_under",
    "settlement_status",
]


def prepare_archive_rows(predictions: pd.DataFrame, game_date: str, snapshot_at_utc: str) -> pd.DataFrame:
    """Convert the current prediction output to the persistent history schema."""
    try:
        game_day = date.fromisoformat(game_date).isoformat()
    except ValueError as exc:
        raise ValueError("--game-date must use YYYY-MM-DD format") from exc
    required = {"player_id", "player_name", "target", "line"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction file is missing columns: {', '.join(sorted(missing))}")
    # p_over remains the raw-model column consumed by the chronological
    # isotonic calibrator. Store the live market-shrunk probability separately.
    probability_col = next(
        (column for column in ("model_p_over_raw", "p_over", "model_p_over") if column in predictions.columns),
        None,
    )
    if probability_col is None:
        raise ValueError("prediction file must contain model_p_over_raw, p_over, or model_p_over")

    out = pd.DataFrame(index=predictions.index)
    out["snapshot_date"] = game_day
    out["snapshot_at_utc"] = snapshot_at_utc
    out["player_id"] = pd.to_numeric(predictions["player_id"], errors="coerce").astype("Int64")
    out["player_name"] = predictions["player_name"].astype(str).str.strip()
    string_columns = (
        "team", "team_abbreviation", "opponent_team", "opponent_abbreviation", "matchup",
        "bookmaker", "bookmaker_over", "bookmaker_under", "line_source", "market_source",
        "market_snapshot_at_utc", "prior_snapshot_at_utc", "market_no_vig_method",
        "best_over_bookmaker", "best_under_bookmaker",
        "availability_status", "availability_gate", "availability_reason", "availability_source",
        "availability_reported_at_utc", "availability_retrieved_at_utc",
        "lineup_status", "lineup_gate", "lineup_reason", "lineup_source", "lineup_reported_at_utc",
        "minutes_projection_status", "minutes_uncertainty_reason", "probability_gate_status",
        "calibration_method", "confidence_grade", "main_risk", "recommended_side",
        "shortlist_status", "shortlist_reason", "probability_distribution",
    )
    for column in string_columns:
        out[column] = predictions[column].fillna("").astype(str).str.strip() if column in predictions else ""
    if "match_date" in predictions:
        match_dates = pd.to_datetime(predictions["match_date"], errors="coerce")
        out["match_date"] = match_dates.dt.strftime("%Y-%m-%d").fillna("")
        # Do not substitute snapshot date when the schedule could not map a player.
        out["game_date"] = out["match_date"]
    else:
        # Legacy prediction files did not have a match_date field; preserve their
        # prior behavior as a compatibility fallback.
        out["match_date"] = ""
        out["game_date"] = game_day
    out["target"] = predictions["target"].astype(str).str.upper().str.strip()
    out["line"] = pd.to_numeric(predictions["line"], errors="coerce")
    out["p_over"] = pd.to_numeric(predictions[probability_col], errors="coerce")
    out["model_p_over_raw"] = pd.to_numeric(predictions.get("model_p_over_raw", out["p_over"]), errors="coerce")
    out["model_p_under_raw"] = pd.to_numeric(predictions.get("model_p_under_raw", pd.Series(index=predictions.index, dtype=float)), errors="coerce")
    # In the current slate pipeline calibrated_p_* is the market-shrunk decision
    # probability, kept separate from raw inputs for honest future backtesting.
    out["market_shrunk_p_over"] = pd.to_numeric(predictions.get("calibrated_p_over", predictions.get("model_p_over")), errors="coerce")
    out["market_shrunk_p_under"] = pd.to_numeric(predictions.get("calibrated_p_under", predictions.get("model_p_under")), errors="coerce")
    for column in ("selected_model_probability", "selected_market_no_vig_probability", "selected_model_edge",
                   "selected_ev_per_1_unit", "probability_cap", "calibration_model_weight", "p_push",
                   "minutes_projection_mean", "minutes_projection_std", "minutes_projection_p10",
                   "minutes_projection_p50", "minutes_projection_p90", "minutes_last5_mean",
                   "minutes_last10_mean", "minutes_last20_mean", "minutes_season_mean",
                   "minutes_previous_season_mean", "minutes_trend_last5_vs_20", "minutes_sample_size",
                   "minutes_current_season_games", "availability_minutes_limit", "lineup_age_hours",
                   "minutes_role_confirmed", "minutes_role_change_detected", "minutes_stable"):
        if column in predictions:
            out[column] = predictions[column]
        else:
            out[column] = pd.NA
    for column in ("closing_line", "closing_odds_over_american", "closing_odds_under_american",
                   "closing_snapshot_at_utc", "clv_line", "clv_price_over", "clv_price_under"):
        out[column] = predictions[column] if column in predictions else pd.NA
    for column in ("availability_retrieved_at_utc", "availability_minutes_limit", "lineup_status",
                   "lineup_gate", "lineup_reason", "lineup_source", "lineup_reported_at_utc",
                   "lineup_age_hours", "minutes_projection_status", "minutes_uncertainty_reason",
                   "probability_gate_status", "calibration_method", "confidence_grade", "main_risk",
                   "recommended_side", "shortlist_status", "shortlist_reason", "probability_distribution"):
        if column in predictions:
            out[column] = predictions[column]
    out["actual_over"] = pd.to_numeric(predictions.get("actual_over", pd.Series(index=predictions.index, dtype=float)), errors="coerce")
    out["actual_stat"] = pd.to_numeric(predictions.get("actual_stat", pd.Series(index=predictions.index, dtype=float)), errors="coerce")
    for column in ("odds_over_american", "odds_under_american"):
        out[column] = pd.to_numeric(predictions.get(column, pd.Series(index=predictions.index, dtype=float)), errors="coerce")
    numeric_market_columns = (
        "opening_line", "source_line_move", "previous_line", "previous_odds_over", "previous_odds_under",
        "line_move_from_previous", "line_move_from_open",
        "over_decimal_odds_move", "under_decimal_odds_move", "market_prior_snapshot_count",
        "market_no_vig_p_over", "market_no_vig_p_under", "market_no_vig_p_over_consensus",
        "market_no_vig_p_under_consensus", "market_books_available", "market_lines_available",
        "market_consensus_line", "market_line_min", "market_line_max", "market_line_spread",
        "market_rows_at_line", "same_book_no_vig_rows", "best_over_odds_american",
        "best_under_odds_american", "market_snapshot_age_hours", "availability_age_hours",
        "availability_minutes_limit", "lineup_age_hours",
        "minutes_projection_mean", "minutes_projection_std", "minutes_projection_p10", "minutes_projection_p50",
        "minutes_projection_p90", "minutes_last5_mean", "minutes_last10_mean", "minutes_last20_mean",
        "minutes_season_mean", "minutes_previous_season_mean", "minutes_trend_last5_vs_20",
        "minutes_sample_size", "minutes_current_season_games", "selected_model_probability",
        "selected_market_no_vig_probability", "selected_model_edge", "selected_ev_per_1_unit",
        "probability_cap", "calibration_model_weight", "p_push",
        "closing_line", "closing_odds_over_american", "closing_odds_under_american",
        "clv_line", "clv_price_over", "clv_price_under",
    )
    for column in numeric_market_columns:
        out[column] = pd.to_numeric(predictions[column], errors="coerce") if column in predictions else float("nan")
    out["settlement_status"] = "PENDING"
    # Keep skipped/insufficient-data prop rows for audit even when no probability
    # is released. The calibrator separately drops rows without p_over/outcomes.
    out = out.dropna(subset=["player_id", "player_name", "target", "line"])
    out = out[out["target"].isin(["PTS", "REB", "AST"])].copy()
    if out.empty:
        raise ValueError("no valid player/stat/line rows found to archive")
    return out[COLUMNS].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-date", required=True, help="snapshot date label (YYYY-MM-DD), used to prevent duplicate snapshots")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--append-snapshot", action="store_true",
        help="allow a second snapshot for a game date; repeated snapshots can overweight a market in calibration",
    )
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"prediction file not found: {args.input}")
    try:
        game_day = date.fromisoformat(args.game_date).isoformat()
    except ValueError:
        parser.error("--game-date must use YYYY-MM-DD format")

    snapshot_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        rows = prepare_archive_rows(pd.read_csv(args.input), game_day, snapshot_at)
    except (ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = pd.read_csv(args.output)
        if "snapshot_date" in existing.columns:
            snapshot_dates = existing["snapshot_date"].copy()
            if "game_date" in existing.columns:
                missing_snapshot = snapshot_dates.isna() | snapshot_dates.astype(str).str.strip().isin(["", "nan", "NaT", "<NA>"])
                snapshot_dates = snapshot_dates.where(~missing_snapshot, existing["game_date"])
        else:
            snapshot_dates = existing["game_date"] if "game_date" in existing.columns else pd.Series(dtype=str)
        normalized_snapshot_dates = pd.to_datetime(snapshot_dates, errors="coerce").dt.strftime("%Y-%m-%d")
        if not args.append_snapshot and normalized_snapshot_dates.eq(game_day).any():
            raise SystemExit(
                f"{args.output} already contains a snapshot for {game_day}. "
                "Use --append-snapshot only if you intentionally want to preserve another pregame market snapshot."
            )
        for column in COLUMNS:
            if column not in existing.columns:
                existing[column] = pd.NA
        rows = pd.concat([existing[COLUMNS], rows], ignore_index=True)
    rows.to_csv(args.output, index=False)
    snapshot_rows = rows.loc[rows["snapshot_date"].astype(str).eq(game_day)]
    odds_count = int(snapshot_rows[["odds_over_american", "odds_under_american"]].notna().all(axis=1).sum())
    dated_games = int(snapshot_rows["match_date"].astype(str).str.len().gt(0).sum())
    print(f"Archived {len(snapshot_rows)} rows for snapshot date {game_day} to {args.output}.")
    print(f"Rows with a scheduled match date: {dated_games}; rows with both prices: {odds_count}. Confirm this was run before tip-off.")


if __name__ == "__main__":
    main()
