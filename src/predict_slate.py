"""Generate league-wide NBA estimates and a ranked, market-aware shortlist.

Without lines, this exports estimates only. With a lines CSV, it scores only
markets supplied by the user or the separate public-page importer. It will not
invent sportsbook lines, market prices, or betting edges.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.nba_client import get_active_players, get_league_game_log
from src.data.seasons import current_season, recent_seasons
from src.data.schedule import get_next_game_map
from src.features.build import build_player_features, build_team_game_metrics, feature_columns
from src.model.predict import Prediction, predict_one
from src.model.train import TrainedModel
from src.market_comparison import (
    compare_market_rows,
    consolidate_same_line_markets,
    no_vig_probabilities,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("predict_slate")
TARGETS = ("PTS", "REB", "AST")


def american_to_decimal(odds: float | int | None) -> float | None:
    """Convert American odds to decimal odds; return None for missing/invalid."""
    if odds is None or pd.isna(odds):
        return None
    value = float(odds)
    if value == 0:
        return None
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def implied_probability(odds: float | int | None) -> float | None:
    decimal = american_to_decimal(odds)
    return (1.0 / decimal) if decimal is not None else None


def expected_profit_per_unit(probability: float | None, odds: float | int | None) -> float | None:
    """Expected net profit on a 1-unit stake, not a guarantee of outcome."""
    decimal = american_to_decimal(odds)
    if probability is None or decimal is None:
        return None
    return float(probability) * (decimal - 1.0) - (1.0 - float(probability))


def _validate_model_feature_context(bundle: TrainedModel, target: str) -> None:
    """Reject stale artifacts whose feature schema predates the current model."""
    saved_names = getattr(bundle, "feature_names", None)
    expected_names = tuple(feature_columns(target))
    if not saved_names:
        raise RuntimeError(
            f"The {target} model artifact has no saved feature schema and is outdated. "
            "Run .\\.venv\\Scripts\\python.exe -m src.train_model before refreshing the slate."
        )
    if tuple(saved_names) != expected_names:
        raise RuntimeError(
            f"The {target} model was trained with an older feature schema. "
            "Retrain all models with .\\.venv\\Scripts\\python.exe -m src.train_model."
        )
    feature_stds = getattr(bundle, "feature_stds", None)
    if not isinstance(feature_stds, dict):
        raise RuntimeError(
            f"The {target} model artifact predates the next-game/opponent-context update. "
            "Run .\\.venv\\Scripts\\python.exe -m src.train_model before refreshing the slate."
        )
    unusable = [
        column for column in ("opp_def_rating", "opp_pace")
        if not _is_positive_finite(feature_stds.get(column))
    ]
    if unusable:
        raise RuntimeError(
            f"The {target} model was trained without varying opponent context ({', '.join(unusable)}). "
            "Rebuild the models with .\\.venv\\Scripts\\python.exe -m src.train_model."
        )


def _is_positive_finite(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(parsed) and parsed > 1e-6)


def _load_league_history() -> pd.DataFrame:
    """Fetch the current season and two seasons of history for rolling features."""
    seasons = recent_seasons(current_season(), count=3)
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            frame = get_league_game_log(season)
        except Exception as exc:
            log.warning("could not load %s game logs: %s", season, exc)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No league game logs could be loaded from stats.nba.com.")
    games = pd.concat(frames, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    games["PLAYER_ID"] = pd.to_numeric(games["PLAYER_ID"], errors="coerce")
    games = games.dropna(subset=["GAME_DATE", "PLAYER_ID"])
    if "GAME_ID" in games.columns:
        games = games.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last")
    return games.sort_values(["GAME_DATE", "PLAYER_ID"], kind="stable").reset_index(drop=True)


def _active_player_map(games: pd.DataFrame) -> dict[int, str]:
    """Use current roster status when available; fall back to recent game logs."""
    try:
        active = get_active_players()
        if not active.empty:
            result = {
                int(row.PLAYER_ID): str(row.PLAYER_NAME)
                for row in active.itertuples(index=False)
                if pd.notna(row.PLAYER_ID) and pd.notna(row.PLAYER_NAME)
            }
            if result:
                log.info("current roster endpoint returned %s active players", len(result))
                return result
    except Exception as exc:
        log.warning("active-roster endpoint unavailable; falling back to recent game logs: %s", exc)
    latest_date = games["GAME_DATE"].max()
    latest_logs = games.loc[games["GAME_DATE"] >= latest_date - pd.Timedelta(days=365)]
    return {
        int(pid): str(group.sort_values("GAME_DATE").iloc[-1]["PLAYER_NAME"])
        for pid, group in latest_logs.groupby("PLAYER_ID", sort=False)
        if pd.notna(pid) and pd.notna(group.iloc[-1].get("PLAYER_NAME"))
    }


def _team_display(city: Any, name: Any, abbreviation: Any) -> tuple[str, str]:
    """Return a readable team name and a normalized three-letter code."""
    city_text = "" if city is None or pd.isna(city) else str(city).strip()
    name_text = "" if name is None or pd.isna(name) else str(name).strip()
    abbr_text = "" if abbreviation is None or pd.isna(abbreviation) else str(abbreviation).strip().upper()
    display = " ".join(part for part in (city_text, name_text) if part)
    return display or abbr_text, abbr_text


def _player_team_metadata(games: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Use current roster team fields first, then the player's latest logged team."""
    metadata: dict[int, dict[str, str]] = {}
    try:
        active = get_active_players()
        if not active.empty and {"PLAYER_ID", "PLAYER_NAME"}.issubset(active.columns):
            for row in active.to_dict(orient="records"):
                try:
                    player_id = int(row["PLAYER_ID"])
                except (TypeError, ValueError):
                    continue
                team, abbreviation = _team_display(
                    row.get("TEAM_CITY"), row.get("TEAM_NAME"), row.get("TEAM_ABBREVIATION")
                )
                if team or abbreviation:
                    metadata[player_id] = {"team": team, "team_abbreviation": abbreviation}
                    team_id = _number_or_none(row.get("TEAM_ID"))
                    if team_id is not None:
                        metadata[player_id]["team_id"] = int(team_id)
    except Exception as exc:
        log.warning("could not load current roster team metadata: %s", exc)

    if "PLAYER_ID" not in games.columns or "GAME_DATE" not in games.columns:
        return metadata
    available = [column for column in ("TEAM_CITY", "TEAM_NAME", "TEAM_ABBREVIATION") if column in games.columns]
    if not available:
        return metadata
    for player_id, group in games.groupby("PLAYER_ID", sort=False):
        try:
            pid = int(player_id)
        except (TypeError, ValueError):
            continue
        current = metadata.get(pid, {})
        if current.get("team_abbreviation"):
            continue
        latest = group.sort_values("GAME_DATE", kind="stable").iloc[-1]
        team, abbreviation = _team_display(
            latest.get("TEAM_CITY"), latest.get("TEAM_NAME"), latest.get("TEAM_ABBREVIATION")
        )
        if team or abbreviation:
            prior = metadata.get(pid, {})
            metadata[pid] = {
                "team": team or str(prior.get("team") or ""),
                "team_abbreviation": abbreviation or str(prior.get("team_abbreviation") or ""),
            }
            team_id = _number_or_none(prior.get("team_id"))
            if team_id is None:
                team_id = _number_or_none(latest.get("TEAM_ID"))
            if team_id is not None:
                metadata[pid]["team_id"] = int(team_id)
    return metadata


def _next_match_for_player(
    player_id: int,
    player_teams: dict[int, dict[str, Any]],
    next_games: dict[str, dict[str, str]],
    line_row: dict[str, Any] | None = None,
    team_id_by_abbreviation: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Join a player's current team to that team's next scheduled NBA game."""
    line_row = line_row or {}
    team_meta = player_teams.get(player_id, {})
    team_abbreviation = team_meta.get("team_abbreviation", "")
    team_name = team_meta.get("team", "")
    def row_text(key: str) -> str:
        value = line_row.get(key)
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    line_team = row_text("team")
    if line_team:
        team_name = line_team
    scheduled = next_games.get(team_abbreviation, {}) if team_abbreviation else {}
    match_date = row_text("match_date") or scheduled.get("match_date", "")
    opponent = row_text("opponent_team") or scheduled.get("opponent_team", "")
    opponent_abbreviation = row_text("opponent_abbreviation") or scheduled.get("opponent_abbreviation", "")
    matchup = row_text("matchup") or scheduled.get("matchup", "")
    if not matchup and team_abbreviation and opponent_abbreviation:
        matchup = (
            f"{team_abbreviation} vs {opponent_abbreviation}"
            if str(scheduled.get("home_away", "")).upper() == "HOME"
            else f"{team_abbreviation} @ {opponent_abbreviation}"
        )
    team_id_by_abbreviation = team_id_by_abbreviation or {}
    team_id_value = team_meta.get("team_id") or team_id_by_abbreviation.get(team_abbreviation)
    opponent_team_id = team_id_by_abbreviation.get(opponent_abbreviation)
    return {
        "team": team_name,
        "team_abbreviation": team_abbreviation,
        "team_id": int(team_id_value) if team_id_value is not None else None,
        "match_date": match_date,
        "opponent_team": opponent,
        "opponent_abbreviation": opponent_abbreviation,
        "opponent_team_id": int(opponent_team_id) if opponent_team_id is not None else None,
        "matchup": matchup,
        "home_away": scheduled.get("home_away", ""),
    }


def _read_line_file(path: Path) -> pd.DataFrame:
    """Read line rows from CSV. Commented-out examples are ignored."""
    try:
        lines = pd.read_csv(path, comment="#")
    except Exception as exc:
        raise ValueError(f"could not read lines CSV {path}: {exc}") from exc
    lines.columns = [str(col).strip().lower() for col in lines.columns]
    if lines.empty:
        raise ValueError(f"{path} has no prop rows. Add player_id (or player_name), target, and line.")
    if "target" not in lines or "line" not in lines:
        raise ValueError("lines CSV must contain target and line columns")
    if "player_id" not in lines and "player_name" not in lines:
        raise ValueError("lines CSV must contain player_id or player_name")
    lines["target"] = lines["target"].astype(str).str.strip().str.upper()
    bad_targets = sorted(set(lines.loc[~lines["target"].isin(TARGETS), "target"].tolist()))
    if bad_targets:
        raise ValueError(f"unsupported target(s): {', '.join(bad_targets)}; use PTS, REB, or AST")
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")
    lines = lines.dropna(subset=["line"]).copy()
    if lines.empty:
        raise ValueError("lines CSV contains no numeric lines")
    return lines


def _number_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _team_ids_by_abbreviation(
    games: pd.DataFrame,
    player_teams: dict[int, dict[str, Any]],
) -> dict[str, int]:
    """Map NBA team abbreviations to team IDs, preferring current-roster metadata."""
    result: dict[str, int] = {}
    for info in player_teams.values():
        abbreviation = str(info.get("team_abbreviation") or "").strip().upper()
        team_id = _number_or_none(info.get("team_id"))
        if abbreviation and team_id is not None:
            result[abbreviation] = int(team_id)

    required = {"TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE"}
    if required.issubset(games.columns):
        historical = games.copy()
        historical["GAME_DATE"] = pd.to_datetime(historical["GAME_DATE"], errors="coerce")
        historical["TEAM_ID"] = pd.to_numeric(historical["TEAM_ID"], errors="coerce")
        historical["TEAM_ABBREVIATION"] = historical["TEAM_ABBREVIATION"].fillna("").astype(str).str.strip().str.upper()
        historical = historical.dropna(subset=["GAME_DATE", "TEAM_ID"]).sort_values("GAME_DATE", kind="stable")
        for row in historical.drop_duplicates("TEAM_ABBREVIATION", keep="last").to_dict(orient="records"):
            abbreviation = row["TEAM_ABBREVIATION"]
            if abbreviation and abbreviation not in result:
                result[abbreviation] = int(row["TEAM_ID"])
    return result


def _build_latest_features(
    games: pd.DataFrame,
    active_players: dict[int, str],
    player_teams: dict[int, dict[str, Any]] | None = None,
    next_games: dict[str, dict[str, str]] | None = None,
    opponent_team_metrics: pd.DataFrame | None = None,
    team_id_by_abbreviation: dict[str, int] | None = None,
) -> dict[tuple[int, str], tuple[pd.Series, str, str]]:
    """Build prediction features on a synthetic row for each player's actual next game."""
    features: dict[tuple[int, str], tuple[pd.Series, str, str]] = {}
    player_teams = player_teams or {}
    next_games = next_games or {}
    team_id_by_abbreviation = team_id_by_abbreviation or {}
    game_groups = {
        int(pid): group.sort_values("GAME_DATE", kind="stable").copy()
        for pid, group in games.groupby("PLAYER_ID", sort=False)
    }
    for player_id, player_name in active_players.items():
        player_games = game_groups.get(player_id)
        if player_games is None or player_games.empty:
            log.info("skip %s (%s): no game log in the last three seasons", player_name, player_id)
            continue

        last_game_date = pd.to_datetime(player_games["GAME_DATE"].max()).date().isoformat()
        match_info = _next_match_for_player(
            player_id, player_teams, next_games,
            team_id_by_abbreviation=team_id_by_abbreviation,
        )
        match_date = pd.to_datetime(match_info.get("match_date"), errors="coerce")
        last_date = pd.to_datetime(player_games["GAME_DATE"].max(), errors="coerce")
        if pd.isna(match_date) or pd.isna(last_date) or match_date.normalize() <= last_date.normalize():
            log.debug(
                "skip %s (%s): no future game date matched to current team %s",
                player_name, player_id, match_info.get("team_abbreviation") or "(unknown)",
            )
            continue
        if not match_info.get("team_id") or not match_info.get("opponent_team_id"):
            log.warning(
                "skip %s (%s): could not map both scheduled teams to NBA team IDs (%s vs %s)",
                player_name, player_id,
                match_info.get("team_abbreviation") or "(unknown)",
                match_info.get("opponent_abbreviation") or "(unknown)",
            )
            continue

        # Append an outcome-free next-game row. The lagged features on this row
        # now include the most recently completed game, unlike the old behavior
        # which returned the feature row for that completed game itself.
        latest = player_games.iloc[-1]
        future_row = latest.to_dict()
        future_row["GAME_DATE"] = match_date.normalize()
        team_abbreviation = str(match_info.get("team_abbreviation") or "").strip().upper()
        opponent_abbreviation = str(match_info.get("opponent_abbreviation") or "").strip().upper()
        home_away = str(match_info.get("home_away") or "").strip().upper()
        if team_abbreviation and opponent_abbreviation and home_away in {"HOME", "AWAY"}:
            future_row["MATCHUP"] = (
                f"{team_abbreviation} vs. {opponent_abbreviation}"
                if home_away == "HOME"
                else f"{team_abbreviation} @ {opponent_abbreviation}"
            )
        if match_info.get("team_id") is not None:
            future_row["TEAM_ID"] = int(match_info["team_id"])
        if match_info.get("opponent_team_id") is not None:
            future_row["OPPONENT_TEAM_ID"] = int(match_info["opponent_team_id"])
        else:
            future_row["OPPONENT_TEAM_ID"] = 0
        if "GAME_ID" in future_row:
            future_row["GAME_ID"] = f"future-{player_id}-{match_date.date().isoformat()}"
        for stat in TARGETS:
            if stat in future_row:
                future_row[stat] = np.nan
        future_games = pd.concat(
            [player_games, pd.DataFrame([future_row])],
            ignore_index=True,
        )

        for target in TARGETS:
            try:
                frame = build_player_features(
                    future_games,
                    target=target,
                    opponent_team_metrics=opponent_team_metrics,
                )
            except (KeyError, ValueError) as exc:
                log.warning("skip %s %s feature build: %s", player_name, target, exc)
                continue
            if frame.empty:
                continue
            next_rows = frame.loc[pd.to_datetime(frame["GAME_DATE"]).dt.normalize().eq(match_date.normalize())]
            if next_rows.empty:
                log.warning("skip %s %s: unable to build next-game features", player_name, target)
                continue
            features[(player_id, target)] = (next_rows.iloc[-1], player_name, last_game_date)
    log.info(
        "built next-game features for %s player/stat combinations across %s players",
        len(features), len({key[0] for key in features}),
    )
    return features

def _prediction_record(
    player_id: int,
    player_name: str,
    target: str,
    last_game_date: str,
    pred: Prediction,
    line_source: str,
    odds_over: float | None = None,
    odds_under: float | None = None,
    bookmaker: str = "",
    player_team: dict[str, str] | None = None,
    match_info: dict[str, str] | None = None,
    market_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    player_team = player_team or {}
    match_info = match_info or {}
    market_info = market_info or {}
    p_over = pred.p_over
    p_under = (1.0 - p_over) if p_over is not None else None
    no_vig_over, no_vig_under, no_vig_method = no_vig_probabilities(
        odds_over,
        odds_under,
        bookmaker_over=market_info.get("bookmaker_over"),
        bookmaker_under=market_info.get("bookmaker_under"),
        bookmaker=bookmaker,
    )
    over_ev = expected_profit_per_unit(p_over, odds_over)
    under_ev = expected_profit_per_unit(p_under, odds_under)
    return {
        "player_id": player_id, "player_name": player_name,
        "team": match_info.get("team") or player_team.get("team", ""),
        "team_abbreviation": match_info.get("team_abbreviation") or player_team.get("team_abbreviation", ""),
        "match_date": match_info.get("match_date", ""), "opponent_team": match_info.get("opponent_team", ""),
        "opponent_abbreviation": match_info.get("opponent_abbreviation", ""), "opponent_team_id": match_info.get("opponent_team_id"), "matchup": match_info.get("matchup", ""),
        "home_away": match_info.get("home_away", ""),
        "target": target, "line": pred.line, "line_source": line_source, "last_game_date": last_game_date,
        "model_expected_value": round(pred.expected_value, 3),
        "interval_80_low": round(pred.interval_80[0], 3),
        "interval_80_high": round(pred.interval_80[1], 3),
        "model_p_over": round(p_over, 4) if p_over is not None else None,
        "model_p_under": round(p_under, 4) if p_under is not None else None,
        "p_over": round(p_over, 4) if p_over is not None else None,
        "actual_over": None,
        "odds_over_american": odds_over, "odds_under_american": odds_under,
        "market_no_vig_p_over": round(no_vig_over, 4) if no_vig_over is not None else None,
        "market_no_vig_p_under": round(no_vig_under, 4) if no_vig_under is not None else None,
        "model_edge_over_vs_market": round(p_over - no_vig_over, 4) if p_over is not None and no_vig_over is not None else None,
        "model_edge_under_vs_market": round(p_under - no_vig_under, 4) if p_under is not None and no_vig_under is not None else None,
        "ev_over_per_1_unit": round(over_ev, 4) if over_ev is not None else None,
        "ev_under_per_1_unit": round(under_ev, 4) if under_ev is not None else None,
        "bookmaker": bookmaker,
        "bookmaker_over": str(market_info.get("bookmaker_over") or "").strip(),
        "bookmaker_under": str(market_info.get("bookmaker_under") or "").strip(),
        "market_source": str(market_info.get("market_source") or ("Flash Odds public board" if line_source == "public_web_import" else "")),
        "market_snapshot_at_utc": market_info.get("market_snapshot_at_utc") or market_info.get("snapshot_at_utc") or "",
        "market_no_vig_method": no_vig_method,
        "previous_line": _number_or_none(market_info.get("previous_line")),
        "previous_odds_over": _number_or_none(market_info.get("previous_odds_over")),
        "previous_odds_under": _number_or_none(market_info.get("previous_odds_under")),
        "prior_snapshot_at_utc": market_info.get("prior_snapshot_at_utc") or "",
        "market_prior_snapshot_count": _number_or_none(market_info.get("market_prior_snapshot_count")),
        "line_move_from_previous": _number_or_none(market_info.get("line_move_from_previous")),
        "over_decimal_odds_move": _number_or_none(market_info.get("over_decimal_odds_move")),
        "under_decimal_odds_move": _number_or_none(market_info.get("under_decimal_odds_move")),
        "line_warning": (
            "No sportsbook line supplied" if pred.line is None else
            "Public web snapshot; verify current line, both prices, and local availability" if line_source == "public_web_import" else
            "User-supplied line; not independently verified" if line_source == "user_supplied" else ""
        ),
    }


def _decorate_and_rank_candidates(
    out: pd.DataFrame,
    *,
    min_edge: float = 0.03,
    min_ev: float = 0.02,
    min_probability: float = 0.53,
    max_data_age_days: int = 14,
    max_market_age_hours: float = 12.0,
    today: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter market rows and rank the ones that pass configured research screens."""
    result = out.copy()
    today_date = (today if today is not None else pd.Timestamp.now()).date()
    freshness: list[int | None] = []
    market_freshness: list[float | None] = []
    statuses: list[str] = []
    reasons: list[str] = []
    sides: list[str | None] = []
    probs: list[float | None] = []
    market_probs: list[float | None] = []
    edges: list[float | None] = []
    evs: list[float | None] = []

    for row in result.to_dict(orient="records"):
        date_value = pd.to_datetime(row.get("last_game_date"), errors="coerce")
        age = max(0, (today_date - date_value.date()).days) if pd.notna(date_value) else None
        freshness.append(age)
        market_age = _number_or_none(row.get("market_snapshot_age_hours"))
        if market_age is None:
            stamp = pd.to_datetime(row.get("market_snapshot_at_utc"), utc=True, errors="coerce")
            if pd.notna(stamp):
                market_age = max(0.0, (pd.Timestamp.now(tz="UTC") - stamp).total_seconds() / 3600.0)
        market_freshness.append(round(market_age, 2) if market_age is not None else None)
        source_value = row.get("bookmaker")
        source_label = "" if source_value is None or pd.isna(source_value) else str(source_value).strip().casefold()
        placeholder_markers = ("test", "example", "sample", "demo", "not_live", "not live")
        is_placeholder = (not source_label) or any(marker in source_label for marker in placeholder_markers)
        line_source_value = str(row.get("line_source") or "").strip().casefold()
        is_public_snapshot = line_source_value == "public_web_import"
        market_stamp = str(row.get("market_snapshot_at_utc") or "").strip()
        market_too_old = market_age is not None and market_age > max_market_age_hours
        if row.get("line") is None or pd.isna(row.get("line")):
            status, reason = "NEEDS_LINE", "Supply a current prop line and both over/under prices."
            side = prob = market_prob = edge = ev = None
        elif market_too_old or (is_public_snapshot and (market_age is None or not market_stamp)):
            status = "STALE_MARKET"
            reason = (
                f"Market snapshot is {market_age:.1f} hours old; limit is {max_market_age_hours:.1f} hours."
                if market_age is not None else
                "Public market row has no capture timestamp; refresh the board before evaluating it."
            )
            side = prob = market_prob = edge = ev = None
        elif age is None or age > max_data_age_days:
            status, reason = "STALE_DATA", f"Last game data is {age if age is not None else 'unknown'} days old; limit is {max_data_age_days} days."
            side = prob = market_prob = edge = ev = None
        elif row.get("odds_over_american") is None or pd.isna(row.get("odds_over_american")) or row.get("odds_under_american") is None or pd.isna(row.get("odds_under_american")):
            status, reason = "NEEDS_BOTH_PRICES", "Supply both over and under prices to estimate market edge and EV."
            side = prob = market_prob = edge = ev = None
        elif is_placeholder:
            status, reason = "PLACEHOLDER_INPUT", "Replace the test/example source label with the real source or sportsbook where you checked the line and both prices."
            side = prob = market_prob = edge = ev = None
        else:
            candidates = []
            over_ev = _number_or_none(row.get("ev_over_per_1_unit"))
            under_ev = _number_or_none(row.get("ev_under_per_1_unit"))
            over_edge = _number_or_none(row.get("model_edge_over_vs_market"))
            under_edge = _number_or_none(row.get("model_edge_under_vs_market"))
            if over_ev is not None and over_edge is not None and row.get("model_p_over") is not None:
                candidates.append((over_ev, over_edge, float(row["model_p_over"]), _number_or_none(row.get("market_no_vig_p_over")), "OVER"))
            if under_ev is not None and under_edge is not None and row.get("model_p_under") is not None:
                candidates.append((under_ev, under_edge, float(row["model_p_under"]), _number_or_none(row.get("market_no_vig_p_under")), "UNDER"))
            if not candidates:
                no_vig_method = str(row.get("market_no_vig_method") or "")
                if no_vig_method == "cross_book_pair_unavailable":
                    reason_text = "Over/under quotes come from different books; no-vig edge is unavailable until matched same-book prices are provided."
                else:
                    reason_text = "Model probability or valid odds are missing."
                status, reason = "NOT_SCORABLE", reason_text
                side = prob = market_prob = edge = ev = None
            else:
                ev, edge, prob, market_prob, side = max(candidates, key=lambda item: item[0])
                if edge >= min_edge and ev >= min_ev and prob >= min_probability:
                    status = "SHORTLIST_CANDIDATE"
                    reason = f"Best modeled side {side}: edge {edge:.1%}, EV {ev:.3f} units per unit staked; verify source and model calibration."
                else:
                    status = "BELOW_THRESHOLDS"
                    reason = f"Best modeled side {side} missed filters (edge >= {min_edge:.1%}, EV >= {min_ev:.3f}, probability >= {min_probability:.1%})."
        statuses.append(status)
        reasons.append(reason)
        sides.append(side)
        probs.append(round(prob, 4) if prob is not None else None)
        market_probs.append(round(market_prob, 4) if market_prob is not None else None)
        edges.append(round(edge, 4) if edge is not None else None)
        evs.append(round(ev, 4) if ev is not None else None)

    result["data_age_days"] = freshness
    result["market_age_hours"] = market_freshness
    result["shortlist_status"] = statuses
    result["shortlist_reason"] = reasons
    result["recommended_side"] = sides
    result["selected_model_probability"] = probs
    result["selected_market_no_vig_probability"] = market_probs
    result["selected_model_edge"] = edges
    result["selected_ev_per_1_unit"] = evs
    shortlist = result.loc[result["shortlist_status"] == "SHORTLIST_CANDIDATE"].copy()
    shortlist = shortlist.sort_values(
        ["selected_ev_per_1_unit", "selected_model_edge", "selected_model_probability"],
        ascending=[False, False, False], kind="stable",
    ).reset_index(drop=True)
    shortlist.insert(0, "rank", range(1, len(shortlist) + 1))
    return result, shortlist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=Path, default=None, help="CSV containing actual lines and optional over/under American odds")
    parser.add_argument("--output", type=Path, default=Path("predictions/latest.csv"))
    parser.add_argument("--shortlist-output", type=Path, default=Path("predictions/shortlist.csv"))
    parser.add_argument("--top", type=int, default=10, help="Maximum number of qualifying candidates to print")
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--min-probability", type=float, default=0.53)
    parser.add_argument("--max-data-age-days", type=int, default=14)
    parser.add_argument("--max-market-age-hours", type=float, default=12.0, help="reject market quotes captured earlier than this many hours ago")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")
    if not 0 <= args.min_edge <= 1 or not 0 <= args.min_probability <= 1:
        parser.error("--min-edge and --min-probability must be between 0 and 1")
    if args.min_ev < 0 or args.max_data_age_days < 0 or args.max_market_age_hours < 0:
        parser.error("--min-ev, --max-data-age-days, and --max-market-age-hours must be non-negative")

    bundles: dict[str, TrainedModel] = {}
    for target in TARGETS:
        bundle = TrainedModel.load(f"artifacts/model_{target}.joblib")
        _validate_model_feature_context(bundle, target)
        bundles[target] = bundle

    games = _load_league_history()
    log.info("loaded %s player-game rows across %s players", len(games), games["PLAYER_ID"].nunique())
    active_players = _active_player_map(games)
    player_teams = _player_team_metadata(games)
    try:
        next_games = get_next_game_map()
        log.info("loaded next scheduled games for %s NBA teams", len(next_games))
        roster_abbreviations = {
            str(info.get("team_abbreviation") or "").strip().upper()
            for info in player_teams.values()
            if str(info.get("team_abbreviation") or "").strip()
        }
        unmatched = sorted(roster_abbreviations - set(next_games))
        if unmatched:
            log.warning(
                "schedule lookup did not match these roster team abbreviations: %s",
                ", ".join(unmatched),
            )
        if not next_games:
            log.error("schedule lookup returned zero teams; next-game projections will be skipped")
    except Exception as exc:
        log.warning("could not load upcoming NBA schedule; next-game predictions will be skipped: %s", exc)
        next_games = {}

    team_id_by_abbreviation = _team_ids_by_abbreviation(games, player_teams)
    opponent_team_metrics = build_team_game_metrics(games)
    if opponent_team_metrics.empty:
        raise RuntimeError(
            "Could not build opponent context from historical player logs. "
            "Required fields include TEAM_ID, OPPONENT_TEAM_ID, PTS, FGA, FTA, OREB and TOV."
        )
    log.info(
        "built opponent context from %s team-game rows across %s teams",
        len(opponent_team_metrics),
        opponent_team_metrics["TEAM_ID"].nunique() if not opponent_team_metrics.empty else 0,
    )
    log.info("preparing next-game features for %s candidate active players", len(active_players))
    feature_lookup = _build_latest_features(
        games, active_players, player_teams, next_games,
        opponent_team_metrics, team_id_by_abbreviation,
    )
    if not feature_lookup:
        raise RuntimeError(
            "No next-game features were prepared. Check the schedule lookup and confirm active players "
            "have a future NBA game with a valid team/opponent mapping."
        )

    records: list[dict[str, Any]] = []

    if args.lines is not None:
        lines = _read_line_file(args.lines)
        name_to_id = {name.strip().casefold(): pid for pid, name in active_players.items()}
        if "PLAYER_NAME" in games.columns:
            for pid, group in games.groupby("PLAYER_ID", sort=False):
                if pd.notna(pid) and pd.notna(group.iloc[-1].get("PLAYER_NAME")):
                    name_to_id.setdefault(str(group.iloc[-1]["PLAYER_NAME"]).strip().casefold(), int(pid))
        for row in lines.to_dict(orient="records"):
            player_id = _number_or_none(row.get("player_id"))
            if player_id is None and str(row.get("player_name", "")).strip():
                player_id = float(name_to_id.get(str(row["player_name"]).strip().casefold(), 0)) or None
            if player_id is None:
                log.warning("skip line row: no matching player ID/name: %s", row)
                continue
            player_id = int(player_id)
            target = str(row["target"]).strip().upper()
            entry = feature_lookup.get((player_id, target))
            if entry is None:
                log.warning("skip line for player_id=%s target=%s: no usable feature history", player_id, target)
                continue
            feature_row, player_name, last_game_date = entry
            provided_name = row.get("player_name")
            display_name = str(provided_name).strip() if provided_name is not None and not pd.isna(provided_name) and str(provided_name).strip() else player_name
            line = float(row["line"])
            pred = predict_one(feature_row, line=line, target=target, bundle=bundles[target])
            odds_over = _number_or_none(row.get("odds_over"))
            odds_under = _number_or_none(row.get("odds_under"))
            bookmaker_value = row.get("bookmaker")
            bookmaker = "" if bookmaker_value is None or pd.isna(bookmaker_value) else str(bookmaker_value).strip()
            line_source = "public_web_import" if bookmaker.casefold().startswith(("probetrics public board", "flash odds public board")) else "user_supplied"
            team_meta = player_teams.get(player_id, {})
            match_info = _next_match_for_player(player_id, player_teams, next_games, row, team_id_by_abbreviation)
            records.append(_prediction_record(
                player_id, display_name, target, last_game_date, pred,
                line_source, odds_over, odds_under, bookmaker, team_meta, match_info, row,
            ))
    else:
        for (player_id, target), (feature_row, player_name, last_game_date) in sorted(feature_lookup.items()):
            pred = predict_one(feature_row, line=None, target=target, bundle=bundles[target])
            team_meta = player_teams.get(player_id, {})
            match_info = _next_match_for_player(player_id, player_teams, next_games, team_id_by_abbreviation=team_id_by_abbreviation)
            records.append(_prediction_record(
                player_id, player_name, target, last_game_date, pred,
                "not_supplied", player_team=team_meta, match_info=match_info,
            ))
    if not records:
        raise RuntimeError("No predictions produced. Confirm input player IDs/names are present in available history.")

    out = pd.DataFrame(records)
    out = compare_market_rows(out)
    # One prediction row per player/stat/line. The price on each side is the
    # best decimal return observed at that exact line; no-vig consensus uses
    # only same-book pairs from any available book rows.
    out = consolidate_same_line_markets(out)
    p_over = pd.to_numeric(out.get("model_p_over"), errors="coerce")
    p_under = pd.to_numeric(out.get("model_p_under"), errors="coerce")
    market_over = pd.to_numeric(out.get("market_no_vig_p_over"), errors="coerce")
    market_under = pd.to_numeric(out.get("market_no_vig_p_under"), errors="coerce")
    out["model_edge_over_vs_market"] = (p_over - market_over).where(market_over.notna(), np.nan)
    out["model_edge_under_vs_market"] = (p_under - market_under).where(market_under.notna(), np.nan)
    out["ev_over_per_1_unit"] = [
        expected_profit_per_unit(_number_or_none(prob), _number_or_none(odds))
        for prob, odds in zip(p_over, out["odds_over_american"])
    ]
    out["ev_under_per_1_unit"] = [
        expected_profit_per_unit(_number_or_none(prob), _number_or_none(odds))
        for prob, odds in zip(p_under, out["odds_under_american"])
    ]
    out, shortlist = _decorate_and_rank_candidates(
        out, min_edge=args.min_edge, min_ev=args.min_ev,
        min_probability=args.min_probability, max_data_age_days=args.max_data_age_days,
        max_market_age_hours=args.max_market_age_hours,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    args.shortlist_output.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(args.shortlist_output, index=False)
    with_line = int(out["line"].notna().sum())
    with_odds = int(out[["odds_over_american", "odds_under_american"]].notna().all(axis=1).sum())
    log.info("wrote %s scored rows to %s", len(out), args.output)
    log.info("rows with a user-supplied line: %s; rows with both odds prices: %s", with_line, with_odds)
    log.info("wrote %s qualified shortlist candidates to %s", len(shortlist), args.shortlist_output)
    if args.lines is None:
        print(f"No prop lines supplied; projections saved to {args.output}. Supply actual lines and both side prices with --lines to score market edges.")
    elif shortlist.empty:
        print("No lines met the shortlist filters. No pick is being recommended.")
        print(out[["player_name", "target", "line", "shortlist_status", "shortlist_reason"]].head(args.top).to_string(index=False))
        print(f"Full details: {args.output}; shortlist: {args.shortlist_output}")
    else:
        cols = ["rank", "player_name", "team", "match_date", "opponent_team", "target", "line", "recommended_side", "selected_model_probability", "selected_market_no_vig_probability", "selected_model_edge", "selected_ev_per_1_unit", "data_age_days", "bookmaker"]
        print("TOP MODELED CANDIDATES — verify current markets; results are experimental.")
        print(shortlist[cols].head(args.top).to_string(index=False))
        print(f"Qualified shortlist: {args.shortlist_output}; full audit: {args.output}")


if __name__ == "__main__":
    main()
