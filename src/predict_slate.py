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
from src.model.minutes import project_minutes
from src.model.prop_probability import (
    calibrate_toward_market,
    exceeds_probability_cap,
    quarter_kelly_fraction,
)
from src.model.train import TrainedModel
from src.market_comparison import (
    compare_market_rows,
    consolidate_same_line_markets,
    no_vig_probabilities,
)
from src.availability import attach_availability, DEFAULT_AVAILABILITY_PATH, DEFAULT_MAX_AGE_HOURS
from src.lineups import attach_lineups, DEFAULT_LINEUP_PATH, DEFAULT_LINEUP_MAX_AGE_HOURS

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


def expected_profit_per_unit(
    probability: float | None,
    odds: float | int | None,
    *,
    p_push: float = 0.0,
) -> float | None:
    """Expected net profit per unit stake, including zero-profit pushes.

    probability is the unconditional chance of winning; p_push is the
    unconditional push chance. The remaining probability mass is a loss.
    """
    decimal = american_to_decimal(odds)
    if probability is None or decimal is None:
        return None
    try:
        p_win = float(probability)
        push = float(p_push)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(p_win) or not np.isfinite(push) or p_win < 0 or push < 0 or p_win + push > 1.000001:
        return None
    p_loss = max(0.0, 1.0 - p_win - push)
    return p_win * (decimal - 1.0) - p_loss


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

def _build_minutes_context(
    games: pd.DataFrame,
    active_players: dict[int, str],
    player_teams: dict[int, dict[str, Any]],
    next_games: dict[str, dict[str, str]],
    team_id_by_abbreviation: dict[str, int],
    *,
    availability_file: Path,
    lineups_file: Path,
    max_availability_age_hours: float,
    max_lineup_age_hours: float,
) -> dict[int, dict[str, Any]]:
    """Build per-player minutes distributions using status/lineup snapshots for the scheduled date."""
    scheduled_rows: list[dict[str, Any]] = []
    for player_id, player_name in active_players.items():
        match = _next_match_for_player(
            player_id, player_teams, next_games,
            team_id_by_abbreviation=team_id_by_abbreviation,
        )
        match_date = pd.to_datetime(match.get("match_date"), errors="coerce")
        if pd.isna(match_date):
            continue
        scheduled_rows.append({
            "player_id": player_id,
            "player_name": player_name,
            "team": match.get("team") or player_teams.get(player_id, {}).get("team", ""),
            "team_abbreviation": match.get("team_abbreviation") or player_teams.get(player_id, {}).get("team_abbreviation", ""),
            "match_date": match_date.date().isoformat(),
            "opponent_team": match.get("opponent_team", ""),
            "opponent_abbreviation": match.get("opponent_abbreviation", ""),
        })
    if not scheduled_rows:
        log.warning("No schedulable match dates; minutes context cannot be built.")
        return {}

    schedule = pd.DataFrame(scheduled_rows)
    schedule = attach_availability(
        schedule, availability_file, max_age_hours=max_availability_age_hours,
    )
    schedule = attach_lineups(
        schedule, lineups_file, max_age_hours=max_lineup_age_hours,
    )
    game_groups = {
        int(pid): group.sort_values("GAME_DATE", kind="stable").copy()
        for pid, group in games.groupby("PLAYER_ID", sort=False)
    }
    result: dict[int, dict[str, Any]] = {}
    for row in schedule.to_dict(orient="records"):
        try:
            player_id = int(row["player_id"])
        except (TypeError, ValueError):
            continue
        history = game_groups.get(player_id, pd.DataFrame())
        availability_gate = str(row.get("availability_gate") or "UNVERIFIED").upper()
        lineup_gate = str(row.get("lineup_gate") or "UNVERIFIED").upper()
        injury_status = str(row.get("availability_status") or "UNKNOWN").upper()
        if availability_gate not in {"CLEAR", "REVIEW", "BLOCKED"}:
            injury_status = availability_gate
        lineup_status = str(row.get("lineup_status") or "UNKNOWN").upper()
        if lineup_gate != "CLEAR":
            lineup_status = "UNKNOWN"
        return_limit = row.get("availability_minutes_limit")
        if availability_gate not in {"CLEAR", "REVIEW"}:
            return_limit = None
        projection = project_minutes(
            history,
            target_team=str(row.get("team_abbreviation") or ""),
            match_date=row.get("match_date"),
            current_season=current_season(),
            lineup_status=lineup_status,
            injury_status=injury_status,
            minutes_limit=return_limit,
        )
        info: dict[str, Any] = {
            "minutes_projection_mean": projection.mean,
            "minutes_projection_std": projection.std,
            "minutes_projection_p10": projection.p10,
            "minutes_projection_p50": projection.p50,
            "minutes_projection_p90": projection.p90,
            "minutes_last5_mean": projection.last5_mean,
            "minutes_last10_mean": projection.last10_mean,
            "minutes_last20_mean": projection.last20_mean,
            "minutes_season_mean": projection.season_mean,
            "minutes_previous_season_mean": projection.previous_season_mean,
            "minutes_trend_last5_vs_20": projection.trend_last5_vs_20,
            "minutes_sample_size": projection.sample_size,
            "minutes_current_season_games": projection.current_season_games,
            "minutes_role_change_detected": projection.role_change_detected,
            "minutes_stable": projection.stable,
            "minutes_projection_status": projection.projection_status,
            "minutes_uncertainty_reason": projection.uncertainty_reason,
            "availability_status": row.get("availability_status", "UNKNOWN"),
            "availability_gate": availability_gate,
            "availability_reason": row.get("availability_reason", ""),
            "availability_source": row.get("availability_source", ""),
            "availability_reported_at_utc": row.get("availability_reported_at_utc", ""),
            "availability_age_hours": row.get("availability_age_hours"),
            "availability_retrieved_at_utc": row.get("availability_retrieved_at_utc", ""),
            "minutes_limit": return_limit,
            "lineup_status": lineup_status,
            "lineup_gate": lineup_gate,
            "lineup_reason": row.get("lineup_reason", ""),
            "lineup_source": row.get("lineup_source", ""),
            "lineup_reported_at_utc": row.get("lineup_reported_at_utc", ""),
            "lineup_age_hours": row.get("lineup_age_hours"),
            "minutes_role_confirmed": bool(lineup_gate == "CLEAR" and lineup_status in {"STARTER", "BENCH"}),
        }
        # Only one schedule row per active player is expected. If duplicate roster
        # entries occur, prefer the most recently reported data and deterministic ID.
        result[player_id] = info
    log.info(
        "built minutes distributions for %s scheduled players; %s have verified starter/bench snapshots",
        len(result), sum(bool(row.get("minutes_role_confirmed")) for row in result.values()),
    )
    return result


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
    p_push = _number_or_none(pred.p_push) or 0.0
    no_vig_over, no_vig_under, no_vig_method = no_vig_probabilities(
        odds_over,
        odds_under,
        bookmaker_over=market_info.get("bookmaker_over"),
        bookmaker_under=market_info.get("bookmaker_under"),
        bookmaker=bookmaker,
    )
    over_ev = expected_profit_per_unit(p_over * (1.0 - p_push), odds_over, p_push=p_push) if p_over is not None else None
    under_ev = expected_profit_per_unit(p_under * (1.0 - p_push), odds_under, p_push=p_push) if p_under is not None else None
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
        "model_p_over_raw": round(p_over, 4) if p_over is not None else None,
        "model_p_under_raw": round(p_under, 4) if p_under is not None else None,
        "minutes_projection_mean": _number_or_none(market_info.get("minutes_projection_mean")),
        "minutes_projection_std": _number_or_none(market_info.get("minutes_projection_std")),
        "minutes_projection_p10": _number_or_none(market_info.get("minutes_projection_p10")),
        "minutes_projection_p50": _number_or_none(market_info.get("minutes_projection_p50")),
        "minutes_projection_p90": _number_or_none(market_info.get("minutes_projection_p90")),
        "minutes_last5_mean": _number_or_none(market_info.get("minutes_last5_mean")),
        "minutes_last10_mean": _number_or_none(market_info.get("minutes_last10_mean")),
        "minutes_last20_mean": _number_or_none(market_info.get("minutes_last20_mean")),
        "minutes_season_mean": _number_or_none(market_info.get("minutes_season_mean")),
        "minutes_previous_season_mean": _number_or_none(market_info.get("minutes_previous_season_mean")),
        "minutes_trend_last5_vs_20": _number_or_none(market_info.get("minutes_trend_last5_vs_20")),
        "minutes_sample_size": _number_or_none(market_info.get("minutes_sample_size")),
        "minutes_current_season_games": _number_or_none(market_info.get("minutes_current_season_games")),
        "minutes_role_change_detected": bool(market_info.get("minutes_role_change_detected", False)),
        "minutes_stable": bool(market_info.get("minutes_stable", False)),
        "minutes_uncertainty_reason": str(market_info.get("minutes_uncertainty_reason") or ""),
        "lineup_status": str(market_info.get("lineup_status") or "UNKNOWN"),
        "lineup_gate": str(market_info.get("lineup_gate") or "UNVERIFIED"),
        "lineup_reason": str(market_info.get("lineup_reason") or ""),
        "lineup_source": str(market_info.get("lineup_source") or ""),
        "lineup_reported_at_utc": str(market_info.get("lineup_reported_at_utc") or ""),
        "lineup_age_hours": _number_or_none(market_info.get("lineup_age_hours")),
        "availability_retrieved_at_utc": str(market_info.get("availability_retrieved_at_utc") or ""),
        "availability_minutes_limit": _number_or_none(market_info.get("availability_minutes_limit")),
        "minutes_role_confirmed": str(market_info.get("minutes_role_confirmed") or "").strip().lower() in {"1", "true", "yes", "confirmed"} or market_info.get("minutes_role_confirmed") is True,
        "calibrated_p_over": None,
        "calibrated_p_under": None,
        "p_push": round(p_push, 5) if pred.p_push is not None else None,
        "probability_distribution": pred.probability_distribution or "",
        "minutes_projection_status": str(market_info.get("minutes_projection_status") or "ROLLING_PROXY_ONLY"),
        "minutes_role_confirmed": str(market_info.get("minutes_role_confirmed") or "").strip().lower() in {"1", "true", "yes", "confirmed"},
        "edge_explanation": str(market_info.get("edge_explanation") or "").strip(),
        "movement_explanation": str(market_info.get("movement_explanation") or "").strip(),
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
        "opening_line": _number_or_none(market_info.get("opening_line")),
        "source_line_move": _number_or_none(market_info.get("source_line_move")),
        "line_move_from_open": _number_or_none(market_info.get("line_move_from_open")),
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


def _calibrate_market_probabilities(out: pd.DataFrame) -> pd.DataFrame:
    """Create market-anchored probabilities and explicit red-flag gates.

    Scored probabilities are conditional on a decisive result (pushes excluded).
    Without a compatible two-sided no-vig market reference, the probability is
    withheld from the scored output rather than treating an uncalibrated value
    as decision-grade.
    """
    result = out.copy()
    raw_over_values: list[float | None] = []
    raw_under_values: list[float | None] = []
    calibrated_over_values: list[float | None] = []
    calibrated_under_values: list[float | None] = []
    weights: list[float | None] = []
    caps: list[float | None] = []
    gates: list[str] = []
    confidence_grades: list[str] = []
    main_risks: list[str] = []
    edge_over_values: list[float | None] = []
    edge_under_values: list[float | None] = []
    ev_over_values: list[float | None] = []
    ev_under_values: list[float | None] = []

    for row in result.to_dict(orient="records"):
        raw_over = _number_or_none(row.get("model_p_over_raw", row.get("model_p_over")))
        raw_under = _number_or_none(row.get("model_p_under_raw", row.get("model_p_under")))
        market_over = _number_or_none(row.get("market_no_vig_p_over"))
        market_under = _number_or_none(row.get("market_no_vig_p_under"))
        if market_under is None and market_over is not None:
            market_under = 1.0 - market_over
        push = _number_or_none(row.get("p_push"))
        push = max(0.0, min(1.0, push or 0.0))
        availability_gate = str(row.get("availability_gate") or "UNVERIFIED").strip().upper()
        minutes_status = str(row.get("minutes_projection_status") or "ROLLING_PROXY_ONLY").strip().upper()
        minutes_confirmed = str(row.get("minutes_role_confirmed") or "").strip().lower() in {"1", "true", "yes", "confirmed"}
        minutes_reliable = minutes_status == "VERIFIED_DISTRIBUTION" and minutes_confirmed
        uncertain = availability_gate != "CLEAR" or not minutes_reliable
        cap = 0.65 if uncertain else 0.72
        weight = 0.50 if not uncertain else 0.25
        calibrated: float | None = None
        gate = "INSUFFICIENT_DATA"

        if raw_over is not None and raw_under is not None and market_over is not None:
            raw_total = raw_over + raw_under
            raw_conditional_over = raw_over / raw_total if raw_total > 1e-9 else raw_over
            raw_flag = exceeds_probability_cap(raw_conditional_over, uncertain=uncertain)
            calibrated = calibrate_toward_market(
                raw_conditional_over, market_over, model_weight=weight
            )
            calibrated_flag = exceeds_probability_cap(calibrated, uncertain=uncertain)
            if raw_flag:
                gate = "RAW_MODEL_CAP_RED_FLAG"
            elif calibrated_flag:
                gate = "CALIBRATED_CAP_RED_FLAG"
            else:
                gate = "CALIBRATED"

        raw_over_values.append(raw_over)
        raw_under_values.append(raw_under)
        calibrated_over_values.append(calibrated)
        calibrated_under_values.append((1.0 - calibrated) if calibrated is not None else None)
        weights.append(weight if calibrated is not None else None)
        caps.append(cap if raw_over is not None else None)
        gates.append(gate)
        confidence_grades.append("B" if gate == "CALIBRATED" and not uncertain else "C")
        main_risks.append(
            "Minutes are rolling-proxy estimates, not a distribution; confirm current rotation, starter status, and injury limits."
            if not minutes_reliable else
            "Check late lineup changes and closing market movement."
        )
        edge_over_values.append(calibrated - market_over if calibrated is not None and market_over is not None else None)
        edge_under_values.append((1.0 - calibrated) - market_under if calibrated is not None and market_under is not None else None)

        odds_over = _number_or_none(row.get("odds_over_american"))
        odds_under = _number_or_none(row.get("odds_under_american"))
        ev_over_values.append(
            expected_profit_per_unit(calibrated * (1.0 - push), odds_over, p_push=push)
            if calibrated is not None else None
        )
        ev_under_values.append(
            expected_profit_per_unit((1.0 - calibrated) * (1.0 - push), odds_under, p_push=push)
            if calibrated is not None else None
        )

    result["model_p_over_raw"] = raw_over_values
    result["model_p_under_raw"] = raw_under_values
    result["calibrated_p_over"] = calibrated_over_values
    result["calibrated_p_under"] = calibrated_under_values
    # model_p_* now mean calibrated P(side | no push); raw outputs remain for audit.
    result["model_p_over"] = calibrated_over_values
    result["model_p_under"] = calibrated_under_values
    result["p_over"] = calibrated_over_values
    result["p_under"] = calibrated_under_values
    result["calibration_model_weight"] = weights
    result["probability_cap"] = caps
    result["probability_gate_status"] = gates
    result["confidence_grade"] = confidence_grades
    result["main_risk"] = main_risks
    result["model_edge_over_vs_market"] = edge_over_values
    result["model_edge_under_vs_market"] = edge_under_values
    result["ev_over_per_1_unit"] = ev_over_values
    result["ev_under_per_1_unit"] = ev_under_values
    result["calibration_method"] = "market_shrinkage; pushes excluded from conditional probability"
    return result


def _documented_reason(value: Any) -> bool:
    text = "" if value is None or pd.isna(value) else str(value).strip().lower()
    return len(text) >= 20 and any(
        term in text for term in ("injury", "role", "minutes", "matchup", "starter", "return", "out")
    )


def _decorate_and_rank_candidates(
    out: pd.DataFrame,
    *,
    min_edge: float = 0.04,
    min_ev: float = 0.0,
    min_probability: float = 0.53,
    max_data_age_days: int = 14,
    max_market_age_hours: float = 2.0,
    require_verified_availability: bool = False,
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
        availability_gate = str(row.get("availability_gate") or "UNVERIFIED").strip().upper()
        availability_reason = str(row.get("availability_reason") or "").strip()
        if availability_gate == "BLOCKED":
            status = "AVAILABILITY_BLOCKED"
            reason = "Current matched player-status report says unavailable. " + availability_reason
            side = prob = market_prob = edge = ev = None
        elif availability_gate == "REVIEW":
            status = "AVAILABILITY_REVIEW"
            reason = "Player status is questionable/doubtful; verify near tip-off. " + availability_reason
            side = prob = market_prob = edge = ev = None
        elif require_verified_availability and availability_gate == "STALE":
            status = "AVAILABILITY_STALE"
            reason = "Verified availability is stale; refresh the report before shortlisting. " + availability_reason
            side = prob = market_prob = edge = ev = None
        elif require_verified_availability and availability_gate != "CLEAR":
            status = "AVAILABILITY_UNVERIFIED"
            reason = "Strict availability mode requires a fresh Available/Probable report. " + availability_reason
            side = prob = market_prob = edge = ev = None
        elif row.get("line") is None or pd.isna(row.get("line")):
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
        elif str(row.get("probability_gate_status") or "").upper() == "INSUFFICIENT_DATA":
            status, reason = "INSUFFICIENT_DATA", "A compatible current two-sided no-vig market reference is missing; no calibrated probability is released."
            side = prob = market_prob = edge = ev = None
        elif str(row.get("probability_gate_status") or "").upper() in {"RAW_MODEL_CAP_RED_FLAG", "CALIBRATED_CAP_RED_FLAG"}:
            status, reason = "PROBABILITY_CAP_REVIEW", "Model probability crosses the uncertainty-adjusted hard cap; investigate missing minutes, role, injury, or market information before betting."
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
                movement = _number_or_none(row.get("line_move_from_open"))
                if movement is None:
                    movement = _number_or_none(row.get("line_move_from_previous"))
                movement_explanation = str(row.get("movement_explanation") or "").strip()
                edge_explanation = row.get("edge_explanation")
                prior_count = _number_or_none(row.get("market_prior_snapshot_count"))
                if abs(edge) > 0.10 and not _documented_reason(edge_explanation):
                    status = "EDGE_RED_FLAG"
                    reason = f"Calibrated edge is {edge:.1%}, above the 10-point review threshold; a specific documented injury/role/minutes/matchup explanation is required."
                elif is_public_snapshot and (prior_count is None or prior_count < 1) and not movement_explanation:
                    status = "LINE_MOVEMENT_UNVERIFIED"
                    reason = "No prior same-market snapshot exists, so movement against this side cannot be checked. Refresh the board again before considering the prop."
                elif movement is not None and (
                    (side == "OVER" and movement > 0.5) or (side == "UNDER" and movement < -0.5)
                ) and len(movement_explanation) < 15:
                    status = "LINE_MOVED_AGAINST_PICK"
                    reason = f"Line movement ({movement:+.1f}) is more than half a point against {side}; document a specific reason or pass."
                elif edge >= min_edge and ev > max(0.0, min_ev) and prob >= min_probability:
                    status = "SHORTLIST_CANDIDATE"
                    reason = f"Calibrated {side}: edge {edge:.1%}, EV {ev:.3f} units per unit staked; minutes remain proxy-based and confidence is capped."
                    if availability_gate != "CLEAR":
                        reason += f" Availability is {availability_gate}; a fresh official report is required. {availability_reason}"
                else:
                    status = "BELOW_THRESHOLDS"
                    reason = f"Best modeled side {side} missed filters (edge >= {min_edge:.1%}, EV > {max(0.0, min_ev):.3f}, probability >= {min_probability:.1%})."
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
    )
    # No duplicate player-game exposure, max two picks per game, quarter-Kelly
    # per pick (<=2% bankroll), and <=5% total daily bankroll exposure.
    accepted: list[dict[str, Any]] = []
    game_counts: dict[str, int] = {}
    player_game_seen: set[str] = set()
    daily_stake = 0.0
    for row_index, row in shortlist.iterrows():
        match_date = str(row.get("match_date") or "").strip()
        team = str(row.get("team_abbreviation") or row.get("team") or "").strip().upper()
        opponent = str(row.get("opponent_abbreviation") or row.get("opponent_team") or "").strip().upper()
        if not match_date or not team or not opponent:
            result.at[row_index, "shortlist_status"] = "GAME_CONTEXT_MISSING"
            result.at[row_index, "shortlist_reason"] = "Scheduled date, team, and opponent are required to control game-level correlation."
            continue
        game_key = "|".join([match_date, *sorted([team, opponent])])
        player_key = f"{game_key}|{row.get('player_id')}"
        if game_counts.get(game_key, 0) >= 2:
            result.at[row_index, "shortlist_status"] = "GAME_EXPOSURE_LIMIT"
            result.at[row_index, "shortlist_reason"] = "Two higher-ranked props from this game already occupy the game exposure budget."
            continue
        if player_key in player_game_seen:
            result.at[row_index, "shortlist_status"] = "CORRELATED_PLAYER_PROP"
            result.at[row_index, "shortlist_reason"] = "Another prop for this player-game is already selected; avoid stacking correlated outcomes."
            continue
        side = str(row.get("recommended_side") or "").upper()
        probability = _number_or_none(row.get("selected_model_probability"))
        odds = _number_or_none(row.get("odds_over_american" if side == "OVER" else "odds_under_american"))
        stake = quarter_kelly_fraction(probability, odds, max_fraction=0.02) if probability is not None else 0.0
        stake = min(stake, max(0.0, 0.05 - daily_stake))
        if stake <= 1e-6:
            result.at[row_index, "shortlist_status"] = "DAILY_EXPOSURE_LIMIT"
            result.at[row_index, "shortlist_reason"] = "Quarter-Kelly stake is zero or the 5% daily bankroll exposure budget is used."
            continue
        accepted_row = row.to_dict()
        accepted_row["suggested_stake_fraction"] = round(stake, 5)
        accepted_row["suggested_stake_percent"] = round(stake * 100.0, 2)
        accepted_row["daily_exposure_after_pick_percent"] = round((daily_stake + stake) * 100.0, 2)
        accepted.append(accepted_row)
        game_counts[game_key] = game_counts.get(game_key, 0) + 1
        player_game_seen.add(player_key)
        daily_stake += stake
    shortlist = pd.DataFrame(accepted)
    if not shortlist.empty:
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
    parser.add_argument("--min-edge", type=float, default=0.04)
    parser.add_argument("--min-ev", type=float, default=0.0, help="minimum expected value; candidates must remain strictly positive EV")
    parser.add_argument("--min-probability", type=float, default=0.53)
    parser.add_argument("--max-data-age-days", type=int, default=14)
    parser.add_argument("--max-market-age-hours", type=float, default=2.0, help="reject market snapshots older than this many hours")
    parser.add_argument("--availability-file", type=Path, default=DEFAULT_AVAILABILITY_PATH, help="CSV of verified player statuses by player ID and scheduled match date")
    parser.add_argument("--max-availability-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS, help="treat older player status reports as stale")
    parser.add_argument("--require-verified-availability", dest="require_verified_availability", action="store_true", default=True, help="require a fresh Available/Probable status (default)")
    parser.add_argument("--allow-unverified-availability", dest="require_verified_availability", action="store_false", help="relax availability gate for research only; never treat as confirmed")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")
    if not 0 <= args.min_edge <= 1 or not 0 <= args.min_probability <= 1:
        parser.error("--min-edge and --min-probability must be between 0 and 1")
    if args.min_ev < 0 or args.max_data_age_days < 0 or args.max_market_age_hours < 0 or args.max_availability_age_hours < 0:
        parser.error("--min-ev, --max-data-age-days, --max-market-age-hours, and --max-availability-age-hours must be non-negative")

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
    if not args.availability_file.exists():
        log.warning(
            "No player-availability snapshot found at %s; rows will be marked UNVERIFIED. "
            "Use the documented CSV template or --require-verified-availability for strict screening.",
            args.availability_file,
        )
    out = attach_availability(
        out, args.availability_file, max_age_hours=args.max_availability_age_hours,
    )
    out = _calibrate_market_probabilities(out)
    # Screen after calibration; thresholds and EV use calibrated probabilities.
    out, shortlist = _decorate_and_rank_candidates(
        out, min_edge=args.min_edge, min_ev=args.min_ev,
        min_probability=args.min_probability, max_data_age_days=args.max_data_age_days,
        max_market_age_hours=args.max_market_age_hours,
        require_verified_availability=args.require_verified_availability,
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
        cols = ["rank", "player_name", "team", "match_date", "opponent_team", "target", "line", "recommended_side", "selected_model_probability", "selected_market_no_vig_probability", "selected_model_edge", "selected_ev_per_1_unit", "availability_status", "availability_gate", "availability_age_hours", "data_age_days", "bookmaker"]
        print("TOP MODELED CANDIDATES — verify current markets; results are experimental.")
        print(shortlist[cols].head(args.top).to_string(index=False))
        print(f"Qualified shortlist: {args.shortlist_output}; full audit: {args.output}")


if __name__ == "__main__":
    main()
