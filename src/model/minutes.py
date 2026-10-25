"""Player-specific playing-time distributions from prior NBA game logs.

The estimator is deliberately conservative: it distinguishes recent role changes
from stable roles, blends seasons early in the year, and exposes uncertainty
instead of fabricating a confirmed rotation. Current starter status is supplied
by a timestamped lineup snapshot when available.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

_SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}
_MAX_REGULATION_MINUTES = 48.0


def parse_minutes(values: pd.Series) -> pd.Series:
    """Parse decimal minutes or MM:SS values to floating-point minutes."""
    numeric = pd.to_numeric(values, errors="coerce")
    missing = numeric.isna() & values.notna()
    if missing.any():
        pieces = values.astype(str).str.extract(r"^\s*(\d+):(\d{1,2})(?:\.(\d+))?\s*$")
        parsed = pd.to_numeric(pieces[0], errors="coerce") + pd.to_numeric(pieces[1], errors="coerce") / 60.0
        numeric.loc[missing] = parsed.loc[missing]
    return numeric.where(numeric.between(0, 65))


def _weighted_mean(values: np.ndarray, decay: float = 0.88) -> float:
    if len(values) == 0:
        return float("nan")
    weights = np.power(decay, np.arange(len(values) - 1, -1, -1, dtype=float))
    return float(np.average(values, weights=weights))


def _robust_std(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 4.5
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust = 1.4826 * mad
    ordinary = float(np.std(values, ddof=1))
    # Blend robust and ordinary dispersion: retain real rotation variability
    # without letting one overtime/early foul-trouble game dominate everything.
    return float(max(2.0, 0.65 * robust + 0.35 * ordinary))


def _season_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class MinutesProjection:
    mean: float | None
    std: float | None
    p10: float | None
    p50: float | None
    p90: float | None
    last5_mean: float | None
    last10_mean: float | None
    last20_mean: float | None
    season_mean: float | None
    previous_season_mean: float | None
    trend_last5_vs_20: float | None
    sample_size: int
    current_season_games: int
    role_change_detected: bool
    stable: bool
    lineup_status: str
    injury_status: str
    projection_status: str
    uncertainty_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_minutes(
    player_games: pd.DataFrame,
    *,
    target_team: str = "",
    match_date: str | pd.Timestamp | None = None,
    current_season: str | None = None,
    lineup_status: str = "UNKNOWN",
    injury_status: str = "UNKNOWN",
    minutes_limit: float | None = None,
) -> MinutesProjection:
    """Estimate expected minutes and a distribution using only pregame history.

    For fewer than 15 games in the current season, the baseline is a 70/30 blend
    of the previous-season and current-season means when both exist. A detected
    recent role shift prioritizes the latest observed rotation while widening the
    distribution. No arbitrary fixed-minute starter assumption is used.
    """
    empty = MinutesProjection(
        None, None, None, None, None, None, None, None, None, None, None,
        0, 0, False, False, lineup_status.upper(), injury_status.upper(),
        "INSUFFICIENT_MINUTES_HISTORY", "No usable pregame minutes history.",
    )
    if player_games.empty or "MIN" not in player_games.columns or "GAME_DATE" not in player_games.columns:
        return empty

    frame = player_games.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"], errors="coerce")
    frame["MIN"] = parse_minutes(frame["MIN"])
    frame = frame.dropna(subset=["GAME_DATE", "MIN"]).sort_values("GAME_DATE", kind="stable")
    if match_date is not None:
        cutoff = pd.to_datetime(match_date, errors="coerce")
        if pd.isna(cutoff):
            raise ValueError("match_date must be a valid date")
        frame = frame.loc[frame["GAME_DATE"].dt.normalize() < cutoff.normalize()]
    frame = frame.loc[frame["MIN"] > 0].copy()

    target_team = str(target_team or "").strip().upper()
    team_changed = False
    if target_team and "TEAM_ABBREVIATION" in frame.columns:
        team_col = frame["TEAM_ABBREVIATION"].fillna("").astype(str).str.strip().str.upper()
        same_team = frame.loc[team_col.eq(target_team)]
        if len(same_team) > 0:
            # Rebuild recent role estimates from this team's games only, even
            # for a player with just one or two games after a trade. Keep the
            # wider frame separately available only as a conservative fallback.
            team_changed = len(same_team) < len(frame)
            frame = same_team.copy()
        elif len(frame) > 0 and team_col.iloc[-1] != target_team:
            team_changed = True

    if frame.empty:
        return empty

    all_values = frame["MIN"].to_numpy(dtype=float)
    last5 = all_values[-5:]
    last10 = all_values[-10:]
    last20 = all_values[-20:]
    last5_mean = float(np.mean(last5))
    last10_mean = float(np.mean(last10))
    last20_mean = float(np.mean(last20))
    trend = last5_mean - last20_mean
    role_change = bool(
        team_changed
        or (
            len(last5) >= 3
            and (abs(trend) >= 4.5 or (last20_mean > 0 and abs(last5_mean / last20_mean - 1.0) >= 0.20))
        )
    )

    current_values = np.array([], dtype=float)
    previous_values = np.array([], dtype=float)
    current_games = 0
    if "SEASON_YEAR" in frame.columns:
        season_keys = frame["SEASON_YEAR"].map(_season_key)
        requested = _season_key(current_season)
        unique = list(dict.fromkeys([value for value in season_keys.tolist() if value]))
        if requested and requested in set(unique):
            current_key = requested
        elif unique:
            current_key = unique[-1]
        else:
            current_key = ""
        if current_key:
            current_values = frame.loc[season_keys.eq(current_key), "MIN"].to_numpy(dtype=float)
            previous_keys = [key for key in unique if key != current_key]
            if previous_keys:
                previous_values = frame.loc[season_keys.eq(previous_keys[-1]), "MIN"].to_numpy(dtype=float)
        current_games = len(current_values)
    if len(current_values) == 0:
        current_values = last20.copy()
        current_games = len(current_values)
    current_season_mean = float(np.mean(current_values)) if len(current_values) else last20_mean
    previous_mean = float(np.mean(previous_values)) if len(previous_values) else float("nan")

    # In the opening weeks, previous-season role is the dominant prior.
    early_season = (
        current_games < 15
        and len(previous_values) >= 5
        and math.isfinite(previous_mean)
        and not role_change
    )
    if role_change:
        # New role signal: use observed recent minutes, but preserve some
        # longer-run stabilizer and materially increase uncertainty.
        prior = previous_mean if math.isfinite(previous_mean) else last20_mean
        mean = 0.60 * _weighted_mean(last5) + 0.25 * _weighted_mean(last10) + 0.15 * prior
        projection_status = "ROLE_CHANGE_ESTIMATE"
    elif early_season:
        mean = 0.70 * previous_mean + 0.30 * current_season_mean
        projection_status = "EARLY_SEASON_PRIOR_BLEND"
    else:
        # Weighted recency plus longer-window shrinkage reduces sensitivity to
        # one or two anomalous games.
        baseline = current_season_mean if current_games >= 15 else last20_mean
        mean = 0.45 * _weighted_mean(last5) + 0.30 * _weighted_mean(last10) + 0.25 * baseline
        projection_status = "HISTORY_SUPPORTED" if len(last20) >= 10 else "LIMITED_HISTORY"

    std = _robust_std(last20)
    if role_change or team_changed:
        std = max(std, 4.5)
    status = str(injury_status or "UNKNOWN").strip().upper()
    lineup = str(lineup_status or "UNKNOWN").strip().upper()
    reason_parts: list[str] = []
    if role_change:
        reason_parts.append(f"recent minutes shifted {trend:+.1f} vs last 20")
        std = max(std, abs(trend) * 0.75)
    if team_changed:
        reason_parts.append("recent team/role context changed")
        std = max(std, 5.0)
    if status in {"QUESTIONABLE", "DOUBTFUL", "UNKNOWN", "STALE", "UNVERIFIED", ""}:
        std = math.sqrt(std * std + (4.0 if status in {"QUESTIONABLE", "DOUBTFUL"} else 2.5) ** 2)
        reason_parts.append(f"availability {status.lower()}")
    if lineup not in {"STARTER", "BENCH"}:
        reason_parts.append("starting role not confirmed")
        std = math.sqrt(std * std + 2.0 ** 2)
    elif role_change:
        reason_parts.append(f"confirmed {lineup.lower()} status with role shift")
    limit = None
    try:
        parsed_limit = float(minutes_limit) if minutes_limit is not None else float("nan")
        if math.isfinite(parsed_limit) and 0 < parsed_limit <= 48:
            limit = parsed_limit
    except (TypeError, ValueError):
        limit = None
    if limit is not None:
        mean = min(mean, limit)
        std = min(std, max(2.0, (limit - mean) / 1.2815515655446004 if limit > mean else 3.0))
        reason_parts.append(f"manual return limit {limit:.0f} min applied")
        projection_status = "RETURN_LIMIT_APPLIED"

    mean = float(np.clip(mean, 0.0, _MAX_REGULATION_MINUTES))
    std = float(np.clip(std, 1.5, 14.0))
    p10 = float(np.clip(mean - 1.2815515655446004 * std, 0.0, 53.0))
    p90 = float(np.clip(mean + 1.2815515655446004 * std, 0.0, 53.0))
    stable = bool(len(last20) >= 10 and std <= 4.0 and not role_change and lineup in {"STARTER", "BENCH"})
    if not reason_parts:
        reason_parts.append("recent/season minutes are reasonably consistent")
    return MinutesProjection(
        mean=round(mean, 2), std=round(std, 2), p10=round(p10, 2),
        p50=round(mean, 2), p90=round(p90, 2),
        last5_mean=round(last5_mean, 2), last10_mean=round(last10_mean, 2),
        last20_mean=round(last20_mean, 2), season_mean=round(current_season_mean, 2),
        previous_season_mean=round(previous_mean, 2) if math.isfinite(previous_mean) else None,
        trend_last5_vs_20=round(trend, 2), sample_size=len(all_values),
        current_season_games=current_games, role_change_detected=role_change,
        stable=stable, lineup_status=lineup, injury_status=status,
        projection_status=projection_status,
        uncertainty_reason="; ".join(reason_parts),
    )
