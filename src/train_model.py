"""Train points/rebounds/assists models on league-wide historical player logs."""
from __future__ import annotations

import logging
import pandas as pd

from src.data.nba_client import get_league_game_log
from src.data.seasons import completed_seasons
from src.features.build import build_player_features, build_team_game_metrics
from src.model.train import train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train_model")
SEASONS = completed_seasons(count=2)
REQUIRED_CONTEXT_FEATURES = (
    "last5_MIN_mean",
    "last5_FGA_mean",
    "last5_FTA_mean",
    "last5_TOV_mean",
    "last5_usage_proxy_mean",
    "last5_usage_proxy_per_minute_mean",
)
MIN_CONTEXT_COVERAGE = 0.80


def validate_context_feature_coverage(features_df: pd.DataFrame, target: str) -> None:
    """Fail instead of training a model while new workload features are unavailable."""
    problems = []
    for column in REQUIRED_CONTEXT_FEATURES:
        if column not in features_df.columns:
            problems.append(f"{column}=missing column")
            continue
        coverage = float(features_df[column].notna().mean()) if len(features_df) else 0.0
        if coverage < MIN_CONTEXT_COVERAGE:
            problems.append(f"{column}={coverage:.1%} non-missing")
    if problems:
        raise RuntimeError(
            f"{target} training data lacks Phase 2 minutes/usage context "
            f"(need at least {MIN_CONTEXT_COVERAGE:.0%} coverage): "
            + ", ".join(problems)
        )


def assemble_training_frame(seasons: list[str] | None = None) -> pd.DataFrame:
    """Fetch all-player regular-season game logs and stack them safely."""
    frames: list[pd.DataFrame] = []
    for season in (seasons or SEASONS):
        log.info("fetching league-wide player game logs for %s", season)
        try:
            df = get_league_game_log(season)
        except Exception as exc:
            log.warning("skip season %s: %s", season, exc)
            continue
        if df.empty:
            log.warning("skip season %s: endpoint returned no games", season)
            continue
        frames.append(df)
        log.info("loaded %s player-game rows from %s", len(df), season)
    if not frames:
        raise RuntimeError("No NBA game data fetched. Check internet access to stats.nba.com and try again later.")
    games = pd.concat(frames, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")
    games["PLAYER_ID"] = pd.to_numeric(games["PLAYER_ID"], errors="coerce")
    games = games.dropna(subset=["GAME_DATE", "PLAYER_ID"])
    if "GAME_ID" in games.columns:
        games = games.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last")
    return games.sort_values(["GAME_DATE", "PLAYER_ID"], kind="stable").reset_index(drop=True)


def main() -> None:
    log.info("training seasons: %s", ", ".join(SEASONS))
    games = assemble_training_frame()
    log.info("assembled %s rows across %s distinct players", len(games), games["PLAYER_ID"].nunique())
    team_metrics = build_team_game_metrics(games)
    if team_metrics.empty:
        raise RuntimeError(
            "Could not build time-aware opponent metrics. Confirm game logs include TEAM_ID, "
            "OPPONENT_TEAM_ID, PTS, FGA, FTA, OREB and TOV; refusing to train with fixed defaults."
        )
    context_stds = team_metrics[["DEF_RATING", "PACE"]].std(ddof=0)
    if context_stds.isna().any() or (context_stds <= 1e-6).any():
        raise RuntimeError(
            "Opponent metrics have no usable variation; check the source logs before training."
        )
    log.info(
        "built time-aware opponent metrics from %s team-game rows across %s teams",
        len(team_metrics), team_metrics["TEAM_ID"].nunique(),
    )
    trained_targets = 0
    for target in ("PTS", "REB", "AST"):
        feats = build_player_features(games, target=target, opponent_team_metrics=team_metrics)
        validate_context_feature_coverage(feats, target)
        feature_context_stds = feats[["opp_def_rating", "opp_pace"]].std(ddof=0)
        if feature_context_stds.isna().any() or (feature_context_stds <= 1e-6).any():
            raise RuntimeError(
                f"{target} features did not receive varying opponent context. "
                "Refusing to train an artifact that would ignore matchup features."
            )
        log.info(
            "target=%s opponent feature stds: defense=%.3f estimated_pace=%.3f",
            target, feature_context_stds["opp_def_rating"], feature_context_stds["opp_pace"],
        )
        if len(feats) < 50:
            log.warning("not enough training rows for %s (%s); skipping", target, len(feats))
            continue
        log.info("training target=%s on %s player-game rows", target, len(feats))
        train(feats, target=target)
        trained_targets += 1
    if trained_targets != 3:
        raise RuntimeError(f"Only {trained_targets}/3 target models were trained; see warnings above.")
    log.info("done. artifacts saved to artifacts/")


if __name__ == "__main__":
    main()
