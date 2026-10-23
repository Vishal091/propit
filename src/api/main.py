"""FastAPI surface for player-stat estimates and explicit supplied-line scoring."""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.nba_client import get_league_game_log
from src.data.seasons import current_season, recent_seasons
from src.features.build import build_player_features
from src.model.predict import predict_one
from src.model.train import TrainedModel

app = FastAPI(
    title="PropIt NBA Prop Research",
    description="Educational NBA player-stat analytics using public NBA statistics. Supply market lines yourself.",
    version="0.2.0",
)


class PredictRequest(BaseModel):
    player_id: int = Field(..., description="stats.nba.com player ID")
    target: str = Field(..., description="One of: PTS, REB, AST")
    line: float | None = Field(default=None, description="Optional user-supplied prop line")
    season: str = Field(default_factory=current_season, description="Latest season, e.g. 2026-27")


class PredictResponse(BaseModel):
    player_id: int
    target: str
    line: float | None
    line_source: str
    p_over: float | None
    expected_value: float
    interval_80: tuple[float, float]
    drivers: list[dict]
    disclaimer: str = (
        "Educational analytics only. Predictions are experimental and not guarantees of accuracy or profit. "
        "The model does not fetch or verify sportsbook odds."
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if req.target not in {"PTS", "REB", "AST"}:
        raise HTTPException(status_code=400, detail="target must be PTS, REB, or AST")
    try:
        frames = []
        for season in recent_seasons(req.season, count=3):
            try:
                frame = get_league_game_log(season)
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise RuntimeError("No game logs returned")
        all_games = pd.concat(frames, ignore_index=True)
        all_games["PLAYER_ID"] = pd.to_numeric(all_games["PLAYER_ID"], errors="coerce")
        games = all_games.loc[all_games["PLAYER_ID"].eq(req.player_id)].copy()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load NBA game data: {exc}") from exc
    if games.empty:
        raise HTTPException(status_code=404, detail="No games found for player in selected/recent seasons")
    try:
        features = build_player_features(games, target=req.target)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if features.empty:
        raise HTTPException(status_code=422, detail="Not enough prior game history to build features")
    try:
        bundle = TrainedModel.load(f"artifacts/model_{req.target}.joblib")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact not found. Run python -m src.train_model first.") from exc
    pred = predict_one(features.iloc[-1], line=req.line, target=req.target, bundle=bundle)
    return PredictResponse(
        player_id=req.player_id, target=pred.target, line=pred.line,
        line_source="user_supplied" if req.line is not None else "not_supplied",
        p_over=round(pred.p_over, 3) if pred.p_over is not None else None,
        expected_value=round(pred.expected_value, 2),
        interval_80=(round(pred.interval_80[0], 2), round(pred.interval_80[1], 2)),
        drivers=pred.drivers,
    )
