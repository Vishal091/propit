"""Chronological calibration and settled-prop backtest.

Input CSV columns:
  game_date,target,p_over,actual_over,odds_over_american,odds_under_american

Alternatively provide actual_stat and line instead of actual_over. Integer-line
pushes are excluded. Probabilities are calibrated on the earliest 70% of dates
and scored only on the later 30%; never use the reported holdout metrics as
proof of profitability without a meaningful sample size.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


def american_to_decimal(odds: float) -> float | None:
    if not math.isfinite(float(odds)) or float(odds) == 0:
        return None
    odds = float(odds)
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def _profit(probability: float, odds: float, outcome: int) -> float:
    decimal = american_to_decimal(odds)
    if decimal is None:
        return float("nan")
    # Net profit per one unit staked.
    return decimal - 1.0 if outcome else -1.0


def prepare_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"game_date", "target", "p_over"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["target"] = df["target"].astype(str).str.upper().str.strip()
    df["p_over"] = pd.to_numeric(df["p_over"], errors="coerce")
    if "actual_over" not in df.columns:
        if not {"actual_stat", "line"}.issubset(df.columns):
            raise ValueError("provide actual_over or both actual_stat and line")
        actual = pd.to_numeric(df["actual_stat"], errors="coerce")
        line = pd.to_numeric(df["line"], errors="coerce")
        df["actual_over"] = np.where(actual == line, np.nan, (actual > line).astype(float))
    df["actual_over"] = pd.to_numeric(df["actual_over"], errors="coerce")
    df = df.dropna(subset=["game_date", "target", "p_over", "actual_over"])
    df = df[df["actual_over"].isin([0, 1]) & df["p_over"].between(0, 1)].copy()
    if "odds_over_american" not in df:
        df["odds_over_american"] = np.nan
    if "odds_under_american" not in df:
        df["odds_under_american"] = np.nan
    return df.sort_values("game_date", kind="stable").reset_index(drop=True)


def _metrics(frame: pd.DataFrame, probability_col: str) -> dict[str, float | int]:
    y = frame["actual_over"].astype(int).to_numpy()
    p = frame[probability_col].clip(1e-6, 1 - 1e-6).to_numpy()
    brier = float(brier_score_loss(y, p))
    loss = float(log_loss(y, p, labels=[0, 1]))
    hit = float(((p >= 0.5) == (y == 1)).mean())
    # Simulate choosing the side with higher positive EV; one unit per wager.
    profits: list[float] = []
    for row, prob, outcome in zip(frame.to_dict("records"), p, y):
        over_odds = row.get("odds_over_american")
        under_odds = row.get("odds_under_american")
        try:
            over_odds, under_odds = float(over_odds), float(under_odds)
        except (TypeError, ValueError):
            continue
        over_dec = american_to_decimal(over_odds)
        under_dec = american_to_decimal(under_odds)
        if over_dec is None or under_dec is None:
            continue
        over_ev = prob * (over_dec - 1) - (1 - prob)
        under_prob = 1 - prob
        under_ev = under_prob * (under_dec - 1) - (1 - under_prob)
        best_ev = max(over_ev, under_ev)
        if best_ev <= 0:
            continue
        if over_ev >= under_ev:
            profits.append(_profit(prob, over_odds, outcome))
        else:
            profits.append(_profit(under_prob, under_odds, 1 - outcome))
    return {
        "n": int(len(frame)),
        "hit_rate": hit,
        "brier": brier,
        "log_loss": loss,
        "bets": len(profits),
        "net_units": float(np.sum(profits)) if profits else 0.0,
        "roi": float(np.mean(profits)) if profits else float("nan"),
    }


def reliability(frame: pd.DataFrame, probability_col: str, n_bins: int = 10) -> pd.DataFrame:
    df = frame.copy()
    df["probability_bin"] = pd.cut(
        df[probability_col].clip(0, 1),
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
    )
    return (
        df.groupby("probability_bin", observed=True)
        .agg(
            count=("actual_over", "size"),
            mean_predicted=(probability_col, "mean"),
            observed_over_rate=("actual_over", "mean"),
        )
        .assign(calibration_gap=lambda x: x["observed_over_rate"] - x["mean_predicted"])
        .reset_index()
    )


def evaluate_group(frame: pd.DataFrame, label: str) -> dict[str, object]:
    dates = sorted(frame["game_date"].dropna().unique())
    if len(dates) < 5:
        raise ValueError(f"{label}: need at least 5 distinct dates; found {len(dates)}")
    split_index = max(1, min(int(len(dates) * 0.7), len(dates) - 1))
    cutoff = dates[split_index]
    train = frame[frame["game_date"] < cutoff].copy()
    test = frame[frame["game_date"] >= cutoff].copy()
    if len(train) < 30 or len(test) < 20:
        raise ValueError(f"{label}: need at least 30 calibration rows and 20 later holdout rows; got {len(train)} and {len(test)}")
    calibrator = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
    calibrator.fit(train["p_over"], train["actual_over"])
    test["calibrated_p_over"] = calibrator.predict(test["p_over"])
    raw = _metrics(test, "p_over")
    calibrated = _metrics(test, "calibrated_p_over")
    return {
        "target": label,
        "train_rows": len(train),
        "holdout_rows": len(test),
        "holdout_start": pd.Timestamp(cutoff).date().isoformat(),
        "raw": raw,
        "calibrated": calibrated,
        "holdout": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV of historical predictions joined to settled results")
    parser.add_argument("--output", type=Path, default=Path("predictions/backtest_holdout.csv"))
    args = parser.parse_args()
    rows = prepare_rows(args.csv)
    if rows.empty:
        raise SystemExit("No valid settled rows found.")
    reports = []
    holdouts = []
    for target, group in rows.groupby("target", sort=True):
        try:
            report = evaluate_group(group, target)
        except ValueError as exc:
            print(f"SKIP {exc}")
            continue
        holdout = report.pop("holdout")
        holdout["calibration_train_end"] = report["holdout_start"]
        holdout["raw_brier"] = report["raw"]["brier"]
        holdout["calibrated_brier"] = report["calibrated"]["brier"]
        holdouts.append(holdout)
        reports.append(report)
    if not reports:
        raise SystemExit("No stat group had enough chronological data for calibration and holdout evaluation.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(holdouts, ignore_index=True).to_csv(args.output, index=False)
    for report in reports:
        print(f"\nTARGET {report['target']} | holdout from {report['holdout_start']}")
        print(f"calibration rows={report['train_rows']} holdout rows={report['holdout_rows']}")
        for kind in ("raw", "calibrated"):
            metric = report[kind]
            roi = metric["roi"]
            roi_text = f"{roi:.1%}" if math.isfinite(roi) else "n/a (no odds/bets)"
            print(
                f"{kind:10s} hit={metric['hit_rate']:.1%} "
                f"Brier={metric['brier']:.4f} logloss={metric['log_loss']:.4f} "
                f"bets={metric['bets']} net={metric['net_units']:+.2f}u ROI={roi_text}"
            )
    print(f"\nHoldout rows with calibrated probabilities: {args.output}")
    print("ROI is a historical simulation, not a guarantee; use only lines/prices actually available before each game.")


if __name__ == "__main__":
    main()
