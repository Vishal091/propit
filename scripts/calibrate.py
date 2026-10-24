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
import json
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
        df.loc[actual.isna() | line.isna(), "actual_over"] = np.nan
    df["actual_over"] = pd.to_numeric(df["actual_over"], errors="coerce")
    df = df.dropna(subset=["game_date", "target", "p_over", "actual_over"])
    df = df[df["actual_over"].isin([0, 1]) & df["p_over"].between(0, 1)].copy()
    if "odds_over_american" not in df:
        df["odds_over_american"] = np.nan
    if "odds_under_american" not in df:
        df["odds_under_american"] = np.nan
    return df.sort_values("game_date", kind="stable").reset_index(drop=True)


def _timestamped_price_row(row: dict[str, object]) -> bool:
    """Only simulate ROI when prices were archived with a timestamp."""
    stamp = row.get("snapshot_at_utc")
    if stamp is None or not str(stamp).strip():
        return False
    parsed = pd.to_datetime(stamp, utc=True, errors="coerce")
    return bool(pd.notna(parsed))


def _metrics(frame: pd.DataFrame, probability_col: str) -> dict[str, float | int]:
    y = frame["actual_over"].astype(int).to_numpy()
    p = frame[probability_col].clip(1e-6, 1 - 1e-6).to_numpy()
    brier = float(brier_score_loss(y, p))
    loss = float(log_loss(y, p, labels=[0, 1]))
    hit = float(((p >= 0.5) == (y == 1)).mean())
    # Probability quality uses all valid holdout rows. Historical ROI is only
    # simulated for archived rows with timestamped prices and both side prices.
    profits: list[float] = []
    timestamped_rows = 0
    priced_rows = 0
    for row, prob, outcome in zip(frame.to_dict("records"), p, y):
        if not _timestamped_price_row(row):
            continue
        timestamped_rows += 1
        try:
            over_odds = float(row.get("odds_over_american"))
            under_odds = float(row.get("odds_under_american"))
        except (TypeError, ValueError):
            continue
        over_dec = american_to_decimal(over_odds)
        under_dec = american_to_decimal(under_odds)
        if over_dec is None or under_dec is None:
            continue
        priced_rows += 1
        over_ev = prob * (over_dec - 1) - (1 - prob)
        under_prob = 1 - prob
        under_ev = under_prob * (under_dec - 1) - (1 - under_prob)
        if max(over_ev, under_ev) <= 0:
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
        "timestamped_rows": timestamped_rows,
        "priced_rows": priced_rows,
        "bets": len(profits),
        "net_units": float(np.sum(profits)) if profits else 0.0,
        "roi": float(np.mean(profits)) if profits else float("nan"),
    }


def reliability(frame: pd.DataFrame, probability_col: str, n_bins: int = 10) -> pd.DataFrame:
    """Return reliability buckets as numeric, JSON-safe values."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    df = frame.copy()
    edges = np.linspace(0, 1, n_bins + 1)
    df["_bin_index"] = pd.cut(
        df[probability_col].clip(0, 1),
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    grouped = (
        df.dropna(subset=["_bin_index"])
        .groupby("_bin_index", observed=True)
        .agg(
            count=("actual_over", "size"),
            mean_predicted=(probability_col, "mean"),
            observed_over_rate=("actual_over", "mean"),
        )
        .reset_index()
    )
    grouped["_bin_index"] = grouped["_bin_index"].astype(int)
    grouped["bin_lower"] = grouped["_bin_index"].map(lambda index: float(edges[index]))
    grouped["bin_upper"] = grouped["_bin_index"].map(lambda index: float(edges[index + 1]))
    grouped["bin_center"] = (grouped["bin_lower"] + grouped["bin_upper"]) / 2.0
    grouped["calibration_gap"] = grouped["observed_over_rate"] - grouped["mean_predicted"]
    return grouped.drop(columns=["_bin_index"]).reset_index(drop=True)


def evaluate_group(
    frame: pd.DataFrame,
    label: str,
    holdout_start: str | None = None,
) -> dict[str, object]:
    dates = sorted(frame["game_date"].dropna().unique())
    if len(dates) < 5:
        raise ValueError(f"{label}: need at least 5 distinct dates; found {len(dates)}")
    if holdout_start:
        try:
            cutoff = pd.Timestamp(holdout_start).normalize()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: invalid saved holdout start {holdout_start!r}") from exc
    else:
        split_index = max(1, min(int(len(dates) * 0.7), len(dates) - 1))
        cutoff = pd.Timestamp(dates[split_index]).normalize()

    train = frame[frame["game_date"] < cutoff].copy()
    test = frame[frame["game_date"] >= cutoff].copy()
    if len(train) < 30 or len(test) < 20:
        raise ValueError(
            f"{label}: need at least 30 calibration rows and 20 later holdout rows; "
            f"got {len(train)} and {len(test)} for cutoff {cutoff.date()}"
        )
    if train["actual_over"].nunique() < 2:
        raise ValueError(f"{label}: calibration rows need both over and under outcomes")
    calibrator = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
    calibrator.fit(train["p_over"], train["actual_over"])
    test["calibrated_p_over"] = calibrator.predict(test["p_over"])
    raw = _metrics(test, "p_over")
    calibrated = _metrics(test, "calibrated_p_over")
    return {
        "target": label,
        "train_rows": len(train),
        "holdout_rows": len(test),
        "calibration_train_end": pd.Timestamp(train["game_date"].max()).date().isoformat(),
        "holdout_start": cutoff.date().isoformat(),
        "raw": raw,
        "calibrated": calibrated,
        "holdout": test,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV of historical predictions joined to settled results")
    parser.add_argument("--output", type=Path, default=Path("predictions/backtest_holdout.csv"))
    parser.add_argument(
        "--cutoffs",
        type=Path,
        default=Path("data/calibration_cutoffs.json"),
        help="Persisted per-stat holdout start dates to keep a stable chronological boundary",
    )
    parser.add_argument(
        "--reset-cutoffs",
        action="store_true",
        help="Recalculate the holdout boundary from current data (starts a new evaluation baseline)",
    )
    args = parser.parse_args()
    rows = prepare_rows(args.csv)
    if rows.empty:
        raise SystemExit("No valid settled rows found.")

    cutoffs: dict[str, str] = {}
    if args.cutoffs.exists() and not args.reset_cutoffs:
        try:
            saved = json.loads(args.cutoffs.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("cutoff file must contain a JSON object of target-to-date values")
            cutoffs = {str(key).upper(): str(value)[:10] for key, value in saved.items()}
        except Exception as exc:
            raise SystemExit(
                f"Could not read saved calibration cutoffs {args.cutoffs}: {exc}. "
                "Inspect the file or rerun with --reset-cutoffs to establish a new baseline."
            ) from exc

    reports = []
    holdouts = []
    for target, group in rows.groupby("target", sort=True):
        target_key = str(target).upper()
        saved_cutoff = cutoffs.get(target_key)
        try:
            report = evaluate_group(group, target_key, holdout_start=saved_cutoff)
        except ValueError as exc:
            print(f"SKIP {exc}")
            continue
        if saved_cutoff is None:
            cutoffs[target_key] = report["holdout_start"]
            print(f"PINNED {target_key} holdout start at {report['holdout_start']} in {args.cutoffs}")
        holdout = report.pop("holdout")
        holdout["calibration_train_end"] = report["calibration_train_end"]
        holdout["calibration_holdout_start"] = report["holdout_start"]
        holdout["raw_brier"] = report["raw"]["brier"]
        holdout["calibrated_brier"] = report["calibrated"]["brier"]
        holdouts.append(holdout)
        reports.append(report)

    if not reports:
        raise SystemExit("No stat group had enough chronological data for calibration and holdout evaluation.")
    args.cutoffs.parent.mkdir(parents=True, exist_ok=True)
    args.cutoffs.write_text(json.dumps(cutoffs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(holdouts, ignore_index=True).to_csv(args.output, index=False)
    for report in reports:
        print(
            f"\nTARGET {report['target']} | train through {report['calibration_train_end']} "
            f"| holdout from {report['holdout_start']}"
        )
        print(f"calibration rows={report['train_rows']} holdout rows={report['holdout_rows']}")
        for kind in ("raw", "calibrated"):
            metric = report[kind]
            roi = metric["roi"]
            roi_text = f"{roi:.1%}" if math.isfinite(roi) else "n/a (no timestamped odds/bets)"
            print(
                f"{kind:10s} n={metric['n']} hit={metric['hit_rate']:.1%} "
                f"Brier={metric['brier']:.4f} logloss={metric['log_loss']:.4f} "
                f"timestamped_prices={metric['timestamped_rows']} eligible_prices={metric['priced_rows']} "
                f"bets={metric['bets']} net={metric['net_units']:+.2f}u ROI={roi_text}"
            )
    print(f"\nHoldout rows with calibrated probabilities: {args.output}")
    print(f"Persistent per-stat holdout boundaries: {args.cutoffs}")
    print("Brier/log loss evaluate probability quality. ROI includes only positive-EV bets with both prices and snapshot_at_utc; it remains a historical simulation.")


if __name__ == "__main__":
    main()
