"""Chronological backtest for PropIt's soccer goal-probability baselines.

Example:
  python -m scripts.backtest_soccer --input data/soccer/matches.csv --min-history 30

This compares independent Poisson with a Dixon-Coles low-score correction on the same
rolling-origin matches. The rho value is an experimental, user-supplied parameter; this
script does not fit it. Use per-match predictions for further calibration analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from src.soccer.backtest import run_rolling_backtest


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronologically backtest soccer goal probabilities")
    parser.add_argument("--input", default="data/soccer/matches.csv", help="normalized soccer matches CSV")
    parser.add_argument("--output", default="predictions/soccer/backtest.csv", help="per-match predictions CSV")
    parser.add_argument("--metrics-output", default="predictions/soccer/backtest_metrics.json", help="metrics report JSON")
    parser.add_argument("--competition-id", help="restrict backtest to one provider competition ID")
    parser.add_argument("--min-history", type=int, default=30, help="minimum prior finished matches before a test date")
    parser.add_argument("--prior-games", type=float, default=5.0, help="shrinkage prior strength in match-equivalents")
    parser.add_argument("--half-life-days", type=float, default=365.0, help="time-decay half-life")
    parser.add_argument("--lookback-days", type=int, default=730, help="maximum age of training observations")
    parser.add_argument("--total-goals-line", type=float, default=2.5, help="total-goals line evaluated in the backtest")
    parser.add_argument("--dixon-coles-rho", type=float, default=-0.05, help="experimental low-score correction; compare holdout metrics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.min_history < 0:
            raise ValueError("--min-history must be non-negative")
        rows = _read_rows(args.input)
        if args.competition_id is not None:
            rows = [row for row in rows if row.get("competition_id") == args.competition_id]
        predictions, report = run_rolling_backtest(
            rows,
            min_history_matches=args.min_history,
            prior_games=args.prior_games,
            half_life_days=args.half_life_days,
            lookback_days=args.lookback_days,
            total_goals_line=args.total_goals_line,
            dixon_coles_rho=args.dixon_coles_rho,
        )
        if not predictions:
            print(
                "No backtest predictions were produced. Import completed matches and/or lower --min-history.",
                file=sys.stderr,
            )
            print(json.dumps(report, indent=2))
            return 2
        _write_rows(args.output, predictions)
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Backtested {report['backtested_matches']} matches from {report['source_scored_matches']} scored source rows.")
        print(f"Predictions: {args.output}")
        print(f"Metrics: {args.metrics_output}")
        for metric in report["models"]:
            total_loss = metric["total_log_loss"]
            total_loss_text = f"{total_loss:.4f}" if isinstance(total_loss, (int, float)) and math.isfinite(total_loss) else "n/a"
            print(
                f"{metric['model']}: n={metric['matches']}, "
                f"1X2 log loss={metric['log_loss_1x2']:.4f}, "
                f"1X2 Brier={metric['brier_1x2']:.4f}, "
                f"accuracy={metric['accuracy_1x2']:.3f}, "
                f"goal-rate MAE={metric['expected_goals_mae']:.3f}, "
                f"totals n={metric['total_matches']}, totals log loss={total_loss_text}"
            )
        print("Lower log loss/Brier is better. Experimental holdout results do not guarantee future performance.")
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        print(f"Soccer backtest failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
