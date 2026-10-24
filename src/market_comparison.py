"""Market comparison, quote freshness and snapshot-to-snapshot movement tracking."""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MARKET_HISTORY = Path("data/prop_market_history.csv")
HISTORY_COLUMNS = [
    "market_snapshot_at_utc", "market_source", "player_id", "player_name", "target",
    "line", "odds_over", "odds_under", "bookmaker", "bookmaker_over", "bookmaker_under",
]


def _normalize_name(value: Any) -> str:
    if value is None:
        value = ""
    else:
        try:
            if pd.isna(value):
                value = ""
        except (TypeError, ValueError):
            pass
    raw = unicodedata.normalize("NFKD", str(value))
    raw = "".join(char for char in raw if not unicodedata.combining(char))
    return "".join(char.casefold() for char in raw if char.isalnum())


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def american_to_decimal(odds: Any) -> float | None:
    value = _numeric(odds)
    if value is None or value == 0:
        return None
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def implied_probability(odds: Any) -> float | None:
    decimal = american_to_decimal(odds)
    return 1.0 / decimal if decimal and decimal > 1 else None


def no_vig_probabilities(
    odds_over: Any,
    odds_under: Any,
    *,
    bookmaker_over: Any = None,
    bookmaker_under: Any = None,
    bookmaker: Any = None,
) -> tuple[float | None, float | None, str]:
    """Normalize a paired two-sided price only when its book provenance is compatible."""
    def clean_book(value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    over_book = clean_book(bookmaker_over)
    under_book = clean_book(bookmaker_under)
    row_book = clean_book(bookmaker)
    if not over_book and not under_book and row_book:
        over_book = under_book = row_book
    elif over_book and not under_book:
        under_book = over_book
    elif under_book and not over_book:
        over_book = under_book

    if over_book and under_book and _normalize_name(over_book) != _normalize_name(under_book):
        return None, None, "cross_book_pair_unavailable"

    over = implied_probability(odds_over)
    under = implied_probability(odds_under)
    if over is None or under is None or over + under <= 0:
        return None, None, "missing_or_invalid_prices"
    total = over + under
    method = "same_book_pair" if (over_book or under_book) else "paired_prices_book_unknown"
    return over / total, under / total, method


def _price_columns(frame: pd.DataFrame) -> tuple[str, str]:
    if {"odds_over", "odds_under"}.issubset(frame.columns):
        return "odds_over", "odds_under"
    return "odds_over_american", "odds_under_american"


def _player_key(row: dict[str, Any]) -> str:
    player_id = _numeric(row.get("player_id"))
    if player_id is not None and player_id > 0:
        return f"id:{int(player_id)}"
    player_name = _normalize_name(row.get("player_name", ""))
    return f"name:{player_name}" if player_name else ""


def _effective_book(row: dict[str, Any], side: str) -> str:
    value = row.get(f"bookmaker_{side}")
    if value is None or pd.isna(value) or not str(value).strip():
        value = row.get("bookmaker")
    return str(value or "").strip()


def _book_key(value: Any) -> str:
    return _normalize_name(value)


def _unique_books(records: list[dict[str, Any]]) -> list[str]:
    books: dict[str, str] = {}
    for row in records:
        for side in ("over", "under"):
            book = _effective_book(row, side)
            key = _book_key(book)
            if key:
                books.setdefault(key, book)
    return list(books.values())


def compare_market_rows(frame: pd.DataFrame, *, now_utc: Any = None) -> pd.DataFrame:
    """Annotate rows with same-line best prices and player/stat market context.

    Rows from different books are only de-vigged individually when both sides
    are compatible with the same bookmaker. Cross-book over/under pairs remain
    usable for price display, but are not mislabelled as no-vig probability.
    """
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    over_col, under_col = _price_columns(result)
    for column, default in (
        ("market_snapshot_at_utc", ""),
        ("market_source", ""),
        ("bookmaker_over", ""),
        ("bookmaker_under", ""),
    ):
        if column not in result.columns:
            result[column] = default

    now = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    records = result.to_dict(orient="records")
    player_keys = [_player_key(row) for row in records]
    targets = [str(row.get("target") or "").strip().upper() for row in records]
    lines = [_numeric(row.get("line")) for row in records]
    no_vig_over_values: list[float | None] = []
    no_vig_under_values: list[float | None] = []
    no_vig_methods: list[str] = []
    ages: list[float | None] = []

    for row, over_col_value, under_col_value in zip(records, result[over_col], result[under_col]):
        p_over, p_under, method = no_vig_probabilities(
            over_col_value,
            under_col_value,
            bookmaker_over=row.get("bookmaker_over"),
            bookmaker_under=row.get("bookmaker_under"),
            bookmaker=row.get("bookmaker"),
        )
        no_vig_over_values.append(p_over)
        no_vig_under_values.append(p_under)
        no_vig_methods.append(method)
        stamp = row.get("market_snapshot_at_utc") or row.get("snapshot_at_utc")
        parsed = pd.to_datetime(stamp, utc=True, errors="coerce")
        ages.append(max(0.0, (now - parsed).total_seconds() / 3600.0) if pd.notna(parsed) else None)

    result["market_no_vig_p_over_consensus"] = np.nan
    result["market_no_vig_p_under_consensus"] = np.nan
    result["market_no_vig_method"] = no_vig_methods
    result["market_snapshot_age_hours"] = ages
    result["market_books_available"] = 0
    result["market_lines_available"] = 0
    result["market_consensus_line"] = np.nan
    result["market_line_min"] = np.nan
    result["market_line_max"] = np.nan
    result["market_line_spread"] = np.nan
    result["market_rows_at_line"] = 0
    result["best_over_odds_american"] = np.nan
    result["best_over_bookmaker"] = ""
    result["best_under_odds_american"] = np.nan
    result["best_under_bookmaker"] = ""
    result["same_book_no_vig_rows"] = 0

    valid_keys = [i for i, key in enumerate(player_keys) if key and targets[i]]
    by_player: dict[tuple[str, str], list[int]] = {}
    by_line: dict[tuple[str, str, float], list[int]] = {}
    for i in valid_keys:
        market_key = (player_keys[i], targets[i])
        by_player.setdefault(market_key, []).append(i)
        if lines[i] is not None:
            by_line.setdefault((*market_key, round(lines[i], 4)), []).append(i)

    for market_key, indices in by_player.items():
        market_records = [records[i] for i in indices]
        books = _unique_books(market_records)
        distinct_lines = sorted({round(lines[i], 4) for i in indices if lines[i] is not None})
        consensus = float(np.median(distinct_lines)) if distinct_lines else None
        lower = min(distinct_lines) if distinct_lines else None
        upper = max(distinct_lines) if distinct_lines else None
        for i in indices:
            result.at[result.index[i], "market_books_available"] = len(books)
            result.at[result.index[i], "market_lines_available"] = len(distinct_lines)
            result.at[result.index[i], "market_consensus_line"] = consensus
            result.at[result.index[i], "market_line_min"] = lower
            result.at[result.index[i], "market_line_max"] = upper
            result.at[result.index[i], "market_line_spread"] = (upper - lower) if lower is not None and upper is not None else np.nan

    for (player_key, target, line), indices in by_line.items():
        market_records = [records[i] for i in indices]
        books = _unique_books(market_records)
        paired_over = [no_vig_over_values[i] for i in indices if no_vig_over_values[i] is not None]
        paired_under = [no_vig_under_values[i] for i in indices if no_vig_under_values[i] is not None]
        consensus_over = float(np.median(paired_over)) if paired_over else np.nan
        consensus_under = float(np.median(paired_under)) if paired_under else np.nan

        valid_over = []
        valid_under = []
        for i in indices:
            over_odds = _numeric(records[i].get(over_col))
            under_odds = _numeric(records[i].get(under_col))
            over_decimal = american_to_decimal(over_odds)
            under_decimal = american_to_decimal(under_odds)
            if over_decimal is not None:
                valid_over.append((over_decimal, over_odds, _effective_book(records[i], "over")))
            if under_decimal is not None:
                valid_under.append((under_decimal, under_odds, _effective_book(records[i], "under")))
        best_over = max(valid_over, key=lambda item: item[0]) if valid_over else (None, None, "")
        best_under = max(valid_under, key=lambda item: item[0]) if valid_under else (None, None, "")

        for i in indices:
            idx = result.index[i]
            result.at[idx, "market_no_vig_p_over_consensus"] = consensus_over
            result.at[idx, "market_no_vig_p_under_consensus"] = consensus_under
            result.at[idx, "market_rows_at_line"] = len(indices)
            result.at[idx, "same_book_no_vig_rows"] = len(paired_over)
            result.at[idx, "best_over_odds_american"] = best_over[1] if best_over[1] is not None else np.nan
            result.at[idx, "best_over_bookmaker"] = best_over[2]
            result.at[idx, "best_under_odds_american"] = best_under[1] if best_under[1] is not None else np.nan
            result.at[idx, "best_under_bookmaker"] = best_under[2]

    for i, method in enumerate(no_vig_methods):
        if method == "cross_book_pair_unavailable" and pd.isna(result.iloc[i]["market_no_vig_p_over_consensus"]):
            idx = result.index[i]
            result.at[idx, "market_no_vig_method"] = "cross_book_pair_unavailable"
    return result


def record_market_snapshot(
    current: pd.DataFrame,
    history_path: Path = DEFAULT_MARKET_HISTORY,
    *,
    observed_at_utc: str | None = None,
    market_source: str = "Flash Odds public board",
) -> pd.DataFrame:
    """Compare a newly fetched board with prior snapshots, then append it to history."""
    if current.empty:
        return current.copy()
    out = current.copy()
    observed = observed_at_utc or datetime.now(timezone.utc).isoformat(timespec="microseconds")
    if "market_snapshot_at_utc" not in out.columns:
        out["market_snapshot_at_utc"] = observed
    else:
        stamps = out["market_snapshot_at_utc"].fillna("").astype(str).str.strip()
        out["market_snapshot_at_utc"] = stamps.where(stamps.ne(""), observed)
    if "market_source" not in out.columns:
        out["market_source"] = market_source
    else:
        out["market_source"] = out["market_source"].fillna("").astype(str).str.strip().replace("", market_source)

    history = pd.DataFrame()
    if history_path.exists() and history_path.stat().st_size:
        try:
            history = pd.read_csv(history_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Could not read market history {history_path}: {exc}") from exc

    history_records = history.to_dict(orient="records") if not history.empty else []
    history_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for old_row in history_records:
        stamp = old_row.get("market_snapshot_at_utc") or old_row.get("snapshot_at_utc")
        parsed = pd.to_datetime(stamp, utc=True, errors="coerce")
        if pd.isna(parsed):
            continue
        identity = (
            _player_key(old_row),
            str(old_row.get("target") or "").strip().upper(),
            _normalize_name(old_row.get("market_source") or old_row.get("bookmaker") or market_source),
        )
        if identity[0] and identity[1]:
            old_copy = dict(old_row)
            old_copy["_parsed_snapshot"] = parsed
            history_by_key.setdefault(identity, []).append(old_copy)

    previous_lines: list[float | None] = []
    previous_over: list[float | None] = []
    previous_under: list[float | None] = []
    previous_stamp: list[str | None] = []
    history_counts: list[int] = []
    line_moves: list[float | None] = []
    over_decimal_moves: list[float | None] = []
    under_decimal_moves: list[float | None] = []

    for row in out.to_dict(orient="records"):
        stamp = pd.to_datetime(row.get("market_snapshot_at_utc"), utc=True, errors="coerce")
        identity = (
            _player_key(row),
            str(row.get("target") or "").strip().upper(),
            _normalize_name(row.get("market_source") or row.get("bookmaker") or market_source),
        )
        eligible = [item for item in history_by_key.get(identity, []) if item["_parsed_snapshot"] < stamp] if pd.notna(stamp) else []
        eligible.sort(key=lambda item: item["_parsed_snapshot"])
        previous = eligible[-1] if eligible else None
        old_line = _numeric(previous.get("line")) if previous else None
        old_over = _numeric(previous.get("odds_over", previous.get("odds_over_american"))) if previous else None
        old_under = _numeric(previous.get("odds_under", previous.get("odds_under_american"))) if previous else None
        current_line = _numeric(row.get("line"))
        current_over = _numeric(row.get("odds_over", row.get("odds_over_american")))
        current_under = _numeric(row.get("odds_under", row.get("odds_under_american")))
        previous_lines.append(old_line)
        previous_over.append(old_over)
        previous_under.append(old_under)
        previous_stamp.append(previous["_parsed_snapshot"].isoformat() if previous else None)
        history_counts.append(len(eligible))
        line_moves.append(current_line - old_line if current_line is not None and old_line is not None else None)
        current_over_decimal = american_to_decimal(current_over)
        old_over_decimal = american_to_decimal(old_over)
        current_under_decimal = american_to_decimal(current_under)
        old_under_decimal = american_to_decimal(old_under)
        over_decimal_moves.append(current_over_decimal - old_over_decimal if current_over_decimal is not None and old_over_decimal is not None else None)
        under_decimal_moves.append(current_under_decimal - old_under_decimal if current_under_decimal is not None and old_under_decimal is not None else None)

    out["previous_line"] = previous_lines
    out["previous_odds_over"] = previous_over
    out["previous_odds_under"] = previous_under
    out["prior_snapshot_at_utc"] = previous_stamp
    out["market_prior_snapshot_count"] = history_counts
    out["line_move_from_previous"] = line_moves
    out["over_decimal_odds_move"] = over_decimal_moves
    out["under_decimal_odds_move"] = under_decimal_moves
    out = compare_market_rows(out)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([history, out], ignore_index=True, sort=False) if not history.empty else out.copy()
    # Keep every distinct observation; this is an audit log, not a latest-only cache.
    snapshot_col = combined["market_snapshot_at_utc"].fillna("").astype(str)
    id_col = combined.apply(_player_key, axis=1)
    target_col = combined.get("target", pd.Series("", index=combined.index)).fillna("").astype(str).str.upper().str.strip()
    source_col = combined.get("market_source", pd.Series("", index=combined.index)).fillna("").astype(str).str.strip()
    combined["_history_key"] = id_col + "|" + target_col + "|" + source_col + "|" + snapshot_col
    combined = combined.drop_duplicates("_history_key", keep="last").drop(columns=["_history_key"])
    combined.to_csv(history_path, index=False)
    return out
