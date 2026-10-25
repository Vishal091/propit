"""Fetch and parse the NBA's official timestamped injury-report PDFs.

The parser only accepts structured rows with a player, team and status. It does
not infer that an unlisted player is available. Unmatched names are retained so
the downstream exact-name/team/date matcher can still make a conservative match.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPORT_ROOT = "https://ak-static.cms.nba.com/referee/injury/"
USER_AGENT = "Mozilla/5.0 PropIt official NBA injury report importer"
EASTERN = ZoneInfo("America/New_York")

_TEAM_ABBREVIATIONS = {
    "ATLANTA HAWKS": "ATL", "BOSTON CELTICS": "BOS", "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA", "CHICAGO BULLS": "CHI", "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL", "DENVER NUGGETS": "DEN", "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW", "HOUSTON ROCKETS": "HOU", "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC", "LA CLIPPERS": "LAC", "LOS ANGELES LAKERS": "LAL",
    "LA LAKERS": "LAL", "MEMPHIS GRIZZLIES": "MEM", "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL", "MINNESOTA TIMBERWOLVES": "MIN", "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK", "OKLAHOMA CITY THUNDER": "OKC", "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI", "PHOENIX SUNS": "PHX", "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC", "SAN ANTONIO SPURS": "SAS", "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA", "WASHINGTON WIZARDS": "WAS",
}
_COLUMNS = ["scheduled_date", "game_time", "matchup", "team_name", "team_abbreviation",
            "player_name", "status", "reason", "report_url", "reported_at_utc"]


def season_page_url(game_date: str | date) -> str:
    """Resolve the official injury page for the NBA season containing game_date."""
    day = pd.Timestamp(game_date).date()
    start_year = day.year if day.month >= 7 else day.year - 1
    season = f"{start_year}-{str(start_year + 1)[-2:]}"
    return f"https://official.nba.com/nba-injury-report-{season}-season/"


def report_timestamp_from_url(url: str) -> datetime | None:
    match = re.search(
        r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{1,2})_(\d{2})(AM|PM)\.pdf",
        url, flags=re.IGNORECASE,
    )
    if not match:
        return None
    date_text, hour_text, minute_text, ampm = match.groups()
    try:
        return datetime.strptime(
            f"{date_text} {int(hour_text):02d}:{minute_text} {ampm.upper()}",
            "%Y-%m-%d %I:%M %p",
        ).replace(tzinfo=EASTERN)
    except ValueError:
        return None


def select_report_url(page_html: str, game_date: str | date) -> str:
    """Select the latest report dated game_date or the preceding calendar day."""
    target = pd.Timestamp(game_date).date()
    candidates: list[tuple[datetime, str]] = []
    soup = BeautifulSoup(page_html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if "Injury-Report_" not in href or not href.lower().endswith(".pdf"):
            continue
        timestamp = report_timestamp_from_url(href)
        if timestamp is None:
            continue
        report_date = timestamp.date()
        if report_date in {target, target - timedelta(days=1)}:
            candidates.append((timestamp, href))
    if not candidates:
        raise ValueError(
            f"No official injury-report PDF was linked for {target.isoformat()} or the previous day. "
            "The official page may not have published that report yet."
        )
    return max(candidates, key=lambda item: item[0])[1]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def _parse_date_cell(value: str) -> date | None:
    raw = _clean(value)
    if not raw:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _row_value(row: list[Any], index: int) -> str:
    return _clean(row[index]) if index < len(row) else ""


def parse_injury_table_rows(
    tables: list[list[list[Any]]],
    *,
    report_date: str | date,
    report_url: str,
    reported_at_utc: str,
    target_date: str | date,
) -> pd.DataFrame:
    """Convert extracted PDF tables to player rows; never infer missing availability."""
    report_day = pd.Timestamp(report_date).date()
    target_day = pd.Timestamp(target_date).date()
    collected: list[dict[str, str]] = []
    for table in tables:
        active_date = report_day
        current_game = ""
        current_time = ""
        current_team = ""
        for raw_row in table:
            if not raw_row:
                continue
            values = [_clean(value) for value in raw_row]
            joined = " ".join(values).casefold()
            if "player" in joined and "status" in joined and ("team" in joined or "game" in joined):
                continue
            # Official reports use Date | Time | Game | Team | Player | Status | Injury/Illness.
            date_cell = _row_value(raw_row, 0)
            parsed_date = _parse_date_cell(date_cell)
            if parsed_date is not None:
                active_date = parsed_date
            time_cell, game_cell, team_cell = (_row_value(raw_row, i) for i in (1, 2, 3))
            player_name, status, reason = (_row_value(raw_row, i) for i in (4, 5, 6))
            if time_cell and "ET" in time_cell.upper():
                current_time = time_cell
            if game_cell and "@" in game_cell:
                current_game = game_cell.upper().replace(" ", "")
            if team_cell and team_cell.upper() in _TEAM_ABBREVIATIONS:
                current_team = team_cell
            elif team_cell and len(team_cell) > 3 and not team_cell.upper().startswith("NOT YET"):
                current_team = team_cell

            if active_date != target_day or not player_name or not status:
                continue
            if "NOT YET SUBMITTED" in " ".join([player_name, status, reason]).upper():
                continue
            if status.casefold() in {"status", "player"}:
                continue
            # PDF extraction can occasionally shift report metadata (for example,
            # "10/25/26 06:45") into the player column. Accept only real-looking
            # names and the statuses used by the official NBA injury report.
            if not any(character.isalpha() for character in player_name):
                continue
            if status.casefold() not in {"available", "probable", "questionable", "doubtful", "out"}:
                continue
            abbreviation = _TEAM_ABBREVIATIONS.get(current_team.upper(), "")
            collected.append({
                "scheduled_date": target_day.isoformat(),
                "game_time": current_time,
                "matchup": current_game,
                "team_name": current_team,
                "team_abbreviation": abbreviation,
                "player_name": player_name,
                "status": status,
                "reason": reason,
                "report_url": report_url,
                "reported_at_utc": reported_at_utc,
            })
    frame = pd.DataFrame(collected, columns=_COLUMNS)
    if not frame.empty:
        # Keep the latest instance of a player/status pair within the selected report.
        frame = frame.drop_duplicates(
            ["scheduled_date", "team_name", "player_name"], keep="last"
        ).reset_index(drop=True)
    return frame


def positioned_words_to_rows(words: list[dict[str, Any]], page_width: float) -> list[list[str]]:
    """Reconstruct seven NBA report columns from PDF word coordinates.

    pdfplumber's automatic table detector can return no tables for this report
    because the source is laid out with positioned text rather than stable cell
    borders. Known post-2024 NBA report column boundaries are scaled to the
    document width and used only as a fallback; the original text is retained.
    """
    if page_width <= 0 or not words:
        return []
    scale = page_width / 842.0
    boundaries = [108.82, 183.55, 255.11, 371.93, 543.48, 655.04]
    boundaries = [value * scale for value in boundaries]
    ordered = sorted(words, key=lambda word: (float(word.get("top", 0.0)), float(word.get("x0", 0.0))))

    lines: list[dict[str, Any]] = []
    for word in ordered:
        try:
            top = float(word.get("top", 0.0))
            x0 = float(word.get("x0", 0.0))
        except (TypeError, ValueError):
            continue
        if not _clean(word.get("text")):
            continue
        if not lines or abs(top - lines[-1]["top"]) > 2.5:
            lines.append({"top": top, "tops": [top], "cells": [[] for _ in range(7)]})
        else:
            # Use the average baseline for a little tolerance in font/character
            # placement without merging separate wrapped rows.
            line = lines[-1]
            line["tops"].append(top)
            line["top"] = sum(line["tops"]) / len(line["tops"])
        column = 0
        while column < len(boundaries) and x0 >= boundaries[column]:
            column += 1
        lines[-1]["cells"][column].append(_clean(word.get("text")))

    rows: list[list[str]] = []
    for line in lines:
        cells = [" ".join(parts).strip() for parts in line["cells"]]
        if any(cells):
            rows.append(cells)
    return rows


def extract_team_submission_metadata(
    tables: list[list[list[Any]]],
    positioned_pages: list[list[list[str]]],
) -> dict[str, Any]:
    """Capture explicit NOT YET SUBMITTED team markers from the raw report rows.

    This does not label any player Available. It records team-level report
    coverage so downstream matching can distinguish an unlisted player on a
    submitted report from a team whose report is explicitly pending.
    """
    not_submitted: set[str] = set()
    unresolved_marker = False
    # Prefer coordinates, which preserve the official seven-column layout.
    # Use table output only if positional extraction yielded no pages at all.
    datasets = positioned_pages if positioned_pages else tables
    for table in datasets:
        current_team = ""
        for raw_row in table:
            if not raw_row:
                continue
            values = [_clean(value) for value in raw_row]
            team_cell = _row_value(raw_row, 3)
            if team_cell and team_cell.upper() in _TEAM_ABBREVIATIONS:
                current_team = _TEAM_ABBREVIATIONS[team_cell.upper()]
            joined = " ".join(values).upper()
            if "NOT YET SUBMITTED" in joined:
                if current_team:
                    not_submitted.add(current_team)
                else:
                    unresolved_marker = True

    return {
        "teams_not_yet_submitted": sorted(not_submitted),
        "team_submission_markers_count": len(not_submitted),
        "team_submission_markers_resolved": not unresolved_marker,
        "team_report_coverage_parsed": bool(datasets),
    }


def fetch_official_injury_report(
    game_date: str | date,
    *,
    timeout_seconds: int = 25,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch the official NBA report for the selected NBA-local game date."""
    target = pd.Timestamp(game_date).date()
    client = session or requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf,*/*"}
    page_url = season_page_url(target)
    page_response = client.get(page_url, headers=headers, timeout=timeout_seconds)
    page_response.raise_for_status()
    report_url = select_report_url(page_response.text, target)
    report_time = report_timestamp_from_url(report_url)
    if report_time is None:
        raise ValueError(f"Could not parse report timestamp from official URL: {report_url}")
    if report_url.startswith("/"):
        report_url = "https://official.nba.com" + report_url
    pdf_response = client.get(report_url, headers=headers, timeout=timeout_seconds)
    pdf_response.raise_for_status()
    if not pdf_response.content.startswith(b"%PDF"):
        raise ValueError("Official injury report link did not return a PDF file.")

    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "PDF parsing dependency is missing. Install dependencies with "
            "python -m pip install -r requirements.txt."
        ) from exc

    extracted_tables: list[list[list[Any]]] = []
    positioned_rows: list[list[list[str]]] = []
    with pdfplumber.open(io.BytesIO(pdf_response.content)) as pdf:
        for page in pdf.pages:
            try:
                extracted_tables.extend(page.extract_tables() or [])
            except Exception:
                # Keep going: some PDF page layouts are text-positioned and have
                # no stable table grid, which is why the word-coordinate fallback
                # below exists.
                pass
            try:
                words = page.extract_words(
                    x_tolerance=1.5, y_tolerance=2.0, keep_blank_chars=False,
                )
                rows = positioned_words_to_rows(words, float(page.width))
                if rows:
                    positioned_rows.append(rows)
            except Exception:
                # The original table parser below still gets its chance.
                continue

    stamp_utc = report_time.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    # The table detector is preferred; if its output is blank or does not produce
    # any player rows, reconstruct rows from known column x-coordinates.
    frame = parse_injury_table_rows(
        extracted_tables,
        report_date=report_time.date(),
        report_url=report_url,
        reported_at_utc=stamp_utc,
        target_date=target,
    )
    parse_method = "pdf_table"
    if frame.empty:
        frame = parse_injury_table_rows(
            positioned_rows,
            report_date=report_time.date(),
            report_url=report_url,
            reported_at_utc=stamp_utc,
            target_date=target,
        )
        parse_method = "positioned_words"
    if frame.empty:
        # More actionable diagnostics, without writing any partial/unverified data.
        all_words = []
        with pdfplumber.open(io.BytesIO(pdf_response.content)) as pdf:
            for page in pdf.pages[:2]:
                try:
                    all_words.extend(page.extract_words())
                except Exception:
                    pass
        preview = " ".join(_clean(word.get("text")) for word in all_words[:70])
        raise ValueError(
            "Downloaded official report but could not parse any player status rows for "
            f"{target.isoformat()}. No current snapshot was written. "
            f"Pages={len(pdf.pages) if 'pdf' in locals() else 'unknown'}; "
            f"table_count={len(extracted_tables)}; positioned_pages={len(positioned_rows)}; "
            f"text_preview={preview[:500]!r}. This diagnostic will help fix any layout change."
        )
    team_submission_meta = extract_team_submission_metadata(extracted_tables, positioned_rows)
    meta = {
        **team_submission_meta,
        "game_date": target.isoformat(),
        "report_page_url": page_url,
        "report_url": report_url,
        "reported_at_utc": stamp_utc,
        "retrieved_at_utc": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
        "rows_parsed": int(len(frame)),
        "players": int(frame["player_name"].nunique()),
        "teams": int(frame["team_abbreviation"].replace("", pd.NA).nunique()),
        "parse_method": parse_method,
        "table_count": int(len(extracted_tables)),
        "positioned_pages": int(len(positioned_rows)),
    }
    return frame, meta
