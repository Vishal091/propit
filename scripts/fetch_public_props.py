"""Best-effort, API-key-free importer for the public Flash Odds NBA prop tables.

The public pages list lines/prices captured from Underdog. This is HTML scraping,
not an official API or an aggregate across bookmakers. Source structure, prices,
coverage, and access are not guaranteed. Verify line, prices, and availability.
Integer lines are skipped because this model does not estimate push probability.
"""
from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger("fetch_public_props")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PAGES = {
    # Flash Odds publishes server-rendered public tables. These lines/prices
    # are captured from Underdog, not an aggregate across books.
    "PTS": "https://flashodds.live/nba/props/points",
    "REB": "https://flashodds.live/nba/props/rebounds",
    "AST": "https://flashodds.live/nba/props/assists",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"
}
ODDS_RE = re.compile(r"(?<!\d)([+\-\u2212\u2013]\s*\d{2,4})(?!\d)")
LINE_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*$")


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return "".join(ch.casefold() for ch in value if ch.isalnum())


def parse_line(value: str) -> float | None:
    match = LINE_RE.match(value or "")
    return float(match.group(1)) if match else None


def parse_odds_and_book(value: str) -> tuple[float | None, str]:
    """Parse labelled odds (e.g. 'DK−128') or bare American odds (e.g. '165')."""
    raw = " ".join((value or "").replace("\xa0", " ").split())
    match = ODDS_RE.search(raw)
    if match:
        token = match.group(1).replace("−", "-").replace("–", "-").replace(" ", "")
        try:
            odds = float(token)
        except ValueError:
            return None, ""
        prefix = raw[:match.start()].strip(" :-|")
        code_match = re.search(r"\b(DK|FD|MGM|CZR|ESPN|BR|FANATICS|HARD ROCK)\b", prefix, re.I)
        book = code_match.group(1).upper() if code_match else (prefix[:24] or "Underdog")
        return odds, book

    # Flash Odds' public HTML renders positive American prices without a plus
    # sign (e.g. 165 means +165); only accept a cell containing a single price.
    bare = re.fullmatch(r"\+?\d{3,4}", raw)
    if bare:
        odds = float(raw.lstrip("+"))
        if 100 <= odds <= 10000:
            return odds, "Underdog"
    return None, ""


def _header_index(headers: list[str], wanted: str) -> int | None:
    wanted = wanted.casefold()
    for idx, value in enumerate(headers):
        if value.strip().casefold() == wanted:
            return idx
    for idx, value in enumerate(headers):
        if wanted in value.strip().casefold():
            return idx
    return None


def _fallback_decimal_line(raw: str) -> float | None:
    """Find the repeated half-point line in a card/row-style prop layout."""
    without_odds = ODDS_RE.sub(" ", raw or "")
    candidates = [float(value) for value in re.findall(r"(?<![\d.])(\d{1,3}\.\d+)(?![\d.])", without_odds)]
    if not candidates:
        return None
    # Prop boards often show the same line in the form/history column and
    # again in the actual Line column. Prefer the most repeated value.
    counts: dict[float, int] = {}
    for value in candidates:
        counts[value] = counts.get(value, 0) + 1
    highest_count = max(counts.values())
    return next(value for value in reversed(candidates) if counts[value] == highest_count)


def _extract_row_odds(raw: str) -> list[tuple[float, str]]:
    """Extract ordered American prices, including unsigned positives, from row text."""
    text = raw or ""
    book_pattern = re.compile(r"\b(DK|FD|MGM|CZR|ESPN|BR|FANATICS|HARD ROCK)\b", re.I)
    tokens: list[tuple[int, int, float]] = []
    signed_spans: list[tuple[int, int]] = []
    for match in ODDS_RE.finditer(text):
        token = match.group(1).replace("−", "-").replace("–", "-").replace(" ", "")
        try:
            tokens.append((match.start(), match.end(), float(token)))
            signed_spans.append((match.start(), match.end()))
        except ValueError:
            continue

    # Positive American odds are sometimes printed without a leading plus.
    for match in re.finditer(r"(?<![\d.\-+\u2212\u2013])\d{3,4}(?![\d.])", text):
        if any(start <= match.start() < end for start, end in signed_spans):
            continue
        try:
            tokens.append((match.start(), match.end(), float(match.group(0))))
        except ValueError:
            continue

    results: list[tuple[float, str]] = []
    for start, end, odds in sorted(tokens, key=lambda item: item[0]):
        prefix = text[max(0, start - 36):start]
        book_matches = list(book_pattern.finditer(prefix))
        book = book_matches[-1].group(1).upper() if book_matches else "Underdog"
        results.append((odds, book))
    return results


def _parse_card_layout(
    soup: BeautifulSoup,
    target: str,
    canonical_players: dict[str, tuple[int, str]],
) -> tuple[list[dict[str, Any]], bool]:
    """Fallback for the div/grid/card layouts used by newer prop-board pages.

    Rather than assuming every site uses an HTML <table>, identify the smallest
    row-like DOM elements that contain a known player, a half-point line and
    both signed prices. This stays conservative: it does not infer missing odds.
    """
    eligible: list[tuple[Any, str, float, list[tuple[float, str]]]] = []
    saw_market_row = False
    for element in soup.find_all(True):
        raw = element.get_text(" ", strip=True)
        if not raw or len(raw) > 1000:
            continue
        odds = _extract_row_odds(raw)
        if len(odds) < 2:
            continue
        line = _fallback_decimal_line(raw)
        if line is None:
            continue
        compact = normalize_name(raw)
        matching = [key for key in canonical_players if key and key in compact]
        if not matching:
            continue
        saw_market_row = True
        if abs(line - round(line)) < 1e-9:
            continue
        eligible.append((element, compact, line, odds))

    # Each player can appear in several nested wrappers. Keep the shortest
    # wrapper that contains all necessary row fields to avoid importing whole
    # sections of the page as one market.
    best: dict[str, tuple[int, str, float, list[tuple[float, str]]]] = {}
    for element, compact, line, odds in eligible:
        raw_length = len(element.get_text(" ", strip=True))
        matching = [key for key in canonical_players if key and key in compact]
        for key in matching:
            current = best.get(key)
            if current is None or raw_length < current[0]:
                best[key] = (raw_length, element.get_text(" ", strip=True), line, odds)

    rows: list[dict[str, Any]] = []
    for key, (length, raw, line, odds) in best.items():
        found = canonical_players[key]
        odds_over, over_book = odds[0]
        odds_under, under_book = odds[1]
        rows.append({
            "player_id": found[0],
            "player_name": found[1],
            "target": target,
            "line": line,
            "odds_over": odds_over,
            "odds_under": odds_under,
            "bookmaker": f"Flash Odds public board (over {over_book}; under {under_book})",
        })
    return rows, saw_market_row


def parse_props_html(
    html: str,
    target: str,
    canonical_players: dict[str, tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    """Extract player, line and both-side prices from a rendered prop table."""
    soup = BeautifulSoup(html, "html.parser")
    rows_out: list[dict[str, Any]] = []
    canonical_players = canonical_players or {}
    matched_tables = 0
    for table in soup.find_all("table"):
        tr_list = table.find_all("tr")
        header_tr = next((tr for tr in tr_list if tr.find_all("th")), None)
        if header_tr is None:
            continue
        header_cells = header_tr.find_all(["th", "td"], recursive=False)
        headers = [cell.get_text(" ", strip=True) for cell in header_cells]
        idx_player = _header_index(headers, "player")
        idx_line = _header_index(headers, "line")
        idx_over = _header_index(headers, "over")
        idx_under = _header_index(headers, "under")
        if None in (idx_player, idx_line, idx_over, idx_under):
            continue
        matched_tables += 1
        for tr in tr_list:
            if tr is header_tr:
                continue
            cells = tr.find_all(["td", "th"], recursive=False)
            if not cells or max(idx_player, idx_line, idx_over, idx_under) >= len(cells):
                continue
            player_cell = cells[idx_player]
            player = player_cell.get_text(" ", strip=True)
            links = [a.get_text(" ", strip=True) for a in player_cell.find_all("a") if a.get_text(" ", strip=True)]
            if links:
                player = links[0]
            line = parse_line(cells[idx_line].get_text(" ", strip=True))
            over_text = cells[idx_over].get_text(" ", strip=True)
            under_text = cells[idx_under].get_text(" ", strip=True)
            odds_over, over_book = parse_odds_and_book(over_text)
            odds_under, under_book = parse_odds_and_book(under_text)
            if not player or line is None or odds_over is None or odds_under is None:
                continue
            if abs(line - round(line)) < 1e-9:
                continue
            player_id = None
            canonical_name = player
            found = canonical_players.get(normalize_name(player))
            if found:
                player_id, canonical_name = found
            rows_out.append({
                "player_id": player_id,
                "player_name": canonical_name,
                "target": target,
                "line": line,
                "odds_over": odds_over,
                "odds_under": odds_under,
                "bookmaker": f"Flash Odds public board (over {over_book}; under {under_book})",
            })
    if matched_tables == 0 or not rows_out:
        # Newer source versions use a grid/card layout rather than a <table>.
        card_rows, saw_market_rows = _parse_card_layout(soup, target, canonical_players)
        if card_rows:
            log.info("parsed %s %s rows from card/grid markup", len(card_rows), target)
            return card_rows
        if saw_market_rows:
            log.info("recognized %s card/grid market rows; none had a supported non-integer line", target)
            return []
        raise ValueError(
            "Could not parse this page as either a Player/Line/Over/Under table or a card/grid of player rows. "
            "The source may now render its markets client-side, or its markup may have changed."
        )
    return rows_out


def load_canonical_players() -> dict[str, tuple[int, str]]:
    """Use this project's public NBA roster helper to match player names to IDs."""
    try:
        from src.data.nba_client import get_active_players
        df = get_active_players()
        if df is None or df.empty or not {"PLAYER_ID", "PLAYER_NAME"}.issubset(df.columns):
            return {}
        result: dict[str, tuple[int, str]] = {}
        for row in df.itertuples(index=False):
            try:
                pid, name = int(getattr(row, "PLAYER_ID")), str(getattr(row, "PLAYER_NAME"))
            except (TypeError, ValueError):
                continue
            result[normalize_name(name)] = (pid, name)
        log.info("loaded %s canonical player names for matching", len(result))
        return result
    except Exception as exc:
        log.warning("could not load NBA roster for ID matching; using exact names: %s", exc)
        return {}


def fetch_rendered_html(url: str, canonical_players: dict[str, tuple[int, str]], timeout_seconds: int = 30) -> str:
    """Render the JS-driven public board in headless Chrome if static HTML is only a shell."""
    try:
        from selenium import webdriver
        from selenium.common.exceptions import TimeoutException, WebDriverException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as exc:
        raise RuntimeError(
            "This board needs browser rendering, but Selenium is missing. "
            "Run .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1440,2200")
    options.add_argument("--user-agent=" + HEADERS["User-Agent"])
    options.page_load_strategy = "eager"

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(45)
        driver.get(url)

        def body_text(current_driver: Any) -> str:
            return str(current_driver.execute_script(
                "return document.body ? document.body.innerText : ''"
            ) or "")

        player_keys = tuple(canonical_players.keys())

        def market_is_rendered(current_driver: Any) -> bool:
            text = body_text(current_driver)
            compact = normalize_name(text)
            return any(key and key in compact for key in player_keys) and len(ODDS_RE.findall(text)) >= 2

        WebDriverWait(driver, timeout_seconds, poll_frequency=0.4).until(market_is_rendered)
        log.info("browser rendered %s page; attempting to load more visible markets", url)

        # Some boards expose 50 rows first and a Load more control.
        for _ in range(8):
            before = body_text(driver)
            load_more = None
            for element in driver.find_elements(By.CSS_SELECTOR, "button, a"):
                label = " ".join((element.text or "").split()).casefold()
                if re.search(r"load(\s+\d+)?\s+more", label) and element.is_displayed() and element.is_enabled():
                    load_more = element
                    break
            if load_more is None:
                break
            try:
                load_more.click()
                WebDriverWait(driver, 8, poll_frequency=0.25).until(
                    lambda current_driver: len(body_text(current_driver)) > len(before)
                )
            except TimeoutException:
                log.warning("Load more control did not reveal additional market text for %s", url)
                break
            except WebDriverException as exc:
                log.warning("could not click Load more on %s: %s", url, exc)
                break

        html = driver.page_source
        log.info("captured %s browser-rendered HTML bytes from %s", len(html), url)
        return html
    except TimeoutException as exc:
        raise RuntimeError(
            f"Chrome opened {url}, but no player rows with sportsbook prices appeared within "
            f"{timeout_seconds} seconds. The page may be blocked or the board may be empty."
        ) from exc
    except WebDriverException as exc:
        raise RuntimeError(
            f"Could not start or control headless Chrome for {url}: {exc}. "
            "Confirm Chrome is installed and Selenium Manager can download the matching driver."
        ) from exc
    finally:
        if driver is not None:
            try:
                driver.quit()
            except WebDriverException:
                pass


def _page_has_more_rows(html: str) -> bool:
    """Detect a public board that explicitly reports more rows than are rendered."""
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(soup.get_text(" ", strip=True).split())
    match = re.search(r"\b(\d+)\s+shown\s+of\s+(\d+)\b", text, re.I)
    if match and int(match.group(1)) < int(match.group(2)):
        return True
    return bool(re.search(r"\bload\s+\d+\s+more\b", text, re.I))


def _merge_market_rows(
    static_rows: list[dict[str, Any]],
    rendered_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge static/browser rows without duplicating the same player/stat/line."""
    merged: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in [*static_rows, *rendered_rows]:
        try:
            key = (normalize_name(str(row["player_name"])), str(row["target"]), float(row["line"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key[0] and key[2] % 1 != 0:
            merged[key] = row
    return list(merged.values())


def fetch_all(output: Path) -> pd.DataFrame:
    players = load_canonical_players()
    session = requests.Session()
    all_rows: list[dict[str, Any]] = []
    for target, url in PAGES.items():
        log.info("fetching public %s lines from %s", target, url)
        try:
            response = session.get(url, headers=HEADERS, timeout=25)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not download {target} public prop page: {exc}") from exc
        requested_url = urlparse(url)
        final_url = urlparse(response.url)
        target_slug = {"PTS": "points", "REB": "rebounds", "AST": "assists"}[target]
        title_soup = BeautifulSoup(response.text, "html.parser")
        page_title = title_soup.title.get_text(" ", strip=True) if title_soup.title else ""
        query_text = " ".join(
            str(value) for values in parse_qs(final_url.query).values() for value in values
        ).casefold()
        route_is_target_specific = target_slug in final_url.path.casefold().split("/")
        query_is_target_specific = target_slug in query_text or target.casefold() in query_text
        title_is_target_specific = target_slug in page_title.casefold()
        if (
            final_url.path.rstrip("/") != requested_url.path.rstrip("/")
            and not route_is_target_specific
            and not query_is_target_specific
            and not title_is_target_specific
        ):
            raise RuntimeError(
                f"{target} source URL redirected to a generic page ({response.url!r}) "
                f"with title {page_title!r}; refusing to label a generic/default board as {target}."
            )
        log.info("resolved %s source URL to %s (%s HTML bytes)", target, response.url, len(response.content))
        try:
            rows = parse_props_html(response.text, target, players)
        except ValueError as static_exc:
            rows = []
            log.warning("static HTML did not expose %s market rows: %s", target, static_exc)

        needs_more = _page_has_more_rows(response.text)
        if needs_more:
            log.info("public %s page indicates that more rows are available; asking the browser to load them", target)
        if not rows or needs_more:
            try:
                rendered_html = fetch_rendered_html(url, players)
                rendered_rows = parse_props_html(rendered_html, target, players)
                if rendered_rows:
                    rows = _merge_market_rows(rows, rendered_rows)
                    log.info("combined static and browser-rendered %s rows into %s unique markets", target, len(rows))
            except (RuntimeError, ValueError) as render_exc:
                if not rows:
                    raise RuntimeError(f"Could not get usable {target} markets from static HTML or headless Chrome: {render_exc}") from render_exc
                log.warning(
                    "browser rendering for %s could not expand the static result (%s); continuing with %s rows",
                    target, render_exc, len(rows),
                )
        log.info("parsed %s usable %s rows", len(rows), target)
        all_rows.extend(rows)
    if not all_rows:
        raise RuntimeError("No usable lines parsed. The source layout may have changed or markets may not be posted.")
    frame = pd.DataFrame(
        all_rows,
        columns=["player_id", "player_name", "target", "line", "odds_over", "odds_under", "bookmaker"],
    )
    frame = frame.dropna(subset=["line", "odds_over", "odds_under"])
    frame = frame.drop_duplicates(subset=["player_name", "target", "line"], keep="first")
    frame = frame.sort_values(["target", "player_name"], kind="stable").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    matched = int(frame["player_id"].notna().sum())
    log.info("wrote %s player markets to %s (%s matched to NBA player IDs)", len(frame), output, matched)
    log.warning("Public-board prices are captured from Underdog and may be delayed or unavailable in your region. Verify all rows.")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/prop_lines_public.csv"))
    args = parser.parse_args()
    try:
        frame = fetch_all(args.output)
    except Exception as exc:
        log.error("public line import failed: %s", exc)
        return 1
    print(f"Imported {len(frame)} player markets into {args.output}.")
    print("Next: .\\.venv\\Scripts\\python.exe -m src.predict_slate --lines data\\prop_lines_public.csv --top 10")
    print("This reads a public webpage, not an official API. Source markup may change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
