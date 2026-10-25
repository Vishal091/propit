"""Public NBA lineup-page ingestion with explicit projected/confirmed provenance.

RotoWire's free daily page displays an expected lineup until its source marks a
team lineup confirmed. This collector never upgrades an unlabeled/projected
lineup to confirmed, and it refuses to import a page for the wrong game date.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from src.player_identity import normalize_player_name

ROTOWIRE_LINEUPS_URL = "https://www.rotowire.com/basketball/nba-lineups.php"
LINEUP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
TEAM_ALIASES = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",
    "WSH": "WAS", "BRK": "BKN", "UTAH": "UTA",
}
VALID_POSITIONS = {"PG", "SG", "SF", "PF", "C"}
OUTPUT_COLUMNS = [
    "player_id", "player_name", "team_abbreviation", "match_date",
    "starting_status", "confirmation_status", "confidence_label",
    "source_status", "reported_at_utc", "source_reported_at_utc",
    "retrieved_at_utc", "source", "reason", "minutes_limit",
]


def normalize_team_abbreviation(value: Any) -> str:
    raw = str(value or "").strip().upper()
    return TEAM_ALIASES.get(raw, raw)


def _html_game_date(soup: BeautifulSoup) -> date | None:
    text = soup.get_text(" ", strip=True)
    match = re.search(
        r"Starting\s+lineups\s+for\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1)
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def _team_code(node: Tag | None) -> str:
    if node is None:
        return ""
    abbr_node = node.select_one(".lineup__abbr, [data-team-abbr], [data-tricode]")
    if abbr_node is not None:
        raw = (
            abbr_node.get("data-team-abbr")
            or abbr_node.get("data-tricode")
            or abbr_node.get_text(" ", strip=True)
        )
        match = re.search(r"\b[A-Z]{2,4}\b", str(raw).upper())
        if match:
            return normalize_team_abbreviation(match.group(0))
    for attr in ("data-team-abbr", "data-tricode", "data-team"):
        raw = node.get(attr)
        if raw:
            match = re.search(r"\b[A-Z]{2,4}\b", str(raw).upper())
            if match:
                return normalize_team_abbreviation(match.group(0))
    text = node.get_text(" ", strip=True).upper()
    match = re.search(r"\b(?:LAL|LAC|NYK|GSW|NOP|SAS|WAS|BKN|UTA|UTAH|PHX|PHI|MIN|MEM|SAC|OKC|ORL|TOR|IND|BOS|ATL|CHA|CHI|CLE|DAL|DEN|DET|HOU|MIA|MIL|POR)\b", text)
    return normalize_team_abbreviation(match.group(0)) if match else ""


def _player_from_list_item(item: Tag) -> tuple[str, str] | None:
    pos_node = item.select_one(".lineup__pos, .lineup__position, [data-position]")
    position = ""
    if pos_node is not None:
        position = str(pos_node.get("data-position") or pos_node.get_text(" ", strip=True)).strip().upper()
    if position not in VALID_POSITIONS:
        # Some markup versions place the position as a short leading label.
        text = item.get_text(" ", strip=True)
        leading = re.match(r"^(PG|SG|SF|PF|C)\b", text.upper())
        if leading:
            position = leading.group(1)
    if position not in VALID_POSITIONS:
        return None

    link = item.select_one("a[title], a[aria-label], a")
    if link is None:
        return None
    name = str(link.get("title") or link.get("aria-label") or link.get_text(" ", strip=True)).strip()
    name = re.sub(r"\s+", " ", name)
    if not name or name.casefold() in {
        "very likely to play", "likely to play", "toss up to play", "unlikely to play",
    }:
        return None
    return position, name


def _side_confirmation_status(card: Tag, team_node: Tag | None, lineup_node: Tag) -> tuple[str, str, str]:
    """Return confirmation only from a side-local explicit source label."""
    candidates: list[Tag] = []
    for node in [lineup_node, *lineup_node.parents]:
        if not isinstance(node, Tag):
            continue
        if node is card.parent:
            break
        candidates.append(node)
        if node is card:
            break

    # A team title or local status node can carry the only explicit label.
    if team_node is not None:
        for selector in (".lineup__status", ".lineup__title", "[data-lineup-status]"):
            for node in team_node.select(selector):
                candidates.insert(0, node)

    for node in candidates:
        label = str(node.get("data-lineup-status") or node.get_text(" ", strip=True)).upper()
        confirmed = "CONFIRMED LINEUP" in label
        expected = "EXPECTED LINEUP" in label
        if confirmed and not expected:
            return "CONFIRMED", "SOURCE_CONFIRMED", "Confirmed Lineup"
        if expected and not confirmed:
            return "PROJECTED", "PROJECTED", "Expected Lineup"
    return "PROJECTED", "LOW", "Lineup label not explicit; treated as projected"


def _roster_indexes(roster: pd.DataFrame | None) -> tuple[dict[tuple[str, str], str], dict[str, set[str]]]:
    exact: dict[tuple[str, str], str] = {}
    by_name: dict[str, set[str]] = {}
    if roster is None or roster.empty or not {"PLAYER_ID", "PLAYER_NAME"}.issubset(roster.columns):
        return exact, by_name
    for row in roster.to_dict(orient="records"):
        key = normalize_player_name(row.get("PLAYER_NAME"))
        pid_raw = row.get("PLAYER_ID")
        if not key or pd.isna(pid_raw):
            continue
        try:
            pid = str(int(float(pid_raw)))
        except (TypeError, ValueError):
            continue
        team = normalize_team_abbreviation(row.get("TEAM_ABBREVIATION"))
        if team:
            exact[(key, team)] = pid
        by_name.setdefault(key, set()).add(pid)
    return exact, by_name


def parse_rotowire_lineups(
    html: str,
    *,
    game_date: str,
    source_url: str = ROTOWIRE_LINEUPS_URL,
    retrieved_at_utc: str | None = None,
    roster: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse the free RotoWire lineup page; projected labels are never promoted."""
    try:
        target_date = date.fromisoformat(game_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("game_date must be YYYY-MM-DD") from exc
    soup = BeautifulSoup(html or "", "html.parser")
    page_date = _html_game_date(soup)
    if page_date is None:
        raise ValueError("Lineup source page date could not be verified; no lineup data was imported.")
    if page_date != target_date:
        raise ValueError(
            f"Lineup source page is for {page_date.isoformat()}, not requested {target_date.isoformat()}; "
            "no lineup data was imported."
        )

    timestamp = pd.to_datetime(
        retrieved_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        utc=True, errors="coerce",
    )
    if pd.isna(timestamp):
        raise ValueError("retrieved_at_utc must be a valid ISO-8601 timestamp")
    stamped = timestamp.isoformat().replace("+00:00", "Z")
    exact, by_name = _roster_indexes(roster)
    records: list[dict[str, Any]] = []
    teams_seen: set[str] = set()
    confirmation_counts = {"CONFIRMED": 0, "PROJECTED": 0}

    cards = soup.select("div.lineup.is-nba")
    if not cards:
        raise ValueError("RotoWire lineup card markup was not found; no lineup data was imported.")

    for card in cards:
        for side in ("visit", "home"):
            team_node = card.select_one(
                f"a.lineup__mteam.is-{side}.white, a.lineup__mteam.is-{side}"
            )
            team = _team_code(team_node)
            if not team:
                # Alternate markup can expose the abbreviation in the side header.
                team_node = card.select_one(f".lineup__team.is-{side}")
                team = _team_code(team_node)
            lineup_node = card.select_one(f"ul.lineup__list.is-{side}")
            if not team or lineup_node is None:
                continue

            players_by_position: dict[str, str] = {}
            for item in lineup_node.select("li"):
                parsed = _player_from_list_item(item)
                if parsed is None:
                    continue
                position, player_name = parsed
                # If the source layout duplicates an item, preserve the first
                # entry in that position and never invent a sixth starter.
                players_by_position.setdefault(position, player_name)
            if not players_by_position:
                continue

            teams_seen.add(team)
            confirmation, confidence, source_status = _side_confirmation_status(card, team_node, lineup_node)
            confirmation_counts[confirmation] += len(players_by_position)
            for position in ("PG", "SG", "SF", "PF", "C"):
                player_name = players_by_position.get(position)
                if not player_name:
                    continue
                key = normalize_player_name(player_name)
                player_id = exact.get((key, team), "")
                if not player_id and len(by_name.get(key, set())) == 1:
                    player_id = next(iter(by_name[key]))

                if confirmation == "CONFIRMED":
                    reason = "RotoWire explicitly labels this team's lineup Confirmed Lineup; verify late changes before tip-off."
                else:
                    reason = "RotoWire expected lineup only; projected role is informational and does not satisfy the strict confirmed-lineup gate."
                records.append({
                    "player_id": player_id,
                    "player_name": player_name,
                    "team_abbreviation": team,
                    "match_date": target_date.isoformat(),
                    "starting_status": "STARTER",
                    "confirmation_status": confirmation,
                    "confidence_label": confidence,
                    "source_status": source_status,
                    "reported_at_utc": stamped,
                    "source_reported_at_utc": "",
                    "retrieved_at_utc": stamped,
                    "source": source_url,
                    "reason": reason,
                    "minutes_limit": "",
                })

    frame = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    if frame.empty:
        raise ValueError("No recognizable starting-five players were parsed; no lineup data was imported.")
    frame = frame.drop_duplicates(
        ["match_date", "team_abbreviation", "player_name"], keep="last"
    ).reset_index(drop=True)
    meta = {
        "game_date": target_date.isoformat(),
        "source": source_url,
        "page_date": page_date.isoformat(),
        "retrieved_at_utc": stamped,
        "rows": int(len(frame)),
        "teams": sorted(teams_seen),
        "teams_count": len(teams_seen),
        "confirmation_counts": confirmation_counts,
        "status_counts": frame["confirmation_status"].value_counts().to_dict(),
    }
    return frame, meta


def fetch_rotowire_lineups(
    game_date: str,
    *,
    url: str = ROTOWIRE_LINEUPS_URL,
    timeout_seconds: int = 20,
    session: requests.Session | None = None,
    roster: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch a target-date lineup page and reject stale/wrong-date page results."""
    try:
        target_date = date.fromisoformat(game_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("game_date must be YYYY-MM-DD") from exc
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["date"] = target_date.isoformat()
    request_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    client = session or requests.Session()
    response = client.get(request_url, headers=LINEUP_HEADERS, timeout=timeout_seconds)
    response.raise_for_status()
    source_url = str(getattr(response, "url", "") or request_url)
    retrieved = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    frame, meta = parse_rotowire_lineups(
        response.text,
        game_date=target_date.isoformat(),
        source_url=source_url,
        retrieved_at_utc=retrieved,
        roster=roster,
    )
    meta["http_status"] = int(getattr(response, "status_code", 200))
    return frame, meta
