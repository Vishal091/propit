"""Name normalization shared by status and lineup snapshots."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

_SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}


def normalize_player_name(value: Any) -> str:
    """Normalize 'Last, First Jr.' and 'First Last' to the same stable key."""
    if value is None:
        return ""
    raw = unicodedata.normalize("NFKD", str(value))
    raw = "".join(char for char in raw if not unicodedata.combining(char))
    raw = raw.replace(".", " ").replace("’", "'")
    if "," in raw:
        last, first = raw.split(",", 1)
        tokens = (first + " " + last).upper().split()
    else:
        tokens = raw.upper().split()
    tokens = [
        re.sub(r"[^A-Z0-9'-]", "", token)
        for token in tokens
        if token.strip().upper().rstrip(".") not in _SUFFIXES
    ]
    tokens = [token for token in tokens if token]
    return " ".join(sorted(tokens))
