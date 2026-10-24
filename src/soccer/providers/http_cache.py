"""Small conditional-GET cache for public soccer-data endpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from src.soccer.schemas import utc_now_iso


class DataProviderError(RuntimeError):
    """The source could not provide a valid response."""


class DataNotAvailableError(DataProviderError):
    """The source does not publish the requested resource."""


class CachedHttpClient:
    def __init__(
        self,
        cache_dir: str | Path,
        timeout_seconds: float = 20.0,
        session: requests.Session | None = None,
        allow_stale_cache: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()
        self.allow_stale_cache = allow_stale_cache

    def fetch_text(self, url: str) -> tuple[str, dict[str, Any]]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Only absolute HTTPS source URLs are permitted")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        body_path = self.cache_dir / f"{key}.body"
        metadata_path = self.cache_dir / f"{key}.json"
        cached_text = body_path.read_text(encoding="utf-8") if body_path.exists() else None
        try:
            cached_meta = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            cached_meta = {}

        headers: dict[str, str] = {}
        if cached_text is not None:
            if cached_meta.get("etag"):
                headers["If-None-Match"] = cached_meta["etag"]
            if cached_meta.get("last_modified"):
                headers["If-Modified-Since"] = cached_meta["last_modified"]

        try:
            response = self.session.get(url, headers=headers, timeout=self.timeout_seconds)
            if response.status_code == 304 and cached_text is not None:
                cached_meta["last_checked_at_utc"] = utc_now_iso()
                cached_meta["cache_status"] = "validated-cache"
                metadata_path.write_text(json.dumps(cached_meta, indent=2), encoding="utf-8")
                return cached_text, cached_meta
            if response.status_code == 404:
                raise DataNotAvailableError(f"Source resource not found: {url}")
            response.raise_for_status()
            text = response.text
            retrieved_at = utc_now_iso()
            metadata = {
                "url": url,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "source_updated_at": response.headers.get("Last-Modified"),
                "source_retrieved_at_utc": retrieved_at,
                "last_checked_at_utc": retrieved_at,
                "cache_status": "fresh",
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            body_path.write_text(text, encoding="utf-8")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return text, metadata
        except DataNotAvailableError:
            raise
        except (requests.RequestException, OSError) as exc:
            if cached_text is not None and self.allow_stale_cache:
                cached_meta = dict(cached_meta)
                cached_meta["cache_status"] = "stale-cache"
                cached_meta["last_check_error"] = type(exc).__name__
                return cached_text, cached_meta
            raise DataProviderError(f"Could not fetch source data from {url}: {exc}") from exc
