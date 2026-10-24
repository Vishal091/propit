"""Create a ZIP backup of PropIt source, data, predictions and trained artifacts.

Intentionally excludes .env, virtual environments, downloaded caches, and prior
backup archives. It never deletes existing backups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_backup_files(root: Path = ROOT) -> list[Path]:
    """Collect a reproducible, allow-listed set of workspace files."""
    found: set[Path] = set()
    for relative in ("README.md", ".gitignore", "requirements.txt"):
        path = root / relative
        if path.is_file():
            found.add(path)

    for directory in ("src", "scripts", "tests", ".github/workflows"):
        folder = root / directory
        if folder.exists():
            found.update(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in {".py", ".html", ".yml", ".yaml", ".txt", ".md"})

    for directory, extensions in (
        ("data", {".csv", ".json", ".jsonl"}),
        ("predictions", {".csv", ".json"}),
        ("artifacts", {".joblib", ".json"}),
    ):
        folder = root / directory
        if folder.exists():
            found.update(
                path for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            )
    return sorted(found, key=lambda path: path.relative_to(root).as_posix().casefold())


def create_backup(output: Path | None = None, root: Path = ROOT) -> dict:
    """Create a timestamped ZIP with per-file checksums and a manifest."""
    root = root.resolve()
    generated = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = generated.strftime("%Y%m%dT%H%M%SZ")
    if output is None:
        target = root / "backups" / f"propit-backup-{stamp}.zip"
    else:
        target = output if output.is_absolute() else root / output
    target.parent.mkdir(parents=True, exist_ok=True)

    files = collect_backup_files(root)
    entries = []
    for path in files:
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    manifest = {
        "product": "PropIt",
        "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
        "root": str(root),
        "files_count": len(entries),
        "files": entries,
        "excluded": [".env", ".venv", ".cache", "previous backups", "Python bytecode"],
    }

    target_resolved = target.resolve()
    if target_resolved in {path.resolve() for path in files}:
        raise ValueError("Backup output cannot overwrite one of its source files.")
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, arcname=path.relative_to(root).as_posix())
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(target), "files_count": len(entries), "size_bytes": target.stat().st_size, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="optional output ZIP path (relative paths are relative to the repository root)")
    args = parser.parse_args()
    result = create_backup(args.output)
    print(f"Backup created: {result['path']}")
    print(f"Included files: {result['files_count']}")
    print(f"ZIP size: {result['size_bytes']:,} bytes")
    print("Excluded secrets (.env), virtual environments, caches, and previous backups.")


if __name__ == "__main__":
    main()
