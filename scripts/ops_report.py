"""Generate a dated JSON and Markdown PropIt operations report."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.operations import build_health_report, render_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/operations"))
    args = parser.parse_args()
    generated = datetime.now(timezone.utc)
    day = generated.date().isoformat()
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_health_report()
    json_path = output_dir / f"operations-{day}.json"
    md_path = output_dir / f"operations-{day}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"Operations status: {report['overall']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"Checks: {report['summary']['ok']} OK, {report['summary']['warning']} warning(s), {report['summary']['errors']} error(s)")


if __name__ == "__main__":
    main()
