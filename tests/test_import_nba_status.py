from __future__ import annotations

import sys

import pytest

from scripts import import_nba_status


def test_importer_prints_expected_report_availability_failure_without_traceback(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise ValueError(
            "Downloaded the official NBA injury report published 2026-10-28 02:45 PM ET, "
            "but it contains game-date rows for 2026-10-28 and none for the requested game date "
            "2026-10-29. The target-date report may not have been published yet. "
            "No current snapshot was written; retry after the NBA publishes the report for that game date."
        )

    monkeypatch.setattr(import_nba_status, "import_report", fail)
    monkeypatch.setattr(sys, "argv", ["import_nba_status", "--game-date", "2026-10-29"])

    with pytest.raises(SystemExit) as exc:
        import_nba_status.main()

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert captured.err.startswith("ERROR: Downloaded the official NBA injury report")
    assert "no current snapshot was written" in captured.err.lower()
    assert "Traceback" not in captured.err
