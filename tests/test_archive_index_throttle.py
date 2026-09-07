"""ensure_index is free when it was checked seconds ago.

The enrichment prefilter (ats_service._needs_enrichment) calls it once per
board, from up to fifty threads per platform. Each call took the build lock,
opened a connection and stat-ed every archive zip; a py-spy dump found 38-68
threads inside it at every sample while the Discord gateway fell 42s behind.
Archives change about weekly, so a check made in the last minute stands.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.jba import archive_index as ai  # noqa: E402


def _count_checks(monkeypatch):
    calls = {"n": 0}
    real = ai._stale_sources

    def _counting(conn):
        calls["n"] += 1
        return real(conn)

    monkeypatch.setattr(ai, "_stale_sources", _counting)
    monkeypatch.setattr(ai, "_last_checked", {})
    return calls


def test_fifty_calls_in_a_row_check_the_archives_once(monkeypatch):
    calls = _count_checks(monkeypatch)
    for _ in range(50):
        ai.ensure_index()
    assert calls["n"] == 1


def test_fifty_threads_at_once_check_the_archives_once(monkeypatch):
    calls = _count_checks(monkeypatch)
    gate = threading.Barrier(50)

    def _go():
        gate.wait(5)
        ai.ensure_index()

    threads = [threading.Thread(target=_go) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert calls["n"] == 1


def test_the_check_is_repeated_once_the_interval_has_passed(monkeypatch):
    calls = _count_checks(monkeypatch)
    now = {"t": 1000.0}
    monkeypatch.setattr(ai.time, "monotonic", lambda: now["t"])
    ai.ensure_index()
    now["t"] += ai._CHECK_INTERVAL_S - 1
    ai.ensure_index()
    assert calls["n"] == 1
    now["t"] += 2
    ai.ensure_index()
    assert calls["n"] == 2


def test_force_always_rebuilds(monkeypatch):
    """A forced call never takes the shortcut: it reaches archive_files every
    time, however recently the archives were checked."""
    _count_checks(monkeypatch)
    ai.ensure_index()                       # a fresh check is now on record
    seen = {"n": 0}
    real = ai.archive_files

    def _counting():
        seen["n"] += 1
        return real()

    monkeypatch.setattr(ai, "archive_files", _counting)
    ai.ensure_index(force=True)
    ai.ensure_index(force=True)
    assert seen["n"] == 2


def test_a_different_index_path_is_checked_on_its_own(monkeypatch, tmp_path):
    """Tests point the index at a fresh tmp dir each; a check made against
    one path must not stand in for another."""
    calls = _count_checks(monkeypatch)
    ai.ensure_index()
    monkeypatch.setattr(ai, "_INDEX_PATH", tmp_path / "other" / "archive_index.db")
    monkeypatch.setattr(ai, "_JOBS_DIR", tmp_path / "other")
    ai.ensure_index()
    assert calls["n"] == 2
