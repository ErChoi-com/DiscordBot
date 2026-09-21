"""Deep stress test and edge-case verification for the broader massive architecture.

Validates:
1. PriorityWorkScheduler: Interactive command preemption over background ATS tasks.
2. Concurrent ETag SQLite commits across multiple worker threads (WAL mode resilience).
3. Geo DB cold load thundering herd protection (_geo_lock).
4. Dedup and Archive Index resilience against malformed, null, and adversarial job rows.
5. Dynamic fleet mutation (adding/deleting slugs mid-traversal) without cursor drift.
"""
from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from services import ats_etags, ats_service, ats_traversal
from services.jba import archive_index, geo_db, geo_priority, merge_data
from services.priority_scheduler import BACKGROUND, INTERACTIVE, PriorityWorkScheduler


# ── 1. Scheduler Priority Preemption Under Heavy Background Load ────────────

def test_priority_scheduler_interactive_preemption_over_ats():
    """Interactive Discord commands must jump ahead of queued background ATS tasks."""
    # 2 general worker threads to induce queueing, 0 reserved for interactive so both take background work
    scheduler = PriorityWorkScheduler(max_workers=2, reserved_interactive=0)

    execution_order = []
    lock = threading.Lock()
    active_gate = threading.Event()
    started = threading.Barrier(3)

    def make_blocking_task(name: str):
        def _task():
            started.wait(timeout=5.0)
            active_gate.wait(timeout=5.0)
            with lock:
                execution_order.append(name)
            return name
        return _task

    def make_task(name: str, duration: float = 0.0):
        def _task():
            if duration > 0:
                time.sleep(duration)
            with lock:
                execution_order.append(name)
            return name
        return _task

    try:
        # 1. Fill the 2 active worker slots with blocking tasks
        f_block1 = scheduler.submit(make_blocking_task("active_bg_1"), tier=BACKGROUND, label="ats_scrape")
        f_block2 = scheduler.submit(make_blocking_task("active_bg_2"), tier=BACKGROUND, label="ats_scrape")
        started.wait(timeout=5.0)  # Wait until both workers are actively executing

        # 2. Queue 5 background tasks (simulating remaining ATS platforms)
        bg_futures = [
            scheduler.submit(make_task(f"queued_bg_{i}"), tier=BACKGROUND, label="ats_scrape")
            for i in range(5)
        ]

        # 3. An interactive user command arrives (.resumebuild)
        f_interactive = scheduler.submit(make_task("user_command"), tier=INTERACTIVE, label="resume_build")

        # 4. Release active workers to process the queue
        active_gate.set()

        # Wait for all to complete
        f_interactive.result(timeout=5.0)
        for f in bg_futures:
            f.result(timeout=5.0)
        f_block1.result(timeout=5.0)
        f_block2.result(timeout=5.0)

        with lock:
            # User command MUST execute before any queued background task
            user_idx = execution_order.index("user_command")
            first_queued_bg_idx = min(execution_order.index(f"queued_bg_{i}") for i in range(5))
            assert user_idx < first_queued_bg_idx, (
                f"Interactive command was starved behind queued background tasks: {execution_order}"
            )
    finally:
        scheduler.shutdown(wait=True)


# ── 2. Concurrent SQLite ETag Commits Across Multiple Threads ───────────────

def test_concurrent_etag_commits_wal_resilience(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Multiple ATS platforms finishing simultaneously and committing to ats_etags.db."""
    test_db = tmp_path / "test_ats_etags.db"
    monkeypatch.setattr(ats_etags, "_DB_PATH", test_db)

    platforms = [f"plat_{i}" for i in range(8)]
    errors = []

    def platform_scrape_worker(platform: str):
        try:
            # 1. Reset platform
            ats_etags.reset(platform)

            # 2. Record 20 boards
            for i in range(20):
                slug = f"{platform}_slug_{i}"
                url = f"https://{platform}.com/{slug}"
                ats_etags.record(
                    platform,
                    slug,
                    url,
                    {"ETag": f'W/"hash_{i}"', "Last-Modified": "Sun, 20 Sep 2026 12:00:00 GMT"},
                )

            # 3. Promote boards that survived deadline
            promoted_slugs = [f"{platform}_slug_{i}" for i in range(15)]
            ats_etags.promote(platform, promoted_slugs)

            # 4. Commit to SQLite (concurrent write)
            ats_etags.commit(platform)

            # 5. Read back conditional headers in next cycle (after reset)
            ats_etags.reset(platform)
            for slug in promoted_slugs:
                url = f"https://{platform}.com/{slug}"
                hdrs = ats_etags.conditional_headers(platform, url)
                assert "If-None-Match" in hdrs or "If-Modified-Since" in hdrs
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=platform_scrape_worker, args=(p,)) for p in platforms]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "ETag commit deadlocked under concurrency"

    assert not errors, f"Concurrent ETag commits produced database errors: {errors}"

    # Verify rows in SQLite
    conn = sqlite3.connect(str(test_db))
    total_stored = conn.execute("SELECT COUNT(*) FROM board_etags").fetchone()[0]
    conn.close()
    assert total_stored == 8 * 15, f"Expected 120 stored validators, found {total_stored}"


# ── 3. Geo DB Cold Load Thundering Herd Protection ──────────────────────────

def test_geo_db_thundering_herd_thread_safety(monkeypatch: pytest.MonkeyPatch):
    """Multiple threads triggering cold geo lookup load simultaneously."""
    # Reset load state
    monkeypatch.setattr(ats_service, "_geo_loaded", False)
    monkeypatch.setattr(ats_service, "_geo_failures", 0)
    monkeypatch.setattr(ats_service, "_geo_next_retry", 0.0)
    load_count = 0
    original_load = ats_service._load_geo_into_globals

    def counting_load():
        nonlocal load_count
        time.sleep(0.01)  # Simulate non-trivial load
        load_count += 1
        original_load()

    monkeypatch.setattr(ats_service, "_load_geo_into_globals", counting_load)

    errors = []

    def reader():
        try:
            for _ in range(5):
                ats_service._ensure_geo_loaded()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"Thundering herd reader threw errors: {errors}"
    # Must only load ONCE despite 20 threads hitting it concurrently
    assert load_count == 1, f"Expected 1 load, but found {load_count} parallel loads (lock failed)"


# ── 4. Malformed, Null, and Adversarial Job Rows in Dedup ────────────────────

def test_archive_index_adversarial_rows_resilience(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Dedup and archive index handle corrupted rows without crashing."""
    monkeypatch.setattr(archive_index, "_INDEX_PATH", tmp_path / "test_archive.db")

    adversarial_jobs = [
        {},  # Empty
        {"job_url": None, "company": None},  # Nulls
        {"job_url": "", "date_posted": None},  # Empty string
        {"job_url": "https://example.com/1", "date_posted": "not_an_iso_date"},
        {"job_url": "https://example.com/2", "date_posted": "2026-09-20T00:00:00Z"},
        {"job_url": "https://example.com/2", "date_posted": "2026-09-20T00:00:00Z"},  # Duplicate
        {"company": "workday_corp", "job_url": "", "title": "Test Engineer"},
        {"job_url": "https://example.com/3", "description": "A" * 100_000},  # Large description
        {"job_url": "https://example.com/4", "location": "Toronto \x00 invalid null byte"},  # Null byte in text
    ]

    # Should not raise any exception
    kept, dropped = archive_index.filter_new_listings(adversarial_jobs)
    assert isinstance(kept, list)
    assert dropped >= 1  # Duplicate https://example.com/2 dropped
    assert any(j.get("job_url") == "https://example.com/2" for j in kept)


# ── 5. Dynamic Fleet Mutation (Adding/Removing Slugs Mid-Traversal) ──────────

def test_fleet_mutation_mid_traversal_stability():
    """Deleting and inserting slugs between cycles maintains smooth bisect resume."""
    # Start with 500 slugs
    original_fleet = [f"slug_{i:04d}" for i in range(500)]
    order = ats_traversal.stable_order(original_fleet)

    # Cycle 1: Ask 50 slugs
    c1_slugs, last_digest, wrapped = ats_traversal.next_slice(order, None, count=50)
    assert len(c1_slugs) == 50
    assert not wrapped
    assert last_digest is not None

    # Now mutate the fleet dynamically (as happens weekly when harvest updates):
    # 1. Delete 50 slugs, including the one last_digest points to!
    mutated_fleet = [s for s in original_fleet if s != c1_slugs[-1] and int(s.split("_")[1]) % 10 != 0]
    # 2. Add 50 brand new slugs
    mutated_fleet.extend([f"new_harvest_{i:04d}" for i in range(50)])

    new_order = ats_traversal.stable_order(mutated_fleet)

    # Cycle 2: Resume from last_digest (even though last_digest was deleted from the fleet!)
    c2_slugs, new_digest, wrapped = ats_traversal.next_slice(new_order, last_digest, count=50)
    assert len(c2_slugs) == 50
    # No crash, no reset to 0, no duplicate of c1_slugs
    overlap = set(c1_slugs) & set(c2_slugs)
    assert not overlap, f"Fleet mutation caused repeat of completed slugs: {overlap}"
