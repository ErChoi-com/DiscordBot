"""The ZipRecruiter scraper's private Playwright launch must go through
browser_service's app-wide dispatch slot: it used to launch a second Chromium
process invisible to the priority gate, so resume scrapes could stall behind
it with no accounting anywhere."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import browser_service
from services.job_service import scrape_ziprecruiter_postings


def test_ziprecruiter_returns_empty_when_slot_unavailable(monkeypatch):
    """Slot denied -> no browser is ever launched and [] comes back."""
    launched = []

    def deny(op_name, timeout_s, detail="", priority=False):
        assert op_name == "ziprecruiter"
        return False

    monkeypatch.setattr(browser_service, "acquire_browser_work_slot", deny)
    monkeypatch.setattr(
        "services.job_service._scrape_ziprecruiter_with_playwright",
        lambda *a, **k: launched.append(1) or [],
    )
    assert scrape_ziprecruiter_postings("python", "Toronto") == []
    assert launched == []


def test_ziprecruiter_releases_slot_even_when_scrape_raises(monkeypatch):
    """Every acquired slot must be released, including on a crash — a leaked
    slot would deadlock all future browser work app-wide."""
    events = []

    monkeypatch.setattr(
        browser_service,
        "acquire_browser_work_slot",
        lambda *a, **k: events.append("acquire") or True,
    )
    monkeypatch.setattr(
        browser_service,
        "release_browser_work_slot",
        lambda: events.append("release"),
    )

    def boom(*a, **k):
        events.append("scrape")
        raise RuntimeError("browser exploded")

    monkeypatch.setattr("services.job_service._scrape_ziprecruiter_with_playwright", boom)
    try:
        scrape_ziprecruiter_postings("python", "Toronto")
    except RuntimeError:
        pass
    assert events == ["acquire", "scrape", "release"]


def test_ziprecruiter_slot_actually_serializes_against_the_real_gate(monkeypatch):
    """Two-thread probe against the REAL _PriorityDispatchGate (not a fake):
    while a fake ZipRecruiter scrape holds the slot, a second bulk acquire
    yields/times out instead of running concurrently."""
    gate = browser_service._PriorityDispatchGate()
    monkeypatch.setattr(browser_service, "_fetch_dispatch_semaphore", gate)

    in_scrape = threading.Event()
    finish_scrape = threading.Event()
    overlap = []

    def slow_scrape(*a, **k):
        in_scrape.set()
        finish_scrape.wait(timeout=5)
        return []

    monkeypatch.setattr(
        "services.job_service._scrape_ziprecruiter_with_playwright", slow_scrape
    )

    worker = threading.Thread(
        target=lambda: scrape_ziprecruiter_postings("python", "Toronto")
    )
    worker.start()
    try:
        assert in_scrape.wait(timeout=5)
        # The slot is held by the scrape: a concurrent bulk acquire must fail.
        overlap.append(gate.acquire(timeout=0.3, priority=False))
    finally:
        finish_scrape.set()
        worker.join(timeout=5)
    assert overlap == [False]
    # After the scrape finishes the slot must be free again.
    assert gate.acquire(timeout=1.0) is True
    gate.release()
