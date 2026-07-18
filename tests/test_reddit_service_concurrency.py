"""Reddit-side Playwright concurrency gate (item 5): concurrent subreddit polls
must never exceed the configured in-flight limit, and gate saturation must fall
through (return None) without feeding the circuit breaker."""
from __future__ import annotations

import threading
import time

from services import browser_service, reddit_service


def _call_scrape(sub: str = "testsub"):
    return reddit_service._scrape_via_playwright(
        sub=sub,
        listing="new",
        tf="day",
        max_items=5,
        cutoff_unix=None,
        seen_lookup=None,
        include_nsfw=False,
        include_spoiler=False,
        wanted_flairs=set(),
        mode="image",
    )


def test_gate_caps_concurrent_playwright_callers(monkeypatch):
    inflight = {"now": 0, "max": 0}
    lock = threading.Lock()

    def fake_ensure_ready() -> bool:
        return True

    def fake_fetch_json(url, timeout_ms=15_000):
        with lock:
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
        time.sleep(0.15)
        with lock:
            inflight["now"] -= 1
        return {"data": {"children": []}}

    monkeypatch.setattr(browser_service, "ensure_ready", fake_ensure_ready)
    monkeypatch.setattr(browser_service, "fetch_json", fake_fetch_json)
    reddit_service._playwright_breaker_open_until = 0.0

    threads = [threading.Thread(target=_call_scrape, args=(f"sub{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert inflight["max"] <= reddit_service._PLAYWRIGHT_GATE_LIMIT


def test_gate_saturation_returns_none_without_breaker_penalty(monkeypatch, capsys):
    monkeypatch.setattr(reddit_service, "_PLAYWRIGHT_GATE_WAIT_SECONDS", 0.1)
    reddit_service._playwright_consecutive_failures = 0
    reddit_service._playwright_breaker_open_until = 0.0

    # Exhaust the gate.
    held = 0
    while reddit_service._playwright_call_gate.acquire(timeout=0):
        held += 1
    try:
        result = _call_scrape()
        assert result is None
        assert "call gate saturated" in capsys.readouterr().out
        # Saturation is congestion, not failure: breaker counter untouched.
        assert reddit_service._playwright_consecutive_failures == 0
    finally:
        for _ in range(held):
            reddit_service._playwright_call_gate.release()


def test_successful_scrape_passes_through_gate(monkeypatch):
    monkeypatch.setattr(browser_service, "ensure_ready", lambda: True)
    monkeypatch.setattr(
        browser_service, "fetch_json", lambda url, timeout_ms=15_000: {"data": {"children": []}}
    )
    reddit_service._playwright_breaker_open_until = 0.0

    assert _call_scrape() == []
    # Gate fully released afterwards.
    for _ in range(reddit_service._PLAYWRIGHT_GATE_LIMIT):
        assert reddit_service._playwright_call_gate.acquire(timeout=0)
    for _ in range(reddit_service._PLAYWRIGHT_GATE_LIMIT):
        reddit_service._playwright_call_gate.release()
