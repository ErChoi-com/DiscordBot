"""Regression guard for the dispatch gate (item 1 baseline).

The fetch dispatch semaphore must fast-fail with a "dispatch saturated" log when
the single dispatch slot is held, and must always be released after a fetch.
"""
from __future__ import annotations

import concurrent.futures
import time

from services import browser_service


class _ImmediateExecutor:
    """Stands in for browser_service._executor without a real thread pool."""

    def submit(self, fn, *args, **kwargs):
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            future.set_exception(exc)
        return future


def test_fetch_json_fast_fails_when_dispatch_saturated(capsys):
    assert browser_service._fetch_dispatch_semaphore.acquire(timeout=0)
    try:
        browser_service._executor = _ImmediateExecutor()
        browser_service._context = object()  # truthy sentinel bypasses the early-return guard

        start = time.monotonic()
        result = browser_service.fetch_json("https://example.com/fake.json", timeout_ms=1000)
        elapsed = time.monotonic() - start

        assert result is None
        # Must fail at slot_wait, never the full dispatch timeout.
        assert elapsed < browser_service._PLAYWRIGHT_FETCH_QUEUE_WAIT_SECONDS + 1.0
        assert "dispatch saturated" in capsys.readouterr().out
    finally:
        browser_service._fetch_dispatch_semaphore.release()


def test_fetch_html_fast_fails_when_dispatch_saturated(capsys):
    assert browser_service._fetch_dispatch_semaphore.acquire(timeout=0)
    try:
        browser_service._executor = _ImmediateExecutor()
        browser_service._context = object()

        result = browser_service.fetch_html("https://example.com/page", timeout_ms=1000)

        assert result is None
        assert "dispatch saturated" in capsys.readouterr().out
    finally:
        browser_service._fetch_dispatch_semaphore.release()


def test_fetch_json_releases_semaphore_after_call():
    browser_service._executor = _ImmediateExecutor()
    browser_service._context = object()

    # _do_fetch_json runs for real against the sentinel context and returns None;
    # what matters is the slot is free again afterwards.
    browser_service.fetch_json("https://example.com/fake.json", timeout_ms=1000)

    assert browser_service._fetch_dispatch_semaphore.acquire(timeout=0)
    browser_service._fetch_dispatch_semaphore.release()


def test_acquire_fetch_slot_available_and_saturated(capsys):
    assert browser_service._acquire_fetch_slot("op_a", 1.0, "https://example.com") is True
    # Slot now held: second acquire must fast-fail and log the op name.
    assert browser_service._acquire_fetch_slot("op_b", 0.1, "https://example.com") is False
    out = capsys.readouterr().out
    assert "op_b" in out and "dispatch saturated" in out
    browser_service._fetch_dispatch_semaphore.release()


def test_saturation_emits_health_event():
    events: list[tuple[str, object]] = []
    browser_service.set_health_hook(lambda event, value=None: events.append((event, value)))
    assert browser_service._fetch_dispatch_semaphore.acquire(timeout=0)
    try:
        assert browser_service._acquire_fetch_slot("fetch_json", 0.1, "u") is False
    finally:
        browser_service._fetch_dispatch_semaphore.release()
    assert ("dispatch_saturated", "fetch_json") in events


def test_raising_health_hook_does_not_break_fast_fail():
    def bad_hook(event, value=None):
        raise RuntimeError("hook exploded")

    browser_service.set_health_hook(bad_hook)
    assert browser_service._fetch_dispatch_semaphore.acquire(timeout=0)
    try:
        # Must still return False cleanly despite the raising hook.
        assert browser_service._acquire_fetch_slot("fetch_json", 0.1, "u") is False
    finally:
        browser_service._fetch_dispatch_semaphore.release()
        browser_service.set_health_hook(None)


# ---------------------------------------------------------------------------
# Priority tier: interactive resume/cover scrapes vs bulk reddit polls
# ---------------------------------------------------------------------------


def _hold_priority_waiter():
    """Start a real priority waiter parked on a busy gate. Returns
    (slot_release_fn, thread) — call the release fn to let the waiter win."""
    import threading

    gate = browser_service._fetch_dispatch_semaphore
    assert gate.acquire(timeout=0)  # make the gate busy
    result: dict = {}

    def _wait():
        result["ok"] = gate.acquire(timeout=10.0, priority=True)
        if result["ok"]:
            gate.release()

    thread = threading.Thread(target=_wait)
    thread.start()
    deadline = time.monotonic() + 2.0
    while gate.priority_waiting == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gate.priority_waiting == 1

    def _finish():
        gate.release()
        thread.join(timeout=5)
        assert result.get("ok") is True

    return _finish


def test_bulk_fetch_yields_while_priority_waiter_present(capsys):
    finish = _hold_priority_waiter()
    try:
        start = time.monotonic()
        assert browser_service._acquire_fetch_slot("fetch_json", 5.0, "u") is False
        # Must refuse immediately, not wait out the bulk window.
        assert time.monotonic() - start < 1.0
        assert "yielding dispatch slot to a priority fetch" in capsys.readouterr().out
    finally:
        finish()


def test_priority_fetch_outwaits_bulk_window_and_wins_slot():
    import threading

    gate = browser_service._fetch_dispatch_semaphore
    assert gate.acquire(timeout=0)
    releaser = threading.Timer(0.4, gate.release)
    releaser.start()
    try:
        # Bulk callers give up after ~3s wait or yield instantly; the priority
        # caller keeps waiting and takes the slot the moment it frees.
        assert browser_service._acquire_fetch_slot("fetch_html", 3.0, "u", priority=True) is True
    finally:
        releaser.join()
        gate.release()
    assert gate.priority_waiting == 0


def test_priority_waiter_counter_resets_after_timeout():
    gate = browser_service._fetch_dispatch_semaphore
    assert gate.acquire(timeout=0)
    try:
        assert browser_service._acquire_fetch_slot("fetch_html", 0.2, "u", priority=True) is False
    finally:
        gate.release()
    assert gate.priority_waiting == 0


def test_priority_yield_emits_health_event():
    events: list[tuple[str, object]] = []
    browser_service.set_health_hook(lambda event, value=None: events.append((event, value)))
    finish = _hold_priority_waiter()
    try:
        assert browser_service._acquire_fetch_slot("fetch_json", 0.1, "u") is False
    finally:
        finish()
        browser_service.set_health_hook(None)
    assert ("dispatch_yield", "fetch_json") in events


def test_late_arriving_priority_beats_already_queued_bulk_waiter():
    """THE regression test for the original design hole: a bulk caller already
    parked inside the gate's wait must NOT win the slot over a priority caller
    that arrives after it. A plain FIFO semaphore fails this 8/8 (verified);
    the condition-variable gate must pass it every time."""
    import threading

    gate = browser_service._fetch_dispatch_semaphore
    for _ in range(4):
        assert gate.acquire(timeout=0)  # busy: both callers must queue
        order: list[str] = []
        lock = threading.Lock()

        def _bulk():
            ok = gate.acquire(timeout=5.0)
            with lock:
                order.append(f"bulk:{ok}")
            if ok:
                gate.release()

        def _priority():
            time.sleep(0.15)  # arrive strictly AFTER the bulk waiter parked
            ok = gate.acquire(timeout=5.0, priority=True)
            with lock:
                order.append(f"priority:{ok}")
            if ok:
                gate.release()

        t_bulk = threading.Thread(target=_bulk)
        t_priority = threading.Thread(target=_priority)
        t_bulk.start()
        t_priority.start()
        time.sleep(0.4)  # both are now waiting; priority arrived second
        gate.release()  # free the slot exactly once
        t_bulk.join(timeout=5)
        t_priority.join(timeout=5)

        # Priority must take the slot FIRST. The parked bulk waiter either
        # yields (wakes, sees the priority waiter, steps aside) or acquires
        # only after the priority caller finished — never wins FIFO-style.
        successes = [item for item in order if item.endswith(":True")]
        assert successes, f"nobody acquired: {order}"
        assert successes[0] == "priority:True", f"bulk won FIFO-style: {order}"
    assert gate.priority_waiting == 0
