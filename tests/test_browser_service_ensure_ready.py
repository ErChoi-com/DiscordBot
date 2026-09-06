"""Single-flight ensure_ready (items 1/2): concurrent callers must not stack
recovery work — only one caller runs the session check/resync while others
bounce off the lock, and cold-start respects the start cooldown."""
from __future__ import annotations

import concurrent.futures
import threading
import time

from services import browser_service


def test_concurrent_ensure_ready_runs_session_check_once(monkeypatch):
    calls = {"count": 0}
    started = threading.Event()

    def fake_check_session() -> bool:
        calls["count"] += 1
        started.set()
        time.sleep(0.4)  # hold the lock long enough for other callers to overlap
        return True

    monkeypatch.setattr(browser_service, "_do_check_session", fake_check_session)
    browser_service._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    browser_service._context = object()
    browser_service._session_valid = True
    # Stale relative to *now*, not an absolute 0.0. The staleness test is
    # `monotonic() - _last_session_check > _SESSION_CHECK_INTERVAL`, and on Linux
    # monotonic() counts from boot -- so 0.0 only reads as stale once the machine
    # has been up longer than the interval (30 min). It always is on a dev box
    # and never is on a freshly booted CI runner, which is why this passed
    # locally and failed in CI with `assert 0 == 1`: the check never ran.
    browser_service._last_session_check = (
        time.monotonic() - browser_service._SESSION_CHECK_INTERVAL - 1
    )

    try:
        threads = [threading.Thread(target=browser_service.ensure_ready) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert calls["count"] == 1, "only the lock holder should run the session check"
    finally:
        browser_service._executor.shutdown(wait=False)


def test_ensure_ready_when_locked_returns_is_ready_without_work(monkeypatch):
    start_calls = {"count": 0}
    monkeypatch.setattr(browser_service, "start", lambda *a, **k: start_calls.update(count=start_calls["count"] + 1) or True)

    browser_service._executor = None
    browser_service._context = None

    assert browser_service._ensure_ready_lock.acquire(blocking=False)
    try:
        # Lock held by "another caller": must return is_ready() (False) immediately.
        assert browser_service.ensure_ready() is False
        assert start_calls["count"] == 0
    finally:
        browser_service._ensure_ready_lock.release()


def test_ensure_ready_respects_start_cooldown(monkeypatch):
    start_calls = {"count": 0}

    def fake_start(path=None):
        start_calls["count"] += 1
        return True

    monkeypatch.setattr(browser_service, "start", fake_start)
    browser_service._executor = None
    browser_service._context = None
    browser_service._last_start_attempt = time.monotonic()  # just attempted

    assert browser_service.ensure_ready() is False
    assert start_calls["count"] == 0, "cooldown must prevent immediate restart attempts"
