from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _reset_browser_and_reddit_globals():
    """Save/restore browser_service and reddit_service module globals so tests
    can freely monkeypatch executor/context/semaphore/breaker state without
    leaking it into other tests."""
    try:
        from services import browser_service
    except ImportError:
        browser_service = None
    try:
        from services import reddit_service
    except ImportError:
        reddit_service = None

    browser_names = (
        "_pw",
        "_context",
        "_executor",
        "_session_valid",
        "_last_session_check",
        "_last_resync_attempt",
        "_last_start_attempt",
        "_last_profile_path",
        "_primary_profile",
        "_runtime_profile",
        "_launched_profile",
        "_health_hook",
    )
    reddit_names = (
        "_playwright_consecutive_failures",
        "_playwright_breaker_open_until",
        "_ratelimit_until",
    )

    saved_browser = (
        {n: getattr(browser_service, n) for n in browser_names if hasattr(browser_service, n)}
        if browser_service
        else {}
    )
    saved_reddit = (
        {n: getattr(reddit_service, n) for n in reddit_names if hasattr(reddit_service, n)}
        if reddit_service
        else {}
    )

    yield

    for name, value in saved_browser.items():
        setattr(browser_service, name, value)
    if browser_service:
        browser_service._fetch_dispatch_semaphore = browser_service._PriorityDispatchGate()
    for name, value in saved_reddit.items():
        setattr(reddit_service, name, value)
    if reddit_service:
        reddit_service._playwright_call_gate = threading.BoundedSemaphore(
            value=reddit_service._PLAYWRIGHT_GATE_LIMIT
        )


@pytest.fixture(autouse=True)
def _shutdown_priority_work_schedulers(monkeypatch):
    """WatcherManager/CommandRouter fall back to constructing their own
    PriorityWorkScheduler when a test doesn't pass one explicitly, and every
    PriorityWorkScheduler() spins up max(4, cpu_count)+1 live daemon threads
    with no automatic cleanup. Left unchecked, a full test run leaks
    hundreds of never-joined threads. Track every instance created during
    the test and signal shutdown at teardown -- this only stops threads
    from continuing to serve the (now-torn-down) test's queue, it does not
    change sizing, so tests that assert the real default worker count
    (e.g. test_uses_all_available_cpu_cores_by_default) are unaffected."""
    from services import priority_scheduler as _ps

    created: list[_ps.PriorityWorkScheduler] = []
    original_init = _ps.PriorityWorkScheduler.__init__

    def _tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(_ps.PriorityWorkScheduler, "__init__", _tracked_init)
    yield
    for scheduler in created:
        scheduler.shutdown(wait=False)


@pytest.fixture(autouse=True)
def _no_dotenv_provider_keys(monkeypatch):
    """GeminiSettings(api_key=None) is NOT keyless: the provider chain falls
    back to real .env openrouter/groq keys, so any test that exhausts the
    fake gemini path would silently make live LLM calls. Tests that want the
    openrouter/groq path must set keys explicitly on their settings object."""
    try:
        from services.resumes import listing
    except ImportError:
        yield
        return
    monkeypatch.setattr(listing, "_openrouter_api_key", lambda: None)
    monkeypatch.setattr(listing, "_groq_api_key", lambda: None)
    yield


@pytest.fixture(autouse=True)
def _clear_profile_catalog_cache():
    """load_structured_profile memoizes by file mtimes; tests that rewrite
    profile files in tmp_path must never see another test's parse."""
    try:
        from services.resumes import structured
    except ImportError:
        yield
        return
    structured._PROFILE_CATALOG_CACHE.clear()
    yield
    structured._PROFILE_CATALOG_CACHE.clear()
