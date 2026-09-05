from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _isolate_resume_telemetry(tmp_path, monkeypatch):
    """Keep listing/cover telemetry and caches out of the real .resume_cache/.

    These are module-level absolute paths, and every writer swallows OSError, so
    a test that reaches one pollutes real state silently. That has already
    happened twice in this repo -- synthetic postings in the job archive
    (see _isolate_job_archive in test_watcher_dedup_resilience) and 7,442
    synthetic records in .interaction_events.log -- and .resume_cache/
    scrape_events.jsonl still carries 4 example.invalid lines from some earlier
    run. test_cover_letter.py isolates these, but a module-scoped fixture only
    protects that module; anything else reaching listing.py writes for real.

    Isolated here so the protection does not depend on which file the next test
    lands in. test_cover_letter.py's own fixture still applies on top -- it also
    stubs provider keys, which this deliberately does not touch.
    """
    try:
        import services.resumes.cover as cover_module
        import services.resumes.listing as listing_module
    except ImportError:
        yield
        return
    monkeypatch.setattr(listing_module, "STRUCTURED_CACHE_ROOT", tmp_path / "resume_cache", raising=False)
    monkeypatch.setattr(listing_module, "STRUCTURED_SELECTION_CACHE_DIR", tmp_path / "sel_cache", raising=False)
    monkeypatch.setattr(listing_module, "STRUCTURED_TELEMETRY_PATH", tmp_path / "telemetry.jsonl", raising=False)
    monkeypatch.setattr(listing_module, "SCRAPE_TELEMETRY_PATH", tmp_path / "scrape.jsonl", raising=False)
    monkeypatch.setattr(cover_module, "COVER_TELEMETRY_PATH", tmp_path / "cover.jsonl", raising=False)
    yield


@pytest.fixture(autouse=True)
def _isolate_interaction_event_log(tmp_path, monkeypatch):
    """Keep interaction telemetry out of the real .interaction_events.log.

    views.log_interaction_event appends to a path at the repo root -- the same
    file the live bot writes. The modal tests call it heavily, so before this
    fixture existed 7,442 of the file's 8,016 lines were synthetic
    ``channel_id=123`` records: 93% test noise buried the real interactions it
    exists to record.

    Autouse and in conftest rather than in one test module, because the writer
    is reachable from any test that builds a view or a modal, and the failure is
    silent -- log_interaction_event swallows every exception.
    """
    try:
        from ui import views
    except ImportError:
        yield
        return
    monkeypatch.setattr(views, "INTERACTION_EVENTS_PATH", tmp_path / "interaction_events.log")
    yield


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
    the test and shut them down at teardown, waiting for the threads to
    actually exit. It does not change sizing, so tests that assert the real
    default worker count (e.g. test_uses_all_available_cpu_cores_by_default)
    are unaffected."""
    from services import priority_scheduler as _ps

    created: list[_ps.PriorityWorkScheduler] = []
    original_init = _ps.PriorityWorkScheduler.__init__

    def _tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(_ps.PriorityWorkScheduler, "__init__", _tracked_init)
    yield
    for scheduler in created:
        # wait=True, which this could not previously afford. The aging thread
        # used to sleep out its sweep interval before noticing shutdown, so a
        # join cost about a second per scheduler; it now waits on a stop event
        # and joins in under two milliseconds.
        #
        # Waiting matters: with wait=False a worker thread from one test is
        # still draining while the next test runs, and three separate files
        # (test_watcher_dedup_resilience, test_jobsettings_modal,
        # test_ats_scrape_scheduling) were failing intermittently under random
        # ordering on exactly the assertions a stray worker would disturb.
        scheduler.shutdown(wait=True)


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


@pytest.fixture(autouse=True)
def _isolate_process_env():
    """load_config pushes every .env value into the real process environment
    (config.py's `os.environ.setdefault(k, v)`), and os.getenv is consulted
    before the parsed .env -- so one test's tmp_path .env silently wins for
    every test that runs after it. test_load_config_reads_groq_api_key read
    the *sibling* test's key whenever the shuffle put that sibling first.
    Snapshot and restore so config tests cannot contaminate each other.
    """
    import os as _os

    saved = dict(_os.environ)
    yield
    _os.environ.clear()
    _os.environ.update(saved)
