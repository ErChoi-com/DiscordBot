"""BrowserServiceHealth telemetry recorders + hook safety (item 10)."""
from __future__ import annotations

import threading
import time

from services import browser_service
from services.health import BrowserServiceHealth, WatcherHealthTracker, build_all_health_embed
from services.priority_scheduler import BACKGROUND, INTERACTIVE, PriorityWorkScheduler


def test_recorder_increments_counters():
    tracker = WatcherHealthTracker()

    tracker.record_browser_event("dispatch_saturated", "fetch_json")
    tracker.record_browser_event("dispatch_saturated", "fetch_html")
    tracker.record_browser_event("dispatch_timeout", "fetch_json")
    tracker.record_browser_event("session_probe", True)
    tracker.record_browser_event("session_probe", False)
    tracker.record_browser_event("resync_attempt", True)
    tracker.record_browser_event("resync_attempt", False)
    tracker.record_browser_event("active_profile", r"C:\bot\chrome_profile_runtime")

    b = tracker.get_browser_health()
    assert b.dispatch_saturated_count == 2
    assert b.dispatch_timeout_count == 1
    assert b.session_probe_success_count == 1
    assert b.session_probe_failure_count == 1
    assert b.resync_attempt_count == 2
    assert b.resync_success_count == 1
    assert b.active_profile_path.endswith("chrome_profile_runtime")
    assert b.last_event_at > 0


def test_unknown_event_is_ignored_but_timestamped():
    tracker = WatcherHealthTracker()
    tracker.record_browser_event("not_a_real_event", 42)
    b = tracker.get_browser_health()
    assert b == BrowserServiceHealth(last_event_at=b.last_event_at)
    assert b.last_event_at > 0


def test_emit_health_swallows_hook_exceptions():
    browser_service.set_health_hook(lambda event, value=None: (_ for _ in ()).throw(RuntimeError("boom")))
    # Must not raise.
    browser_service._emit_health("dispatch_saturated", "fetch_json")


def test_emit_health_noop_without_hook():
    browser_service.set_health_hook(None)
    browser_service._emit_health("session_probe", True)  # must not raise


def test_hook_wiring_end_to_end():
    tracker = WatcherHealthTracker()
    browser_service.set_health_hook(tracker.record_browser_event)

    browser_service._emit_health("active_profile", "runtime_x")
    browser_service._emit_health("session_probe", False)

    b = tracker.get_browser_health()
    assert b.active_profile_path == "runtime_x"
    assert b.session_probe_failure_count == 1


def test_dispatch_yield_event_increments_counter():
    """dispatch_yield is emitted by browser_service._acquire_fetch_slot whenever
    a bulk (reddit) caller steps aside for a priority (.resumebuild) caller --
    it was previously emitted but silently dropped since record_browser_event
    had no branch for it."""
    tracker = WatcherHealthTracker()
    tracker.record_browser_event("dispatch_yield", "fetch_json")
    tracker.record_browser_event("dispatch_yield", "fetch_json")
    assert tracker.get_browser_health().dispatch_yield_count == 2


def test_last_probe_success_reflects_most_recent_probe_not_ever_succeeded():
    """One early success must not permanently mask a later, ongoing outage."""
    tracker = WatcherHealthTracker()
    assert tracker.get_browser_health().last_probe_success is None  # no probe yet

    tracker.record_browser_event("session_probe", True)
    assert tracker.get_browser_health().last_probe_success is True

    for _ in range(5):
        tracker.record_browser_event("session_probe", False)
    b = tracker.get_browser_health()
    assert b.last_probe_success is False
    assert b.session_probe_success_count == 1  # cumulative counters still accumulate
    assert b.session_probe_failure_count == 5


def test_browser_icon_reflects_ongoing_outage_not_a_stale_early_success():
    """Regression guard for the embed itself: the old condition
    (`failure_count == 0 or success_count > 0`) stayed green forever after a
    single early success. Build the real embed and check the rendered text."""
    tracker = WatcherHealthTracker()
    tracker.record_browser_event("session_probe", True)  # one lucky early success
    for _ in range(20):
        tracker.record_browser_event("session_probe", False)  # sustained outage since

    embed = build_all_health_embed(
        tracker,
        channel_job_tasks={},
        channel_names={},
        active_job_watchers=0,
        active_reddit_watchers=0,
    )
    browser_field = next(f for f in embed.fields if f.name == "Browser (Playwright)")
    assert "🔴" in browser_field.value
    assert "🟢" not in browser_field.value


def test_browser_icon_is_green_when_most_recent_probe_succeeded():
    tracker = WatcherHealthTracker()
    for _ in range(3):
        tracker.record_browser_event("session_probe", False)
    tracker.record_browser_event("session_probe", True)  # recovered

    embed = build_all_health_embed(
        tracker,
        channel_job_tasks={},
        channel_names={},
        active_job_watchers=0,
        active_reddit_watchers=0,
    )
    browser_field = next(f for f in embed.fields if f.name == "Browser (Playwright)")
    assert "🟢" in browser_field.value


def test_all_health_embed_omits_work_queue_field_without_scheduler_stats():
    """scheduler_stats defaults to None so callers that don't pass it (or old
    tests) keep working -- the field must not appear at all in that case."""
    tracker = WatcherHealthTracker()
    embed = build_all_health_embed(
        tracker, channel_job_tasks={}, channel_names={}, active_job_watchers=0, active_reddit_watchers=0
    )
    assert not any(f.name == "Work Queue" for f in embed.fields)


def test_all_health_embed_reports_real_queue_depth_from_scheduler():
    """Drive a real PriorityWorkScheduler (not a fake stats dict) so this
    proves the actual queued/active counts scheduler.stats() reports land in
    the embed, split by tier."""
    scheduler = PriorityWorkScheduler(max_workers=1)
    try:
        started = threading.Event()
        release = threading.Event()

        def _blocker():
            started.set()
            release.wait(timeout=5)

        # Occupy the single worker so subsequent submissions queue up behind it.
        scheduler.submit(_blocker, tier=BACKGROUND)
        assert started.wait(timeout=5)

        scheduler.submit(time.sleep, 0, tier=BACKGROUND)
        scheduler.submit(time.sleep, 0, tier=INTERACTIVE)

        tracker = WatcherHealthTracker()
        embed = build_all_health_embed(
            tracker,
            channel_job_tasks={},
            channel_names={},
            active_job_watchers=0,
            active_reddit_watchers=0,
            scheduler_stats=scheduler.stats(),
        )
        queue_field = next(f for f in embed.fields if f.name == "Work Queue")
        assert "Queued: **2** (1 interactive / 1 background)" in queue_field.value
        assert "Active: **1**/1 workers" in queue_field.value
    finally:
        release.set()
        scheduler.shutdown()
