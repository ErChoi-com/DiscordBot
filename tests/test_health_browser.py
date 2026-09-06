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
        assert "1 running · 0 idle · 2 waiting · 1 workers" in queue_field.value
        assert "Waiting: 1 interactive · 0 boosted · 1 background" in queue_field.value
        # Full pool with work still waiting is the state that hung interactive
        # commands, so it must read as an alert rather than as ordinary load.
        assert queue_field.value.startswith("🔴")
        assert "**Saturated**" in queue_field.value
        assert "Running:" in queue_field.value
        # Internal tier names must not leak into the readout; the conventional
        # term stands in for them.
        assert "PROMOTED" not in queue_field.value
        assert "aged" not in queue_field.value
        # Nothing was cancelled here, so the warning line stays absent -- but
        # the zero is still reported in the totals.
        assert "0 cancelled" in queue_field.value
    finally:
        release.set()
        scheduler.shutdown()


def test_all_health_embed_surfaces_work_abandoned_before_it_ran():
    """Dropped work has no caller left to report it, so the embed is the only
    place it can surface. Driven through a real scheduler and a real
    cancellation rather than a hand-written stats dict."""
    scheduler = PriorityWorkScheduler(max_workers=1, reserved_interactive=0)
    try:
        started = threading.Event()
        release = threading.Event()

        def _blocker():
            started.set()
            release.wait(timeout=5)

        blocker = scheduler.submit(_blocker, tier=BACKGROUND)
        assert started.wait(timeout=5)

        doomed = scheduler.submit(time.sleep, 0, tier=BACKGROUND)
        assert doomed.cancel() is True

        release.set()
        blocker.result(timeout=5)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and scheduler.stats()["dropped"] == 0:
            time.sleep(0.01)

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
        assert "**1 cancelled**" in queue_field.value
    finally:
        release.set()
        scheduler.shutdown()


def test_queue_field_reports_every_scheduler_number():
    """The readout is a status surface, not a summary: every counter
    scheduler.stats() exposes has to appear somewhere, including the zeros.
    An earlier pass hid empty categories for brevity, which made the visible
    numbers stop adding up to the stated totals."""
    from services.health import _format_queue_field

    stats = {
        "queued": 3, "queued_interactive": 1, "queued_promoted": 2,
        "queued_background": 0, "active": 5, "workers": 12,
        "reserved_interactive": 2, "completed": 776, "dropped": 4,
        "promoted": 117,
        "in_flight": [{"label": "ats_scrape:lever", "age_seconds": 90}],
    }
    value = _format_queue_field(stats)

    assert "5 running" in value
    assert "7 idle" in value          # workers - active, derived but stated
    assert "3 waiting" in value
    assert "12 workers (2 reserved)" in value
    assert "ats_scrape:lever" in value
    # Zero categories are printed, so the split visibly sums to the total.
    assert "1 interactive · 2 boosted · 0 background" in value
    assert "776 completed" in value
    assert "4 cancelled" in value
    assert "117 boosted" in value
    # Compact: the whole readout stays inside a handful of short lines.
    assert len(value.splitlines()) <= 5, value


def test_queue_field_names_tiers_in_conventional_terms():
    """The scheduler's PROMOTED tier is priority aging -- a task whose priority
    was raised because it had waited too long. "aged-up" named the mechanism
    and meant nothing without the source; "priority-boosted" is the standard
    term and describes what happened to the task."""
    from services.health import _format_queue_field

    stats = {
        "queued": 3, "queued_interactive": 1, "queued_promoted": 2,
        "queued_background": 0, "active": 4, "workers": 12,
        "reserved_interactive": 2, "completed": 10, "dropped": 0,
        "promoted": 99, "in_flight": [],
    }
    value = _format_queue_field(stats)
    assert "boosted" in value
    assert "aged" not in value
    assert "bumped" not in value
    # Internal tier constants never reach the reader.
    assert "PROMOTED" not in value
    assert "INTERACTIVE" not in value


def test_queue_field_reports_long_running_ages_in_readable_units():
    """The runaway scrapes this surface exists for sit in the tens of
    thousands of seconds, where the raw number means nothing at a glance."""
    from services.health import compact_age, _format_queue_field

    assert compact_age(42) == "42s"
    assert compact_age(312) == "5m"
    assert compact_age(24956) == "6.9h"

    saturated = {
        "queued": 2, "queued_interactive": 1, "queued_promoted": 1,
        "queued_background": 0, "active": 12, "workers": 12,
        "reserved_interactive": 2, "completed": 776, "dropped": 0,
        "promoted": 117,
        "in_flight": [{"label": "ats_scrape:workable", "age_seconds": 24956}],
    }
    value = _format_queue_field(saturated)
    assert "`ats_scrape:workable` 6.9h" in value
    assert "24956" not in value
