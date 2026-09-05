"""A refused ATS platform must be distinguishable from a quiet one.

`scrape_ats_platform` returns a list. A platform whose every request came back
429 returns `[]`, and so does a platform that genuinely has no openings today --
the return value cannot tell them apart, so the health tracker recorded both as
a successful scrape and nothing in the pipeline ever said a platform had gone
dark. Measured live on 2026-09-04: workable answered 429 in ~0.1s for every
sampled board across five endpoint and header variants, and the scrape reported
success with zero jobs.
"""
from __future__ import annotations

from services.health import WatcherHealthTracker


def test_an_empty_scrape_is_recorded_as_silent_rather_than_as_a_success():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("workable", 0)

    ph = tracker.get_ats_health().per_platform["workable"]
    assert ph.consecutive_silent == 1
    # Still not an error: the platform did not crash, and calling it one would
    # make every genuinely quiet platform look broken.
    assert ph.last_was_error is False
    assert ph.total_errors == 0


def test_silence_accumulates_across_runs():
    tracker = WatcherHealthTracker()
    for _ in range(4):
        tracker.record_ats_platform_result("workable", 0)

    assert tracker.get_ats_health().per_platform["workable"].consecutive_silent == 4


def test_a_scrape_that_returns_jobs_clears_the_silence():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("lever", 0)
    tracker.record_ats_platform_result("lever", 0)
    tracker.record_ats_platform_result("lever", 12, new_count=3)

    ph = tracker.get_ats_health().per_platform["lever"]
    assert ph.consecutive_silent == 0
    assert ph.last_nonempty_at > 0.0


def test_an_error_is_not_counted_as_silence():
    """Errors already have their own counter. Counting a raising scrape as
    silence too would double-report it and let one failure mode mask the other.
    """
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("icims", 0, error="boom")

    ph = tracker.get_ats_health().per_platform["icims"]
    assert ph.total_errors == 1
    assert ph.last_was_error is True
    assert ph.consecutive_silent == 0


def test_a_platform_is_only_reported_once_it_passes_the_threshold():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("workable", 0)
    tracker.record_ats_platform_result("workable", 0)
    assert tracker.silent_ats_platforms(threshold=3) == []

    tracker.record_ats_platform_result("workable", 0)
    assert tracker.silent_ats_platforms(threshold=3) == [("workable", 3)]


def test_an_intermittent_platform_never_trips_the_alarm():
    """One empty run is ordinary -- a board can simply have nothing open. The
    alarm is for a platform that stays empty, so anything that yields in
    between must reset and never accumulate toward the threshold.
    """
    tracker = WatcherHealthTracker()
    for _ in range(10):
        tracker.record_ats_platform_result("greenhouse", 0)
        tracker.record_ats_platform_result("greenhouse", 5)

    assert tracker.silent_ats_platforms(threshold=3) == []


def test_the_longest_silent_platform_is_reported_first():
    tracker = WatcherHealthTracker()
    for _ in range(3):
        tracker.record_ats_platform_result("breezy", 0)
    for _ in range(7):
        tracker.record_ats_platform_result("workable", 0)

    assert tracker.silent_ats_platforms(threshold=3) == [("workable", 7), ("breezy", 3)]


def test_a_yielding_platform_is_never_reported_as_silent():
    tracker = WatcherHealthTracker()
    for _ in range(5):
        tracker.record_ats_platform_result("lever", 20, new_count=2)

    assert tracker.silent_ats_platforms(threshold=1) == []


# ── fleet coverage: how much of each platform a cycle actually reached ───────

def test_coverage_is_recorded_per_cycle():
    """The fan-out's submitted/completed counts existed only in a print, so
    "how many companies did we ask?" could not be answered from inside the bot.
    Reasoning about it from the printed *cancelled* count instead is how
    "43-100% reached" got misread as "2% reached".
    """
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("workday", 95, submitted=14990, completed=14989)

    ph = tracker.get_ats_health().per_platform["workday"]
    assert ph.last_submitted == 14990
    assert ph.last_completed == 14989


def test_coverage_reports_the_reached_fraction_not_the_cancelled_one():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("icims", 40, submitted=13343, completed=7214)

    row = tracker.ats_fleet_coverage()[0]
    assert row["platform"] == "icims"
    assert row["reached_pct"] == 54.1     # 7214/13343, not the 45.9% cancelled
    assert row["submitted"] == 13343


def test_coverage_is_recorded_even_when_the_cycle_errored():
    """A cycle that raised after reaching 9,000 of 10,000 companies is a
    different failure from one that reached 12, and the counts are the only
    thing that distinguishes them.
    """
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result(
        "greenhouse", 0, error="timed out", submitted=10807, completed=9033
    )

    ph = tracker.get_ats_health().per_platform["greenhouse"]
    assert ph.last_was_error is True
    assert ph.last_completed == 9033


def test_a_platform_with_no_recorded_cycle_is_omitted_not_reported_as_zero():
    """0% reached and "never ran" are different, and reporting the second as
    the first is the exact shape of mistake this instrumentation exists to
    prevent.
    """
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("lever", 12)          # no counts supplied
    assert tracker.ats_fleet_coverage() == []


def test_lifetime_coverage_accumulates_across_cycles():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("ashby", 5, submitted=1000, completed=500)
    tracker.record_ats_platform_result("ashby", 5, submitted=1000, completed=900)

    row = tracker.ats_fleet_coverage()[0]
    assert row["reached_pct"] == 90.0            # last cycle only
    assert row["lifetime_reached_pct"] == 70.0   # 1400/2000 across both


def test_the_worst_covered_platform_is_reported_first():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("workday", 1, submitted=100, completed=99)
    tracker.record_ats_platform_result("icims", 1, submitted=100, completed=43)
    tracker.record_ats_platform_result("lever", 1, submitted=100, completed=70)

    assert [r["platform"] for r in tracker.ats_fleet_coverage()] == ["icims", "lever", "workday"]
