"""Surfacing fleet coverage and platform silence in `.health`.

`ats_fleet_coverage()` and `silent_ats_platforms()` both answered questions
nothing could ask: the tracker recorded them and no embed ever read them, so
from Discord a platform that reached 43% of its fleet and one that reached all
of it looked identical, and a platform silent for a day looked like a platform
with nothing to post.

The other half of this is the field budget. Discord rejects an embed field over
1024 characters with a 400 that takes the whole reply down, and the Platforms
field was capped by a fixed `[:12]` slice -- which drops four of sixteen
platforms without saying so, and still overflows once the lines grow.
"""
from __future__ import annotations

from services.health import (
    WatcherHealthTracker,
    _FIELD_VALUE_LIMIT,
    _format_ats_coverage_field,
    _join_within_limit,
    build_all_health_embed,
    build_channel_health_embed,
)


def _all_embed(tracker: WatcherHealthTracker):
    return build_all_health_embed(
        tracker,
        channel_job_tasks={},
        channel_names={},
        active_job_watchers=0,
        active_reddit_watchers=0,
    )


def _ats_embed(tracker: WatcherHealthTracker):
    _main, ats = build_channel_health_embed(
        tracker,
        channel_id=1,
        channel_name="jobs",
        job_task_alive=False,
        reddit_task_alive=False,
        active_job_watchers=0,
        active_reddit_watchers=0,
    )
    return ats


def _field(embed, name: str):
    return next((f for f in embed.fields if f.name == name), None)


# ── the numbers reach Discord at all ─────────────────────────────────────────

def test_coverage_reaches_the_all_watcher_embed():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("icims", 40, submitted=13343, completed=7214)

    field = _field(_all_embed(tracker), "ATS Fleet Coverage")
    assert field is not None
    assert "icims" in field.value
    assert "54.1%" in field.value          # reached, not the 45.9% cancelled
    assert "7,214/13,343" in field.value


def test_coverage_reaches_the_channel_ats_embed():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("icims", 40, submitted=13343, completed=7214)

    field = _field(_ats_embed(tracker), "Fleet Coverage")
    assert field is not None
    assert "54.1%" in field.value


def test_the_worst_covered_platform_reads_first():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("workday", 1, submitted=100, completed=99)
    tracker.record_ats_platform_result("icims", 1, submitted=100, completed=43)

    value = _field(_all_embed(tracker), "ATS Fleet Coverage").value
    assert value.index("icims") < value.index("workday")


def test_a_tracker_with_nothing_to_say_gets_no_field():
    """An empty field is worse than an absent one: it reads as "coverage is
    zero" rather than "no cycle has run yet"."""
    assert _format_ats_coverage_field(WatcherHealthTracker()) is None
    assert _field(_all_embed(WatcherHealthTracker()), "ATS Fleet Coverage") is None


def test_a_platform_that_ran_but_reported_no_counts_is_not_shown_as_zero():
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("lever", 12)
    assert _format_ats_coverage_field(tracker) is None


# ── silence, which coverage alone cannot show ────────────────────────────────

def test_a_silent_platform_is_called_out_beside_its_coverage():
    tracker = WatcherHealthTracker()
    for _ in range(4):
        tracker.record_ats_platform_result("workable", 0, submitted=200, completed=200)

    value = _format_ats_coverage_field(tracker)
    assert "workable" in value
    assert "silent 4" in value
    # It reached its whole fleet, so the problem is the endpoint, not the cycle.
    assert "100.0% reached" in value


def test_a_platform_silent_without_any_cycle_is_still_reported():
    """A platform refused before it submits anything has no coverage row. That
    is the loudest symptom there is, and keying the field on coverage alone
    would drop it entirely.
    """
    tracker = WatcherHealthTracker()
    for _ in range(3):
        tracker.record_ats_platform_result("workable", 0)

    value = _format_ats_coverage_field(tracker)
    assert "workable" in value
    assert "no cycle recorded" in value


def test_one_quiet_run_is_not_called_out():
    """Empty runs are ordinary. Flagging the first would make the field noise."""
    tracker = WatcherHealthTracker()
    tracker.record_ats_platform_result("breezy", 0, submitted=50, completed=50)

    value = _format_ats_coverage_field(tracker)
    assert "silent" not in value


def test_recovering_clears_the_silence_note():
    tracker = WatcherHealthTracker()
    for _ in range(5):
        tracker.record_ats_platform_result("workable", 0, submitted=200, completed=200)
    tracker.record_ats_platform_result("workable", 5, submitted=200, completed=200)

    assert "silent" not in _format_ats_coverage_field(tracker)


# ── the field budget ─────────────────────────────────────────────────────────

def test_a_full_sixteen_platform_fleet_still_fits_the_field():
    tracker = WatcherHealthTracker()
    names = [
        "applicantpro", "ashby", "bamboohr", "breezy", "greenhouse", "icims",
        "jazzhr", "jobvite", "lever", "paylocity", "recruitee", "rippling",
        "smartrecruiters", "teamtailor", "workable", "workday",
    ]
    for i, name in enumerate(names):
        tracker.record_ats_platform_result(
            name, 100 + i, new_count=i, submitted=14990, completed=14990 - i
        )

    for embed, field_name in (
        (_all_embed(tracker), "ATS Fleet Coverage"),
        (_ats_embed(tracker), "Fleet Coverage"),
        (_ats_embed(tracker), "Platforms"),
    ):
        value = _field(embed, field_name).value
        assert 0 < len(value) <= _FIELD_VALUE_LIMIT, field_name


def test_everything_that_fits_is_kept():
    lines = ["x" * 100 for _ in range(5)]
    assert _join_within_limit(lines) == "\n".join(lines)


def test_what_does_not_fit_is_counted_rather_than_dropped_silently():
    lines = [f"line{i} " + "x" * 90 for i in range(40)]
    out = _join_within_limit(lines)

    assert len(out) <= _FIELD_VALUE_LIMIT
    shown = [ln for ln in out.split("\n") if ln.startswith("line")]
    assert 0 < len(shown) < 40
    assert f"+{40 - len(shown)} more" in out


def test_the_overflow_line_is_itself_inside_the_budget():
    """Appending "+N more" after fitting the lines is how a field that just
    fit becomes a 400. The reservation has to happen before the last line is
    accepted, not after.
    """
    line = "x" * 101
    lines = [line] * 20
    out = _join_within_limit(lines)
    assert len(out) <= _FIELD_VALUE_LIMIT
    assert out.endswith("more")


def test_no_overflow_line_when_every_line_was_kept():
    lines = ["short"] * 3
    assert _join_within_limit(lines) == "short\nshort\nshort"


def test_a_last_line_is_not_evicted_to_make_room_for_a_summary_of_nothing():
    """Reserving space for "+N more" unconditionally would drop a final line
    that fit, then report "+1 more" -- strictly worse than showing it.
    """
    # Sized so the two lines plus their separator land exactly on the budget:
    # reserving for an overflow line that is not needed would push it over and
    # evict the second line.
    lines = ["a" * 511, "b" * 512]
    out = _join_within_limit(lines)
    assert len(out) == _FIELD_VALUE_LIMIT
    assert out == "a" * 511 + "\n" + "b" * 512


def test_a_single_line_over_the_whole_budget_says_so_rather_than_vanishing():
    out = _join_within_limit(["y" * 2000])
    assert out
    assert len(out) <= _FIELD_VALUE_LIMIT
    assert "too long" in out


def test_no_lines_is_the_empty_string():
    assert _join_within_limit([]) == ""
