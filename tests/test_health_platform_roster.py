"""`.health` must list every ATS platform, not only the ones that reported.

`per_platform` is written by `record_ats_platform_result`, which the scrape
loop calls when a platform finishes. So it is a log of what ran, and rendering
straight from it produced three separate blind spots:

  - before the first cycle the Platforms field was absent entirely, so a fresh
    bot showed no ATS platforms at all;
  - after one platform reported it listed one of sixteen;
  - a platform disabled in config never reported, so it could never appear --
    and bamboohr is disabled by default while carrying the largest fleet of the
    sixteen (21,291 boards), making the platform most worth knowing about the
    one guaranteed to be invisible.

The roster fixes all three: every platform appears, with an explicit state for
the ones that have not reported, because "returned nothing" and "was never
asked" are exactly what this subsystem exists to tell apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.health import (  # noqa: E402
    _FIELD_VALUE_LIMIT,
    WatcherHealthTracker,
    build_all_health_embed,
    build_channel_health_embed,
)

SIXTEEN = [
    "applicantpro", "ashby", "bamboohr", "breezy", "greenhouse", "icims",
    "jazzhr", "jobvite", "lever", "paylocity", "recruitee", "rippling",
    "smartrecruiters", "teamtailor", "workable", "workday",
]


def _all(tracker):
    return build_all_health_embed(
        tracker, channel_job_tasks={}, channel_names={},
        active_job_watchers=0, active_reddit_watchers=0,
    )


def _ats(tracker):
    _main, ats = build_channel_health_embed(
        tracker, channel_id=1, channel_name="jobs",
        job_task_alive=False, reddit_task_alive=False,
        active_job_watchers=0, active_reddit_watchers=0,
    )
    return ats


def _field(embed, prefix):
    return next((f for f in embed.fields if f.name.startswith(prefix)), None)


# ── every platform appears ───────────────────────────────────────────────────

def test_all_sixteen_platforms_appear_before_any_cycle_has_run():
    """The field used to be missing entirely here, so a freshly started bot
    reported no ATS platforms whatsoever."""
    t = WatcherHealthTracker()
    t.set_ats_roster(SIXTEEN)

    for embed, prefix in ((_ats(t), "Platforms"), (_all(t), "ATS Platforms")):
        field = _field(embed, prefix)
        assert field is not None, prefix
        for name in SIXTEEN:
            assert name in field.value, f"{name} missing from {prefix}"


def test_all_sixteen_still_appear_when_only_one_has_reported():
    t = WatcherHealthTracker()
    t.set_ats_roster(SIXTEEN)
    t.record_ats_platform_result("lever", 12, new_count=3)

    value = _field(_ats(t), "Platforms").value
    for name in SIXTEEN:
        assert name in value


def test_the_field_title_counts_them_so_a_gap_is_visible():
    t = WatcherHealthTracker()
    t.set_ats_roster(SIXTEEN)
    assert _field(_ats(t), "Platforms").name == "Platforms (16)"


def test_a_platform_that_has_not_run_says_so_rather_than_reading_as_zero():
    """"0 scraped" and "never asked" are different, and showing the second as
    the first is the exact confusion this instrumentation exists to prevent.
    """
    t = WatcherHealthTracker()
    t.set_ats_roster(["lever"])
    assert "no cycle yet" in _field(_ats(t), "Platforms").value


# ── disabled platforms ───────────────────────────────────────────────────────

def test_a_disabled_platform_is_named_not_omitted():
    """bamboohr ships disabled and holds the largest fleet. Omitting it made
    the single most consequential setting invisible from Discord.
    """
    t = WatcherHealthTracker()
    t.set_ats_roster(SIXTEEN, disabled=["bamboohr"])

    value = _field(_ats(t), "Platforms").value
    assert "bamboohr" in value
    assert "disabled in config" in value


def test_a_disabled_platform_is_not_mistaken_for_a_broken_one():
    t = WatcherHealthTracker()
    t.set_ats_roster(["bamboohr"], disabled=["bamboohr"])
    line = _field(_ats(t), "Platforms").value
    assert "❌" not in line


def test_enabling_a_platform_changes_what_is_shown():
    t = WatcherHealthTracker()
    t.set_ats_roster(SIXTEEN, disabled=["bamboohr"])
    assert "disabled in config" in _field(_ats(t), "Platforms").value

    t.set_ats_roster(SIXTEEN)
    assert "disabled in config" not in _field(_ats(t), "Platforms").value


# ── reported platforms keep their detail ────────────────────────────────────

def test_a_reported_platform_still_shows_its_counts():
    t = WatcherHealthTracker()
    t.set_ats_roster(["lever"])
    t.record_ats_platform_result("lever", 12, new_count=3)

    value = _field(_ats(t), "Platforms").value
    assert "12" in value and "3" in value


def test_a_platform_that_returned_nothing_is_not_shown_as_a_success():
    """A refused platform returns [] and so does a quiet one; a green tick on
    both is what made workable look healthy for 37 consecutive runs.
    """
    t = WatcherHealthTracker()
    t.set_ats_roster(["workable"])
    t.record_ats_platform_result("workable", 0)
    assert "✅" not in _field(_ats(t), "Platforms").value


def test_a_long_silence_is_called_out_on_the_platform_line():
    t = WatcherHealthTracker()
    t.set_ats_roster(["workable"])
    for _ in range(5):
        t.record_ats_platform_result("workable", 0)
    assert "silent 5" in _field(_ats(t), "Platforms").value


def test_an_errored_platform_still_reads_as_an_error():
    t = WatcherHealthTracker()
    t.set_ats_roster(["icims"])
    t.record_ats_platform_result("icims", 0, error="boom")
    assert "❌" in _field(_ats(t), "Platforms").value


# ── robustness ───────────────────────────────────────────────────────────────

def test_a_platform_that_reports_without_being_rostered_is_still_shown():
    """The roster must not become a filter. A platform reporting results is
    proof it exists, whatever the roster says.
    """
    t = WatcherHealthTracker()
    t.set_ats_roster(["lever"])
    t.record_ats_platform_result("brandnew", 4)
    assert "brandnew" in _field(_ats(t), "Platforms").value


def test_no_roster_and_no_results_shows_no_field_rather_than_an_empty_one():
    assert _field(_ats(WatcherHealthTracker()), "Platforms") is None


def test_a_duplicated_roster_entry_is_listed_once():
    t = WatcherHealthTracker()
    t.set_ats_roster(["lever", "lever", "ashby"])
    assert _field(_ats(t), "Platforms").name == "Platforms (2)"


def test_sixteen_reported_platforms_still_fit_the_discord_field_limit():
    """Discord rejects a field over 1024 characters with a 400 that takes the
    whole reply down, and the roster makes this field longer than it was.
    """
    t = WatcherHealthTracker()
    t.set_ats_roster(SIXTEEN)
    for i, name in enumerate(SIXTEEN):
        t.record_ats_platform_result(name, 14_990 + i, new_count=i, error="x" * 30)

    for embed, prefix in ((_ats(t), "Platforms"), (_all(t), "ATS Platforms")):
        value = _field(embed, prefix).value
        assert 0 < len(value) <= _FIELD_VALUE_LIMIT, prefix


def test_the_scrape_loop_registers_the_roster():
    """Source-level pin: the roster is only useful if the loop declares it, and
    a capability nothing calls is the recurring failure in this codebase.
    """
    import inspect
    from watchers import manager

    compact = " ".join(
        inspect.getsource(manager.WatcherManager._run_ats_scrape_loop).split()
    )
    assert "set_ats_roster(" in compact
    assert "disabled=" in compact


# ── a refused platform is not a quiet one ───────────────────────────────────

def test_a_refused_platform_is_distinguished_from_a_silent_one():
    """The breaker stops asking a platform that refuses everything, and that
    was visible only in the log. A refused platform returns [] exactly like a
    quiet one, which is how workable looked ordinary for 45 consecutive runs.
    """
    t = WatcherHealthTracker()
    t.set_ats_roster(["workable", "lever"])
    t.record_ats_platform_result("workable", 0)
    t.record_ats_platform_result("lever", 5)
    t.set_ats_refusing({"workable": 25})

    value = _field(_ats(t), "Platforms").value
    assert "refused 25× and stopped" in value
    assert "⛔" in value


def test_a_platform_refused_before_it_ever_reported_is_still_named():
    """A platform cut off on its first cycle has no per_platform entry at all,
    so keying the note on a recorded result would hide the worst case.
    """
    t = WatcherHealthTracker()
    t.set_ats_roster(["workable"])
    t.set_ats_refusing({"workable": 25})

    value = _field(_ats(t), "Platforms").value
    assert "refused 25× and stopped" in value
    assert "no cycle yet" not in value


def test_recovery_clears_the_refusal_note():
    t = WatcherHealthTracker()
    t.set_ats_roster(["workable"])
    t.record_ats_platform_result("workable", 0)
    t.set_ats_refusing({"workable": 25})
    assert "refused" in _field(_ats(t), "Platforms").value

    t.set_ats_refusing({})
    assert "refused" not in _field(_ats(t), "Platforms").value


def test_a_platform_that_is_not_refusing_reads_normally():
    t = WatcherHealthTracker()
    t.set_ats_roster(["lever"])
    t.record_ats_platform_result("lever", 5, new_count=2)
    t.set_ats_refusing({"workable": 25})
    assert "⛔" not in _field(_ats(t), "Platforms").value


def test_the_loop_snapshots_the_breaker_after_the_cycle():
    """Each platform resets its own breaker as its fan-out starts, so reading
    the report before the gather would describe the previous cycle.
    """
    import inspect
    from watchers import manager

    src = inspect.getsource(manager.WatcherManager._run_ats_scrape_loop)
    compact = " ".join(src.split())
    assert "set_ats_refusing(self._ats_refusing())" in compact
    assert compact.index("asyncio.gather") < compact.index("set_ats_refusing")


def test_a_checkout_without_the_breaker_reports_nothing_rather_than_raising():
    """The breaker is instrumentation a given checkout of ats_service may not
    carry. Losing the report must never cost the scrape that produced it.
    """
    from watchers.manager import WatcherManager
    from services import ats_service

    saved = getattr(ats_service, "refusal_report", None)
    try:
        if saved is not None:
            delattr(ats_service, "refusal_report")
        assert WatcherManager._ats_refusing() == {}
    finally:
        if saved is not None:
            ats_service.refusal_report = saved
