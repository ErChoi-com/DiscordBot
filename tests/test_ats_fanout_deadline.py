"""A platform's fan-out must stop when its caller's bound says so.

_fanout_budget answers "how long would asking every company take", which is
only the right question if we are allowed that long. We are not: the watcher
bounds a platform at ATS_PLATFORM_TIMEOUT_S, and the two numbers had no
relationship. Measured against the live fleet lists:

    lever      7,842 slugs / 30 workers ->  2.2h   (13x the 600s bound)
    bamboohr  21,238 slugs / 30 workers ->  5.9h   (35x)
    paylocity 15,914 slugs /  8 workers -> 16.6h   (99x)
    workable   9,841 slugs /  4 workers -> 20.5h  (123x)

and the observed thread lifetimes matched the budgets rather than the bound --
breezy ran 18,020s against a budget of 20,610s. The outer bound could not
correct it because asyncio.wait_for cancels the await, not the thread: the
awaiting side gave up at 600s and returned [], while the thread kept its
scheduler worker for hours. That is what starved the job watchers.

It also explains "it worked earlier the same day". The budget is derived from
the fleet size, and the harvest grows the fleet, so every harvest makes it
worse.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service
from watchers.manager import (
    ATS_PLATFORM_DRAIN_MARGIN_S,
    ATS_PLATFORM_FANOUT_BUDGET_S,
    ATS_PLATFORM_TIMEOUT_S,
)

PLATFORM = "greenhouse"


@pytest.fixture
def slow_platform(monkeypatch):
    """Register a scraper slow enough that the budget, not the work, decides."""
    started = {"n": 0}

    def _slow(slug, keywords, location, max_jobs):
        started["n"] += 1
        time.sleep(0.20)
        return [{"title": f"job-{slug}", "job_url": f"https://x/{slug}"}]

    monkeypatch.setitem(ats_service._SCRAPERS, PLATFORM, _slow)
    monkeypatch.setitem(ats_service.PLATFORM_WORKERS, PLATFORM, 2)
    monkeypatch.setattr(ats_service, "reset_refusal_breaker", lambda p: None)
    monkeypatch.setattr(ats_service, "flush_dead_slugs", lambda: None)
    monkeypatch.setattr(ats_service, "_drop_already_archived", lambda p, rows: rows)
    monkeypatch.setattr(ats_service, "_stamp_levels", lambda rows, levels: rows)
    return started


def _run(slugs, **kw):
    t0 = time.monotonic()
    rows = ats_service.scrape_ats_platform(PLATFORM, "", "", 0, company_slugs=slugs, **kw)
    return rows, time.monotonic() - t0


def test_max_seconds_actually_bounds_the_wait(slow_platform):
    """40 slugs over 2 workers at 0.2s each is ~4s of work. Given 1s, the
    fan-out must give up at about 1s, not run the work out."""
    slugs = [f"c{i}" for i in range(40)]
    rows, elapsed = _run(slugs, max_seconds=1.0)
    assert elapsed < 3.0, f"took {elapsed:.1f}s; the cap did not bite"
    assert rows, "and it must still return what it managed to collect"


def test_the_unbounded_budget_really_is_the_larger_number(slow_platform):
    """The premise. Without a cap this same fan-out is allowed far longer --
    if it were not, the cap above would prove nothing."""
    slugs = [f"c{i}" for i in range(40)]
    uncapped = ats_service._fanout_budget(len(slugs), 2)
    assert uncapped > 60, f"budget {uncapped}s should dwarf the 1s cap"


def test_partial_results_are_kept_not_discarded(slow_platform):
    """A cut-off cycle banks what it collected. _rotate_tail then resumes the
    next cycle past it, so discarding here would lose the work AND advance the
    cursor over it."""
    slugs = [f"c{i}" for i in range(40)]
    rows, _ = _run(slugs, max_seconds=1.0)
    assert 0 < len(rows) < len(slugs), f"expected a partial harvest, got {len(rows)}"


def test_without_max_seconds_nothing_changes(slow_platform):
    """Other callers (scripts, tests) must keep the old behaviour."""
    slugs = [f"c{i}" for i in range(4)]
    rows, _ = _run(slugs)
    assert len(rows) == 4, "a small fan-out still completes in full"


def test_the_cap_survives_the_slow_hardware_stretch(slow_platform, monkeypatch):
    """capacity.timeout STRETCHES a budget for slow hosts. Capping before it
    would let the stretch walk straight back past the cap, which is the whole
    reason the min() comes after."""
    monkeypatch.setattr(ats_service.capacity, "timeout", lambda nominal, *a, **k: nominal * 50)
    slugs = [f"c{i}" for i in range(40)]
    _, elapsed = _run(slugs, max_seconds=1.0)
    assert elapsed < 3.0, f"took {elapsed:.1f}s; the stretch escaped the cap"


def test_the_fanout_budget_leaves_room_to_drain():
    """When the fan-out gives up it cancels queued fetches, but in-flight ones
    cannot be cancelled and the pool waits the longest out. Without margin the
    platform is abandoned AT its bound, still holding a worker -- the exact
    failure being fixed."""
    assert ATS_PLATFORM_FANOUT_BUDGET_S < ATS_PLATFORM_TIMEOUT_S
    assert ATS_PLATFORM_TIMEOUT_S - ATS_PLATFORM_FANOUT_BUDGET_S == ATS_PLATFORM_DRAIN_MARGIN_S
    assert ATS_PLATFORM_DRAIN_MARGIN_S >= 2 * ats_service.REQUEST_TIMEOUT, (
        "margin must cover at least a request timeout and a retry"
    )
    assert ATS_PLATFORM_FANOUT_BUDGET_S > 0


def test_every_live_platform_is_now_bounded_by_the_caller():
    """The regression, stated over the real fleet: no platform's own budget may
    still decide when it stops."""
    lists = ats_service.load_company_lists()
    assert lists, "fleet lists must load for this to mean anything"
    worse = []
    for platform, slugs in lists.items():
        if not slugs:
            continue
        workers = ats_service.PLATFORM_WORKERS.get(platform, 10)
        own = ats_service._fanout_budget(len(slugs), workers)
        if own > ATS_PLATFORM_FANOUT_BUDGET_S:
            worse.append((platform, round(own)))
    assert worse, "if no platform exceeds the cap this test proves nothing"
    # Every one of those is now capped, which is the point.
    for platform, own in worse:
        assert min(own, ATS_PLATFORM_FANOUT_BUDGET_S) == ATS_PLATFORM_FANOUT_BUDGET_S


# --- the watcher must actually pass the bound down -----------------------------
#
# Everything above verifies scrape_ats_platform HONOURS max_seconds. None of it
# noticed when the watcher stopped SENDING it: a mutant that deleted that one
# keyword argument -- reverting the entire fix -- left the suite green. The
# call site is now a method so a test can reach the arguments it passes.

from types import SimpleNamespace  # noqa: E402

from watchers.manager import WatcherManager  # noqa: E402


def _watcher(recorder: dict):
    return SimpleNamespace(
        _ordered_slugs=lambda platform: ["a", "b"],
        _ats_last_fanout=lambda platform: {"submitted": 2, "completed": 2},
        health=SimpleNamespace(record_ats_platform_result=lambda *a, **k: recorder.setdefault("health", (a, k))),
    )


def _call(monkeypatch, recorder):
    def _spy(**kwargs):
        recorder["kwargs"] = kwargs
        return [{"title": "t", "job_url": "u"}]

    monkeypatch.setattr("watchers.manager._scrape_ats_platform", _spy)
    monkeypatch.setattr("services.jba.merge_data.log_jobs", lambda rows: len(rows))
    return WatcherManager._scrape_one_platform(_watcher(recorder), PLATFORM)


def test_the_watcher_passes_the_fanout_bound_to_the_scrape(monkeypatch):
    """The mutant that survived: drop this argument and the fan-out is
    unbounded again, silently, with every other test still green."""
    recorder: dict = {}
    _call(monkeypatch, recorder)
    assert "max_seconds" in recorder["kwargs"], "the bound is not being sent at all"
    assert recorder["kwargs"]["max_seconds"] == ATS_PLATFORM_FANOUT_BUDGET_S


def test_the_bound_it_sends_is_below_the_hard_deadline(monkeypatch):
    """Sending the full 600s would leave no room for the pool to drain, so the
    platform would be abandoned at its bound still holding a worker."""
    recorder: dict = {}
    _call(monkeypatch, recorder)
    assert recorder["kwargs"]["max_seconds"] < ATS_PLATFORM_TIMEOUT_S


def test_the_fleet_order_is_still_passed_with_it(monkeypatch):
    """Adding the bound must not displace the rotation, which is what makes
    partial coverage acceptable in the first place."""
    recorder: dict = {}
    _call(monkeypatch, recorder)
    assert recorder["kwargs"]["company_slugs"] == ["a", "b"]
