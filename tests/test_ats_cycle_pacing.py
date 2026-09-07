"""What the ATS cycle loop does when a cycle does not finish cleanly.

Two things hid the same failure. When a cycle errored (the gather timing out),
the loop left its pacing state untouched and started the next cycle at once --
while the failed cycle's platform threads were still alive, because a thread
cannot be cancelled. Every platform was re-queued behind its own predecessor.
And when a platform's bound expired while it was still in that queue, the log
said "timed out (0/0 companies reached)", which reads as a scrape that ran and
found nobody. Eighteen aged tasks behind ten busy workers, for hours, described
as eighteen quiet vendors.

These drive one real cycle of the loop, the way test_ats_scrape_scheduling.py
does, rather than inspecting its source.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service  # noqa: E402
from services.health import WatcherHealthTracker  # noqa: E402
from services.priority_scheduler import PriorityWorkScheduler  # noqa: E402
from state.store import RuntimeStore  # noqa: E402
from watchers import manager as manager_module  # noqa: E402
from watchers.manager import WatcherManager  # noqa: E402


class _Config:
    def __init__(self, tmp_path: Path) -> None:
        self.base_dir = tmp_path
        self.ats_bamboohr_enabled = True


def _manager(tmp_path: Path) -> tuple[WatcherManager, WatcherHealthTracker]:
    store = RuntimeStore(tmp_path / ".bot_state.json")
    store.channel_job_settings = {1: {"enabled": True}}
    health = WatcherHealthTracker()
    manager = WatcherManager(
        client=object(), config=_Config(tmp_path), store=store, health=health,
        scheduler=PriorityWorkScheduler(max_workers=4),
    )
    return manager, health


async def _drive(manager: WatcherManager, until, settle: float, deadline: float = 20.0) -> None:
    """Run the loop until `until()` holds, let it run `settle` more seconds so
    anything it would do next has had the chance to happen, then stop it."""
    task = asyncio.create_task(manager._run_ats_scrape_loop())
    end = asyncio.get_running_loop().time() + deadline
    while asyncio.get_running_loop().time() < end and not until():
        await asyncio.sleep(0.02)
    assert until(), "the cycle never reached the state under test"
    await asyncio.sleep(settle)
    assert not task.done(), "the loop itself must survive a failed cycle"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_an_errored_cycle_waits_its_interval_instead_of_retrying_at_once(tmp_path, monkeypatch):
    """A gather that times out is a cycle that ran: its threads are still
    going. The next cycle must wait the interval like a finished one would,
    not start immediately on top of it."""
    manager, health = _manager(tmp_path)
    monkeypatch.setattr(manager_module, "_ATS_PLATFORMS", ("greenhouse", "lever"))
    monkeypatch.setattr(manager_module, "_BAMBOOHR", "bamboohr")

    started = {"n": 0}

    async def _gather_times_out(platforms, scrape_one):
        started["n"] += 1
        # Yield once, as the real gather always does, so a loop that retries
        # at once spins visibly rather than starving the test of the loop.
        await asyncio.sleep(0)
        raise asyncio.TimeoutError()

    monkeypatch.setattr(manager, "_gather_ats_platforms", _gather_times_out)

    # Half a second is thousands of iterations for a loop that retries at once.
    asyncio.run(_drive(manager, lambda: started["n"] >= 1, settle=0.5))
    assert started["n"] == 1, f"the loop started {started['n']} cycles in the time one failure should have paced"
    # And the day's allowance moved, in health as well as in the loop: the
    # errored cycle was a cycle.
    ats = health.get_ats_health()
    assert ats.scrapes_today == 1
    assert "TimeoutError" in ats.last_cycle_error


def test_a_cycle_that_finished_and_then_errored_is_not_counted_twice(tmp_path, monkeypatch, capsys):
    """The daily allowance is four cycles. A cycle that scraped everything and
    then failed in its housekeeping tail (archive commit, GeoNames sync) has
    used one of them, not two."""
    manager, health = _manager(tmp_path)
    monkeypatch.setattr(manager_module, "_ATS_PLATFORMS", ("greenhouse",))
    monkeypatch.setattr(manager_module, "_BAMBOOHR", "bamboohr")
    monkeypatch.setattr(manager, "_ordered_slugs", lambda platform: ["a"])
    monkeypatch.setattr(manager_module, "_scrape_ats_platform", lambda **kw: [{"title": "t", "job_url": "u"}])
    monkeypatch.setattr("services.jba.merge_data.log_jobs", lambda rows: len(rows))

    real_to_thread = manager._tracked_to_thread

    async def _housekeeping_fails(fn, *args, **kwargs):
        if kwargs.get("label") == manager_module.scheduler_labels.ats_scrape_label("greenhouse"):
            return await real_to_thread(fn, *args, **kwargs)
        raise RuntimeError("git push refused")

    monkeypatch.setattr(manager, "_tracked_to_thread", _housekeeping_fails)

    asyncio.run(_drive(manager, lambda: bool(health.get_ats_health().last_cycle_error), settle=0.3))
    ats = health.get_ats_health()
    assert "git push refused" in ats.last_cycle_error
    assert ats.scrapes_today == 1


def test_a_platform_bounded_before_it_started_says_so_rather_than_zero_of_zero(tmp_path, monkeypatch, capsys):
    """Zero submitted means the thread had not reached its fan-out when the
    bound expired -- still queued, or still ordering. The line must say that,
    because "0/0 companies reached" sent the last diagnosis looking inside the
    scraper for time that was spent in the scheduler queue."""
    manager, health = _manager(tmp_path)
    monkeypatch.setattr(manager_module, "_ATS_PLATFORMS", ("greenhouse",))
    monkeypatch.setattr(manager_module, "_BAMBOOHR", "bamboohr")
    monkeypatch.setattr(manager_module, "ATS_PLATFORM_TIMEOUT_S", 0.05)
    # The cycle bound is derived from the platform bound; hold it open so the
    # platform's own bound, not the cycle's, is what fires.
    monkeypatch.setattr(manager_module, "ats_cycle_timeout", lambda platforms, workers: 30)
    monkeypatch.setattr(ats_service, "LAST_FANOUT", {})
    monkeypatch.setattr(manager, "_ordered_slugs", lambda platform: ["a"])
    monkeypatch.setattr("services.jba.merge_data.log_jobs", lambda rows: len(rows))

    def _slow_scrape(**kwargs):
        time.sleep(0.4)          # outlives the 0.05s bound without ever fanning out
        return []

    monkeypatch.setattr(manager_module, "_scrape_ats_platform", _slow_scrape)

    asyncio.run(_drive(
        manager,
        lambda: "greenhouse" in health.get_ats_health().per_platform,
        settle=0.6,              # let the thread finish so nothing outlives the test
    ))
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "greenhouse timed out or errored" in l]
    assert lines, "no timeout line was printed; output was: " + out
    line = lines[0]
    assert "no fan-out recorded" in line
    assert "queued" in line
    assert "0/0" not in line
