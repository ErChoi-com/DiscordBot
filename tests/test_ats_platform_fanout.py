"""The ATS fan-out must leave the pool room for the other watchers.

Every platform used to be submitted at once. Each one is its own scheduler
task, so nothing was hidden from the worker accounting -- but the scheduler
only orders the queue, and these tasks run for minutes with nothing able to
preempt them. Eighteen platforms against ten general workers did not share
the pool with the job and reddit watchers, it took the pool.

Measured on the real host while the job watchers had produced nothing for six
hours: workers=12 active=12, `completed` frozen at 20 across three consecutive
watchdog dumps, one lever scrape at 2,584s, three job scrapes queued behind it.
Glassdoor and ZipRecruiter both returned rows the whole time when run by hand
-- 3 and 10 respectively -- so the sources were fine. They were never asked.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from watchers import manager as manager_module
from watchers.manager import WatcherManager, ats_cycle_timeout, ats_platform_slots

REAL_HOST = (12, 2)  # workers, reserved_interactive -- this bot's actual shape


def _manager(workers: int, reserved: int):
    """Enough of a WatcherManager to call the real fan-out method on.

    Deliberately not a constructed WatcherManager: that needs a discord client,
    config and store, none of which the fan-out touches. The method under test
    is the real one.
    """
    return SimpleNamespace(
        scheduler=SimpleNamespace(
            stats=lambda: {"workers": workers, "reserved_interactive": reserved}
        )
    )


async def _run(fake_self, platforms, scrape_one):
    return await WatcherManager._gather_ats_platforms(fake_self, platforms, scrape_one)


def test_the_fanout_never_takes_every_general_worker():
    """The whole point. If ATS may hold every worker that can run it, the job
    and reddit watchers cannot run at all while a cycle is in flight."""
    for workers in range(2, 33):
        for reserved in range(0, max(1, workers // 4) + 1):
            general = workers - reserved
            slots = ats_platform_slots(workers, reserved)
            assert 1 <= slots, "a host must always be able to make progress"
            if general > 1:
                assert slots < general, (
                    f"workers={workers} reserved={reserved}: ATS may take {slots} "
                    f"of {general} general workers, leaving nothing for the watchers"
                )


def test_the_real_host_keeps_half_the_general_pool_free():
    """12 workers, 2 reserved for interactive -> 10 general, 5 for ATS."""
    assert ats_platform_slots(*REAL_HOST) == 5


def test_reserved_interactive_workers_are_not_counted_as_available():
    """ATS is background work and cannot run on a reserved-interactive worker,
    so counting those would hand ATS more of the pool than intended."""
    assert ats_platform_slots(12, 2) < ats_platform_slots(12, 0)


def test_a_single_worker_host_still_runs_platforms():
    """Degenerate hosts must not compute a zero-slot gate and deadlock."""
    for workers, reserved in [(1, 0), (2, 1), (2, 0), (3, 2)]:
        assert ats_platform_slots(workers, reserved) >= 1


def test_no_more_than_the_slot_count_run_at_once():
    """Measured against the real method, not a re-implementation of it."""
    in_flight = 0
    peak = 0
    seen: list[str] = []

    async def scrape_one(platform: str):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.01)
            seen.append(platform)
            return [{"platform": platform}]
        finally:
            in_flight -= 1

    platforms = [f"p{i}" for i in range(18)]
    results = asyncio.run(_run(_manager(*REAL_HOST), platforms, scrape_one))

    assert peak == 5, f"gate allowed {peak} concurrent platforms, not 5"
    assert sorted(seen) == sorted(platforms), "every platform must still be scraped"
    assert [r[0]["platform"] for r in results] == platforms, "results stay in order"


def test_the_cycle_bound_is_derived_from_the_gate_not_the_pool():
    """ats_cycle_timeout counts waves. A wave is now the gate width, so passing
    the worker count would under-estimate the cycle and cancel its last wave --
    the exact failure that function was written to prevent."""
    captured: list[tuple[int, int]] = []
    real = manager_module.ats_cycle_timeout

    def _spy(platform_count: int, workers: int) -> int:
        captured.append((platform_count, workers))
        return real(platform_count, workers)

    manager_module.ats_cycle_timeout = _spy
    try:
        asyncio.run(_run(_manager(*REAL_HOST), [f"p{i}" for i in range(18)],
                         lambda p: asyncio.sleep(0, result=[])))
    finally:
        manager_module.ats_cycle_timeout = real

    assert captured == [(18, 5)], f"got {captured}; 12 here would halve the budget"


def test_the_bound_actually_covers_the_waves_the_gate_creates():
    """Narrowing concurrency without widening the bound would time out every
    cycle. 18 platforms through 5 slots is 4 waves, not 2."""
    slots = ats_platform_slots(*REAL_HOST)
    budget = ats_cycle_timeout(18, slots)
    waves_needed = -(-18 // slots)
    assert budget >= waves_needed * manager_module.ATS_PLATFORM_TIMEOUT_S
    assert budget > ats_cycle_timeout(18, REAL_HOST[0]), "must be longer than before"
    assert budget <= manager_module.ATS_CYCLE_TIMEOUT_CAP_S


def test_a_slow_platform_does_not_hold_a_slot_it_has_finished_with():
    """The gate is released on failure too, or one erroring platform would
    permanently shrink the pool for the rest of the cycle."""
    async def scrape_one(platform: str):
        if platform in {"p0", "p1", "p2", "p3", "p4"}:
            raise RuntimeError("platform down")
        await asyncio.sleep(0.01)
        return [{"platform": platform}]

    with pytest.raises(RuntimeError):
        asyncio.run(_run(_manager(*REAL_HOST), [f"p{i}" for i in range(18)], scrape_one))

    # And with the errors swallowed the way the real caller does it, the
    # remaining platforms still all get a turn.
    async def tolerant(platform: str):
        try:
            return await scrape_one(platform)
        except RuntimeError:
            return []

    results = asyncio.run(
        _run(_manager(*REAL_HOST), [f"p{i}" for i in range(18)], tolerant)
    )
    assert sum(1 for r in results if r) == 13
