"""The loop watch names what blocked the event loop, and how long for.

Real stalls: a coroutine that holds the loop with time.sleep, watched by the
real thread, at a shortened tick so the test takes under two seconds.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import loop_watch  # noqa: E402


@pytest.fixture(autouse=True)
def _fast_ticks(monkeypatch):
    monkeypatch.setattr(loop_watch, "_TICK_S", 0.05)
    monkeypatch.setattr(loop_watch, "_active", None)


def _hold_the_loop_for(seconds: float) -> None:
    """The offending call, named so its frame is recognisable in the dump."""
    time.sleep(seconds)


def test_a_blocked_loop_is_reported_with_its_stack_and_its_length():
    lines: list[str] = []
    watch = loop_watch.LoopWatch(stale_after=0.2, log=lines.append)

    async def _main():
        watch.start()
        await asyncio.sleep(0.15)          # a few healthy ticks first
        _hold_the_loop_for(0.7)            # the stall, on the loop thread
        await asyncio.sleep(0.3)           # long enough for the thread to see it back
        watch.stop()

    asyncio.run(_main())

    stall_reports = [l for l in lines if "has not ticked" in l]
    assert len(stall_reports) == 1, lines
    assert "_hold_the_loop_for" in stall_reports[0], "the stack must name the blocking call"
    assert "time.sleep" in stall_reports[0] or "sleep(" in stall_reports[0]
    back = [l for l in lines if "is back after" in l]
    assert len(back) == 1, lines
    assert len(watch.stalls) == 1
    length, frame = watch.stalls[0]
    assert 0.5 <= length <= 1.5, f"stall measured as {length}s for a 0.7s block"


def test_a_healthy_loop_reports_nothing():
    lines: list[str] = []
    watch = loop_watch.LoopWatch(stale_after=0.2, log=lines.append)

    async def _main():
        watch.start()
        for _ in range(6):
            await asyncio.sleep(0.1)       # never holds the loop
        watch.stop()

    asyncio.run(_main())
    assert lines == []
    assert watch.stalls == []


def test_the_stack_is_read_from_another_thread_while_the_loop_runs():
    """sys._current_frames from the watcher thread sees the loop thread's
    frame -- the mechanism the whole thing rests on."""
    seen: dict[str, str] = {}
    watch = loop_watch.LoopWatch(stale_after=10)

    async def _main():
        watch.start()
        done = threading.Event()

        def _read():
            time.sleep(0.05)               # let the loop thread reach the hold below
            seen["stack"] = watch.loop_stack()
            done.set()

        threading.Thread(target=_read).start()
        _hold_the_loop_for(0.2)            # be somewhere recognisable when read
        done.wait(2)
        watch.stop()

    asyncio.run(_main())
    assert "_hold_the_loop_for" in seen["stack"]


def test_stop_ends_the_watcher_thread():
    watch = loop_watch.LoopWatch(stale_after=0.2)

    async def _main():
        watch.start()
        await asyncio.sleep(0.1)
        watch.stop()

    asyncio.run(_main())
    watch._thread.join(2)
    assert not watch._thread.is_alive()


def test_start_is_idempotent_per_process():
    async def _main():
        first = loop_watch.start(stale_after=1)
        second = loop_watch.start(stale_after=1)
        assert first is second
        first.stop()

    asyncio.run(_main())


def test_busy_periods_below_the_stall_threshold_are_still_named():
    """Many sub-threshold blocks -- each too short to be a stall -- still delay
    everything queued behind them. The per-window histogram says where the
    loop was on the ticks that ran late."""
    lines: list[str] = []
    watch = loop_watch.LoopWatch(stale_after=10, log=lines.append)
    watch.late_after = 0.08
    watch.report_every = 0.5

    async def _main():
        watch.start()
        for _ in range(6):
            _hold_the_loop_for(0.12)       # late, never a stall
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.7)           # let a report window close
        watch.stop()

    asyncio.run(_main())
    reports = [l for l in lines if "ran late" in l]
    assert reports, lines
    assert "_hold_the_loop_for" in reports[0]
    assert not any("has not ticked" in l for l in lines), "no stall: none of the blocks reached the threshold"
    assert watch.late_windows and any("_hold_the_loop_for" in k for k in watch.late_windows[0])


def test_a_quiet_window_prints_no_histogram():
    lines: list[str] = []
    watch = loop_watch.LoopWatch(stale_after=10, log=lines.append)
    watch.report_every = 0.2

    async def _main():
        watch.start()
        await asyncio.sleep(0.6)
        watch.stop()

    asyncio.run(_main())
    assert lines == []
