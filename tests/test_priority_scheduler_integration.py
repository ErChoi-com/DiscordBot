from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commands.handlers import CommandRouter
from services.health import WatcherHealthTracker
from services.priority_scheduler import PriorityWorkScheduler
from state.store import RuntimeStore
from watchers.manager import WatcherManager


class _Config:
    def __init__(self, tmp_path: Path) -> None:
        self.resume_profiles_dir = tmp_path / "resumes"
        self.resume_cache_dir = tmp_path / ".resume_cache"
        self.resume_profiles_dir.mkdir(parents=True, exist_ok=True)
        self.resume_cache_dir.mkdir(parents=True, exist_ok=True)
        self.main_user_profile_key = "owner-profile"
        self.base_dir = tmp_path
        self.discord_history_check_limit = 20


def _blocking(tag: str, order: list[str], lock: threading.Lock, duration: float = 0.05) -> str:
    time.sleep(duration)
    with lock:
        order.append(tag)
    return tag


def _gated(tag: str, order: list[str], lock: threading.Lock, started: threading.Event, release: threading.Event) -> str:
    """Like _blocking, but announces that it is running and then waits to be let
    go, so a test can sequence on those two facts instead of on a sleep long
    enough to "probably" have happened. The sleep version failed under a loaded
    machine -- the ordering it asserted was real, the timing assumption was not.
    """
    started.set()
    assert release.wait(timeout=10), f"{tag} was never released"
    with lock:
        order.append(tag)
    return tag


async def _await_event(event: threading.Event, what: str, timeout: float = 10.0) -> None:
    """Wait on a threading.Event without blocking the loop the tasks run on."""
    await asyncio.wait_for(asyncio.to_thread(event.wait, timeout), timeout=timeout + 1)
    assert event.is_set(), f"timed out waiting for {what}"


async def _await_queued(scheduler: PriorityWorkScheduler, count: int, timeout: float = 10.0) -> None:
    """Wait until `count` tasks are actually queued, rather than sleeping and
    hoping the submission landed."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while scheduler.stats()["queued"] < count:
        assert loop.time() < deadline, f"only {scheduler.stats()['queued']} of {count} tasks queued"
        await asyncio.sleep(0.005)


def test_watcher_manager_and_command_router_share_one_scheduler_by_default(tmp_path: Path) -> None:
    """The whole point of a shared scheduler is that both domains submit to the
    same queue -- if app.py's wiring regresses to two separate schedulers, the
    priority arbitration silently stops doing anything."""
    config = _Config(tmp_path)
    store = RuntimeStore(tmp_path / ".bot_state.json")
    health = WatcherHealthTracker()

    manager = WatcherManager(client=object(), config=config, store=store, health=health)
    router = CommandRouter(client=object(), config=config, store=store, watcher_manager=manager, health=health)

    assert router.scheduler is manager.scheduler


def test_shared_scheduler_lets_interactive_work_preempt_queued_background_work(tmp_path: Path) -> None:
    """End-to-end: WatcherManager._tracked_to_thread (background watcher path)
    and CommandRouter._run_interactive (resumebuild-style path) both funnel
    through the same PriorityWorkScheduler, and interactive work dispatched
    after background work still runs first."""
    config = _Config(tmp_path)
    store = RuntimeStore(tmp_path / ".bot_state.json")
    health = WatcherHealthTracker()
    scheduler = PriorityWorkScheduler(max_workers=1)

    manager = WatcherManager(client=object(), config=config, store=store, health=health, scheduler=scheduler)
    router = CommandRouter(
        client=object(), config=config, store=store, watcher_manager=manager, health=health, scheduler=scheduler
    )

    order: list[str] = []
    lock = threading.Lock()

    started = threading.Event()
    release = threading.Event()

    async def _drive() -> None:
        # Occupy the single worker so both submissions below queue up behind it,
        # and hold it there until they demonstrably have.
        blocker = asyncio.create_task(
            manager._tracked_to_thread(_gated, "blocker", order, lock, started, release)
        )
        await _await_event(started, "the blocker to occupy the worker")

        background_task = asyncio.create_task(
            manager._tracked_to_thread(_blocking, "job_watcher_scrape", order, lock, 0.01, cost=50)
        )
        await _await_queued(scheduler, 1)
        interactive_task = asyncio.create_task(
            router._run_interactive(_blocking, "resumebuild", order, lock, 0.01, cost=1)
        )
        await _await_queued(scheduler, 2)

        # Both are queued behind the held worker, so what follows is the
        # scheduler's ordering decision and nothing else.
        release.set()
        await asyncio.gather(blocker, background_task, interactive_task)

    asyncio.run(_drive())

    assert order == ["blocker", "resumebuild", "job_watcher_scrape"]


def test_drain_active_work_waits_for_in_flight_interactive_command(tmp_path: Path) -> None:
    """CommandRouter._run_interactive must hold WatcherManager.work_guard() --
    otherwise graceful shutdown's drain_active_work() has zero visibility into
    an in-flight .resumebuild and reports 'drained' while the command is still
    running, letting shutdown abandon it."""
    config = _Config(tmp_path)
    store = RuntimeStore(tmp_path / ".bot_state.json")
    health = WatcherHealthTracker()
    scheduler = PriorityWorkScheduler(max_workers=2)

    manager = WatcherManager(client=object(), config=config, store=store, health=health, scheduler=scheduler)
    router = CommandRouter(
        client=object(), config=config, store=store, watcher_manager=manager, health=health, scheduler=scheduler
    )

    async def _drive() -> None:
        order: list[str] = []
        lock = threading.Lock()
        started = threading.Event()
        release = threading.Event()
        interactive_task = asyncio.create_task(
            router._run_interactive(_gated, "resumebuild", order, lock, started, release)
        )
        # Held mid-flight until released, rather than sleeping and assuming it
        # started -- that assumption is what broke this test under load.
        await _await_event(started, "the interactive command to start")

        # If _run_interactive didn't hold work_guard(), this would be set even
        # though the command above is still mid-flight.
        assert not manager._work_idle_event.is_set()

        release.set()
        drained = await manager.drain_active_work(timeout=5)
        assert drained is True
        await interactive_task
        assert order == ["resumebuild"]

    asyncio.run(_drive())
    scheduler.shutdown()


def test_resumebuild_completes_while_watcher_work_occupies_every_general_worker(
    tmp_path: Path,
) -> None:
    """End-to-end reproduction of the outage, through the real wiring.

    The ATS cycle submits one task per platform (16) into a pool of 12, and
    those tasks run for hours because the 600s bound around them cancels an
    await, not a thread. Every worker ends up held, and a .resumebuild
    submitted in that window never gets one -- it was correctly ranked first
    in the queue the entire time, which bought it nothing.

    Here: WatcherManager fills every general worker via its own submission
    path, then a command goes through CommandRouter._run_interactive. It must
    finish without any watcher work being released.
    """
    config = _Config(tmp_path)
    store = RuntimeStore(tmp_path / ".bot_state.json")
    health = WatcherHealthTracker()
    scheduler = PriorityWorkScheduler(max_workers=3, reserved_interactive=1)

    manager = WatcherManager(client=object(), config=config, store=store, health=health, scheduler=scheduler)
    router = CommandRouter(
        client=object(), config=config, store=store, watcher_manager=manager, health=health, scheduler=scheduler
    )

    async def _drive() -> None:
        release = threading.Event()
        occupied = threading.Semaphore(0)

        def _long_platform_scrape() -> str:
            occupied.release()
            assert release.wait(timeout=10), "watcher work was never released"
            return "scrape"

        # More submissions than the pool can hold, exactly like the platform
        # fan-out that caused this.
        watcher_tasks = [
            asyncio.create_task(
                manager._tracked_to_thread(_long_platform_scrape, label="ats_scrape:workable")
            )
            for _ in range(6)
        ]

        # Both general workers are now inside a scrape that will not return.
        for _ in range(2):
            await asyncio.to_thread(occupied.acquire)
        stats = scheduler.stats()
        assert stats["active"] == 2, stats
        assert stats["queued"] == 4, stats

        # The reserved worker carries the command through regardless.
        result = await asyncio.wait_for(
            router._run_interactive(lambda: "resume built", label="resume_rewrite"),
            timeout=5,
        )
        assert result == "resume built"

        # Proof the command did not simply wait for a scrape to finish: none
        # of them have been allowed to return yet.
        assert not release.is_set()
        assert scheduler.stats()["active"] == 2

        # The scheduler can also now say what is holding the pool, which is
        # what turns "full" into a diagnosis.
        holding = {entry["label"] for entry in scheduler.stats()["in_flight"]}
        assert holding == {"ats_scrape:workable"}, holding

        release.set()
        await asyncio.gather(*watcher_tasks)

    try:
        asyncio.run(_drive())
    finally:
        scheduler.shutdown()


def test_scheduler_sized_to_all_usable_cpu_cores() -> None:
    import os

    scheduler = PriorityWorkScheduler()
    try:
        assert scheduler.stats()["workers"] == max(4, os.cpu_count() or 4)
    finally:
        scheduler.shutdown()


def test_watchdog_stats_line_reports_scheduler_state(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The watchdog loop (30s cadence in production) is the monitoring surface
    for this scheduler -- assert it actually prints scheduler stats so an hour
    of runtime is observable in the console log."""
    config = _Config(tmp_path)
    store = RuntimeStore(tmp_path / ".bot_state.json")
    health = WatcherHealthTracker()
    manager = WatcherManager(client=object(), config=config, store=store, health=health)

    async def _one_cycle() -> None:
        task = asyncio.create_task(manager._run_watchdog(interval_seconds=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_one_cycle())
    captured = capsys.readouterr()
    assert "[watchdog] scheduler:" in captured.out
    assert "workers=" in captured.out
