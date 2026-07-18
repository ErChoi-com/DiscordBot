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

    async def _drive() -> None:
        # Occupy the single worker so both submissions below queue up behind it.
        blocker = asyncio.create_task(manager._tracked_to_thread(_blocking, "blocker", order, lock, 0.15))
        await asyncio.sleep(0.02)

        background_task = asyncio.create_task(
            manager._tracked_to_thread(_blocking, "job_watcher_scrape", order, lock, 0.01, cost=50)
        )
        await asyncio.sleep(0.01)
        interactive_task = asyncio.create_task(
            router._run_interactive(_blocking, "resumebuild", order, lock, 0.01, cost=1)
        )

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
        interactive_task = asyncio.create_task(
            router._run_interactive(_blocking, "resumebuild", order, lock, 0.1)
        )
        await asyncio.sleep(0.02)  # let it actually start

        # If _run_interactive didn't hold work_guard(), this would return True
        # immediately even though the command above is still mid-flight.
        drained_immediately = manager._work_idle_event.is_set()
        assert not drained_immediately

        drained = await manager.drain_active_work(timeout=5)
        assert drained is True
        await interactive_task

    asyncio.run(_drive())
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
