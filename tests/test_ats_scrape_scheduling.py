from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.health import WatcherHealthTracker
from services.priority_scheduler import PriorityWorkScheduler
from state.store import RuntimeStore
from watchers import manager as manager_module
from watchers.manager import WatcherManager


class _Config:
    def __init__(self, tmp_path: Path) -> None:
        self.base_dir = tmp_path
        self.ats_bamboohr_enabled = True


def test_ats_scrape_submits_one_scheduler_task_per_platform_and_survives_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old implementation ran every ATS platform inside a nested
    ThreadPoolExecutor invisible to the scheduler's own worker accounting.
    This asserts the replacement: each platform is its own scheduler
    submission with its own label, results from all platforms are still
    aggregated, and one platform failing doesn't stop the others."""
    config = _Config(tmp_path)
    store = RuntimeStore(tmp_path / ".bot_state.json")
    store.channel_job_settings = {1: {"enabled": True}}
    health = WatcherHealthTracker()
    scheduler = PriorityWorkScheduler(max_workers=4)
    manager = WatcherManager(client=object(), config=config, store=store, health=health, scheduler=scheduler)

    fake_platforms = ("greenhouse", "lever", "ashby")
    monkeypatch.setattr(manager_module, "_ATS_PLATFORMS", fake_platforms)
    monkeypatch.setattr(manager_module, "_BAMBOOHR", "bamboohr")

    calls_lock = threading.Lock()
    calls: list[str] = []

    def fake_scrape_ats_platform(platform: str, **kwargs):
        with calls_lock:
            calls.append(platform)
        if platform == "lever":
            raise RuntimeError("simulated platform failure")
        return [{"title": f"{platform} job"}]

    monkeypatch.setattr(manager_module, "_scrape_ats_platform", fake_scrape_ats_platform)
    monkeypatch.setattr("services.jba.merge_data.log_jobs", lambda jobs: len(jobs))

    async def _run_one_cycle() -> None:
        """Run exactly one cycle, waiting for it rather than timing it.

        This slept a flat 0.5s and cancelled. That is a wall-clock budget on
        work that grows: the cycle loads every platform's fleet before it
        scrapes, and the fleet is now 132,865 slugs across eighteen platforms
        against the six it was written for. It went from passing to failing
        with no code change, which is the worst way for a test to break --
        it reads as the scrape losing a platform.

        So wait for the observable end of the cycle, with the timeout only as
        a backstop. Fast when the machine is fast, and it does not rot.
        """
        task = asyncio.create_task(manager._run_ats_scrape_loop())
        deadline = asyncio.get_running_loop().time() + 30.0
        while asyncio.get_running_loop().time() < deadline:
            with calls_lock:
                done = len(calls) >= len(fake_platforms)
            # The health record is written after the gather, so it -- not the
            # call list -- is what says the cycle finished rather than merely
            # started every platform.
            if done and len(health.get_ats_health().per_platform) >= len(fake_platforms):
                break
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run_one_cycle())

    try:
        assert set(calls) == set(fake_platforms)  # every platform attempted despite one failing

        stats = scheduler.stats()
        for platform in fake_platforms:
            entry = stats["label_costs"].get(f"ats_scrape:{platform}")
            assert entry is not None, f"expected a per-platform label for {platform}"
            assert entry["samples"] >= 1

        ats_health = health.get_ats_health()
        lever_health = ats_health.per_platform.get("lever")
        assert lever_health is not None
        assert lever_health.total_errors >= 1
        assert lever_health.last_was_error is True

        greenhouse_health = ats_health.per_platform.get("greenhouse")
        assert greenhouse_health is not None
        assert greenhouse_health.total_jobs >= 1
        assert greenhouse_health.last_was_error is False
    finally:
        scheduler.shutdown()
