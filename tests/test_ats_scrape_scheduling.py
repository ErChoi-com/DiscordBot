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
        task = asyncio.create_task(manager._run_ats_scrape_loop())
        await asyncio.sleep(0.5)  # let the first (immediate, no-gating) cycle finish
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
