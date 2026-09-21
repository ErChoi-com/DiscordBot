import asyncio
import os
import pytest
from config import AppConfig, load_config
from watchers.manager import (
    dynamic_ats_platform_slots,
    ats_platform_slots,
    ats_cycle_timeout,
    ATS_PLATFORM_TIMEOUT_S,
    ATS_PLATFORM_FANOUT_BUDGET_S,
)


def test_dynamic_ats_platform_slots_idle():
    # 12 workers, 2 reserved -> 10 general. Idle: other_in_flight=0, queued=0
    stats = {
        "workers": 12,
        "reserved_interactive": 2,
        "active": 0,
        "in_flight": [],
        "queued_interactive": 0,
        "queued_promoted": 0,
        "queued_background": 0,
    }
    slots = dynamic_ats_platform_slots(stats)
    # Ceiling is general - 1 = 10 - 1 = 9
    assert slots == 9


def test_dynamic_ats_platform_slots_with_active_watchers():
    # 12 workers, 2 reserved -> 10 general. 3 channel watchers in flight, 2 queued
    stats = {
        "workers": 12,
        "reserved_interactive": 2,
        "active": 3,
        "in_flight": [
            ("job_scrape:123", 10.0),
            ("job_scrape:456", 12.0),
            ("reddit_scrape:wallpapers", 5.0),
            ("ats_scrape:greenhouse", 20.0),  # ATS doesn't count against itself
        ],
        "queued_interactive": 0,
        "queued_promoted": 1,
        "queued_background": 1,
    }
    slots = dynamic_ats_platform_slots(stats)
    # dynamic_available = 10 - 3 (other in flight) - 2 (queued) = 5
    assert slots == 5


def test_dynamic_ats_platform_slots_heavy_load_respects_floor():
    # 12 workers, 2 reserved -> 10 general. 8 other watchers running, 5 queued
    stats = {
        "workers": 12,
        "reserved_interactive": 2,
        "active": 8,
        "in_flight": [(f"job_scrape:{i}", 5.0) for i in range(8)],
        "queued_interactive": 2,
        "queued_promoted": 1,
        "queued_background": 2,
    }
    slots = dynamic_ats_platform_slots(stats)
    # floor is max(1, 10 // 3) = 3
    assert slots == 3


def test_dynamic_ats_platform_slots_override():
    stats = {"workers": 12, "reserved_interactive": 2}
    assert dynamic_ats_platform_slots(stats, override=7) == 7


def test_ats_cycle_timeout_adaptation():
    # 18 platforms with 9 workers -> 2 waves. 1800s per platform -> 3600s
    t = ats_cycle_timeout(18, 9, platform_timeout=1800)
    assert t == 3600

    # Custom timeout: 600s
    t_fast = ats_cycle_timeout(18, 6, platform_timeout=600)
    assert t_fast == 1800


def test_app_config_ats_timeouts(tmp_path, monkeypatch):
    monkeypatch.setenv("ATS_PLATFORM_TIMEOUT_SECONDS", "2400")
    monkeypatch.setenv("ATS_PLATFORM_DRAIN_MARGIN_SECONDS", "200")

    cfg = load_config()
    assert cfg.ats_platform_timeout_seconds == 2400
    assert cfg.ats_platform_drain_margin_seconds == 200
    assert cfg.ats_platform_fanout_budget_seconds == 2200


def test_dynamic_gating_simulation():
    async def _runner():
        active_ats = 0
        max_concurrent_seen = 0
        cond = asyncio.Condition()
        current_allowed = 4

        async def worker(pid):
            nonlocal active_ats, max_concurrent_seen
            async with cond:
                while True:
                    if active_ats < current_allowed:
                        active_ats += 1
                        if active_ats > max_concurrent_seen:
                            max_concurrent_seen = active_ats
                        break
                    await cond.wait()
            try:
                await asyncio.sleep(0.01)
            finally:
                async with cond:
                    active_ats -= 1
                    cond.notify_all()

        tasks = [asyncio.create_task(worker(i)) for i in range(8)]
        await asyncio.gather(*tasks)
        assert max_concurrent_seen <= 4
        assert active_ats == 0

    asyncio.run(_runner())
