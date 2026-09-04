"""Hardware-aware pool scaling.

These tests drive *fake* hardware (patched cpu/memory limits and synthetic
cgroup files) rather than asserting against whatever machine runs the suite, so
they mean the same thing on a 12-core dev box and a 1-core CI runner.
"""
from __future__ import annotations

import pytest

from services import capacity


@pytest.fixture(autouse=True)
def _clear_overrides(monkeypatch):
    """The env overrides short-circuit detection, so unset them for every test
    and let the ones that care set them explicitly."""
    monkeypatch.delenv("BOT_WORKER_SCALE", raising=False)
    monkeypatch.delenv("BOT_MAX_WORKERS", raising=False)


def _fake_hardware(monkeypatch, cpus: float, memory_gb: float | None):
    monkeypatch.setattr(capacity, "cpu_limit", lambda: cpus)
    monkeypatch.setattr(
        capacity, "memory_limit_bytes",
        lambda: None if memory_gb is None else int(memory_gb * 1024 ** 3),
    )


# ---------------------------------------------------------------------------
# Scaling direction
# ---------------------------------------------------------------------------

def test_reference_hardware_leaves_tuned_values_untouched(monkeypatch):
    _fake_hardware(monkeypatch, capacity.REFERENCE_CPUS, capacity.REFERENCE_MEMORY_GB)

    for nominal in (5, 6, 20, 30, 50):
        assert capacity.workers(nominal, minimum=2) == nominal


def test_small_vps_scales_pools_down_hard(monkeypatch):
    _fake_hardware(monkeypatch, 1, 1.0)  # 1 core, 1 GB

    assert capacity.workers(50, minimum=4) < 12
    assert capacity.workers(30, minimum=4) < 8
    # The caller's floor is still honoured -- I/O-bound pools stay usable.
    assert capacity.workers(50, minimum=4) >= 4


def test_memory_constrains_even_when_cpu_is_plentiful(monkeypatch):
    """A 32-core box with 1 GB must not open 50 connections: memory is the
    binding constraint and the scale factor takes the *minimum* of the two."""
    _fake_hardware(monkeypatch, 32, 1.0)

    generous_cpu_only = 50
    assert capacity.workers(50, minimum=2) < generous_cpu_only // 2


def test_cpu_constrains_even_when_memory_is_plentiful(monkeypatch):
    _fake_hardware(monkeypatch, 1, 128.0)

    assert capacity.workers(50, minimum=2) < 12


def test_upward_scaling_requires_an_explicit_maximum(monkeypatch):
    """Big hardware must not silently increase load on third-party sites."""
    _fake_hardware(monkeypatch, 64, 256.0)

    # Politeness-capped pool: no `maximum`, so it stays at the tuned value.
    assert capacity.workers(30, minimum=4) == 30
    # Local-resource pool: opts in, so it grows.
    assert capacity.workers(6, minimum=2, maximum=12) == 12


def test_upward_scaling_is_bounded_by_max_scale(monkeypatch):
    """An enormous box should not produce an unbounded pool."""
    _fake_hardware(monkeypatch, 4096, 8192.0)

    assert capacity.workers(6, minimum=2, maximum=1000) <= int(6 * capacity.MAX_SCALE)


def test_scale_factor_never_collapses_below_min_scale(monkeypatch):
    _fake_hardware(monkeypatch, 0.1, 0.1)
    assert capacity.scale_factor() == capacity.MIN_SCALE


def test_workers_never_returns_less_than_one(monkeypatch):
    _fake_hardware(monkeypatch, 0.1, 0.1)
    assert capacity.workers(1) >= 1
    assert capacity.workers(2, minimum=1) >= 1


def test_minimum_above_nominal_does_not_invert_the_range(monkeypatch):
    """A caller floor larger than the tuned value must not produce a pool
    smaller than the floor, nor crash on an inverted clamp."""
    _fake_hardware(monkeypatch, 1, 1.0)
    assert capacity.workers(2, minimum=8) >= 2


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------

def test_worker_scale_override_forces_the_factor(monkeypatch):
    _fake_hardware(monkeypatch, capacity.REFERENCE_CPUS, capacity.REFERENCE_MEMORY_GB)
    monkeypatch.setenv("BOT_WORKER_SCALE", "0.5")

    assert capacity.scale_factor() == 0.5
    assert capacity.workers(30, minimum=2) == 15


def test_worker_scale_override_can_scale_up(monkeypatch):
    _fake_hardware(monkeypatch, 1, 1.0)
    monkeypatch.setenv("BOT_WORKER_SCALE", "2")

    assert capacity.scale_factor() == 2.0
    assert capacity.workers(6, minimum=2, maximum=20) == 12


def test_invalid_overrides_fall_back_to_detection(monkeypatch):
    _fake_hardware(monkeypatch, capacity.REFERENCE_CPUS, capacity.REFERENCE_MEMORY_GB)
    for bad in ("abc", "", "-1", "0"):
        monkeypatch.setenv("BOT_WORKER_SCALE", bad)
        assert capacity.scale_factor() == pytest.approx(1.0)


def test_max_workers_is_a_hard_ceiling_over_everything(monkeypatch):
    _fake_hardware(monkeypatch, 64, 256.0)
    monkeypatch.setenv("BOT_MAX_WORKERS", "3")

    assert capacity.workers(50, minimum=8) == 3
    assert capacity.workers(6, minimum=2, maximum=100) == 3


def test_invalid_max_workers_is_ignored(monkeypatch):
    _fake_hardware(monkeypatch, capacity.REFERENCE_CPUS, capacity.REFERENCE_MEMORY_GB)
    monkeypatch.setenv("BOT_MAX_WORKERS", "not-a-number")
    assert capacity.workers(30, minimum=2) == 30


# ---------------------------------------------------------------------------
# Container (cgroup) detection -- the case where os.cpu_count() lies
# ---------------------------------------------------------------------------

def test_cgroup_v2_cpu_quota_is_read_as_fractional_cores(tmp_path):
    quota = tmp_path / "cpu.max"
    quota.write_text("150000 100000\n", encoding="utf-8")

    assert capacity.cgroup_cpu_limit(v2=quota) == pytest.approx(1.5)


def test_cgroup_v2_max_means_unlimited(tmp_path):
    quota = tmp_path / "cpu.max"
    quota.write_text("max 100000\n", encoding="utf-8")

    missing = tmp_path / "absent"
    assert capacity.cgroup_cpu_limit(v2=quota, v1_quota=missing, v1_period=missing) is None


def test_cgroup_v1_quota_and_period_are_combined(tmp_path):
    v2 = tmp_path / "absent-v2"
    quota = tmp_path / "cpu.cfs_quota_us"
    period = tmp_path / "cpu.cfs_period_us"
    quota.write_text("50000\n", encoding="utf-8")
    period.write_text("100000\n", encoding="utf-8")

    assert capacity.cgroup_cpu_limit(v2=v2, v1_quota=quota, v1_period=period) == pytest.approx(0.5)


def test_cgroup_v1_negative_quota_means_unlimited(tmp_path):
    v2 = tmp_path / "absent-v2"
    quota = tmp_path / "cpu.cfs_quota_us"
    period = tmp_path / "cpu.cfs_period_us"
    quota.write_text("-1\n", encoding="utf-8")
    period.write_text("100000\n", encoding="utf-8")

    assert capacity.cgroup_cpu_limit(v2=v2, v1_quota=quota, v1_period=period) is None


def test_cgroup_limit_absent_when_no_files_exist(tmp_path):
    missing = tmp_path / "nope"
    assert capacity.cgroup_cpu_limit(v2=missing, v1_quota=missing, v1_period=missing) is None


def test_container_cpu_quota_overrides_host_core_count(monkeypatch, tmp_path):
    """The whole point: a 2-core container on a 64-core host must size for 2."""
    monkeypatch.setattr(capacity, "cgroup_cpu_limit", lambda **_kw: 2.0)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    assert capacity.cpu_limit() == pytest.approx(2.0)


def test_meminfo_is_parsed_into_bytes(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:        2048000 kB\nMemFree:          100000 kB\n", encoding="utf-8"
    )
    assert capacity._meminfo_bytes(meminfo) == 2048000 * 1024


def test_meminfo_missing_returns_none(tmp_path):
    assert capacity._meminfo_bytes(tmp_path / "absent") is None


def test_describe_reports_hardware_without_raising(monkeypatch):
    _fake_hardware(monkeypatch, 2, 4.0)
    summary = capacity.describe()

    assert "cpu=2" in summary
    assert "4.0GB" in summary
    assert "scale=" in summary


def test_describe_tolerates_undetectable_memory(monkeypatch):
    _fake_hardware(monkeypatch, 2, None)
    assert "unknown" in capacity.describe()


def test_real_hardware_detection_returns_sane_values():
    """Smoke test against the actual host: whatever it is, the values must be
    self-consistent rather than zero, negative, or absurd."""
    assert capacity.cpu_limit() > 0
    memory = capacity.memory_limit_bytes()
    assert memory is None or memory > 0
    assert 0 < capacity.scale_factor() <= capacity.MAX_SCALE


# ---------------------------------------------------------------------------
# Memory affordability gate (semantic model) and memory-proportional caps
# ---------------------------------------------------------------------------

def test_can_afford_rejects_allocation_larger_than_host_memory(monkeypatch):
    _fake_hardware(monkeypatch, 1, 1.0)
    assert capacity.can_afford(2.0) is False


def test_can_afford_accepts_when_host_has_room(monkeypatch):
    _fake_hardware(monkeypatch, 4, 8.0)
    assert capacity.can_afford(2.0) is True


def test_can_afford_defaults_to_yes_when_memory_is_undetectable(monkeypatch):
    """Failing to detect memory must not silently disable features."""
    _fake_hardware(monkeypatch, 4, None)
    assert capacity.can_afford(64.0) is True


def test_scaled_cap_shrinks_on_small_hosts_but_never_grows(monkeypatch):
    _fake_hardware(monkeypatch, 1, 1.0)
    assert capacity.scaled_cap(80_000, minimum=5_000) < 80_000
    assert capacity.scaled_cap(80_000, minimum=5_000) >= 5_000

    _fake_hardware(monkeypatch, 128, 512.0)
    assert capacity.scaled_cap(80_000, minimum=5_000) == 80_000


def test_scaled_cap_honours_its_floor(monkeypatch):
    """Where the scaled value would fall below the floor, the floor wins."""
    _fake_hardware(monkeypatch, 0.1, 0.1)
    # 10_000 * MIN_SCALE (0.08) == 800, which is below the 5_000 floor.
    assert capacity.scaled_cap(10_000, minimum=5_000) == 5_000


# ---------------------------------------------------------------------------
# Timeouts: these scale INVERSELY to pool sizes
# ---------------------------------------------------------------------------

def test_timeouts_stretch_on_slow_hardware(monkeypatch):
    """The interaction that matters: pools shrink AND each item gets slower, so
    a fixed budget that was generous on the reference box becomes marginal."""
    _fake_hardware(monkeypatch, 1, 1.0)

    assert capacity.workers(50, minimum=2) < 50        # pool scaled down
    assert capacity.timeout(90) > 90                   # budget scaled up


def test_timeouts_are_unchanged_on_reference_hardware(monkeypatch):
    _fake_hardware(monkeypatch, capacity.REFERENCE_CPUS, capacity.REFERENCE_MEMORY_GB)

    assert capacity.timeout(90) == pytest.approx(90, rel=0.02)
    assert capacity.latency_multiplier() == pytest.approx(1.0, rel=0.02)


def test_timeouts_never_tighten_on_fast_hardware(monkeypatch):
    """Shortening budgets on a big box would invent failures that do not
    currently happen, for no gain."""
    _fake_hardware(monkeypatch, 128, 512.0)

    assert capacity.timeout(90) == 90
    assert capacity.latency_multiplier() == 1.0


def test_timeout_stretch_is_bounded(monkeypatch):
    """A mis-detected host must not turn a 90s budget into an hour."""
    _fake_hardware(monkeypatch, 0.1, 0.1)

    assert capacity.timeout(90) <= 90 * capacity.MAX_TIMEOUT_STRETCH
    assert capacity.latency_multiplier() <= capacity.MAX_TIMEOUT_STRETCH


def test_timeout_respects_a_caller_supplied_stretch_cap(monkeypatch):
    _fake_hardware(monkeypatch, 1, 1.0)
    assert capacity.timeout(90, maximum_multiplier=1.5) == pytest.approx(135)


def test_timeout_caller_cap_below_one_is_ignored(monkeypatch):
    """A cap under 1.0 would mean 'shorten', which this API must never do."""
    _fake_hardware(monkeypatch, 1, 1.0)
    assert capacity.timeout(90, maximum_multiplier=0.1) == 90


def test_scrape_cache_ttl_stays_under_the_stretched_subprocess_budget(monkeypatch):
    """Both derive from the same multiplier, so the cache TTL must not overtake
    the scrape budget it is meant to sit just under."""
    for cpus, gb in ((1, 1.0), (4, 8.0), (12, 16.0), (64, 256.0)):
        _fake_hardware(monkeypatch, cpus, gb)
        assert capacity.timeout(55.0) < capacity.timeout(90)
