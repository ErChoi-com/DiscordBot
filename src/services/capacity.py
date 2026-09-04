"""Scale thread-pool sizes to the hardware the bot is actually running on.

The pool sizes throughout this codebase (ATS fetch pools, JBA per-platform
pools, the scrape fan-out, Reddit batches) were tuned on a 12-core / 16 GB
machine. Run unchanged on a 1-core 1 GB VPS they oversubscribe badly: 50
concurrent Workday fetches on one core means every request competes for the same
CPU to do TLS and JSON parsing, and the memory ceiling is hit long before the
network is saturated.

`workers()` scales a tuned number in both directions, but upward scaling is
opt-in per call site via `maximum=`. That asymmetry is deliberate: some of these
pool sizes are politeness ceilings against third-party sites, not local resource
limits, and a bigger box must not mean more aggressive scraping. Pools that only
consume local resources (the scheduler, the scrape fan-out) pass an explicit
`maximum` and do grow on larger hardware.

Detection is container-aware. `os.cpu_count()` reports the *host's* cores inside
a cgroup-limited container, which is exactly the case where scaling down matters
most, so cgroup v2/v1 CPU quota and memory limits are read first.

Overrides:
    BOT_WORKER_SCALE   positive float; forces the scale factor (>1 scales up)
    BOT_MAX_WORKERS    int; hard ceiling applied to every pool
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from services import platform_support

# The machine the current pool sizes were tuned on. Scaling is relative to this,
# so that box (and anything larger) keeps today's behaviour exactly.
REFERENCE_CPUS = 12.0
REFERENCE_MEMORY_GB = 16.0

# I/O-bound pools stay useful well below their tuned size -- threads spend most
# of their life blocked on the network -- so never collapse past this fraction
# purely on hardware. Callers still set their own absolute floor, which is the
# real protection; this only stops the factor reaching zero. Kept low enough
# that a 1-core/1GB box and a 2-core/2GB box get visibly different pools rather
# than both bottoming out on the same clamp.
MIN_SCALE = 0.08

# Ceiling on upward scaling, for callers that opt in via `maximum=`. Past this,
# extra threads stop buying throughput on I/O-bound work and just add memory and
# context-switching, so a 128-core box should not open 10x the connections.
MAX_SCALE = 4.0

_CGROUP_V2_CPU = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
_CGROUP_V2_MEM = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MEM = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_MEMINFO = Path("/proc/meminfo")


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def cgroup_cpu_limit(
    v2: Path = _CGROUP_V2_CPU,
    v1_quota: Path = _CGROUP_V1_QUOTA,
    v1_period: Path = _CGROUP_V1_PERIOD,
) -> float | None:
    """Effective cores from a cgroup CPU quota, or None if unlimited/absent."""
    try:
        raw = v2.read_text(encoding="utf-8").split()
    except OSError:
        raw = []
    if len(raw) == 2 and raw[0] != "max":
        try:
            v2_quota, v2_period = int(raw[0]), int(raw[1])
            if v2_quota > 0 and v2_period > 0:
                return v2_quota / v2_period
        except ValueError:
            pass

    quota = _read_int(v1_quota)
    period = _read_int(v1_period)
    if quota is not None and quota > 0 and period is not None and period > 0:
        return quota / period
    return None


def cpu_limit() -> float:
    """Cores available to this process, honouring cgroup quota and affinity."""
    cgroup = cgroup_cpu_limit()
    if cgroup is not None:
        return max(0.1, cgroup)
    try:
        affinity = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        if affinity > 0:
            return float(affinity)
    except (AttributeError, OSError):
        pass
    return float(os.cpu_count() or 1)


def _meminfo_bytes(meminfo: Path = _MEMINFO) -> int | None:
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def memory_limit_bytes() -> int | None:
    """Memory available to this process, or None if it cannot be determined.

    A cgroup limit wins over physical RAM: a 2 GB container on a 64 GB host must
    size its pools for 2 GB. Absurdly large cgroup values mean "unlimited" and
    are ignored in favour of the physical total.
    """
    physical = _meminfo_bytes() or platform_support.total_physical_memory()

    for path in (_CGROUP_V2_MEM, _CGROUP_V1_MEM):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        # cgroup v1 signals "no limit" with a near-word-size sentinel.
        if limit <= 0 or limit >= (1 << 62):
            continue
        return min(limit, physical) if physical else limit

    return physical


def scale_factor() -> float:
    """How far to scale tuned pool sizes, relative to the reference machine.

    1.0 means "reference hardware, use the tuned value". Below 1 means scale
    down, above 1 means there is headroom to scale up. Bound by whichever of CPU
    or memory is scarcer, since both constrain concurrent scrapes.
    """
    override = (os.environ.get("BOT_WORKER_SCALE") or "").strip()
    if override:
        try:
            value = float(override)
            if value > 0:
                return min(value, MAX_SCALE)
        except ValueError:
            pass

    factor = cpu_limit() / REFERENCE_CPUS

    memory = memory_limit_bytes()
    if memory:
        factor = min(factor, (memory / 1024 ** 3) / REFERENCE_MEMORY_GB)

    return max(MIN_SCALE, min(MAX_SCALE, factor))


def workers(nominal: int, minimum: int = 1, maximum: int | None = None) -> int:
    """Scale a tuned pool size to this hardware.

    `maximum` gates upward scaling and defaults to `nominal`, i.e. down-only.
    Pass it explicitly for pools that consume only local resources; leave it
    unset for pools whose size is really a politeness ceiling against a
    third-party site, where more hardware must not mean more load on them.
    """
    ceiling = nominal if maximum is None else max(maximum, minimum)
    floor = max(1, min(minimum, ceiling))

    scaled = math.ceil(nominal * scale_factor())
    scaled = max(floor, min(scaled, ceiling))

    hard_cap = (os.environ.get("BOT_MAX_WORKERS") or "").strip()
    if hard_cap:
        try:
            limit = int(hard_cap)
            if limit > 0:
                scaled = min(scaled, limit)
        except ValueError:
            pass
    return max(1, scaled)


# Ceiling on how far a timeout may stretch on slow hardware. Without a cap, a
# badly mis-detected host could turn a 90s budget into something that hangs a
# watcher cycle for the better part of an hour.
MAX_TIMEOUT_STRETCH = 3.0


def latency_multiplier() -> float:
    """How much longer local work takes here than on the reference machine.

    Sub-linear (1/sqrt) rather than a straight inverse of `scale_factor()`.
    A straight inverse both over-corrects -- work does not slow in exact
    proportion to a factor that is itself min(cpu_ratio, memory_ratio) -- and
    saturates the cap almost immediately, which would give a 4-core Pi and a
    1-core VPS identical timeouts. The square root keeps them distinguishable
    across the whole range.

    Never below 1.0: tightening timeouts on a fast host would invent failures
    that do not currently happen, for no gain.
    """
    return max(1.0, min(MAX_TIMEOUT_STRETCH, 1.0 / math.sqrt(scale_factor())))


def timeout(nominal: float, maximum_multiplier: float = MAX_TIMEOUT_STRETCH) -> float:
    """Stretch a CPU- or operation-bound timeout to suit slower hardware.

    This scales in the OPPOSITE direction to `workers()`, and the two interact:
    scaling a pool down makes each item slower to get through, so a fixed
    timeout that was generous on the reference box becomes marginal on a small
    one. Timeouts, not pool sizes, are what actually break on slow hardware.

    Use for whole-operation budgets (a scrape subprocess, a LaTeX compile, a
    browser launch). Do NOT use for per-request network timeouts: a slow local
    CPU does not make a remote server answer any slower, so stretching those
    just delays the detection of a genuinely dead endpoint.
    """
    multiplier = min(latency_multiplier(), max(1.0, maximum_multiplier))
    return nominal * multiplier


def can_afford(required_gb: float) -> bool:
    """True if the host plausibly has room for a `required_gb` allocation.

    Returns True when memory cannot be determined: refusing to load a feature
    because detection failed would be worse than attempting it. Callers should
    treat this as "is this a good idea", not as a hard guarantee.
    """
    memory = memory_limit_bytes()
    if not memory:
        return True
    return (memory / 1024 ** 3) >= required_gb


def scaled_cap(nominal: int, minimum: int) -> int:
    """Scale a memory-proportional cap (dedup sets, seen-URL lists).

    Down-only: these caps trade memory for dedup accuracy, and growing them on a
    big host would silently increase steady-state memory for no user-visible
    gain.
    """
    return max(minimum, min(nominal, int(nominal * min(1.0, scale_factor()))))


def describe() -> str:
    """One-line hardware summary for the /health dashboard and startup logs."""
    memory = memory_limit_bytes()
    memory_text = f"{memory / 1024 ** 3:.1f}GB" if memory else "unknown"
    return f"cpu={cpu_limit():.3g} mem={memory_text} scale={scale_factor():.2f}"
