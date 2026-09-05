"""Sixteen ATS platforms have to fit inside one cycle's time budget.

Each platform is a single scheduler submission, so at most `scheduler workers`
of them run at once and the rest queue. That makes the number of platforms, the
scheduler's width and the two timeouts one arithmetic problem:

    waves = ceil(platforms / workers)
    waves * ATS_PLATFORM_TIMEOUT_S  must fit  ATS_CYCLE_TIMEOUT_S

Adding a platform is free until it pushes the wave count up, and then it is not
free at all: the whole last wave is cancelled by the outer timeout, and those
platforms are simply never asked. Nothing reports that as a fault -- the cycle
returns what it gathered and looks like a success.

This came up for real when BambooHR was switched on, taking the roster from
fifteen platforms to sixteen. On twelve workers both are two waves, so it cost
nothing; on five workers fifteen is three waves and sixteen is four, which
would not fit. The arithmetic was accidental until it was written down.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import capacity  # noqa: E402
from services.ats_service import ATS_PLATFORMS  # noqa: E402
from watchers import manager as M  # noqa: E402


def _waves(platforms: int, workers: int) -> int:
    return math.ceil(platforms / max(workers, 1))


def _fits(platforms: int, workers: int) -> bool:
    return (_waves(platforms, workers) * M.ATS_PLATFORM_TIMEOUT_S
            <= M.ats_cycle_timeout(platforms, workers))


def _scheduler_workers() -> int:
    """The same expression PriorityWorkScheduler sizes itself with."""
    return max(2, min(32, round(capacity.cpu_limit())))


# ── the budget on this host ──────────────────────────────────────────────────

def test_every_platform_fits_the_cycle_on_this_host():
    """The one that matters: if this fails, some platforms are being cancelled
    by the outer timeout on every cycle and nothing says so.
    """
    workers = _scheduler_workers()
    assert _fits(len(ATS_PLATFORMS), workers), (
        f"{len(ATS_PLATFORMS)} platforms over {workers} workers is "
        f"{_waves(len(ATS_PLATFORMS), workers)} waves x "
        f"{M.ATS_PLATFORM_TIMEOUT_S}s > {M.ATS_CYCLE_TIMEOUT_S}s"
    )


def test_the_cycle_budget_is_a_whole_number_of_platform_budgets():
    """A bound that is not a multiple of the platform bound wastes the
    remainder: a wave that cannot finish is worse than not starting it.
    """
    for workers in (1, 2, 4, 5, 8, 12, 16, 32):
        assert M.ats_cycle_timeout(16, workers) % M.ATS_PLATFORM_TIMEOUT_S == 0


def test_every_host_width_gets_enough_room_for_all_its_waves():
    """The whole point of deriving it: no host silently drops a wave."""
    for workers in range(1, 33):
        assert _fits(len(ATS_PLATFORMS), workers), workers


# ── the arithmetic itself ────────────────────────────────────────────────────

@pytest.mark.parametrize("workers,expected_waves", [
    (16, 1), (12, 2), (8, 2), (6, 3), (5, 4), (4, 4), (2, 8),
])
def test_wave_count_for_sixteen_platforms(workers, expected_waves):
    assert _waves(16, workers) == expected_waves


def test_the_bamboohr_addition_costs_nothing_at_twelve_workers():
    """Fifteen and sixteen platforms are both two waves there, which is why
    switching BambooHR on did not change the cycle's shape.
    """
    assert _waves(15, 12) == _waves(16, 12) == 2


def test_a_narrow_host_is_where_the_sixteenth_platform_bites():
    """The boundary, stated rather than left to be discovered. At five
    scheduler workers -- a 5-core host or container -- fifteen platforms fit
    exactly (3 x 600s = 1800s) and sixteen do not (4 x 600s = 2400s).

    So switching BambooHR on is free on this 12-core host and is NOT free on a
    5-core one, where the last wave would be cancelled wholesale. That is a
    real deployment caveat, not a hypothetical: the fix there is to raise
    ATS_CYCLE_TIMEOUT_S or switch BambooHR back off in settings.toml.
    """
    assert _waves(15, 5) == 3
    assert _waves(16, 5) == 4
    # Under the old fixed 1800s bound the sixteenth platform lost a whole wave.
    assert 3 * M.ATS_PLATFORM_TIMEOUT_S <= 1800
    assert 4 * M.ATS_PLATFORM_TIMEOUT_S > 1800
    # Derived, both fit, because the bound grew with the host.
    assert _fits(15, 5) is True
    assert _fits(16, 5) is True
    assert M.ats_cycle_timeout(16, 5) == 2400


def test_a_host_with_as_many_workers_as_platforms_is_one_wave():
    assert _fits(len(ATS_PLATFORMS), len(ATS_PLATFORMS)) is True


# ── the constants are the ones actually used ────────────────────────────────

def test_the_loop_uses_the_named_budgets_not_bare_numbers():
    """The point of naming them is that the test and the loop cannot drift.
    Literal 600/1800 in the loop would leave this passing while the real
    timeouts moved.
    """
    import inspect
    compact = " ".join(inspect.getsource(M.WatcherManager._run_ats_scrape_loop).split())
    assert "timeout=ATS_PLATFORM_TIMEOUT_S" in compact
    assert "timeout=ats_cycle_timeout(" in compact
    assert "timeout=600" not in compact
    assert "timeout=1800" not in compact


def test_the_derived_bound_is_capped_so_a_cycle_always_returns():
    """The rail that stops the derivation running away. Today's sixteen
    platforms on one worker is 9,600s, comfortably under it -- so the cap is
    tested where it actually binds, at a roster large enough to need more than
    twenty-four waves. Without it, a growing roster on a small container would
    be handed a bound longer than the six-hour gap between cycles and the loop
    would stop scheduling itself.
    """
    assert M.ats_cycle_timeout(16, 1) == 9600            # under the cap
    assert M.ats_cycle_timeout(40, 1) == M.ATS_CYCLE_TIMEOUT_CAP_S   # would be 24,000s
    assert M.ats_cycle_timeout(200, 2) == M.ATS_CYCLE_TIMEOUT_CAP_S
    assert M.ATS_CYCLE_TIMEOUT_CAP_S < 6 * 3600


def test_degenerate_inputs_do_not_produce_a_zero_budget():
    """A zero bound would cancel the cycle instantly, every time."""
    assert M.ats_cycle_timeout(0, 0) >= M.ATS_PLATFORM_TIMEOUT_S
    assert M.ats_cycle_timeout(1, 0) >= M.ATS_PLATFORM_TIMEOUT_S
