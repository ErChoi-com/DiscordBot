"""Each platform is asked as much of its fleet as its budget can actually finish.

The fan-out used to be handed the whole fleet and cut off at its budget. With
135,475 slugs across eighteen platforms and 1,633 of them known to post in a
wanted country, every cycle submitted work it would mostly cancel; the in-flight
survivors of the cancelled tail then held their scheduler workers for 3-4x the
budget, and the next cycle's platforms queued behind them until their own bound
expired before they had started -- "timed out (0/0 companies reached)" on all
eighteen, every cycle, for hours.

So the ask is sized from evidence: the geo-preferred head always, plus a slice
of the rotated tail that is what the last cycle completed (a cut-off), a little
more than that (it finished), or a pessimistic floor (nothing recorded yet).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service, ats_traversal  # noqa: E402
from watchers.manager import (  # noqa: E402
    ATS_FANOUT_GROWTH,
    ATS_PLATFORM_FANOUT_BUDGET_S,
    WatcherManager,
    ats_fanout_cap,
    fanout_reached,
)


# ── the sizing rule itself ────────────────────────────────────────────────────

def test_a_platform_that_has_not_reported_gets_the_head_and_the_floor():
    assert ats_fanout_cap(fleet_size=10_000, head_size=400, floor=450,
                          last_submitted=0, last_completed=0) == 850


def test_a_cut_off_cycle_is_asked_what_it_completed():
    """Completed under a cut-off is a direct measurement of what the budget
    affords on this host against this vendor today."""
    assert ats_fanout_cap(10_000, 400, 100, last_submitted=5_000, last_completed=2_512) == 2_512


def test_a_cut_off_never_shrinks_below_the_head_and_floor():
    """A cycle that completed almost nothing (refusals, a dead vendor) must
    not talk the next one down to nothing; the floor is a floor."""
    assert ats_fanout_cap(10_000, 400, 450, last_submitted=5_000, last_completed=3) == 850


def test_a_cycle_that_finished_everything_is_offered_more():
    got = ats_fanout_cap(10_000, 0, 0, last_submitted=800, last_completed=800)
    assert got == 1_000
    assert got == -(-800 * 125 // 100)  # ceil(800 * 1.25), spelled out


def test_growth_is_gentle_by_design():
    """Overshooting is paid for in in-flight slugs the pool must drain past
    the budget, each a board fetch plus an enrichment pass -- doubling would
    hand that bill to every platform that finds its edge."""
    assert 1.0 < ATS_FANOUT_GROWTH <= 1.5


def test_the_ask_never_exceeds_the_fleet():
    assert ats_fanout_cap(50, 10, 450, 0, 0) == 50
    assert ats_fanout_cap(50, 10, 0, last_submitted=50, last_completed=50) == 50


def test_the_head_is_always_asked_whole_even_when_it_alone_exceeds_the_floor():
    """The head is the part that yields. Sizing it down would be optimising
    for the wrong half of the fleet."""
    assert ats_fanout_cap(10_000, head_size=900, floor=15, last_submitted=0, last_completed=0) == 915
    assert ats_fanout_cap(10_000, head_size=900, floor=15, last_submitted=915, last_completed=300) == 915


def test_a_head_larger_than_the_fleet_is_clamped_not_an_error():
    assert ats_fanout_cap(fleet_size=5, head_size=9, floor=0, last_submitted=0, last_completed=0) == 5


def test_a_growth_below_one_cannot_shrink_a_finished_platform():
    assert ats_fanout_cap(10_000, 0, 0, 800, 800, growth=0.5) == 800


# ── the floor: what the pool is guaranteed to drain ───────────────────────────

@pytest.mark.parametrize("workers", [1, 2, 4, 8, 30, 50])
def test_the_floor_is_the_inverse_of_the_fanout_budget(workers):
    """Exactly as many fetches as _fanout_budget says fit in the budget when
    every one of them stalls, and not one more wave."""
    cap = ats_service.fanout_capacity(ATS_PLATFORM_FANOUT_BUDGET_S, workers)
    assert ats_service._fanout_budget(cap, workers) <= ATS_PLATFORM_FANOUT_BUDGET_S
    assert ats_service._fanout_budget(cap + workers, workers) > ATS_PLATFORM_FANOUT_BUDGET_S


def test_a_budget_too_small_for_one_wave_still_asks_one_wave():
    assert ats_service.fanout_capacity(1.0, 8) == 8
    assert ats_service.fanout_capacity(0, 8) == 8


def test_the_floor_scales_with_the_pool_width():
    """workable runs 4 workers by measured necessity, workday 50. The same
    budget affords them very different fleets."""
    narrow = ats_service.fanout_capacity(ATS_PLATFORM_FANOUT_BUDGET_S, 4)
    wide = ats_service.fanout_capacity(ATS_PLATFORM_FANOUT_BUDGET_S, 50)
    assert wide == narrow * 50 // 4


def test_fanout_workers_is_the_width_the_scrape_actually_runs_at(monkeypatch):
    """One definition, or the sizing assumes a pool the scrape does not have."""
    monkeypatch.setattr(ats_service.capacity, "workers", lambda nominal, minimum=1, maximum=None: 7)
    assert ats_service.fanout_workers("greenhouse", 100) == 7
    assert ats_service.fanout_workers("greenhouse", 3) == 3, "never wider than the fleet"
    assert ats_service.fanout_workers("greenhouse", 0) == 1


def test_the_scrape_uses_fanout_workers_for_its_pool(monkeypatch):
    """The seam the sizing reads must be the one the scrape reads, or a change
    to either quietly desynchronises them."""
    seen: dict = {}

    def _spy(platform, fleet_size):
        seen["args"] = (platform, fleet_size)
        return 1

    monkeypatch.setattr(ats_service, "fanout_workers", _spy)
    monkeypatch.setitem(ats_service._SCRAPERS, "greenhouse", lambda *a: [])
    monkeypatch.setattr(ats_service, "reset_refusal_breaker", lambda p: None)
    monkeypatch.setattr(ats_service, "flush_dead_slugs", lambda: None)
    monkeypatch.setattr(ats_service, "_drop_already_archived", lambda p, rows: rows)
    ats_service.scrape_ats_platform("greenhouse", "", "", 0, company_slugs=["a", "b", "c"])
    assert seen["args"] == ("greenhouse", 3)


# ── the ordering the scrape is handed ────────────────────────────────────────

class _Store:
    def __init__(self, channels):
        self.channel_job_settings = channels


class _Config:
    def __init__(self, base_dir):
        self.base_dir = base_dir


def _manager(tmp_path, channels=None):
    mgr = WatcherManager.__new__(WatcherManager)
    mgr.store = _Store(channels or {})
    mgr.config = _Config(tmp_path)
    return mgr


@pytest.fixture
def last_fanout(monkeypatch):
    """What the last cycle reached, per platform -- the seam the sizing and
    the rotation both read."""
    counts: dict[str, dict[str, int]] = {}
    monkeypatch.setattr(
        WatcherManager,
        "_ats_last_fanout",
        lambda self, platform: dict(counts.get(platform) or {"submitted": 0, "completed": 0}),
    )
    return counts


@pytest.fixture
def one_worker(monkeypatch):
    """Pin the pool to one worker so the floor is a small, host-independent
    number: 480s / 30s per fetch, less one wave = 15 boards."""
    monkeypatch.setattr(ats_service, "fanout_workers", lambda platform, fleet_size: 1)
    return ats_service.fanout_capacity(ATS_PLATFORM_FANOUT_BUDGET_S, 1)


def _fleet(monkeypatch, platform, slugs, preferred=()):
    from services.jba import geo_priority

    monkeypatch.setattr(ats_service, "load_company_lists", lambda: {platform: list(slugs)})
    monkeypatch.setattr(geo_priority, "slugs_for", lambda country, *a, **k: frozenset(preferred))


def test_the_first_cycle_asks_the_head_and_one_floor_of_tail(monkeypatch, tmp_path, last_fanout, one_worker):
    fleet = [f"c{i}" for i in range(1_000)]
    _fleet(monkeypatch, "lever", fleet, preferred={"c5", "c9"})
    mgr = _manager(tmp_path, {1: {"enabled": True, "location": "Canada"}})

    asked = mgr._ordered_slugs("lever")
    # The head's own order is the digest rotation (see _rotate_head); what
    # matters here is that it is first and whole.
    assert set(asked[:2]) == {"c5", "c9"}, "the preferred head, first and whole"
    assert len(asked) == 2 + one_worker
    assert len(set(asked)) == len(asked)


def test_the_tail_slice_is_the_front_of_the_rotation(monkeypatch, tmp_path, last_fanout, one_worker):
    """Slicing must not disturb where the rotation starts, or the resume
    cursor and the slice would disagree about which boards were asked."""
    fleet = [f"c{i}" for i in range(1_000)]
    _fleet(monkeypatch, "lever", fleet)
    mgr = _manager(tmp_path)

    asked = mgr._ordered_slugs("lever")
    assert asked == ats_traversal.rotate(fleet, None)[: one_worker]


def test_a_finished_cycle_advances_the_slice_and_widens_it(monkeypatch, tmp_path, last_fanout, one_worker):
    """The property the whole change rests on: consecutive cycles that finish
    walk the fleet in adjacent, growing steps, without gaps or repeats."""
    fleet = [f"c{i}" for i in range(1_000)]
    _fleet(monkeypatch, "lever", fleet)
    mgr = _manager(tmp_path)
    full = ats_traversal.rotate(fleet, None)

    first = mgr._ordered_slugs("lever")
    last_fanout["lever"] = {"submitted": len(first), "completed": len(first)}
    second = mgr._ordered_slugs("lever")

    assert len(second) == -(-len(first) * 125 // 100), "grew by ATS_FANOUT_GROWTH"
    assert second == full[len(first): len(first) + len(second)], "picks up exactly where the first left off"


def test_a_cut_off_cycle_shrinks_the_next_ask_to_what_fit(monkeypatch, tmp_path, last_fanout, one_worker):
    fleet = [f"c{i}" for i in range(1_000)]
    _fleet(monkeypatch, "lever", fleet)
    mgr = _manager(tmp_path)

    asked = mgr._ordered_slugs("lever")                   # 15
    for _ in range(3):                                    # 19, 24, 30: widening
        last_fanout["lever"] = {"submitted": len(asked), "completed": len(asked)}
        asked = mgr._ordered_slugs("lever")
    assert len(asked) == 30

    last_fanout["lever"] = {"submitted": 30, "completed": 22}   # found its edge
    asked = mgr._ordered_slugs("lever")
    assert len(asked) == 22


def test_the_head_is_never_sliced_away(monkeypatch, tmp_path, last_fanout, one_worker):
    """A head larger than the floor is still asked whole; the slice comes out
    of the tail only."""
    fleet = [f"c{i}" for i in range(1_000)]
    head = {f"c{i}" for i in range(40)}
    _fleet(monkeypatch, "lever", fleet, preferred=head)
    mgr = _manager(tmp_path, {1: {"enabled": True, "location": "Canada"}})

    asked = mgr._ordered_slugs("lever")
    assert set(asked[:40]) == head
    assert len(asked) == 40 + one_worker


def test_repeated_finished_cycles_cover_the_whole_fleet(monkeypatch, tmp_path, last_fanout, one_worker):
    """A smaller ask does not mean the same companies every cycle. It means
    the fleet is walked in shorter steps -- each of which completes."""
    fleet = [f"c{i}" for i in range(300)]
    _fleet(monkeypatch, "lever", fleet, preferred={"c1"})
    mgr = _manager(tmp_path, {1: {"enabled": True, "location": "Canada"}})

    asked_ever: set[str] = set()
    for _ in range(40):
        asked = mgr._ordered_slugs("lever")
        asked_ever.update(asked)
        last_fanout["lever"] = {"submitted": len(asked), "completed": len(asked)}
        if asked_ever == set(fleet):
            break
    assert asked_ever == set(fleet)


def test_a_fleet_smaller_than_the_floor_is_asked_whole(monkeypatch, tmp_path, last_fanout, one_worker):
    _fleet(monkeypatch, "lever", ["a", "b", "c"])
    assert sorted(_manager(tmp_path)._ordered_slugs("lever")) == ["a", "b", "c"]


def test_the_ask_is_announced_with_the_fleet_it_came_from(monkeypatch, tmp_path, last_fanout, one_worker, capsys):
    """The line that would have named the failure in the first place: how
    many boards were handed over, out of how many, and what the last cycle
    reached. Without it the only trace of a fan-out was its cut-off."""
    fleet = [f"c{i}" for i in range(1_000)]
    _fleet(monkeypatch, "lever", fleet, preferred={"c5"})
    mgr = _manager(tmp_path, {1: {"enabled": True, "location": "Canada"}})
    last_fanout["lever"] = {"submitted": 16, "completed": 16}

    mgr._ordered_slugs("lever")
    out = capsys.readouterr().out
    assert "[ats-scrape] lever: asking 20 of 1,000 boards (1 preferred, 19 on rotation" in out
    assert "last cycle reached 16/16" in out


def test_the_cap_reads_the_pool_width_the_scrape_will_use(monkeypatch, tmp_path, last_fanout):
    """Width comes from ats_service.fanout_workers, not a copy of its formula."""
    seen: list = []

    def _width(platform, fleet_size):
        seen.append((platform, fleet_size))
        return 2

    monkeypatch.setattr(ats_service, "fanout_workers", _width)
    _fleet(monkeypatch, "lever", [f"c{i}" for i in range(100)])
    asked = _manager(tmp_path)._ordered_slugs("lever")
    assert seen == [("lever", 100)]
    assert len(asked) == ats_service.fanout_capacity(ATS_PLATFORM_FANOUT_BUDGET_S, 2)


# ── what a timeout says about itself ─────────────────────────────────────────

def test_a_platform_that_never_fanned_out_says_so():
    text = fanout_reached({"submitted": 0, "completed": 0})
    assert "no fan-out recorded" in text
    assert "queued" in text
    assert "0/0" not in text


def test_a_platform_that_ran_reports_its_coverage():
    assert fanout_reached({"submitted": 13_696, "completed": 2_512}) == "2,512/13,696 companies reached"
