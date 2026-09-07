"""After a cut-off, the measurement wins over the floor, and a head that does
not fit rotates like a tail.

First capped cycle, measured: thirteen platforms reached 100% of what they were
asked; icims completed 103 of 600 and workday 1,050 of 1,424. The floor had
promised 450 boards would fit in 480s because it assumes one request per
board, and an icims board is a sitemap plus a page per posting. Re-asking the
floor every cycle would cut icims off at the same 103 boards forever -- all of
them inside its 150-board head, so neither the rest of the head nor any of the
tail would ever be asked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service, ats_traversal  # noqa: E402
from watchers.manager import ATS_PLATFORM_FANOUT_BUDGET_S, WatcherManager, ats_fanout_cap  # noqa: E402


# ── the rule ─────────────────────────────────────────────────────────────────

def test_a_cut_off_is_asked_what_it_completed_not_the_floor():
    """icims, measured: 103 of 600, floor 450, head 150, pool 30."""
    assert ats_fanout_cap(13_307, 150, 450, last_submitted=600, last_completed=103, min_tail=30) == 180


def test_the_cut_off_ask_keeps_one_wave_of_tail_so_discovery_never_stops():
    assert ats_fanout_cap(10_000, 100, 450, last_submitted=550, last_completed=20, min_tail=8) == 108


def test_a_cut_off_that_reached_past_the_floor_keeps_what_it_reached():
    assert ats_fanout_cap(15_007, 674, 750, last_submitted=1_424, last_completed=1_050, min_tail=50) == 1_050


def test_without_min_tail_the_old_floor_rule_still_holds():
    assert ats_fanout_cap(10_000, 400, 450, last_submitted=5_000, last_completed=3) == 850


def test_min_tail_never_exceeds_the_floor():
    assert ats_fanout_cap(10_000, 100, 20, last_submitted=200, last_completed=5, min_tail=500) == 120


def test_a_finished_cycle_still_grows_from_the_floor_or_more():
    assert ats_fanout_cap(10_000, 100, 450, last_submitted=550, last_completed=550, min_tail=8) == 688


# ── the head rotates when a cycle stopped inside it ──────────────────────────

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
    counts: dict[str, dict[str, int]] = {}
    monkeypatch.setattr(
        WatcherManager, "_ats_last_fanout",
        lambda self, platform: dict(counts.get(platform) or {"submitted": 0, "completed": 0}),
    )
    return counts


@pytest.fixture
def one_worker(monkeypatch):
    monkeypatch.setattr(ats_service, "fanout_workers", lambda platform, fleet_size: 1)
    return ats_service.fanout_capacity(ATS_PLATFORM_FANOUT_BUDGET_S, 1)


def _fleet(monkeypatch, platform, slugs, preferred=()):
    from services.jba import geo_priority

    monkeypatch.setattr(ats_service, "load_company_lists", lambda: {platform: list(slugs)})
    monkeypatch.setattr(geo_priority, "slugs_for", lambda country, *a, **k: frozenset(preferred))


CANADA = {1: {"enabled": True, "location": "Canada"}}


def test_a_head_that_fits_keeps_its_order(monkeypatch, tmp_path, last_fanout, one_worker):
    fleet = [f"c{i}" for i in range(400)]
    head = {f"c{i}" for i in range(20)}
    _fleet(monkeypatch, "icims", fleet, preferred=head)
    mgr = _manager(tmp_path, CANADA)

    first = mgr._ordered_slugs("icims")
    last_fanout["icims"] = {"submitted": len(first), "completed": len(first)}
    second = mgr._ordered_slugs("icims")
    assert second[:20] == first[:20], "the head is stable while it fits"


def test_a_cycle_cut_off_inside_the_head_moves_the_head(monkeypatch, tmp_path, last_fanout, one_worker):
    """103 of a 150-board head reached: the next cycle starts at the 104th,
    and the 103 already asked come round again at the back of the head."""
    fleet = [f"c{i}" for i in range(400)]
    head_set = {f"c{i}" for i in range(150)}
    _fleet(monkeypatch, "icims", fleet, preferred=head_set)
    mgr = _manager(tmp_path, CANADA)

    first = mgr._ordered_slugs("icims")
    assert set(first[:150]) == head_set
    last_fanout["icims"] = {"submitted": len(first), "completed": 103}

    second = mgr._ordered_slugs("icims")
    assert set(second[:150]) == head_set, "still the whole head, first"
    assert second[:47] == first[103:150], "resumes where the cut-off landed"
    assert second[47:150] == first[:103], "and the asked ones go to the back"
    assert len(second) == 150 + 1, "the ask is head plus one wave of tail"


def test_repeated_cut_offs_give_every_head_board_a_turn(monkeypatch, tmp_path, last_fanout, one_worker):
    fleet = [f"c{i}" for i in range(400)]
    head_set = {f"c{i}" for i in range(150)}
    _fleet(monkeypatch, "icims", fleet, preferred=head_set)
    mgr = _manager(tmp_path, CANADA)

    asked: set[str] = set()
    for _ in range(3):
        ordered = mgr._ordered_slugs("icims")
        asked.update(ordered[:60])
        last_fanout["icims"] = {"submitted": len(ordered), "completed": 60}
    assert asked == head_set, "three cycles of 60 cover a head of 150"


def test_the_head_cursor_survives_a_restart(monkeypatch, tmp_path, last_fanout, one_worker):
    fleet = [f"c{i}" for i in range(400)]
    head_set = {f"c{i}" for i in range(150)}
    _fleet(monkeypatch, "icims", fleet, preferred=head_set)

    first = _manager(tmp_path, CANADA)._ordered_slugs("icims")
    last_fanout["icims"] = {"submitted": len(first), "completed": 40}
    _manager(tmp_path, CANADA)._ordered_slugs("icims")     # advances and persists
    last_fanout["icims"] = {"submitted": 0, "completed": 0}  # fresh process, nothing reported yet
    resumed = _manager(tmp_path, CANADA)._ordered_slugs("icims")
    assert resumed[0] == first[40]


def test_the_head_cursor_does_not_disturb_the_tail_cursor(monkeypatch, tmp_path, last_fanout, one_worker):
    """The two cursors share a state file and must not overwrite each other."""
    fleet = [f"c{i}" for i in range(400)]
    head_set = {f"c{i}" for i in range(10)}
    _fleet(monkeypatch, "lever", fleet, preferred=head_set)
    mgr = _manager(tmp_path, CANADA)
    tail_full = ats_traversal.rotate([s for s in fleet if s not in head_set], None)

    first = mgr._ordered_slugs("lever")                     # 10 head + 15 tail
    last_fanout["lever"] = {"submitted": len(first), "completed": len(first)}   # finished: tail advances by 15
    second = mgr._ordered_slugs("lever")
    assert second[10:] == tail_full[15: 15 + (len(second) - 10)]

    last_fanout["lever"] = {"submitted": len(second), "completed": 4}           # cut inside the head
    third = mgr._ordered_slugs("lever")
    assert third[:6] == second[4:10], "head rotated by 4"
    assert third[10] == second[10], "tail did not move: the cut-off never reached it"
