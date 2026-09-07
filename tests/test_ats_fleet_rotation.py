"""The tail a cut-off ATS cycle cancels must not be the same companies forever.

A cycle is bounded at 600s per platform and 1800s overall. When it runs out --
icims has come in at 43% of its fleet -- the unfinished submissions are
cancelled, and in fleet-file order the cancelled set is identical on every
cycle. Those companies are never asked, not once, for as long as the file order
holds.

Rotating what follows the geo-preferred head fixes that without asking for
anything extra: the same fleet is submitted, so the request count, the
politeness ceiling and the platform's own load are all unchanged. Only which
companies sit in the part that gets cancelled moves.

The cursor is a digest rather than an index because the fleet is not stable --
the harvest adds slugs weekly and reconciliation removes them, and an integer
offset silently skips or repeats entries when that happens underneath it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_traversal as T  # noqa: E402
from watchers.manager import WatcherManager  # noqa: E402


# ── the rotation primitive ───────────────────────────────────────────────────

def test_rotation_returns_the_whole_fleet():
    """The property everything else rests on: this reorders, it never filters.
    If it can lose a slug then wiring it in front of the scraper shrinks
    coverage, which is the opposite of what it is for.
    """
    fleet = [f"c{i}" for i in range(500)]
    got = T.rotate(fleet, T.digest_for("c123"))
    assert sorted(got) == sorted(fleet)
    assert len(got) == len(set(got))


def test_no_cursor_starts_at_the_top_of_digest_order():
    fleet = ["a", "b", "c", "d"]
    assert T.rotate(fleet, None) == [s for _, s in T.stable_order(fleet)]


def test_the_cursor_slug_itself_is_not_repeated_at_the_front():
    """Resuming *after* the last reached company, not at it -- resuming at it
    would re-ask the same board on every cycle and never advance.
    """
    fleet = [f"c{i}" for i in range(50)]
    order = [s for _, s in T.stable_order(fleet)]
    got = T.rotate(fleet, T.digest_for(order[10]))
    assert got[0] == order[11]
    assert got[-1] == order[10]


def test_a_cursor_at_the_end_wraps_to_the_beginning():
    fleet = ["a", "b", "c"]
    order = [s for _, s in T.stable_order(fleet)]
    assert T.rotate(fleet, T.digest_for(order[-1])) == order


def test_a_cursor_whose_slug_is_gone_resumes_at_its_neighbourhood():
    """The fleet changes weekly. A removed cursor must not restart the walk --
    that would re-ask the front of the fleet and strand the far side again.
    """
    fleet = [f"c{i}" for i in range(200)]
    order = [s for _, s in T.stable_order(fleet)]
    victim = order[100]
    cursor = T.digest_for(victim)

    survivors = [s for s in fleet if s != victim]
    got = T.rotate(survivors, cursor)
    assert got[0] == order[101]
    assert sorted(got) == sorted(survivors)


def test_digest_order_is_not_alphabetical_order():
    """Fleet files are alphabetical, so an alphabetical walk cancels the same
    letters every time. The digest is what makes any prefix a cross-section.
    """
    fleet = [f"company{i:03d}" for i in range(200)]
    assert T.rotate(fleet, None) != sorted(fleet)


def test_rotating_an_empty_fleet_is_not_an_error():
    assert T.rotate([], None) == []


# ── turning "it reached N" back into a cursor ────────────────────────────────

def test_the_cursor_lands_on_the_last_company_actually_reached():
    rotated = ["a", "b", "c", "d"]
    assert T.cursor_after(rotated, 2) == T.digest_for("b")


def test_reaching_nothing_leaves_the_cursor_alone():
    """A cycle that asked nobody learned nothing. Advancing on it would step
    over companies that were never asked -- the exact failure being fixed.
    """
    assert T.cursor_after(["a", "b"], 0) is None
    assert T.cursor_after(["a", "b"], -5) is None


def test_a_completed_pass_lands_on_the_final_entry():
    rotated = ["a", "b", "c"]
    assert T.cursor_after(rotated, 3) == T.digest_for("c")
    assert T.cursor_after(rotated, 9_999) == T.digest_for("c")


def test_a_cursor_from_an_empty_rotation_is_none():
    assert T.cursor_after([], 5) is None


# ── the scrape loop actually using it ────────────────────────────────────────

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


@pytest.fixture(autouse=True)
def _clean_fanout(monkeypatch):
    """What the last cycle reached, per platform, without going through the
    scraper's module-level record of it.

    The rotation only ever asks `_ats_last_fanout`, so that is the seam to
    stand in for. Reading the scraper's own dict here would tie these tests to
    where those counts happen to be kept today.
    """
    counts: dict[str, dict[str, int]] = {}
    monkeypatch.setattr(
        WatcherManager,
        "_ats_last_fanout",
        lambda self, platform: dict(counts.get(platform) or {"submitted": 0, "completed": 0}),
    )
    return counts


@pytest.fixture(autouse=True)
def _uncapped(monkeypatch):
    """These tests are about where the walk resumes, not how far it steps. The cap that sizes
    each cycle's ask has its own file (test_ats_fanout_cap.py); here it is
    held open, so a narrow host does not truncate a 100-slug fleet and turn
    a rotation assertion into a sizing one.
    """
    monkeypatch.setattr(
        WatcherManager,
        "_fanout_cap",
        lambda self, platform, fleet_size, head_size: fleet_size,
    )


def _fleet(monkeypatch, platform, slugs):
    from services import ats_service
    monkeypatch.setattr(ats_service, "load_company_lists", lambda: {platform: list(slugs)})


def test_a_cut_off_cycle_moves_the_next_cycles_tail(monkeypatch, tmp_path, _clean_fanout):
    """The whole point, measured end to end: the companies cancelled by one
    cycle are asked by the next.
    """
    fleet = [f"c{i}" for i in range(100)]
    _fleet(monkeypatch, "icims", fleet)
    mgr = _manager(tmp_path)

    first = mgr._ordered_slugs("icims")
    # The cycle reached 43 of 100 -- the measured icims worst case.
    _clean_fanout["icims"] = {"submitted": 100, "completed": 43}

    second = mgr._ordered_slugs("icims")
    assert second[0] == first[43]
    # Everything the first cycle cancelled is now at the front, so the second
    # cycle reaches it before running out again.
    assert second[:57] == first[43:]
    assert sorted(second) == sorted(fleet)


def test_repeated_partial_cycles_eventually_cover_the_whole_fleet(monkeypatch, tmp_path, _clean_fanout):
    """In fleet order these 57 companies are never asked. Three passes at 43%
    is enough to offer every one of them a turn.
    """
    fleet = [f"c{i}" for i in range(100)]
    _fleet(monkeypatch, "icims", fleet)
    mgr = _manager(tmp_path)

    asked: set[str] = set()
    for _ in range(3):
        ordered = mgr._ordered_slugs("icims")
        asked.update(ordered[:43])
        _clean_fanout["icims"] = {"submitted": 100, "completed": 43}

    assert asked == set(fleet)


def test_the_cursor_survives_a_restart(monkeypatch, tmp_path, _clean_fanout):
    """A bot restart between cycles must not send the walk back to the top --
    that is how the far side of the fleet stays unasked across restarts.
    """
    fleet = [f"c{i}" for i in range(100)]
    _fleet(monkeypatch, "icims", fleet)

    first = _manager(tmp_path)._ordered_slugs("icims")
    _clean_fanout["icims"] = {"submitted": 100, "completed": 40}
    _manager(tmp_path)._ordered_slugs("icims")          # advances and persists

    _clean_fanout.clear()                               # a fresh process
    resumed = _manager(tmp_path)._ordered_slugs("icims")
    assert resumed[0] == first[40]


def test_a_platform_that_completes_its_fleet_is_unaffected(monkeypatch, tmp_path, _clean_fanout):
    """workday completed 14,989 of 14,990. Rotation must not disturb a platform
    that already reaches everyone.
    """
    fleet = [f"c{i}" for i in range(100)]
    _fleet(monkeypatch, "workday", fleet)
    mgr = _manager(tmp_path)

    first = mgr._ordered_slugs("workday")
    _clean_fanout["workday"] = {"submitted": 100, "completed": 100}
    assert sorted(mgr._ordered_slugs("workday")) == sorted(first)


def test_the_preferred_head_stays_at_the_front_every_cycle(monkeypatch, tmp_path, _clean_fanout):
    """The head is geo-preferred: it is there because it should be asked first
    on every cycle, so the rotation must not carry it away.
    """
    from services.jba import geo_priority

    fleet = [f"c{i}" for i in range(100)]
    _fleet(monkeypatch, "lever", fleet)
    monkeypatch.setattr(
        geo_priority, "slugs_for", lambda country, *a, **k: frozenset({"c5", "c9"})
    )
    mgr = _manager(tmp_path, {1: {"enabled": True, "location": "Canada"}})

    for completed in (30, 60, 90):
        ordered = mgr._ordered_slugs("lever")
        assert set(ordered[:2]) == {"c5", "c9"}
        assert sorted(ordered) == sorted(fleet)
        _clean_fanout["lever"] = {"submitted": 100, "completed": completed}


def test_the_head_is_not_counted_against_the_tail_cursor(monkeypatch, tmp_path, _clean_fanout):
    """`completed` counts the whole fan-out. Advancing the tail cursor by it
    would skip as many tail companies as there are preferred boards, every
    cycle -- a permanent blind spot proportional to the head.
    """
    from services.jba import geo_priority

    fleet = [f"c{i}" for i in range(100)]
    _fleet(monkeypatch, "lever", fleet)
    head = {f"c{i}" for i in range(10)}
    monkeypatch.setattr(geo_priority, "slugs_for", lambda country, *a, **k: frozenset(head))
    mgr = _manager(tmp_path, {1: {"enabled": True, "location": "Canada"}})

    first = mgr._ordered_slugs("lever")
    _clean_fanout["lever"] = {"submitted": 100, "completed": 30}  # 10 head + 20 tail

    second = mgr._ordered_slugs("lever")
    assert second[10] == first[30]      # 20 into the tail, not 30


def test_each_platform_keeps_its_own_cursor(monkeypatch, tmp_path, _clean_fanout):
    from services import ats_service

    fleet = [f"c{i}" for i in range(60)]
    monkeypatch.setattr(
        ats_service, "load_company_lists", lambda: {"lever": list(fleet), "ashby": list(fleet)}
    )
    mgr = _manager(tmp_path)

    lever_first = mgr._ordered_slugs("lever")
    ashby_first = mgr._ordered_slugs("ashby")
    _clean_fanout["lever"] = {"submitted": 60, "completed": 25}

    assert mgr._ordered_slugs("lever")[0] == lever_first[25]
    assert mgr._ordered_slugs("ashby")[0] == ashby_first[0]


def test_a_corrupt_state_file_costs_one_rotation_not_the_scrape(monkeypatch, tmp_path, _clean_fanout):
    fleet = [f"c{i}" for i in range(40)]
    _fleet(monkeypatch, "lever", fleet)
    (tmp_path / ".bot_state.ats_rotation.json").write_text("{ truncated", encoding="utf-8")

    got = _manager(tmp_path)._ordered_slugs("lever")
    assert sorted(got) == sorted(fleet)


def test_state_is_only_written_once_a_cycle_has_reported(monkeypatch, tmp_path, _clean_fanout):
    """A first run with no fan-out yet has nothing to record, and writing a
    cursor for it would claim progress that never happened.
    """
    _fleet(monkeypatch, "lever", ["a", "b", "c"])
    _manager(tmp_path)._ordered_slugs("lever")
    assert not (tmp_path / ".bot_state.ats_rotation.json").exists()


def test_progress_is_recorded_for_reporting(monkeypatch, tmp_path, _clean_fanout):
    fleet = [f"c{i}" for i in range(50)]
    _fleet(monkeypatch, "lever", fleet)
    mgr = _manager(tmp_path)

    mgr._ordered_slugs("lever")
    _clean_fanout["lever"] = {"submitted": 50, "completed": 50}
    mgr._ordered_slugs("lever")

    state = T.load_state(tmp_path / ".bot_state.ats_rotation.json")
    entry = state["platforms"]["lever"]
    assert entry["last_digest"]
    assert entry["cycle_wraps"] == 1     # it got all the way round
