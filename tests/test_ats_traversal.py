"""Walking the whole ATS fleet instead of its alphabetic head.

The bug this guards is not hypothetical. Because the company list is submitted
in file order and the tail is cancelled at the budget, the same head was asked
every cycle; once its jobs were archived, dedup dropped them and ATS output
went from 8,434 rows (2026-08-06) to 30 (2026-08-07) on every platform at once.

The properties that matter are coverage ones -- every company gets a turn, no
company is skipped, and the walk survives both a restart and a fleet that
changes size underneath it. They are tested by actually walking, not by
asserting on internals.
"""
from __future__ import annotations

import json

import pytest

from services import ats_traversal as T


def _fleet(n: int, prefix: str = "co") -> list[str]:
    return [f"{prefix}{i:05d}" for i in range(n)]


# ── ordering ─────────────────────────────────────────────────────────────────

def test_digest_is_stable_across_calls():
    assert T.digest_for("acme") == T.digest_for("acme")
    assert T.digest_for("acme") != T.digest_for("acmf")


def test_order_is_deterministic():
    fleet = _fleet(500)
    assert T.stable_order(fleet) == T.stable_order(list(reversed(fleet)))


def test_order_breaks_the_alphabetic_bias():
    """The whole point. In file order the first 100 of an A..Z fleet are all
    A's and B's; in digest order they must be spread across the alphabet.
    """
    fleet = sorted(f"{c}{i:03d}" for c in "abcdefghijklmnopqrstuvwxyz" for i in range(40))
    head_letters = {s[0] for s in fleet[:100]}
    walked = [s for _, s in T.stable_order(fleet)[:100]]
    assert len(head_letters) <= 3, "fixture is not alphabetically clustered"
    assert len({s[0] for s in walked}) > 15, "digest order is still clustered by letter"


def test_order_dedupes_and_drops_blanks():
    got = [s for _, s in T.stable_order(["a", "a", "  ", "b", "", " a "])]
    assert sorted(got) == ["a", "b"]


# ── the coverage property ────────────────────────────────────────────────────

def test_repeated_cycles_cover_every_company_exactly_once_per_pass():
    """The property the module exists for: walk the fleet in slices and every
    company is asked exactly once before any is asked twice.
    """
    fleet = _fleet(1000)
    order = T.stable_order(fleet)
    seen: list[str] = []
    last = None
    for _ in range(10):                      # 10 slices x 100 = one full pass
        picked, last, _wrapped = T.next_slice(order, last, 100)
        seen.extend(picked)

    assert len(seen) == 1000
    assert len(set(seen)) == 1000, "a company was asked twice before the pass finished"
    assert set(seen) == set(fleet), "a company was never asked"


def test_the_walk_wraps_and_reports_it():
    fleet = _fleet(50)
    order = T.stable_order(fleet)
    picked, last, wrapped = T.next_slice(order, None, 30)
    assert not wrapped and len(picked) == 30
    picked2, last2, wrapped2 = T.next_slice(order, last, 30)
    assert wrapped2, "running off the end was not reported as a wrap"
    assert len(picked2) == 30


def test_asking_for_more_than_the_fleet_returns_it_once_not_looped():
    fleet = _fleet(10)
    picked, _last, _w = T.next_slice(T.stable_order(fleet), None, 500)
    assert len(picked) == 10
    assert len(set(picked)) == 10


def test_an_empty_fleet_is_not_an_error():
    assert T.next_slice([], None, 10) == ([], None, False)


def test_a_zero_count_asks_for_nothing():
    picked, last, wrapped = T.next_slice(T.stable_order(_fleet(10)), None, 0)
    assert picked == [] and wrapped is False


# ── a fleet that changes under the cursor ────────────────────────────────────

def test_a_grown_fleet_resumes_without_skipping_or_repeating():
    """The harvest adds slugs weekly. An integer offset would drift; a digest
    must still resume in the right place.
    """
    fleet = _fleet(200)
    first, last, _ = T.next_slice(T.stable_order(fleet), None, 50)

    grown = fleet + _fleet(200, prefix="new")
    second, _last2, _ = T.next_slice(T.stable_order(grown), last, 50)

    assert not (set(first) & set(second)), "resuming re-asked companies already done"


def test_a_cursor_whose_company_is_gone_still_resumes():
    """Reconciliation removes slugs. A cursor pointing at a deleted company
    must resume at its neighbourhood, not raise and not restart at the head.
    """
    fleet = _fleet(200)
    order = T.stable_order(fleet)
    _picked, last, _ = T.next_slice(order, None, 50)

    survivors = [s for _d, s in order if _d != last]
    shrunk = T.stable_order(survivors)
    resumed, _l, _w = T.next_slice(shrunk, last, 50)

    assert len(resumed) == 50
    assert not (set(resumed) & set(_picked)), "restarted at the head after a removal"


def test_a_completely_replaced_fleet_does_not_raise():
    _picked, last, _ = T.next_slice(T.stable_order(_fleet(50)), None, 10)
    picked, _l, _w = T.next_slice(T.stable_order(_fleet(50, prefix="zz")), last, 10)
    assert len(picked) == 10


# ── state, restarts ──────────────────────────────────────────────────────────

def test_the_cursor_survives_a_restart(tmp_path):
    """A restart must not send the walk back to the head -- that is the failure
    the old design had permanently.
    """
    path = tmp_path / "traversal.json"
    fleet = _fleet(500)

    state = T.load_state(path)
    first = T.plan_cycle(state, "greenhouse", fleet, 100, now=1.0)
    T.save_state(path, state)

    reloaded = T.load_state(path)            # a fresh process
    second = T.plan_cycle(reloaded, "greenhouse", fleet, 100, now=2.0)

    assert len(first) == len(second) == 100
    assert not (set(first) & set(second)), "restart re-asked the same companies"


def test_platforms_advance_independently(tmp_path):
    state = {"platforms": {}}
    fleet = _fleet(300)
    a1 = T.plan_cycle(state, "greenhouse", fleet, 50, now=1.0)
    b1 = T.plan_cycle(state, "lever", fleet, 50, now=1.0)
    assert a1 == b1, "same fleet and no prior cursor should give the same slice"

    a2 = T.plan_cycle(state, "greenhouse", fleet, 50, now=2.0)
    assert not (set(a1) & set(a2))
    assert state["platforms"]["lever"]["slices"] == 1, "lever advanced when greenhouse did"


def test_a_missing_state_file_reads_as_empty(tmp_path):
    assert T.load_state(tmp_path / "nope.json") == {"platforms": {}}


def test_a_corrupt_state_file_reads_as_empty_rather_than_raising(tmp_path):
    p = tmp_path / "traversal.json"
    p.write_text("{ not json", encoding="utf-8")
    assert T.load_state(p) == {"platforms": {}}


def test_a_state_file_of_the_wrong_shape_reads_as_empty(tmp_path):
    p = tmp_path / "traversal.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert T.load_state(p) == {"platforms": {}}


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    p = tmp_path / "traversal.json"
    state = {"platforms": {}}
    T.record_progress(state, "lever", "abc", False, now=1.0)
    T.save_state(p, state)
    assert p.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(p.read_text(encoding="utf-8"))["platforms"]["lever"]["last_digest"] == "abc"


def test_a_crashed_cycle_re_asks_its_slice_rather_than_skipping_it(tmp_path):
    """plan_cycle mutates state but does not persist it. If the cycle dies
    before save_state, the next run must re-ask that slice -- re-asking is
    cheap, skipping means those companies wait a whole extra pass.
    """
    path = tmp_path / "traversal.json"
    fleet = _fleet(300)
    T.save_state(path, {"platforms": {}})

    state = T.load_state(path)
    doomed = T.plan_cycle(state, "greenhouse", fleet, 50, now=1.0)
    # crash here: no save_state

    after = T.load_state(path)
    retry = T.plan_cycle(after, "greenhouse", fleet, 50, now=2.0)
    assert retry == doomed


# ── wraps and reporting ──────────────────────────────────────────────────────

def test_a_full_pass_is_counted(tmp_path):
    state = {"platforms": {}}
    fleet = _fleet(100)
    for i in range(5):                        # 5 x 40 = 200 = two passes
        T.plan_cycle(state, "ashby", fleet, 40, now=float(i))
    assert state["platforms"]["ashby"]["cycle_wraps"] >= 1


def test_coverage_reports_how_long_a_pass_takes():
    state = {"platforms": {"paylocity": {"last_digest": "x", "cycle_wraps": 1, "slices": 9}}}
    cov = T.coverage(state, "paylocity", fleet_size=26160, count=1000)
    assert cov["cycles_per_pass"] == 27        # ceil(26160/1000)
    assert cov["full_passes"] == 1
    assert cov["slices_done"] == 9
    assert cov["started"] is True


def test_coverage_on_an_untouched_platform_says_so():
    cov = T.coverage({"platforms": {}}, "jobvite", fleet_size=1501, count=250)
    assert cov["started"] is False
    assert cov["full_passes"] == 0


@pytest.mark.parametrize("size,count", [(1, 1), (2, 1), (7, 3), (1000, 7), (26160, 1000)])
def test_a_full_pass_covers_the_fleet_for_any_slice_size(size, count):
    """Coverage must not depend on the fleet dividing evenly by the slice."""
    fleet = _fleet(size)
    order = T.stable_order(fleet)
    seen: set[str] = set()
    last = None
    cycles = -(-size // count)
    for _ in range(cycles):
        picked, last, _w = T.next_slice(order, last, count)
        seen.update(picked)
    assert seen == set(fleet), f"missed {len(set(fleet) - seen)} of {size}"


# ── dead boards must not eat slice positions ─────────────────────────────────

def test_skipped_boards_do_not_consume_slice_positions():
    """26% of the fleet carries a live dead mark and for lever it is 70%, so a
    slice of 250 raw slugs asks only 75 companies and the cycle ends early with
    its budget unspent. The slice must be 250 *askable* boards.
    """
    fleet = _fleet(2000)
    dead = {s for i, s in enumerate(sorted(fleet)) if i % 4}      # 75% dead
    picked, _last, _w = T.next_slice(
        T.stable_order(fleet), None, 100, is_skipped=lambda s: s in dead
    )
    assert len(picked) == 100
    assert not (set(picked) & dead), "a dead board was asked"


def test_the_cursor_advances_past_skipped_boards():
    """If the cursor only moved over boards that were asked, the walk would
    re-scan the same run of dead marks every cycle and never get past it.
    """
    fleet = _fleet(600)
    order = T.stable_order(fleet)
    dead = {s for _d, s in order[:200]}          # a solid run of dead at the front

    first, last, _ = T.next_slice(order, None, 50, is_skipped=lambda s: s in dead)
    second, _l2, _ = T.next_slice(order, last, 50, is_skipped=lambda s: s in dead)

    assert not (set(first) & set(second)), "the walk re-asked boards after skipping"


def test_a_slice_landing_entirely_in_dead_boards_still_moves_the_walk_on():
    """Nothing is asked, but the cursor must still advance -- otherwise this
    cycle is retried forever and the fleet past it is never reached.
    """
    fleet = _fleet(500)
    order = T.stable_order(fleet)
    all_dead = {s for _d, s in order}
    state = {"platforms": {}}

    picked = T.plan_cycle(state, "lever", fleet, 50, now=1.0,
                          is_skipped=lambda s: s in all_dead, max_scan=50)
    assert picked == []
    moved = state["platforms"]["lever"]["last_digest"]
    assert moved is not None, "the cursor did not move over an all-dead slice"

    picked2 = T.plan_cycle(state, "lever", fleet, 50, now=2.0,
                           is_skipped=lambda s: s in all_dead, max_scan=50)
    assert state["platforms"]["lever"]["last_digest"] != moved, "the walk stalled"
    assert picked2 == []


def test_max_scan_bounds_the_work_when_dead_boards_are_dense():
    """A pathological run of dead marks must cost a bounded scan, not a walk of
    the whole fleet. Falling short of the requested count is correct here.
    """
    fleet = _fleet(10_000)
    order = T.stable_order(fleet)
    seen: list[str] = []
    picked, _l, _w = T.next_slice(
        order, None, 100,
        is_skipped=lambda s: seen.append(s) or True,   # everything dead, count the probes
        max_scan=300,
    )
    assert picked == []
    assert len(seen) == 300, f"scanned {len(seen)}, expected the 300 cap"


def test_every_live_board_is_still_covered_across_a_full_walk():
    """Skipping must not cost coverage: the live half of the fleet must all be
    reached, not just the part near the front.
    """
    fleet = _fleet(1000)
    order = T.stable_order(fleet)
    dead = {s for i, s in enumerate(sorted(fleet)) if i % 2}
    live = set(fleet) - dead

    seen: set[str] = set()
    last = None
    for _ in range(20):
        picked, last, _w = T.next_slice(order, last, 50, is_skipped=lambda s: s in dead)
        seen.update(picked)

    assert seen == live, f"missed {len(live - seen)} live boards"


def test_no_predicate_behaves_exactly_as_before():
    """Backward compatibility: the wiring is not live yet, so the default path
    must be untouched.
    """
    fleet = _fleet(300)
    order = T.stable_order(fleet)
    assert T.next_slice(order, None, 50) == T.next_slice(order, None, 50, is_skipped=None)
