"""The half-hourly scrapability monitor.

Two properties carry the whole thing and both are tested here rather than
assumed: it must never write ATS state (a three-board sample must not be able
to condemn companies), and its silence streak must survive a restart (the
in-process tracker resets, which is exactly when the streak matters most).

Hermetic -- no network. The scraping itself is exercised live by running the
script; what is tested here is every decision it makes about the results.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import monitor_ats_scrapability as mon  # noqa: E402


# ── the read-only guarantee ──────────────────────────────────────────────────

def _fake_ats_module():
    """A stand-in carrying the three writers the real scrape path has."""
    mod = types.SimpleNamespace()
    mod.written = []
    mod._mark_dead = lambda platform, slug: mod.written.append(("dead", platform, slug))
    mod._mark_alive = lambda platform, slug: mod.written.append(("alive", platform, slug))
    mod.flush_dead_slugs = lambda: mod.written.append(("flush",))
    return mod


def test_every_dead_slug_writer_is_neutralised():
    mod = _fake_ats_module()
    patched = mon.neutralise_state_writes(mod)

    assert set(patched) == {"_mark_dead", "_mark_alive", "flush_dead_slugs"}

    # The replacements must actually be inert, not merely present.
    mod._mark_dead("greenhouse", "acme")
    mod._mark_alive("greenhouse", "acme")
    mod.flush_dead_slugs()
    assert mod.written == [], "a neutralised writer still wrote"


def test_the_replacements_keep_the_signatures_the_scraper_calls_them_with():
    """A no-op with the wrong arity would raise inside the scrape path and be
    swallowed as a scraper failure -- reporting the platform broken when it is
    the monitor that is.
    """
    mod = _fake_ats_module()
    mon.neutralise_state_writes(mod)
    mod._mark_dead("workable", "acme")      # two positional args
    mod._mark_alive("workable", "acme")
    mod.flush_dead_slugs()                  # none


def test_a_writer_that_cannot_be_found_is_reported_rather_than_skipped():
    """main refuses to run on an incomplete list. If a future rename means a
    writer is missed, this must surface as a missing name, not a quiet pass.
    """
    mod = _fake_ats_module()
    del mod._mark_alive
    patched = mon.neutralise_state_writes(mod)

    assert "_mark_alive" not in patched
    assert {"_mark_dead", "flush_dead_slugs"} == set(patched)


# ── folding one run's result into the state ──────────────────────────────────

def test_an_empty_run_counts_as_silence():
    state = {"platforms": {}}
    entry = mon.update_platform(state, "workable", 0, now=100.0)
    assert entry["consecutive_silent"] == 1
    assert entry["last_was_error"] is False


def test_silence_accumulates_across_runs():
    state = {"platforms": {}}
    for i in range(4):
        mon.update_platform(state, "workable", 0, now=100.0 + i)
    assert state["platforms"]["workable"]["consecutive_silent"] == 4
    assert state["platforms"]["workable"]["runs"] == 4


def test_a_run_that_returns_rows_clears_the_streak_and_stamps_the_time():
    state = {"platforms": {}}
    mon.update_platform(state, "lever", 0, now=100.0)
    mon.update_platform(state, "lever", 0, now=200.0)
    entry = mon.update_platform(state, "lever", 12, now=300.0)

    assert entry["consecutive_silent"] == 0
    assert entry["last_nonempty_at"] == 300.0


def test_an_error_is_not_folded_in_as_silence():
    """A scraper that raised has its own diagnosis. Counting it as silence too
    would let one failure mode mask the other.
    """
    state = {"platforms": {}}
    entry = mon.update_platform(state, "icims", 0, now=100.0, error="Timeout")

    assert entry["consecutive_silent"] == 0
    assert entry["last_was_error"] is True
    assert entry["last_error"] == "Timeout"


def test_an_error_does_not_reset_a_streak_that_was_already_running():
    """The platform has not proven it can answer, so the streak must survive."""
    state = {"platforms": {}}
    for _ in range(3):
        mon.update_platform(state, "workable", 0, now=100.0)
    mon.update_platform(state, "workable", 0, now=200.0, error="Timeout")

    assert state["platforms"]["workable"]["consecutive_silent"] == 3


# ── the alarm ────────────────────────────────────────────────────────────────

def test_a_platform_alarms_only_once_it_reaches_the_threshold():
    state = {"platforms": {}}
    mon.update_platform(state, "workable", 0, now=1.0)
    mon.update_platform(state, "workable", 0, now=2.0)
    assert mon.silent_platforms(state, threshold=3) == []

    mon.update_platform(state, "workable", 0, now=3.0)
    assert mon.silent_platforms(state, threshold=3) == [("workable", 3)]


def test_the_longest_silence_is_reported_first():
    state = {"platforms": {}}
    for _ in range(3):
        mon.update_platform(state, "breezy", 0, now=1.0)
    for _ in range(9):
        mon.update_platform(state, "workable", 0, now=1.0)

    assert mon.silent_platforms(state, threshold=3) == [("workable", 9), ("breezy", 3)]


def test_an_intermittent_platform_never_alarms():
    """One empty run is ordinary -- a board can simply have nothing open."""
    state = {"platforms": {}}
    for i in range(10):
        mon.update_platform(state, "greenhouse", 0, now=float(i))
        mon.update_platform(state, "greenhouse", 5, now=float(i) + 0.5)

    assert mon.silent_platforms(state, threshold=3) == []


# ── persistence: the point of the whole file ─────────────────────────────────

def test_the_streak_survives_a_restart(tmp_path):
    """The in-process tracker resets on restart, which is precisely when a
    multi-run streak matters. State on disk is what makes the alarm reachable.
    """
    path = tmp_path / "monitor.json"

    state = mon.load_state(path)
    for _ in range(2):
        mon.update_platform(state, "workable", 0, now=1.0)
    mon.save_state(path, state)

    reloaded = mon.load_state(path)           # a fresh process
    assert reloaded["platforms"]["workable"]["consecutive_silent"] == 2
    assert mon.silent_platforms(reloaded, threshold=3) == []

    mon.update_platform(reloaded, "workable", 0, now=2.0)
    assert mon.silent_platforms(reloaded, threshold=3) == [("workable", 3)]


def test_a_missing_state_file_reads_as_empty(tmp_path):
    assert mon.load_state(tmp_path / "nope.json") == {"platforms": {}}


def test_a_corrupt_state_file_reads_as_empty_rather_than_raising(tmp_path):
    """Losing the streak makes the next alarm late. Refusing to start makes it
    never come at all, so the corrupt case must degrade, not raise.
    """
    path = tmp_path / "monitor.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert mon.load_state(path) == {"platforms": {}}


def test_a_state_file_of_the_wrong_shape_reads_as_empty(tmp_path):
    path = tmp_path / "monitor.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert mon.load_state(path) == {"platforms": {}}


def test_save_state_replaces_atomically_and_leaves_no_temp_file(tmp_path):
    """A run interrupted mid-write must not leave a half-file that the next
    run reads as corrupt and discards the streak from.
    """
    path = tmp_path / "monitor.json"
    state = {"platforms": {}}
    mon.update_platform(state, "lever", 3, now=1.0)
    mon.save_state(path, state)

    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(path.read_text(encoding="utf-8"))["platforms"]["lever"]["runs"] == 1


# ── sampling ─────────────────────────────────────────────────────────────────

def test_sample_draws_the_requested_number_of_confirmed_live_slugs(tmp_path):
    (tmp_path / "lever.json").write_text(
        json.dumps({f"c{i}": "2026-09-01" for i in range(50)}), encoding="utf-8")
    got = mon.sample_slugs("lever", 3, tmp_path, seed="fixed", dead_dir=tmp_path / "no_dead")
    assert len(got) == 3
    assert len(set(got)) == 3


def test_sample_is_capped_by_what_the_platform_actually_has(tmp_path):
    (tmp_path / "tiny.json").write_text(json.dumps({"only": "2026-09-01"}), encoding="utf-8")
    assert mon.sample_slugs("tiny", 5, tmp_path, seed="fixed", dead_dir=tmp_path / "no_dead") == ["only"]


def test_a_platform_with_no_confirmed_live_file_samples_nothing(tmp_path):
    assert mon.sample_slugs("ghost", 3, tmp_path, seed="fixed", dead_dir=tmp_path / "no_dead") == []


def test_an_empty_confirmed_live_file_samples_nothing(tmp_path):
    (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
    assert mon.sample_slugs("empty", 3, tmp_path, seed="fixed", dead_dir=tmp_path / "no_dead") == []


def test_the_sample_is_redrawn_between_runs_rather_than_pinned():
    """A fixed sample would report a platform healthy on three boards that
    happen to still work, and never notice the rest of the fleet rotting.
    """
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as d:
        checked = _P(d)
        (checked / "big.json").write_text(
            json.dumps({f"c{i}": "2026-09-01" for i in range(200)}), encoding="utf-8")
        draws = {tuple(mon.sample_slugs("big", 3, checked, seed=f"run{i}", dead_dir=checked / "no_dead")) for i in range(8)}

    assert len(draws) > 1, "every run drew the same boards"


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_sampling_never_returns_duplicates(tmp_path, count):
    """A duplicate would be scraped twice and double-counted toward the row
    total, making a platform look healthier than its sample showed.
    """
    (tmp_path / "p.json").write_text(
        json.dumps({f"c{i}": "2026-09-01" for i in range(20)}), encoding="utf-8")
    got = mon.sample_slugs("p", count, tmp_path, seed="fixed", dead_dir=tmp_path / "no_dead")
    assert len(got) == len(set(got)) == count
