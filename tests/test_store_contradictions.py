"""A slug cannot be both confirmed-live and dead, and 137 of them were.

Two processes write these stores. validate_ats_slugs probes a board and records
the verdict in both -- apply_checked drops a slug out of the confirmed-live map
the moment it is found dead, so the validator's own answers never contradict
each other. ats_service is the other writer: the running bot marks a slug dead
when a real scrape cycle finds it gone, and it has no idea the confirmed-live
store exists.

So a slug confirmed live in one validation run and found dead by the bot a week
later stays in both files, and nothing ever removed it -- reconcile_stores only
looked at slugs missing from the company lists. Measured across the fleet
before this: 137 slugs in both stores on eight platforms, 101 of them recruitee.

It is not cosmetic. monitor_ats_scrapability samples the confirmed-live store
to ask whether a platform still answers, and ats_service returns nothing for a
dead-marked slug without making a request. Sampling a contradiction records the
platform as silent -- a false alarm manufactured by the one tool whose entire
job is telling real silence from noise, and silence is how this repo finds
broken scrapers.

Fixed on both sides: the validator repairs the stores every daily run, and the
monitor filters dead marks itself so a mark the bot wrote an hour ago is
respected before any repair has run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import monitor_ats_scrapability as mon  # noqa: E402
import validate_ats_slugs as v  # noqa: E402


def _write(path: Path, slugs, stamp="2026-09-01"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({s: stamp for s in slugs}), encoding="utf-8")


# ── the validator repairs the stores ────────────────────────────────────────

def test_a_slug_in_both_stores_loses_its_confirmed_live_entry():
    dead = {"gone": "2026-09-01"}
    checked = {"gone": "2026-08-01", "here": "2026-09-01"}
    stats = v.reconcile_stores(dead, checked, ["gone", "here"])

    assert "gone" not in checked
    assert checked == {"here": "2026-09-01"}
    assert stats["contradictions"] == 1


def test_the_dead_mark_survives_the_repair():
    """The dead mark is the claim backed by a board that actually refused; a
    confirmed-live entry only records that it answered once, earlier."""
    dead = {"gone": "2026-09-01"}
    v.reconcile_stores(dead, {"gone": "2026-08-01"}, ["gone"])
    assert dead == {"gone": "2026-09-01"}


def test_a_newer_confirmed_live_entry_still_loses():
    """Deliberate. A confirmed-live entry is refreshed by the validator's own
    TTL sweep, so a later date does not mean a later probe -- while a dead mark
    is only ever written by something that just asked and was refused.
    """
    dead = {"gone": "2026-01-01"}
    checked = {"gone": "2026-09-04"}
    v.reconcile_stores(dead, checked, ["gone"])
    assert checked == {}


def test_stores_that_do_not_overlap_are_left_alone():
    dead = {"a": "2026-09-01"}
    checked = {"b": "2026-09-01"}
    stats = v.reconcile_stores(dead, checked, ["a", "b"])
    assert dead == {"a": "2026-09-01"} and checked == {"b": "2026-09-01"}
    assert stats["contradictions"] == 0


def test_the_repair_is_reported_not_silent(capsys):
    lines = []
    v.reconcile_stores({"gone": "x"}, {"gone": "y"}, ["gone"], log=lines.append,
                       platform="recruitee")
    assert any("both" in l and "recruitee" in l for l in lines)


def test_a_refused_reconcile_repairs_nothing():
    """An empty candidate list makes every mark look orphaned, and refusing is
    how that is survived. The refusal has to cover this repair too, or a
    truncated harvest silently empties the confirmed-live store instead.
    """
    dead = {"gone": "2026-09-01"}
    checked = {"gone": "2026-08-01"}
    stats = v.reconcile_stores(dead, checked, [])
    assert stats["refused"] == 1
    assert checked == {"gone": "2026-08-01"}


def test_the_caller_saves_a_repair_that_dropped_no_orphans():
    """The save was guarded on orphan counts alone, so a run whose only change
    was this repair computed it and threw it away."""
    import inspect

    src = " ".join(inspect.getsource(v._run).split())
    assert 'recon["contradictions"]' in src
    assert src.index('recon["contradictions"]') < src.index("save_checked(platform, checked_map)")


# ── the monitor does not sample one ─────────────────────────────────────────

def test_a_dead_marked_slug_is_never_sampled(tmp_path):
    checked, dead = tmp_path / "checked", tmp_path / "dead"
    _write(checked / "lever.json", [f"c{i}" for i in range(10)])
    _write(dead / "lever.json", [f"c{i}" for i in range(9)])

    for seed in range(20):
        assert mon.sample_slugs("lever", 3, checked, seed=seed, dead_dir=dead) == ["c9"]


def test_a_platform_whose_every_live_slug_is_dead_marked_samples_nothing(tmp_path):
    """Better than sampling one anyway: "no boards to ask" is a different
    report from "asked three and got nothing", and only the second is silence.
    """
    checked, dead = tmp_path / "checked", tmp_path / "dead"
    _write(checked / "lever.json", ["a", "b"])
    _write(dead / "lever.json", ["a", "b"])
    assert mon.sample_slugs("lever", 3, checked, seed=1, dead_dir=dead) == []


def test_a_missing_dead_file_means_nothing_is_filtered(tmp_path):
    checked = tmp_path / "checked"
    _write(checked / "lever.json", ["a", "b", "c"])
    assert len(mon.sample_slugs("lever", 3, checked, seed=1, dead_dir=tmp_path / "none")) == 3


def test_a_corrupt_dead_file_does_not_take_the_sample_with_it(tmp_path):
    """Losing the filter costs a possible false alarm; treating the platform as
    having no boards costs the check itself."""
    checked, dead = tmp_path / "checked", tmp_path / "dead"
    _write(checked / "lever.json", ["a", "b", "c"])
    dead.mkdir()
    (dead / "lever.json").write_text("{not json", encoding="utf-8")
    assert len(mon.sample_slugs("lever", 3, checked, seed=1, dead_dir=dead)) == 3


def test_the_filter_is_on_by_default(tmp_path):
    """A caller that forgets the argument must not silently lose the
    protection -- that is how the confirmed-live store came to disagree with
    the dead store in the first place.
    """
    import inspect

    sig = inspect.signature(mon.sample_slugs)
    assert sig.parameters["dead_dir"].default is None
    src = " ".join(inspect.getsource(mon.sample_slugs).split())
    assert "dead_dir = DEAD_DIR if dead_dir is None else dead_dir" in src


def test_the_monitor_reads_the_same_store_ats_service_writes():
    """The filter is only worth anything if it reads the file the bot marks."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from services import ats_service

    assert mon.DEAD_DIR.name == "dead_slugs"
    assert mon.DEAD_DIR.resolve() == Path(ats_service._DEAD_SLUG_DIR).resolve()
    # And the same one the validator repairs, or the two halves of the fix
    # would be maintaining different files.
    assert mon.DEAD_DIR.resolve() == Path(v.DEAD_DIR).resolve()
