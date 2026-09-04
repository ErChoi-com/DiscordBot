"""The outage guard: the only thing standing between a network failure and
mass dead-marking.

A probe that fails returns "not live", so an endpoint being down is
indistinguishable from every company on it having closed. The validator writes
dead marks, the bot reads them, and a mark suppresses a real company for the
full TTL -- so a single validation run during an outage can silently remove a
whole platform from the bot's results.

The guard asks slugs *already confirmed live* whether the endpoint answers, and
discards the platform's results if none of them do. None of it had a test.

Everything here is hermetic: PROBES is stubbed, so no network.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import validate_ats_slugs as v  # noqa: E402


# ── the gate in front of the guard ──────────────────────────────────────────

@pytest.mark.parametrize("probed, live, expected", [
    (20, 0, True),      # total collapse, enough probes to mean it
    (20, 9, True),      # just under half
    (20, 10, False),    # exactly half is not a collapse
    (19, 0, False),     # too few probes for the rate to mean anything
    (100, 49, True),
    (100, 50, False),
])
def test_collapse_gate_boundaries(probed, live, expected):
    assert v.collapse_suspected(probed, live) is expected


def test_a_run_that_probed_nothing_is_not_a_collapse():
    """0/0 read as a 0% rate sends every skipped platform down the outage path.

    Most platforms probe nothing on most runs once the confirmed-live store is
    warm, so this is the common case, not an edge one.
    """
    assert v.collapse_suspected(0, 0) is False


# ── the guard itself ────────────────────────────────────────────────────────

def _stub_probe(monkeypatch, platform, fn):
    monkeypatch.setitem(v.PROBES, platform, fn)


def test_endpoint_is_down_when_every_known_live_slug_fails(monkeypatch):
    _stub_probe(monkeypatch, "greenhouse", lambda slug: False)
    assert v.endpoint_healthy("greenhouse", ["a", "b", "c"]) is False


def test_one_answering_canary_is_enough(monkeypatch):
    """A real cull leaves the endpoint answering; only a dead endpoint silences
    every control at once. Requiring all of them would discard genuine
    closures, which is the failure this guard must not cause."""
    _stub_probe(monkeypatch, "greenhouse", lambda slug: slug == "c")
    assert v.endpoint_healthy("greenhouse", ["a", "b", "c"]) is True


def test_a_raising_probe_counts_as_a_failure_not_a_pass(monkeypatch):
    """An exception is not an answer, and must not be read as health."""
    def boom(slug):
        raise RuntimeError("connection reset")

    _stub_probe(monkeypatch, "greenhouse", boom)
    assert v.endpoint_healthy("greenhouse", ["a", "b"]) is False


def test_a_raising_probe_does_not_hide_a_healthy_one(monkeypatch):
    def flaky(slug):
        if slug == "a":
            raise RuntimeError("connection reset")
        return True

    _stub_probe(monkeypatch, "greenhouse", flaky)
    assert v.endpoint_healthy("greenhouse", ["a", "b"]) is True


def test_no_canaries_assumes_healthy(monkeypatch):
    """Documented trade-off, pinned so a change to it is deliberate.

    With no confirmed-live slugs there is nothing to compare against. Returning
    False would stall a platform's first run forever, since the confirmed-live
    file only gets written by a run that is allowed to record results. The cost
    is that a first-ever run during an outage marks that platform dead -- in
    practice every platform now has a confirmed-live file, which is what keeps
    this safe.
    """
    def explode(slug):  # must not be consulted at all
        raise AssertionError("probed despite having no canaries")

    _stub_probe(monkeypatch, "greenhouse", explode)
    assert v.endpoint_healthy("greenhouse", []) is True


def test_the_guard_logs_when_it_trips(monkeypatch):
    """The discard has to be visible; a silent one looks like a quiet run."""
    _stub_probe(monkeypatch, "greenhouse", lambda slug: False)
    lines: list[str] = []
    v.endpoint_healthy("greenhouse", ["a"], log=lines.append)
    assert lines and "endpoint as down" in lines[-1]


# ── canary selection ────────────────────────────────────────────────────────

def test_canaries_are_the_most_recently_confirmed():
    checked = {"old": "2026-01-01", "newest": "2026-09-01", "mid": "2026-05-01"}
    assert v.pick_canaries(checked, {}, count=2) == ["newest", "mid"]


def test_canaries_exclude_slugs_already_marked_dead():
    """A slug that is already dead cannot report on the endpoint's health --
    it is expected to fail, so including it would fake an outage."""
    checked = {"gone": "2026-09-02", "here": "2026-09-01"}
    assert v.pick_canaries(checked, {"gone": "2026-09-03"}) == ["here"]


def test_canary_count_is_capped():
    checked = {f"s{i}": f"2026-09-{i:02d}" for i in range(1, 20)}
    assert len(v.pick_canaries(checked, {})) == v.ENDPOINT_CANARIES


def test_no_confirmed_live_slugs_yields_no_canaries():
    assert v.pick_canaries({}, {}) == []


# ── folding verdicts back in ────────────────────────────────────────────────

def test_live_slugs_are_revived_and_dead_slugs_dated():
    dead = {"closed": "2026-01-01", "reopened": "2026-01-01"}
    counts = v.apply_results(
        dead, {"_live": ["reopened"], "_dead": ["closed", "newly"]},
        dt.date(2026, 9, 4))
    assert "reopened" not in dead
    assert dead["newly"] == "2026-09-04"
    assert dead["closed"] == "2026-09-04"
    assert counts == {"added": 1, "revived": 1, "redated": 1}


def test_unknown_slugs_are_left_exactly_as_they_were():
    """A timeout is not evidence. Treating it as either verdict is how a slow
    network turns into a wave of dead marks."""
    dead = {"already": "2026-01-01"}
    counts = v.apply_results(
        dead,
        # A run that timed out on two slugs and resolved nothing else. Empty
        # lists alone would not prove anything: the assertion has to be that
        # slugs which *were* seen but unresolved are still left alone.
        {"_live": [], "_dead": [], "_unknown": ["timed-out", "already"]},
        dt.date(2026, 9, 4))
    assert dead == {"already": "2026-01-01"}, "an unresolved slug was re-dated"
    assert "timed-out" not in dead, "a timeout was recorded as dead"
    assert counts == {"added": 0, "revived": 0, "redated": 0}
