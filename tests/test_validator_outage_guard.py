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
import json
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


# ── --dry-run must report what a real run would do, and write nothing ───────
#
# Two separate promises, and the second is the one that rots quietly: a dry run
# that writes nothing but reports different numbers than the real run is worse
# than no dry run, because it is trusted. dead_total was read off the map from
# *before* the run's own verdicts (0 where the real run wrote 1), and
# checked_total was missing from dry runs altogether -- so the JSON summary,
# which is consumed by tooling, had a different shape depending on the flag.

@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Redirect every data directory the validator touches into tmp_path."""
    for name in ("harvest", "dead", "checked", "company"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(v, "HARVEST_DIR", tmp_path / "harvest")
    monkeypatch.setattr(v, "DEAD_DIR", tmp_path / "dead")
    monkeypatch.setattr(v, "CHECKED_DIR", tmp_path / "checked")
    monkeypatch.setattr(v, "COMPANY_DIR", tmp_path / "company")
    (tmp_path / "harvest" / "lever.json").write_text(
        json.dumps(["alive", "alive2", "gone"]), encoding="utf-8")
    monkeypatch.setitem(v.PROBES, "lever", lambda slug: slug.startswith("alive"))
    return tmp_path


def _summary(capsys, dry):
    args = ["--platform", "lever", "--json", "--budget-seconds", "0"]
    if dry:
        args.append("--dry-run")
    assert v.main(args) == 0
    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):])["platforms"]["lever"]


def test_dry_run_writes_nothing(sandbox, capsys):
    _summary(capsys, dry=True)
    assert list((sandbox / "dead").iterdir()) == []
    assert list((sandbox / "checked").iterdir()) == []


def test_a_real_run_does_write(sandbox, capsys):
    """The guard must not disable validation outright."""
    _summary(capsys, dry=False)
    assert (sandbox / "dead" / "lever.json").exists()
    assert (sandbox / "checked" / "lever.json").exists()


def test_dry_run_summary_matches_the_real_one(sandbox, capsys):
    """Same keys and same values -- this is what makes a preview worth reading."""
    dry = _summary(capsys, dry=True)
    real = _summary(capsys, dry=False)
    assert dry == real


def test_dry_run_reports_the_dead_total_it_would_leave_behind(sandbox, capsys):
    """Not the total from before its own verdicts.

    "How many will be dead after this" is the question a dry run is run to
    answer, and it was answered with the count from before the run.
    """
    dry = _summary(capsys, dry=True)
    assert dry["dead_total"] == 1
    assert dry["added"] == 1


def test_dry_run_reports_the_confirmed_live_total(sandbox, capsys):
    """checked_total was absent on dry runs, so the shape depended on the flag."""
    dry = _summary(capsys, dry=True)
    assert dry["checked_total"] == 2


# ── the shared confirmed-live update ────────────────────────────────────────

def test_apply_checked_records_live_and_drops_dead():
    checked = {"stale-dead": "2026-09-01"}
    out = v.apply_checked(
        checked, {"_live": ["a", "b"], "_dead": ["stale-dead"]},
        dt.date(2026, 9, 4))
    assert out["a"] == "2026-09-04" and out["b"] == "2026-09-04"
    assert "stale-dead" not in out, "a slug found dead is still confirmed-live"


def test_apply_checked_expires_entries_past_twice_the_recheck_window():
    today = dt.date(2026, 9, 4)
    old = (today - dt.timedelta(days=v.LIVE_RECHECK_DAYS * 2 + 1)).isoformat()
    recent = (today - dt.timedelta(days=1)).isoformat()
    out = v.apply_checked({"old": old, "recent": recent},
                          {"_live": [], "_dead": []}, today)
    assert "old" not in out and "recent" in out


# ── --sample records; --audit-sample measures ───────────────────────────────
#
# Two sampling flags with overlapping names and opposite purposes. --sample
# narrows this run's targets and its verdicts are written; --audit-sample draws
# uniformly from the whole population and writes nothing. The help text for
# --sample used to read "for measuring live rates", which pointed anyone
# wanting a measurement at the one that both mutates and measures the wrong
# population -- targets exclude slugs already marked dead, so its rate is of
# survivors. Measured on real data that difference was 97.5% against 38.8%.

def test_sample_cannot_reach_slugs_already_marked_dead():
    """The bias, stated as a test: 80% of this population is dead and the
    sample draws none of it."""
    today = dt.date(2026, 9, 4)
    candidates = [f"s{i}" for i in range(100)]
    dead = {f"s{i}": today.isoformat() for i in range(80)}

    with_dead = v.select_targets("lever", dead, recheck_dead=False, limit=None,
                                 sample=20, today=today)
    assert len(with_dead) == 20
    assert not [s for s in with_dead if s in dead], (
        "a survivor-only sample was treated as a population sample")


def test_audit_sample_draws_slugs_that_are_already_marked_dead(sandbox):
    """Including known-dead slugs is the entire point of the audit.

    Excluding them is exactly what makes the working pass's rate an
    overestimate, so an audit that inherited the same exclusion would measure
    the same wrong thing while looking authoritative.

    The dead map has to be real here rather than stubbed: an earlier version of
    this test replaced load_candidates, which left the dead map empty, so a
    mutant that filtered known-dead slugs out of the audit changed nothing and
    survived.
    """
    population = [f"s{i}" for i in range(60)]
    (sandbox / "harvest" / "lever.json").write_text(
        json.dumps(population), encoding="utf-8")
    stamp = dt.date.today().isoformat()
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({f"s{i}": stamp for i in range(50)}), encoding="utf-8")

    seen: list[str] = []

    def probe(slug):
        seen.append(slug)
        return False

    v.audit_live_rate("lever", 40, probe=probe, workers=1, log=lambda m: None)
    assert len(seen) == 40
    dead_sampled = [s for s in seen if int(s[1:]) < 50]
    assert dead_sampled, "the audit skipped every known-dead slug"


def test_audit_sample_is_a_random_draw_not_a_prefix(monkeypatch):
    """A prefix would sample whatever order the files happen to be in.

    "not confined to a prefix" has to be asserted as inequality against the
    actual prefix -- checking that the highest index is large passes for a
    prefix too, which is how that mutant survived once.
    """
    population = [f"s{i}" for i in range(50)]
    monkeypatch.setattr(v, "load_candidates", lambda p: list(population))
    seen: list[str] = []
    v.audit_live_rate("lever", 30, probe=lambda s: seen.append(s) or False,
                      workers=1, log=lambda m: None)
    assert sorted(seen) != sorted(population[:30]), "the audit took a prefix"


def test_audit_writes_nothing(sandbox, capsys):
    """"Deliberately read-only": an audit that recorded verdicts would reshape
    the population it is measuring."""
    v.audit_live_rate("lever", 3, probe=lambda s: False, workers=1,
                      log=lambda m: None)
    assert list((sandbox / "dead").iterdir()) == []
    assert list((sandbox / "checked").iterdir()) == []


def test_sample_verdicts_are_recorded(sandbox, capsys):
    """The counterpart: --sample is working, not measuring, so it must write."""
    assert v.main(["--platform", "lever", "--sample", "3", "--json",
                   "--budget-seconds", "0"]) == 0
    assert (sandbox / "dead" / "lever.json").exists(), (
        "--sample stopped recording; it is a working pass, not an audit")


# ── reconciling the stores against the company lists ────────────────────────
#
# prune_existing makes filter fixes retroactive by removing junk from the
# harvest. Nothing removed the marks that junk had already collected, so they
# sat in the files the bot loads on every scrape until the 90-day TTL happened
# to reach them: 539 orphaned dead marks and 19 orphaned confirmed-live ones,
# and the sample is exactly the junk list -- '100', 'ads.txt', '2fwww',
# 'en-ca', 'llms.txt'.

def test_marks_for_slugs_no_longer_harvested_are_dropped():
    dead = {"real": "2026-09-01", "ads.txt": "2026-09-01", "100": "2026-09-01"}
    checked = {"real2": "2026-09-01", "favicon.png": "2026-09-01"}
    stats = reconcile = v.reconcile_stores(
        dead, checked, ["real", "real2"] + [f"pad{i}" for i in range(100)])
    assert set(dead) == {"real"}
    assert set(checked) == {"real2"}
    assert stats["dead_orphans"] == 2 and stats["checked_orphans"] == 1
    assert reconcile["refused"] == 0


def test_an_empty_candidate_list_refuses_instead_of_deleting_everything():
    """The catastrophic case, and the reason this function has a guard.

    A missing or unreadable harvest file makes load_candidates return nothing,
    which makes *every* mark look orphaned. Deleting on that input would wipe
    the platform's entire memory of what is dead, silently, and the next scrape
    would re-probe tens of thousands of known-dead boards.
    """
    dead = {"a": "2026-09-01", "b": "2026-09-01"}
    checked = {"c": "2026-09-01"}
    stats = v.reconcile_stores(dead, checked, [])
    assert stats["refused"] == 1
    assert set(dead) == {"a", "b"} and set(checked) == {"c"}


def test_an_implausible_drop_share_refuses():
    """A truncated candidate list is not empty, so the empty check alone is not
    enough -- the share of the store being dropped is the real signal."""
    dead = {f"s{i}": "2026-09-01" for i in range(100)}
    checked = {}
    stats = v.reconcile_stores(dead, checked, ["s0", "s1", "s2"])
    assert stats["refused"] == 1
    assert len(dead) == 100, "a truncated candidate list deleted 97 marks"


def test_a_handful_of_orphans_is_allowed_on_a_small_store():
    """A rate-only guard leaves small stores permanently un-reconcilable.

    rippling holds 35 dead marks, so four orphans is 11% and would be refused
    forever. Dropping at most RECONCILE_MIN_DROP cannot do real damage even if
    the candidate list is wrong, and the empty-list check still covers the
    catastrophic case.
    """
    dead = {f"s{i}": "2026-09-01" for i in range(8)}
    stats = v.reconcile_stores(dead, {}, ["s0", "s1", "s2", "s3", "s4"])
    assert stats["refused"] == 0 and stats["dead_orphans"] == 3
    assert len(dead) == 5


def test_a_drop_share_within_the_cap_proceeds():
    """The real numbers are ~1%, so the cap must not block ordinary use."""
    dead = {f"s{i}": "2026-09-01" for i in range(100)}
    stats = v.reconcile_stores(dead, {}, [f"s{i}" for i in range(95)])
    assert stats["refused"] == 0 and stats["dead_orphans"] == 5
    assert len(dead) == 95


def test_reconciliation_refusal_is_logged():
    """A silent refusal looks identical to nothing needing reconciliation."""
    lines: list[str] = []
    v.reconcile_stores({"a": "2026-09-01"}, {}, [], log=lines.append,
                       platform="lever")
    assert lines and "refusing" in lines[-1]


def test_reconciliation_is_reported_in_both_run_modes(sandbox, capsys):
    """Same dry/real parity rule as the rest of the summary."""
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({"alive": "2026-09-01", "ads.txt": "2026-09-01"}),
        encoding="utf-8")
    dry = _summary(capsys, dry=True)
    assert dry["dead_orphans"] == 1

    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({"alive": "2026-09-01", "ads.txt": "2026-09-01"}),
        encoding="utf-8")
    real = _summary(capsys, dry=False)
    assert real["dead_orphans"] == 1


def test_a_dry_run_does_not_write_the_reconciled_stores(sandbox, capsys):
    original = json.dumps({"alive": "2026-09-01", "ads.txt": "2026-09-01"})
    (sandbox / "dead" / "lever.json").write_text(original, encoding="utf-8")
    _summary(capsys, dry=True)
    assert (sandbox / "dead" / "lever.json").read_text(encoding="utf-8") == original


def test_a_real_run_persists_the_reconciled_stores(sandbox, capsys):
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({"alive": "2026-09-01", "ads.txt": "2026-09-01"}),
        encoding="utf-8")
    _summary(capsys, dry=False)
    on_disk = json.loads((sandbox / "dead" / "lever.json").read_text(encoding="utf-8"))
    assert "ads.txt" not in on_disk, "the orphan survived a real run"
