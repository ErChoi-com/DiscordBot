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


# ── live_icims must not mistake iCIMS's own site for a tenant ───────────────
#
# An iCIMS infrastructure host answers 200 and redirects to the vendor's site:
# www4.icims.com/sitemap.xml ends at www.icims.com. live_icims trusted the
# status alone, so it read that as a live board -- the same failure live_bamboohr
# already guards against by checking the final URL. Invented slugs 404 properly,
# which is why the probe-discrimination check never caught this: the blind spot
# is real hostnames that are not tenants.

def _icims_request(monkeypatch, table):
    """Stub _request as {host: (status, final_url)}."""
    def fake(url, **kwargs):
        host = url.split("//", 1)[1].split(".icims.com", 1)[0]
        status, final = table.get(host, (404, url))
        return status, b"", final
    monkeypatch.setattr(v, "_request", fake)


def test_a_200_that_redirects_off_the_tenant_host_is_not_a_board(monkeypatch):
    _icims_request(monkeypatch, {
        "www4": (200, "https://www.icims.com/"),
        "careers-www4": (404, "https://careers-www4.icims.com/sitemap.xml"),
    })
    assert v.live_icims("www4") is False


def test_a_real_tenant_keeps_its_hostname_and_is_live(monkeypatch):
    _icims_request(monkeypatch, {
        "careers-kearneyco": (200, "https://careers-kearneyco.icims.com/sitemap.xml"),
    })
    assert v.live_icims("kearneyco") is True


def test_the_second_host_form_is_still_tried_after_a_redirect(monkeypatch):
    """The redirect rules out that form, not the company.

    Returning False on the first form would lose every board whose other form
    is the live one -- and the two forms are mutually exclusive, so that is
    roughly half of them.
    """
    _icims_request(monkeypatch, {
        "careers-acme": (200, "https://www.icims.com/"),
        "acme": (200, "https://acme.icims.com/sitemap.xml"),
    })
    assert v.live_icims("acme") is True


def test_an_unreachable_form_still_raises_rather_than_reporting_dead(monkeypatch):
    """Unchanged behaviour, pinned: a refusal is not evidence of absence."""
    _icims_request(monkeypatch, {
        "careers-acme": (None, None),
        "acme": (404, "https://acme.icims.com/sitemap.xml"),
    })
    with pytest.raises(v.Unreachable):
        v.live_icims("acme")


# ── partition_unscrapeable must not condemn a guess ─────────────────────────

@pytest.mark.parametrize("platform, slug", [("bamboohr", "mx51"), ("icims", "www4")])
def test_a_numbered_infra_label_is_probed_not_retired(platform, slug):
    """mx51.bamboohr.com serves tenant JSON from its own host and had been
    dead-marked since 2026-09-01 without ever being asked.

    The harvest-time filter that drops these is sound -- nothing is lost by not
    collecting www4. Retiring an *existing* candidate on the same guess is not:
    the contract for that is "cannot correspond to a board whatever the network
    says", and a numbered label can. Now it costs one probe and the network
    answers.
    """
    retired, _ = v.partition_unscrapeable(platform, [slug])
    assert retired == []


@pytest.mark.parametrize("slug", ["www", "api", "cdn", "embed"])
def test_a_bare_infra_label_is_still_retired_without_a_probe(slug):
    """Bare labels are host roles, not tenants -- the guess is safe there."""
    retired, _ = v.partition_unscrapeable("bamboohr", [slug])
    assert retired == [slug]


def test_the_workday_impossibility_is_still_retired():
    """The case this function exists for: wd1|wd1|careers would have to resolve
    wd1.wd1.myworkdayjobs.com, which cannot exist. Not a guess."""
    retired, _ = v.partition_unscrapeable("workday", ["wd1|wd1|careers"])
    assert retired == ["wd1|wd1|careers"]


# ── every host-based probe must reject an off-host redirect ─────────────────
#
# Five of the seven subdomain platforms needed this rule, each found separately:
# bamboohr and jazzhr had it, live_icims was fixed last, and breezy and
# recruitee were found by auditing the rest. The measured false positives:
# api.breezy.hr answers 200 and lands on developer.breezy.hr (Breezy's API
# docs), blog.recruitee.com lands on recruitee.com/blog, and
# login.recruitee.com lands on loginsoftware.recruitee.com -- a *different
# company's* board, which would credit that tenant's jobs to the slug "login".
#
# Invented slugs 404 properly on all of them, so check_probe_discrimination.py
# passes either way. The blind spot is real hostnames that are not tenants.

_HOST_PROBES = [
    ("bamboohr", "live_bamboohr", "{slug}.bamboohr.com", "https://www.bamboohr.com/"),
    ("jazzhr", "live_jazzhr", "{slug}.applytojob.com", "https://www.jazzhr.com/job-seekers"),
    ("breezy", "live_breezy", "{slug}.breezy.hr", "https://developer.breezy.hr/reference/overview"),
    ("recruitee", "live_recruitee", "{slug}.recruitee.com", "https://recruitee.com/blog"),
    ("teamtailor", "live_teamtailor", "{slug}.teamtailor.com", "https://www.teamtailor.com/"),
]


@pytest.mark.parametrize("platform, fn_name, host_tpl, elsewhere", _HOST_PROBES)
def test_a_200_that_lands_off_the_tenant_host_is_not_live(
        monkeypatch, platform, fn_name, host_tpl, elsewhere):
    monkeypatch.setattr(v, "_request", lambda url, **kw: (200, b'{"x":1}', elsewhere))
    assert getattr(v, fn_name)("acme") is False


@pytest.mark.parametrize("platform, fn_name, host_tpl, elsewhere", _HOST_PROBES)
def test_a_200_that_stays_on_the_tenant_host_is_live(
        monkeypatch, platform, fn_name, host_tpl, elsewhere):
    """The other half: the guard must not condemn real tenants."""
    host = host_tpl.format(slug="acme")
    body = b'{"meta":{"totalCount":1},"result":[{"id":1}]}'
    monkeypatch.setattr(v, "_request",
                        lambda url, **kw: (200, body, f"https://{host}/careers/list"))
    assert getattr(v, fn_name)("acme") is True


def test_recruitee_does_not_credit_another_tenants_board_to_this_slug(monkeypatch):
    """login.recruitee.com really does land on loginsoftware.recruitee.com.

    Substring matching the vendor domain rather than the full host would accept
    it, since both end in .recruitee.com -- and the jobs of a real company would
    be filed under the slug "login".
    """
    monkeypatch.setattr(v, "_request", lambda url, **kw: (
        200, b"[]", "https://loginsoftware.recruitee.com/api/offers/"))
    assert v.live_recruitee("login") is False


@pytest.mark.parametrize("platform, fn_name, host_tpl, elsewhere", _HOST_PROBES)
def test_a_non_200_is_still_decided_normally(monkeypatch, platform, fn_name,
                                             host_tpl, elsewhere):
    """The host check only applies to a 200. A 404 is still dead and a refusal
    is still Unreachable -- reading either as "off host" would turn an outage
    into a wave of dead marks."""
    monkeypatch.setattr(v, "_request", lambda url, **kw: (404, b"", elsewhere))
    assert getattr(v, fn_name)("acme") is False

    monkeypatch.setattr(v, "_request", lambda url, **kw: (None, b"", None))
    with pytest.raises(v.Unreachable):
        getattr(v, fn_name)("acme")


@pytest.mark.parametrize("platform, fn_name, host_tpl, elsewhere", _HOST_PROBES)
@pytest.mark.parametrize("status", [403, 429, 503])
def test_a_refusal_is_unreachable_even_when_the_final_url_is_elsewhere(
        monkeypatch, platform, fn_name, host_tpl, elsewhere, status):
    """The host check must apply only to a 200.

    A 403/429/503 is a refusal, and refusals carry whatever final URL the
    vendor's error page happens to have. Letting the host check answer for them
    turns every refusal into a dead mark -- the false-closure shape that once
    had lever and iCIMS marking companies dead when one host merely refused.
    A 404 hides this, since dead is the right answer there either way, which is
    why these three statuses get their own case.
    """
    monkeypatch.setattr(v, "_request", lambda url, **kw: (status, b"", elsewhere))
    with pytest.raises(v.Unreachable):
        getattr(v, fn_name)("acme")


# ── smartrecruiters: existence, not "has jobs" ──────────────────────────────
#
# The postings API answers 200 for anything, so the probe read totalFound and
# a zero ended it -- conflating "no such company" with "a real company
# advertising nothing today". Measured: of six candidates returning zero, four
# had a real careers page. The bot skips dead slugs, so the first job each of
# them posted would have been missed until a recheck cleared the mark.

def _sr_responses(monkeypatch, postings_body, careers=None):
    def fake(url, **kwargs):
        if "api.smartrecruiters.com" in url:
            return 200, postings_body, url
        status, final = careers or (200, "https://jobs.smartrecruiters.com")
        return status, b"", final
    monkeypatch.setattr(v, "_request", fake)


def test_a_company_with_postings_needs_no_second_request(monkeypatch):
    """The ordinary case must still cost one request."""
    calls: list[str] = []

    def fake(url, **kwargs):
        calls.append(url)
        return 200, b'{"totalFound": 3}', url

    monkeypatch.setattr(v, "_request", fake)
    assert v.live_smartrecruiters("acme") is True
    assert len(calls) == 1, "the careers page was fetched despite a live count"


def test_zero_postings_but_a_real_careers_page_is_live(monkeypatch):
    _sr_responses(monkeypatch, b'{"totalFound": 0}',
                  careers=(200, "https://careers.smartrecruiters.com/acme"))
    assert v.live_smartrecruiters("acme") is True


def test_zero_postings_and_a_bounce_to_the_root_is_dead(monkeypatch):
    """An unknown slug is redirected to the bare jobs host, losing the slug."""
    _sr_responses(monkeypatch, b'{"totalFound": 0}',
                  careers=(200, "https://jobs.smartrecruiters.com"))
    assert v.live_smartrecruiters("acme") is False


def test_the_careers_page_match_is_case_insensitive(monkeypatch):
    """SmartRecruiters echoes the slug with its own capitalisation."""
    _sr_responses(monkeypatch, b'{"totalFound": 0}',
                  careers=(200, "https://careers.smartrecruiters.com/AcMe"))
    assert v.live_smartrecruiters("acme") is True


def test_a_refusal_on_the_careers_page_is_not_a_dead_mark(monkeypatch):
    """"No jobs" plus "the second endpoint refused" is not evidence of absence.

    Reading it as one would mark companies dead during an outage, which is the
    false-closure shape the rest of this module already guards against.
    """
    for status in (403, 429, 503):
        _sr_responses(monkeypatch, b'{"totalFound": 0}', careers=(status, None))
        with pytest.raises(v.Unreachable):
            v.live_smartrecruiters("acme")


def test_a_404_on_the_careers_page_is_dead(monkeypatch):
    _sr_responses(monkeypatch, b'{"totalFound": 0}', careers=(404, None))
    assert v.live_smartrecruiters("acme") is False


def test_an_unparseable_postings_body_is_still_unknown(monkeypatch):
    """Unchanged: truncation must not read as "no postings" and fall through to
    a verdict. It biases toward dead precisely where it matters, because a
    company with postings returns the long response."""
    monkeypatch.setattr(v, "_request",
                        lambda url, **kw: (200, b'{"totalFound": ', url))
    with pytest.raises(v.Unreachable):
        v.live_smartrecruiters("acme")


@pytest.mark.parametrize("slug, final", [
    ("jobs", "https://jobs.smartrecruiters.com"),
    ("careers", "https://careers.smartrecruiters.com"),
    ("smartrecruiters", "https://jobs.smartrecruiters.com"),
])
def test_a_slug_matching_the_bounce_host_is_not_treated_as_existing(
        monkeypatch, slug, final):
    """The match needs the path separator, not just the substring.

    The bounce target is jobs.smartrecruiters.com, so a company slugged "jobs"
    or "careers" appears in the URL of the very redirect that means "no such
    company" -- and a substring test would read its own failure as success.
    A mutation dropping the leading slash survived every other test here.
    """
    _sr_responses(monkeypatch, b'{"totalFound": 0}', careers=(200, final))
    assert v.live_smartrecruiters(slug) is False


def test_a_200_with_no_final_url_is_not_treated_as_existing(monkeypatch):
    """Silence is not evidence of existence.

    _request can return a 200 with no final URL, and defaulting that to "the
    company exists" would confirm slugs on the strength of a missing field --
    the same silence-reads-as-health shape guarded against elsewhere here.
    """
    _sr_responses(monkeypatch, b'{"totalFound": 0}', careers=(200, None))
    assert v.live_smartrecruiters("acme") is False


# ── revival audit ───────────────────────────────────────────────────────────
#
# A dead mark coming back live is normal; companies repost. A platform whose
# dead marks come back live in bulk is a broken probe, because those marks were
# never earned. This is the only measure that would have caught the
# smartrecruiters postings-count bug: its live rate looked plausible, it said no
# to invented slugs and yes to known-live ones, and no scraper contradicted a
# mark -- but 150 of its 212 dead marks answered live when re-asked.

def test_a_platform_whose_dead_marks_answer_live_raises_the_alarm(sandbox):
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({f"s{i}": "2026-09-01" for i in range(20)}), encoding="utf-8")
    out = v.revival_rate("lever", 20, probe=lambda slug: True, workers=1,
                         log=lambda m: None)
    assert out["revived"] == 20
    assert out["rate"] == 100.0
    assert out["alarm"] is True


def test_dead_marks_that_stay_dead_do_not_alarm(sandbox):
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({f"s{i}": "2026-09-01" for i in range(20)}), encoding="utf-8")
    out = v.revival_rate("lever", 20, probe=lambda slug: False, workers=1,
                         log=lambda m: None)
    assert out["revived"] == 0 and out["alarm"] is False


def test_a_handful_of_revivals_is_not_an_alarm(sandbox):
    """Real baseline is 0-20%; companies do repost. Alarming on any revival
    would fire constantly and be ignored, which is worse than not having it."""
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({f"s{i}": "2026-09-01" for i in range(20)}), encoding="utf-8")
    out = v.revival_rate("lever", 20, probe=lambda slug: slug in ("s1", "s2"),
                         workers=1, log=lambda m: None)
    assert out["rate"] == 10.0 and out["alarm"] is False


def test_a_tiny_sample_cannot_alarm(sandbox):
    """Two of three is 67% and means nothing. icims genuinely reads 33% on a
    six-slug sample, and that must not be reported as a broken probe."""
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({f"s{i}": "2026-09-01" for i in range(3)}), encoding="utf-8")
    out = v.revival_rate("lever", 3, probe=lambda slug: slug != "s0",
                         workers=1, log=lambda m: None)
    assert out["rate"] > 50 and out["alarm"] is False, "alarmed on 3 samples"


def test_a_platform_with_no_dead_marks_is_not_an_error(sandbox):
    out = v.revival_rate("lever", 10, probe=lambda slug: True, workers=1,
                         log=lambda m: None)
    assert out == {"sampled": 0, "revived": 0, "rate": 0.0, "dead_total": 0,
                   "alarm": False}


def test_the_revival_audit_writes_nothing(sandbox):
    """Recording these verdicts would repair the evidence the audit exists to
    report -- the same reason audit_live_rate is read-only."""
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({f"s{i}": "2026-09-01" for i in range(12)}), encoding="utf-8")
    before = (sandbox / "dead" / "lever.json").read_text(encoding="utf-8")
    v.revival_rate("lever", 12, probe=lambda slug: True, workers=1,
                   log=lambda m: None)
    assert (sandbox / "dead" / "lever.json").read_text(encoding="utf-8") == before
    assert list((sandbox / "checked").iterdir()) == []


def test_the_alarm_is_visible_in_the_log(sandbox):
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({f"s{i}": "2026-09-01" for i in range(20)}), encoding="utf-8")
    lines: list[str] = []
    v.revival_rate("lever", 20, probe=lambda slug: True, workers=1,
                   log=lines.append)
    assert any("condemning companies that exist" in line for line in lines)


def test_the_revival_audit_flag_reaches_the_run(sandbox, capsys):
    """Pins the wiring, not the measure.

    revival_rate can be entirely correct while --revival-audit never calls it --
    a mutation that stubbed the call site to an empty dict left every other test
    in this section passing, because they all call revival_rate directly.
    """
    (sandbox / "dead" / "lever.json").write_text(
        json.dumps({f"s{i}": "2026-09-01" for i in range(12)}), encoding="utf-8")

    assert v.main(["--platform", "lever", "--revival-audit", "12", "--json",
                   "--limit", "0", "--budget-seconds", "0"]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{"):])["platforms"]["lever"]

    assert "revival" in summary, "--revival-audit produced no revival section"
    # The stub probe in `sandbox` calls anything starting with "alive" live, so
    # these twelve all read dead -- the point is that real numbers came back.
    assert summary["revival"]["dead_total"] == 12
    assert summary["revival"]["sampled"] == 12


# ── giving up on a platform that is refusing everything ─────────────────────
#
# A 429 is Unreachable, not dead, so no mark is ever wrong because of one -- but
# that is exactly why the live-rate collapse guard never fires for it: it
# watches dead verdicts, and a refused probe produces none. So a quota-exhausted
# platform spent its whole budget learning nothing. Workable is metered per
# window rather than rate-limited, and was observed refusing 40 of 40 probes;
# an earlier pass came back 91% refused and left the address throttled for
# minutes, so continuing also makes the next run worse.

def _storm_probe(refuse_all=True):
    def probe(slug):
        if refuse_all:
            raise v.Unreachable("429")
        return True
    return probe


def test_a_platform_refusing_everything_stops_early():
    slugs = [f"s{i}" for i in range(200)]
    lines: list[str] = []
    res = v.validate_platform("workable", slugs, probe=_storm_probe(),
                              workers=2, budget_seconds=None, log=lines.append)
    assert res["probed"] < len(slugs), "the whole list was probed anyway"
    assert res["deferred"] == len(slugs) - res["probed"]
    assert any("refused -- stopping" in line for line in lines)


def test_no_verdict_is_invented_when_it_stops():
    """Stopping must not turn refusals into dead marks -- that is the failure
    the refusal handling exists to prevent."""
    res = v.validate_platform("workable", [f"s{i}" for i in range(100)],
                              probe=_storm_probe(), workers=2, log=lambda m: None)
    assert res["dead"] == 0 and res["live"] == 0
    assert res["_dead"] == [] and res["_live"] == []


def test_a_healthy_platform_is_never_stopped():
    slugs = [f"s{i}" for i in range(120)]
    lines: list[str] = []
    res = v.validate_platform("greenhouse", slugs, probe=lambda s: True,
                              workers=2, log=lines.append)
    assert res["probed"] == len(slugs)
    assert not any("refused -- stopping" in line for line in lines)


def test_a_partly_flaky_platform_keeps_going():
    """Half-refused runs still resolve half of what they touch, and those
    verdicts are worth having. Only a near-total refusal is worthless."""
    def flaky(slug):
        if int(slug[1:]) % 2:
            raise v.Unreachable("timeout")
        return False

    res = v.validate_platform("icims", [f"s{i}" for i in range(120)],
                              probe=flaky, workers=2, log=lambda m: None)
    assert res["probed"] == 120
    assert res["dead"] == 60


def test_a_short_run_cannot_trip_the_storm_guard():
    """Below the minimum, a run of refusals means nothing -- three timeouts in
    a row is an ordinary network hiccup, not an exhausted quota."""
    slugs = [f"s{i}" for i in range(8)]
    res = v.validate_platform("workable", slugs, probe=_storm_probe(),
                              workers=2, log=lambda m: None)
    assert res["probed"] == len(slugs), "stopped on too small a sample"


def test_the_budget_message_and_the_storm_message_are_distinct():
    """"Budget spent" and "the platform is refusing us" call for different
    responses, and reading one as the other wastes a debugging cycle."""
    lines: list[str] = []
    v.validate_platform("workable", [f"s{i}" for i in range(60)],
                        probe=_storm_probe(), workers=2, log=lines.append)
    assert not any("budget spent" in line for line in lines)


# ── rippling: probe the endpoint the scraper reads ──────────────────────────
#
# The public board page and the ATS API disagree: the page 200s for companies
# whose API board 404s. Because the probe read the page and _scrape_rippling
# reads the API, the two traded the same slugs indefinitely -- the validator
# revived the mark, the scraper's next pass got a 404 and marked it dead again.
# qucareers, rabot-energy and whitehatgaming were each marked dead on
# 2026-09-03 and reported live the following day, three times out of three.
#
# This is the same rule live_recruitee already states: a company the probe calls
# live has to be one the bot can actually pull jobs from.

def test_rippling_probes_the_api_not_the_board_page(monkeypatch):
    asked: list[str] = []

    def fake(url, **kwargs):
        asked.append(url)
        return 200, b"[]", url

    monkeypatch.setattr(v, "_request", fake)
    v.live_rippling("acme")
    assert asked, "no request was made"
    assert "api.rippling.com" in asked[0], asked
    assert "ats.rippling.com" not in asked[0], (
        "the board page 200s for companies the scraper cannot read")


def test_a_company_the_scraper_cannot_read_stays_dead(monkeypatch):
    """The API 404 is the verdict that matters -- reviving on the board page is
    what created the loop."""
    monkeypatch.setattr(v, "_request", lambda url, **kw: (404, b"", url))
    assert v.live_rippling("qucareers") is False


def test_a_rippling_board_with_no_jobs_is_still_live(monkeypatch):
    """The API answers 200 with an empty list for a real company advertising
    nothing. Reading that as dead would reintroduce exactly the conflation
    smartrecruiters was fixed for."""
    monkeypatch.setattr(v, "_request", lambda url, **kw: (200, b"[]", url))
    assert v.live_rippling("webber-restaurant-group") is True


def test_a_rippling_refusal_is_unreachable(monkeypatch):
    for status in (403, 429, 503):
        monkeypatch.setattr(v, "_request", lambda url, **kw: (status, b"", url))
        with pytest.raises(v.Unreachable):
            v.live_rippling("acme")


def test_the_rippling_probe_and_scraper_share_an_endpoint():
    """Pins the alignment itself.

    Either side drifting to a different URL brings the loop back, and the loop
    is invisible in any single run -- each side looks locally correct.
    """
    import inspect

    from services import ats_service

    probe_src = inspect.getsource(v.live_rippling)
    scraper_src = inspect.getsource(ats_service._scrape_rippling)
    path = "/platform/api/ats/v1/board/"
    assert path in probe_src, "the probe stopped using the ATS API"
    assert path in scraper_src, "the scraper moved off the ATS API"


# ── per-platform pacing ─────────────────────────────────────────────────────
#
# recruitee had been given workable's measured numbers -- four workers at a
# quarter-second -- without a curve of its own, and refused 28% of a real run at
# that pace. Its own measurements, 429s per 60 probes from cold: 17 at 8.6/s,
# 14 at 5.3/s, 0 at 2.4/s. Two workers at half a second refused none of 100.
#
# It also holds a grudge: the same 3-worker/1.0s setting that refused nothing
# from cold refused 9 of 60 straight after a fast burst, so one impatient run
# spoils the next.

def test_no_dict_in_the_module_has_a_duplicated_key():
    """WORKERS listed "recruitee" twice, as 8 and then as 4.

    Python keeps the last, so it worked -- but the first entry was dead and
    misleading, and editing it would have changed nothing while looking like it
    had. Static, so it catches the next one for free.
    """
    import ast
    import inspect

    src = inspect.getsource(v)
    offenders = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        offenders += [(node.lineno, k) for k in set(keys) if keys.count(k) > 1]
    assert not offenders, f"duplicated dict keys: {offenders}"


def test_every_platform_has_a_worker_setting():
    """A platform missing from WORKERS silently takes the generic default,
    which is how recruitee ran at four times a safe pace."""
    for platform in v.PLATFORMS:
        assert platform in v.WORKERS, f"{platform} has no measured worker count"


def test_recruitee_is_paced_below_its_measured_refusal_threshold():
    """Rate is what matters, not worker count.

    The delay is per worker, so the ceiling is workers/delay. recruitee refused
    heavily at 5/s and above and refused nothing at 2.5/s, so the configured
    ceiling has to stay well under the refusing range -- raising either value
    alone silently undoes the other.
    """
    ceiling = v.WORKERS["recruitee"] / v.DELAYS["recruitee"]
    assert ceiling <= 4.0, f"recruitee ceiling is {ceiling}/s, it refuses from ~5/s"


def test_a_paced_platform_actually_sleeps(monkeypatch):
    """The delay has to reach validate_platform, not merely be configured."""
    slept: list[float] = []
    monkeypatch.setattr(v.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setitem(v.PROBES, "recruitee", lambda slug: True)
    v.validate_platform("recruitee", ["a", "b", "c"], workers=1,
                        log=lambda m: None)
    assert slept, "no pacing delay was applied"
    assert all(s == v.DELAYS["recruitee"] for s in slept), slept


def test_an_unpaced_platform_does_not_sleep(monkeypatch):
    """Pacing costs throughput, so it must not spread to platforms that never
    asked for it."""
    slept: list[float] = []
    monkeypatch.setattr(v.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setitem(v.PROBES, "greenhouse", lambda slug: True)
    v.validate_platform("greenhouse", ["a", "b", "c"], workers=1,
                        log=lambda m: None)
    assert slept == []
