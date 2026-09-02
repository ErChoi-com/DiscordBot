"""Tests for the ATS slug liveness validator.

Hermetic per pytest.ini: no probe here touches the network. What is being tested
is the decision logic -- what counts as dead, what a timeout must not do, and
that the file written is byte-compatible with what ats_service reads.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_ats_slugs as v  # noqa: E402

TODAY = _dt.date(2026, 9, 1)


# --------------------------------------------------------------------------
# What a status code means
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    (200, True), (404, False), (410, False), (422, False),
])
def test_decide_definitive(status, expected):
    assert v._decide(status) is expected


@pytest.mark.parametrize("status", [None, 500, 502, 503, 429, 403])
def test_decide_treats_ambiguous_as_unreachable(status):
    """A 5xx or a timeout says nothing about whether the company exists.

    Counting these as dead would suppress live companies for a week every time
    a platform had a bad minute.
    """
    with pytest.raises(v.Unreachable):
        v._decide(status)


# --------------------------------------------------------------------------
# Verdict handling
# --------------------------------------------------------------------------

def test_unknown_never_marks_dead():
    dead: dict[str, str] = {}
    result = v.validate_platform(
        "greenhouse", ["a", "b"],
        probe=lambda s: (_ for _ in ()).throw(v.Unreachable("timeout")),
        workers=2, log=lambda m: None,
    )
    assert result["unknown"] == 2 and result["dead"] == 0
    changes = v.apply_results(dead, result, TODAY)
    assert dead == {} and changes["added"] == 0


def test_probe_crash_is_unknown_not_dead():
    """A bug in a probe must not be indistinguishable from 'company is gone'."""
    result = v.validate_platform(
        "greenhouse", ["a"],
        probe=lambda s: (_ for _ in ()).throw(ValueError("boom")),
        workers=1, log=lambda m: None,
    )
    assert result["unknown"] == 1 and result["dead"] == 0


def test_live_slug_is_revived():
    dead = {"acme": "2026-08-01", "gone": "2026-08-01"}
    result = v.validate_platform("greenhouse", ["acme", "gone"],
                                 probe=lambda s: s == "acme", workers=2,
                                 log=lambda m: None)
    changes = v.apply_results(dead, result, TODAY)
    assert "acme" not in dead
    assert dead["gone"] == "2026-09-01"
    assert changes["revived"] == 1 and changes["redated"] == 1


def test_dead_slug_is_dated_today():
    dead: dict[str, str] = {}
    result = v.validate_platform("greenhouse", ["gone"], probe=lambda s: False,
                                 workers=1, log=lambda m: None)
    v.apply_results(dead, result, TODAY)
    assert dead == {"gone": "2026-09-01"}


# --------------------------------------------------------------------------
# Target selection
# --------------------------------------------------------------------------

def _seed(tmp_path, monkeypatch, platform, companies, harvest=(), dead=None):
    (tmp_path / "ats_companies").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ats_harvest").mkdir(parents=True, exist_ok=True)
    (tmp_path / "dead_slugs").mkdir(parents=True, exist_ok=True)
    # Go through COMPANY_FILES rather than assuming "{platform}_companies.json":
    # paylocity's upstream list is named differently, so seeding it by the
    # default pattern writes a file nothing reads and the platform silently
    # validates zero companies.
    (tmp_path / "ats_companies" / v.COMPANY_FILES[platform]).write_text(
        json.dumps(list(companies)))
    (tmp_path / "ats_harvest" / f"{platform}.json").write_text(json.dumps(list(harvest)))
    if dead is not None:
        (tmp_path / "dead_slugs" / f"{platform}.json").write_text(json.dumps(dead))
    monkeypatch.setattr(v, "COMPANY_DIR", tmp_path / "ats_companies")
    monkeypatch.setattr(v, "HARVEST_DIR", tmp_path / "ats_harvest")
    monkeypatch.setattr(v, "DEAD_DIR", tmp_path / "dead_slugs")
    # Redirect this too, or a test that persists results writes fixture slugs
    # into the repository's real data -- which is exactly what happened.
    monkeypatch.setattr(v, "CHECKED_DIR", tmp_path / "ats_checked")


def test_candidates_union_upstream_and_harvest(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["a", "b"], harvest=["b", "c"])
    assert v.load_candidates("lever") == ["a", "b", "c"]


def test_unchecked_slugs_are_probed_before_known_dead(tmp_path, monkeypatch):
    """With --limit bounding a CI run, spending the budget on slugs nobody has
    ever probed beats re-confirming what we already believe."""
    _seed(tmp_path, monkeypatch, "lever", ["olddead", "fresh"],
          dead={"olddead": "2026-01-01"})
    targets = v.select_targets("lever", v.load_dead("lever"), recheck_dead=False,
                               limit=1, sample=None, today=TODAY)
    assert targets == ["fresh"]


def test_recent_dead_marks_are_skipped(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["recent"], dead={"recent": "2026-08-30"})
    targets = v.select_targets("lever", v.load_dead("lever"), recheck_dead=False,
                               limit=None, sample=None, today=TODAY)
    assert targets == []


def test_stale_dead_marks_are_reprobed(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["old"], dead={"old": "2026-07-01"})
    targets = v.select_targets("lever", v.load_dead("lever"), recheck_dead=False,
                               limit=None, sample=None, today=TODAY)
    assert targets == ["old"]


def test_recheck_dead_overrides_the_window(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["recent"], dead={"recent": "2026-08-30"})
    targets = v.select_targets("lever", v.load_dead("lever"), recheck_dead=True,
                               limit=None, sample=None, today=TODAY)
    assert targets == ["recent"]


def test_corrupt_date_is_reprobed(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["weird"], dead={"weird": "not-a-date"})
    targets = v.select_targets("lever", v.load_dead("lever"), recheck_dead=False,
                               limit=None, sample=None, today=TODAY)
    assert targets == ["weird"]


# --------------------------------------------------------------------------
# On-disk contract with ats_service
# --------------------------------------------------------------------------

def test_saved_file_matches_ats_service_format(tmp_path, monkeypatch):
    """ats_service reads {slug: 'YYYY-MM-DD'}; anything else silently resets
    the platform's dead marks."""
    _seed(tmp_path, monkeypatch, "lever", [])
    v.save_dead("lever", {"b": "2026-09-01", "a": "2026-08-01"})
    raw = (tmp_path / "dead_slugs" / "lever.json").read_text()
    parsed = json.loads(raw)
    assert parsed == {"a": "2026-08-01", "b": "2026-09-01"}
    assert list(parsed) == ["a", "b"], "must be sorted for stable diffs"
    assert raw.startswith("{\n"), "must be indented, matching _save_dead_slugs"


def test_legacy_list_format_is_read_not_discarded(tmp_path, monkeypatch):
    """An older revision stored a bare list. Reading it as empty would
    resurrect every dead slug on the platform at once."""
    _seed(tmp_path, monkeypatch, "lever", [], dead=["a", "b"])
    dead = v.load_dead("lever")
    assert set(dead) == {"a", "b"}
    assert all(len(x) == 10 for x in dead.values())


def test_corrupt_dead_file_reads_as_empty(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", [])
    (tmp_path / "dead_slugs" / "lever.json").write_text("{ not json")
    assert v.load_dead("lever") == {}


def test_save_leaves_no_temp_file(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", [])
    v.save_dead("lever", {"a": "2026-09-01"})
    assert [p.name for p in (tmp_path / "dead_slugs").iterdir()] == ["lever.json"]


def test_purge_drops_only_expired_marks():
    dead = {"old": "2026-01-01", "recent": "2026-08-25"}
    removed = v.purge_expired(dead, TODAY, ttl_days=90)
    assert removed == 1 and dead == {"recent": "2026-08-25"}


# --------------------------------------------------------------------------
# Platform-specific liveness rules, each a bug found by calibration
# --------------------------------------------------------------------------

def test_bamboohr_redirect_to_marketing_site_is_dead(monkeypatch):
    """An unknown BambooHR tenant answers 200 and redirects to www."""
    monkeypatch.setattr(v, "_request",
                        lambda *a, **k: (200, b"<!DOCTYPE html>", "https://www.bamboohr.com/"))
    assert v.live_bamboohr("nosuchtenant") is False


def test_bamboohr_json_on_tenant_host_is_live(monkeypatch):
    monkeypatch.setattr(v, "_request",
                        lambda *a, **k: (200, b'{"meta":{}}',
                                         "https://acme.bamboohr.com/careers/list"))
    assert v.live_bamboohr("acme") is True


def test_icims_tries_both_host_conventions(monkeypatch):
    """46 of 120 sampled slugs resolved only prefixed, 34 only bare, none both."""
    seen = []

    def fake(url, **kwargs):
        seen.append(url)
        return (200 if url.startswith("https://acme.icims.com") else 404), b"", url

    monkeypatch.setattr(v, "_request", fake)
    assert v.live_icims("acme") is True
    assert any("careers-acme.icims.com" in u for u in seen)
    assert any("//acme.icims.com" in u for u in seen)


def test_icims_already_prefixed_slug_is_not_double_prefixed(monkeypatch):
    seen = []

    def fake(url, **kwargs):
        seen.append(url)
        return 404, b"", url

    monkeypatch.setattr(v, "_request", fake)
    assert v.live_icims("careers-gbrx") is False
    assert not any("careers-careers-" in u for u in seen)


def test_workday_posts_a_body(monkeypatch):
    """Without a JSON body a live tenant answers 500 and a dead one 422, so an
    empty POST cannot tell them apart."""
    captured = {}

    def fake(url, *, method="GET", payload=None, **kwargs):
        captured["method"] = method
        captured["payload"] = payload
        return 200, b"", url

    monkeypatch.setattr(v, "_request", fake)
    assert v.live_workday("acme|wd1|external") is True
    assert captured["method"] == "POST"
    assert "limit" in captured["payload"]


def test_workday_malformed_triple_is_dead(monkeypatch):
    monkeypatch.setattr(v, "_request", lambda *a, **k: (200, b"", ""))
    assert v.live_workday("not-a-triple") is False


# --------------------------------------------------------------------------
# Run-level behaviour
# --------------------------------------------------------------------------

def test_nothing_to_probe_is_success_not_failure(tmp_path, monkeypatch):
    """Every slug being inside its recheck window is a healthy steady state."""
    _seed(tmp_path, monkeypatch, "lever", ["a"], dead={"a": "2026-08-30"})
    monkeypatch.setattr(v._dt, "date", type("D", (), {
        "today": staticmethod(lambda: TODAY),
        "fromisoformat": staticmethod(_dt.date.fromisoformat)}))
    assert v.main(["--platform", "lever"]) == 0


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["gone"])
    monkeypatch.setitem(v.PROBES, "lever", lambda s: False)
    assert v.main(["--platform", "lever", "--dry-run"]) == 0
    assert v.load_dead("lever") == {}


def test_results_are_persisted(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["gone", "alive"])
    monkeypatch.setitem(v.PROBES, "lever", lambda s: s == "alive")
    assert v.main(["--platform", "lever"]) == 0
    dead = v.load_dead("lever")
    assert "gone" in dead and "alive" not in dead


def test_lever_checks_both_regions(monkeypatch):
    """Lever's US and EU deployments are separate and a company exists in
    exactly one: amicustherapeutics is 200 on api.eu.lever.co and 404 on
    api.lever.co. Checking only the US host marks every EU company dead."""
    seen = []

    def fake(url, **kwargs):
        seen.append(url)
        return (200 if "api.eu.lever.co" in url else 404), b"", url

    monkeypatch.setattr(v, "_request", fake)
    assert v.live_lever("amicustherapeutics") is True
    assert any("//api.lever.co/" in u for u in seen)
    assert any("//api.eu.lever.co/" in u for u in seen)


def test_lever_dead_only_when_both_regions_miss(monkeypatch):
    monkeypatch.setattr(v, "_request", lambda url, **k: (404, b"", url))
    assert v.live_lever("nosuchco") is False


def test_lever_unreachable_in_every_region_is_not_dead(monkeypatch):
    monkeypatch.setattr(v, "_request", lambda url, **k: (503, b"", url))
    with pytest.raises(v.Unreachable):
        v.live_lever("acme")


# --------------------------------------------------------------------------
# Wall-clock budget
# --------------------------------------------------------------------------

def test_validation_stops_when_its_budget_is_spent():
    """Platforms differ by an order of magnitude in how fast they answer --
    Workday needs a bodied POST, iCIMS two requests per slug -- so one slow
    platform must not consume the whole CI job."""
    clock = {"t": 0.0}
    seen = []

    def probe(slug):
        seen.append(slug)
        clock["t"] += 10.0
        return True

    result = v.validate_platform(
        "greenhouse", [f"co{i}" for i in range(400)], probe=probe, workers=2,
        budget_seconds=100.0, now=lambda: clock["t"], log=lambda m: None)
    assert result["probed"] < 400
    assert result["deferred"] == 400 - result["probed"]


def test_deferred_slugs_are_not_marked_dead():
    """Whatever the budget cut off must keep its existing state, not be
    recorded as anything -- it is simply picked up on the next run."""
    clock = {"t": 0.0}

    def probe(slug):
        clock["t"] += 10.0
        return False

    dead: dict[str, str] = {}
    result = v.validate_platform(
        "greenhouse", [f"co{i}" for i in range(400)], probe=probe, workers=2,
        budget_seconds=50.0, now=lambda: clock["t"], log=lambda m: None)
    v.apply_results(dead, result, TODAY)
    assert len(dead) == result["dead"] < 400


def test_budget_never_skips_the_first_chunk():
    """A budget of zero must still make progress rather than probing nothing
    forever."""
    result = v.validate_platform(
        "greenhouse", ["a", "b"], probe=lambda s: True, workers=2,
        budget_seconds=0.0, now=lambda: 10_000.0, log=lambda m: None)
    assert result["probed"] == 2


def test_no_budget_probes_everything():
    result = v.validate_platform(
        "greenhouse", [f"co{i}" for i in range(50)], probe=lambda s: True,
        workers=4, budget_seconds=None, log=lambda m: None)
    assert result["probed"] == 50 and result["deferred"] == 0


# --------------------------------------------------------------------------
# Concurrent writers
# --------------------------------------------------------------------------

def test_marks_written_during_the_run_are_not_clobbered(tmp_path, monkeypatch):
    """The bot marks these same files from its own scrape cycles.

    On this machine it wrote ~3,500 greenhouse marks during a single validation
    pass. Writing back the map loaded at the start would silently discard every
    one of them -- and because both writers are individually atomic, the loss
    looks like clean data rather than corruption.
    """
    _seed(tmp_path, monkeypatch, "lever", ["gone"], dead={"pre": "2026-08-20"})

    result = v.validate_platform("lever", ["gone"], probe=lambda s: False,
                                 workers=1, log=lambda m: None)
    # Another writer lands a mark after this run loaded the map.
    concurrent = v.load_dead("lever")
    concurrent["written-by-the-bot"] = "2026-09-01"
    v.save_dead("lever", concurrent)

    v.save_dead_merged("lever", result, TODAY, ttl_days=90)

    final = v.load_dead("lever")
    assert "written-by-the-bot" in final, "concurrent mark was clobbered"
    assert final["gone"] == "2026-09-01", "this run's verdict was lost"
    assert "pre" in final, "pre-existing mark was lost"


def test_merge_still_revives_a_slug_found_live(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["back"], dead={"back": "2026-08-20"})
    result = v.validate_platform("lever", ["back"], probe=lambda s: True,
                                 workers=1, log=lambda m: None)
    v.save_dead_merged("lever", result, TODAY, ttl_days=90)
    assert "back" not in v.load_dead("lever")


def test_merge_applies_the_ttl_purge(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["x"], dead={"ancient": "2020-01-01"})
    result = v.validate_platform("lever", [], probe=lambda s: True, workers=1,
                                 log=lambda m: None)
    changes = v.save_dead_merged("lever", result, TODAY, ttl_days=90)
    assert changes["expired"] == 1
    assert "ancient" not in v.load_dead("lever")


# --------------------------------------------------------------------------
# Unbiased audit sampling
# --------------------------------------------------------------------------

def test_audit_samples_the_whole_population_not_just_unprobed(tmp_path, monkeypatch):
    """The working pass probes never-probed slugs first, so its live rate
    excludes everything already known dead and reads far too high -- 97.5%
    against 38.8% on the same data. The audit must sample uniformly.
    """
    _seed(tmp_path, monkeypatch, "lever", [f"co{i}" for i in range(100)],
          dead={f"co{i}": "2026-08-30" for i in range(90)})
    seen = []

    def probe(slug):
        seen.append(slug)
        return int(slug[2:]) >= 90        # only the 10 unprobed ones are live

    out = v.audit_live_rate("lever", 100, probe=probe, workers=2,
                            log=lambda m: None)
    assert out["sampled"] == 100
    # A biased pass would have found 100%; the true population rate is 10%.
    assert out["rate"] == 10.0


def test_audit_is_read_only(tmp_path, monkeypatch):
    """Folding audit verdicts into the dead map would let the measurement
    reshape the population it is measuring."""
    _seed(tmp_path, monkeypatch, "lever", ["a", "b"], dead={"a": "2026-08-30"})
    before = v.load_dead("lever")
    v.audit_live_rate("lever", 2, probe=lambda s: False, workers=1,
                      log=lambda m: None)
    assert v.load_dead("lever") == before


def test_audit_handles_a_population_smaller_than_the_sample(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["only"])
    out = v.audit_live_rate("lever", 500, probe=lambda s: True, workers=1,
                            log=lambda m: None)
    assert out["sampled"] == 1 and out["rate"] == 100.0


def test_audit_on_empty_population_is_not_an_error(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", [])
    out = v.audit_live_rate("lever", 50, probe=lambda s: True, workers=1,
                            log=lambda m: None)
    assert out["sampled"] == 0 and out["rate"] == 0.0


def test_audit_rate_ignores_unknowns(tmp_path, monkeypatch):
    """A rate over attempts rather than answers would let timeouts read as
    deaths, the same mistake the CI gate had."""
    _seed(tmp_path, monkeypatch, "lever", ["a", "b", "c", "d"])

    def probe(slug):
        if slug in ("a", "b"):
            raise v.Unreachable("timeout")
        return slug == "c"

    out = v.audit_live_rate("lever", 4, probe=probe, workers=2, log=lambda m: None)
    assert out["unknown"] == 2
    assert out["rate"] == 50.0        # 1 live of 2 decided, not of 4 attempted


def test_audit_is_time_bounded():
    """The audit is the one network loop that annotates rather than works, so
    it must never outlast the run. Paylocity probes at two workers and refuses
    connections under load, which is exactly where an unbounded sample bites.
    """
    clock = {"t": 0.0}

    def probe(slug):
        clock["t"] += 10.0
        return True

    result = v.validate_platform("greenhouse", [f"co{i}" for i in range(400)],
                                 probe=probe, workers=2, budget_seconds=60.0,
                                 now=lambda: clock["t"], log=lambda m: None)
    assert result["probed"] < 400 and result["deferred"] > 0


def test_audit_passes_a_budget_through(monkeypatch, tmp_path):
    seen = {}
    real = v.validate_platform

    def spy(platform, slugs, **kwargs):
        seen.update(kwargs)
        return real(platform, slugs, **kwargs)

    _seed(tmp_path, monkeypatch, "lever", ["a", "b"])
    monkeypatch.setattr(v, "validate_platform", spy)
    v.audit_live_rate("lever", 2, probe=lambda s: True, workers=1,
                      log=lambda m: None)
    assert seen.get("budget_seconds") == v.AUDIT_BUDGET_SECONDS


def test_paylocity_concurrency_stays_at_two():
    """Measured over 60-slug batches, unreachable responses by worker count:
    2 -> 0, 4 -> 20, 8 -> 48, 12 -> 53. Raw throughput keeps climbing past two
    workers while useful throughput falls, so a well-meaning bump here buys
    refusals that have to be re-probed later."""
    assert v.WORKERS["paylocity"] == 2


def test_every_platform_has_a_worker_count():
    for platform in v.PLATFORMS:
        assert v.WORKERS.get(platform), platform


def test_uncorroborated_slugs_are_probed_first(tmp_path, monkeypatch):
    """Appearing in BOTH lists predicts liveness; either alone does not.

        platform     both   harvest-only   upstream-only
        greenhouse  72.0%          42.0%           44.0%
        bamboohr    79.2%          51.0%           50.0%
        workday     81.6%          62.0%    0.0% (n=6,213)
        lever       41.7%          11.7%            1.7%

    A bounded run retires dead slugs before the bot spends a request on each
    every cycle, so it should start with the mostly-dead group.
    """
    _seed(tmp_path, monkeypatch, "lever",
          ["corroborated"], harvest=["corroborated", "harvest-only"])
    targets = v.select_targets("lever", {}, recheck_dead=False, limit=None,
                               sample=None, today=TODAY)
    assert targets[0] == "harvest-only"
    assert set(targets) == {"corroborated", "harvest-only"}


def test_upstream_only_slugs_are_not_treated_as_corroborated(tmp_path, monkeypatch):
    """Ordering on "is it upstream" gets this exactly backwards.

    Workday carries 6,213 slugs that upstream has and our harvest does not, and
    a 50-slug sample of them was 0% live -- the deadest group on any platform.
    A key of "in upstream" sorts them last, behind corroborated slugs that are
    81.6% live, which is the opposite of what a bounded run should probe.
    """
    _seed(tmp_path, monkeypatch, "workday",
          ["corroborated", "upstream-only"], harvest=["corroborated"])
    targets = v.select_targets("workday", {}, recheck_dead=False, limit=1,
                               sample=None, today=TODAY)
    assert targets == ["upstream-only"]


def test_ordering_changes_the_order_not_the_set(tmp_path, monkeypatch):
    """Everything is still reached; the priority only decides when."""
    _seed(tmp_path, monkeypatch, "lever", ["a", "b"], harvest=["c", "d"])
    targets = v.select_targets("lever", {}, recheck_dead=False, limit=None,
                               sample=None, today=TODAY)
    assert set(targets) == {"a", "b", "c", "d"}


def test_unchecked_still_outrank_stale_dead_marks(tmp_path, monkeypatch):
    """The corroboration ordering must not promote a known-dead slug above one
    nobody has ever probed."""
    _seed(tmp_path, monkeypatch, "lever", ["olddead"], harvest=["fresh"],
          dead={"olddead": "2026-01-01"})
    targets = v.select_targets("lever", v.load_dead("lever"), recheck_dead=False,
                               limit=1, sample=None, today=TODAY)
    assert targets == ["fresh"]


# --------------------------------------------------------------------------
# Structurally unscrapeable identifiers
# --------------------------------------------------------------------------

def test_impossible_identifiers_are_not_probed():
    """wd1|wd1|careers would have to resolve wd1.wd1.myworkdayjobs.com, which
    does not exist. Upstream ships 6,068 of that shape -- 47% of its Workday
    list -- and each costs a request here and one per scrape cycle in the bot
    until a mark lands."""
    bad, keep = v.partition_unscrapeable(
        "workday", ["wd1|wd1|careers", "acme|wd1|external", "alcon|wd5|job"])
    assert bad == ["wd1|wd1|careers", "alcon|wd5|job"]
    assert keep == ["acme|wd1|external"]


def test_normalizable_identifiers_are_still_probed():
    """The conservative half. iCIMS "careers-2u" is not broken -- the extractor
    would merely rewrite it to "2u", and ats_service probes both host forms, so
    the board is reachable. Treating 3,255 of those as impossible would retire
    live companies."""
    bad, keep = v.partition_unscrapeable("icims", ["careers-2u", "theharrispoll"])
    assert bad == []
    assert keep == ["careers-2u", "theharrispoll"]


def test_unknown_platform_partitions_nothing():
    bad, keep = v.partition_unscrapeable("nosuchplatform", ["a", "b"])
    assert bad == [] and keep == ["a", "b"]


def test_unscrapeable_slugs_are_marked_dead(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "workday", ["wd1|wd1|careers", "acme|wd1|external"])
    monkeypatch.setitem(v.PROBES, "workday", lambda s: True)
    assert v.main(["--platform", "workday"]) == 0
    dead = v.load_dead("workday")
    assert "wd1|wd1|careers" in dead
    assert "acme|wd1|external" not in dead


def test_unscrapeable_slugs_stay_out_of_the_live_rate(tmp_path, monkeypatch, capsys):
    """The rate feeds the CI gate that detects a broken endpoint, so it has to
    measure what the network said. Folding in entries that were never asked
    would drive Workday's rate to near zero on the first run and fail the build
    for a cleanup."""
    _seed(tmp_path, monkeypatch, "workday",
          ["wd1|wd1|careers", "wd3|wd1|x", "acme|wd1|external"])
    monkeypatch.setitem(v.PROBES, "workday", lambda s: True)
    v.main(["--platform", "workday"])
    out = capsys.readouterr().out
    assert "live rate 100.0%" in out


# --------------------------------------------------------------------------
# Concurrency lock
# --------------------------------------------------------------------------

def test_a_second_validation_run_is_refused(tmp_path, monkeypatch):
    """Two runs both read the dead map, probe for minutes, then write back.

    save_dead_merged re-reads immediately before writing, which narrows the
    window to a single write, but narrowing a race is not closing it. The
    harvester learned this the hard way when four sweeps ran at once.
    """
    _seed(tmp_path, monkeypatch, "lever", ["a"])
    monkeypatch.setitem(v.PROBES, "lever", lambda s: True)
    dead_dir = tmp_path / "dead_slugs"
    (dead_dir / v._harvest.LOCK_NAME).write_text("99999")
    assert v.main(["--platform", "lever"]) == 2


def test_a_dry_run_takes_no_lock(tmp_path, monkeypatch):
    """It writes nothing, so it has nothing to serialise against -- and being
    blocked from measuring because a real run is in progress would be
    gratuitous."""
    _seed(tmp_path, monkeypatch, "lever", ["a"])
    monkeypatch.setitem(v.PROBES, "lever", lambda s: True)
    dead_dir = tmp_path / "dead_slugs"
    (dead_dir / v._harvest.LOCK_NAME).write_text("99999")
    assert v.main(["--platform", "lever", "--dry-run"]) == 0


def test_lock_is_released_after_a_normal_run(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["a"])
    monkeypatch.setitem(v.PROBES, "lever", lambda s: True)
    assert v.main(["--platform", "lever"]) == 0
    assert not (tmp_path / "dead_slugs" / v._harvest.LOCK_NAME).exists()


def test_paylocity_reads_the_redirect_rather_than_the_body(monkeypatch):
    """A missing Paylocity board 302s to a "Job Not Found" shell; a real one
    answers 200. Following redirects collapses both to 200 and forces reading
    the body to tell them apart. Not following makes the status the answer --
    0.2s against 1.4s for a dead board, and a redirect is structural where a
    page title can be reworded."""
    seen = {}

    def fake(url, *, method="GET", payload=None, timeout=None,
             follow_redirects=True):
        seen["follow"] = follow_redirects
        return 302, b"", url

    monkeypatch.setattr(v, "_request", fake)
    assert v.live_paylocity("00000000-0000-0000-0000-000000000000") is False
    assert seen["follow"] is False, "redirects must not be followed"


def test_paylocity_two_hundred_is_a_live_board(monkeypatch):
    monkeypatch.setattr(v, "_request",
                        lambda url, **k: (200, b"", url))
    assert v.live_paylocity("2eba1a0a-d60f-4fd8-95ab-90b070f1d9f2") is True


def test_paylocity_server_error_is_unknown(monkeypatch):
    """A 5xx says nothing about the company, same as everywhere else."""
    monkeypatch.setattr(v, "_request", lambda url, **k: (503, b"", url))
    with pytest.raises(v.Unreachable):
        v.live_paylocity("2eba1a0a-d60f-4fd8-95ab-90b070f1d9f2")


def test_other_probes_still_follow_redirects(monkeypatch):
    """BambooHR depends on following one: an unknown tenant redirects to the
    marketing site, and the final URL is what gives it away."""
    seen = {}

    def fake(url, **kwargs):
        seen["follow"] = kwargs.get("follow_redirects", True)
        return 200, b'{"meta":{}}', "https://acme.bamboohr.com/careers/list"

    monkeypatch.setattr(v, "_request", fake)
    assert v.live_bamboohr("acme") is True
    assert seen["follow"] is True


# --------------------------------------------------------------------------
# Memory of live answers
# --------------------------------------------------------------------------

def _seed_checked(tmp_path, platform, checked):
    d = tmp_path / "ats_checked"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{platform}.json").write_text(json.dumps(checked))


def test_recently_confirmed_slugs_are_not_reprobed(tmp_path, monkeypatch):
    """A live answer used to be recorded nowhere, so every run re-probed every
    company it had ever confirmed. Measured on a full run: 39,043 probes,
    36,438 live, and the due count unchanged -- lever probed 2,469 slugs and
    still reported 2,469 due. The budget went almost entirely on re-confirming
    companies already known to be there."""
    _seed(tmp_path, monkeypatch, "lever", ["confirmed", "never-checked"])
    _seed_checked(tmp_path, "lever", {"confirmed": "2026-08-30"})
    targets = v.select_targets("lever", {}, recheck_dead=False, limit=None,
                               sample=None, today=TODAY)
    assert targets == ["never-checked"]


def test_confirmation_expires_so_closures_are_noticed(tmp_path, monkeypatch):
    """Thirty days sits well inside the 90-day dead TTL, so a board that closes
    is still caught within a month."""
    _seed(tmp_path, monkeypatch, "lever", ["old-confirmation"])
    _seed_checked(tmp_path, "lever", {"old-confirmation": "2026-06-01"})
    targets = v.select_targets("lever", {}, recheck_dead=False, limit=None,
                               sample=None, today=TODAY)
    assert targets == ["old-confirmation"]


def test_recheck_dead_still_forces_everything(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["confirmed"])
    _seed_checked(tmp_path, "lever", {"confirmed": "2026-08-30"})
    targets = v.select_targets("lever", {}, recheck_dead=True, limit=None,
                               sample=None, today=TODAY)
    assert targets == ["confirmed"]


def test_live_answers_are_recorded(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, "lever", ["alive", "gone"])
    result = v.validate_platform("lever", ["alive", "gone"],
                                 probe=lambda s: s == "alive", workers=2,
                                 log=lambda m: None)
    v.save_dead_merged("lever", result, TODAY, ttl_days=90)
    assert v.load_checked("lever") == {"alive": "2026-09-01"}


def test_a_slug_found_dead_stops_counting_as_confirmed(tmp_path, monkeypatch):
    """Otherwise a company that closes would be skipped as recently-confirmed
    while also carrying a dead mark."""
    _seed(tmp_path, monkeypatch, "lever", ["was-alive"])
    _seed_checked(tmp_path, "lever", {"was-alive": "2026-08-30"})
    result = v.validate_platform("lever", ["was-alive"], probe=lambda s: False,
                                 workers=1, log=lambda m: None)
    v.save_dead_merged("lever", result, TODAY, ttl_days=90)
    assert "was-alive" not in v.load_checked("lever")
    assert "was-alive" in v.load_dead("lever")


# --------------------------------------------------------------------------
# Shared time budget
# --------------------------------------------------------------------------

def test_slack_from_fast_platforms_reaches_the_slow_one(tmp_path, monkeypatch):
    """A fixed per-platform ceiling wastes what a fast platform does not use.

    Since the confirmed-live store landed, five platforms finish in seconds
    while Paylocity defers thousands of slugs at two workers -- so the unused
    time is exactly what the slow one needs.
    """
    budgets = {}
    real = v.validate_platform

    def spy(platform, slugs, **kwargs):
        budgets[platform] = kwargs.get("budget_seconds")
        return real(platform, slugs, **kwargs)

    for p in v.PLATFORMS:
        slug = "acme|wd1|external" if p == "workday" else "acmecorp"
        _seed(tmp_path, monkeypatch, p, [slug])
        monkeypatch.setitem(v.PROBES, p, lambda s: True)
    monkeypatch.setattr(v, "validate_platform", spy)

    total = 700.0
    v.main(["--budget-seconds", "10", "--total-budget-seconds", str(total)])
    # The first platform is offered an equal share of the whole; the last is
    # offered nearly all of it, because the ones before it returned instantly
    # and their unused time stayed in the pool. Derived from the platform count
    # rather than written in, so adding a platform does not silently retune
    # what this asserts.
    share = total / len(v.PLATFORMS)
    first, last = v.PLATFORMS[0], v.PLATFORMS[-1]
    assert 0.9 * share <= budgets[first] <= share
    assert budgets[last] > 0.85 * total
    assert budgets[last] > 5 * budgets[first]


def test_per_platform_ceiling_still_applies_without_a_total(tmp_path, monkeypatch):
    seen = {}
    real = v.validate_platform

    def spy(platform, slugs, **kwargs):
        seen[platform] = kwargs.get("budget_seconds")
        return real(platform, slugs, **kwargs)

    _seed(tmp_path, monkeypatch, "lever", ["a"])
    monkeypatch.setitem(v.PROBES, "lever", lambda s: True)
    monkeypatch.setattr(v, "validate_platform", spy)
    v.main(["--platform", "lever", "--budget-seconds", "42"])
    assert seen["lever"] == 42


def test_a_total_budget_never_shortens_the_per_platform_one(tmp_path, monkeypatch):
    """The total is there to hand out slack, not to take time away -- a run
    that sets both should not probe less than one that sets only the
    per-platform ceiling."""
    seen = {}
    real = v.validate_platform

    def spy(platform, slugs, **kwargs):
        seen[platform] = kwargs.get("budget_seconds")
        return real(platform, slugs, **kwargs)

    _seed(tmp_path, monkeypatch, "lever", ["a"])
    monkeypatch.setitem(v.PROBES, "lever", lambda s: True)
    monkeypatch.setattr(v, "validate_platform", spy)
    v.main(["--platform", "lever", "--budget-seconds", "500",
            "--total-budget-seconds", "10"])
    assert seen["lever"] >= 500


# --------------------------------------------------------------------------
# An endpoint outage must not be recorded as a mass closure
# --------------------------------------------------------------------------

def _dead_now(tmp_path):
    """Dead marks on disk; absent means none were recorded."""
    f = tmp_path / "dead_slugs" / "greenhouse.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _collapse_setup(tmp_path, monkeypatch, probe, checked_live=True):
    slugs = [f"co{i}" for i in range(40)]
    _seed(tmp_path, monkeypatch, "greenhouse", slugs)
    if checked_live:
        # Confirmed live long enough ago to be due again, so they are probed
        # rather than skipped -- but still the best control the run has.
        _seed_checked(tmp_path, "greenhouse", {s: "2026-01-01" for s in slugs})
    monkeypatch.setitem(v.PROBES, "greenhouse", probe)
    return slugs


def test_a_dead_endpoint_does_not_mark_every_company_dead(tmp_path, monkeypatch):
    """Nothing in the counts distinguishes a dead endpoint from a mass
    closure: a failed probe and an absent company are both "not live"."""
    _collapse_setup(tmp_path, monkeypatch, lambda s: False)
    v.main(["--platform", "greenhouse"])
    assert _dead_now(tmp_path) == {}


def test_a_genuine_cull_is_still_recorded(tmp_path, monkeypatch):
    """The guard must not swallow real closures. Here the endpoint answers --
    the slugs it confirmed most recently still come back live -- so a low rate
    is the truth and the dead marks have to land."""
    slugs = [f"co{i:02d}" for i in range(40)]
    _seed(tmp_path, monkeypatch, "greenhouse", slugs)
    # Distinct dates, so which slugs serve as the control is deterministic
    # rather than whatever a tie-break happens to order first.
    _seed_checked(tmp_path, "greenhouse",
                  {s: f"2026-01-{i + 1:02d}" for i, s in enumerate(slugs)})
    survivors = set(slugs[-v.ENDPOINT_CANARIES:])
    monkeypatch.setitem(v.PROBES, "greenhouse", lambda s: s in survivors)
    v.main(["--platform", "greenhouse"])
    dead = _dead_now(tmp_path)
    assert not (survivors & set(dead))
    assert len(dead) == len(slugs) - len(survivors)


def test_a_refusal_is_not_a_closure():
    """403 is what a blocked or rate-limited client is served, so it cannot
    mean the company is gone -- reading it as "absent" would record a block as
    a mass closure and suppress live boards for the whole dead TTL.

    Precautionary: sampling known-dead slugs found greenhouse, ashby and lever
    answer 404 and workday 404/422, so no platform currently returns 403 for a
    missing board."""
    with pytest.raises(v.Unreachable):
        v._decide(403)


@pytest.mark.parametrize("status", [404, 410, 422])
def test_a_real_absence_is_still_a_closure(status):
    assert v._decide(status) is False


def test_a_healthy_run_pays_nothing_for_the_guard(tmp_path, monkeypatch):
    """The check costs live requests, so it must only fire on a collapse."""
    calls = []

    def probe(slug):
        calls.append(slug)
        return True

    slugs = _collapse_setup(tmp_path, monkeypatch, probe)
    v.main(["--platform", "greenhouse"])
    assert len(calls) == len(slugs), "guard probed on a healthy run"


def test_a_small_batch_is_not_second_guessed(tmp_path, monkeypatch):
    """Three slugs all dead is unremarkable; the guard is for collapses large
    enough to mean something."""
    _seed(tmp_path, monkeypatch, "greenhouse", ["aco", "bco", "cco"])
    _seed_checked(tmp_path, "greenhouse", {"aco": "2026-01-01"})
    monkeypatch.setitem(v.PROBES, "greenhouse", lambda s: False)
    v.main(["--platform", "greenhouse"])
    assert len(json.loads(
        (tmp_path / "dead_slugs" / "greenhouse.json").read_text())) == 3


def test_a_raising_probe_counts_as_a_failed_canary(tmp_path, monkeypatch):
    """A network drop raises rather than returning False; if that escaped, the
    guard would crash on exactly the outage it exists to catch."""
    def probe(slug):
        raise OSError("network down")

    _collapse_setup(tmp_path, monkeypatch, probe)
    v.main(["--platform", "greenhouse"])
    assert _dead_now(tmp_path) == {}


def test_the_guard_reports_itself(tmp_path, monkeypatch):
    """A silently discarded platform looks identical to one with nothing to do,
    so the run has to say the endpoint was down."""
    _collapse_setup(tmp_path, monkeypatch, lambda s: False)
    out = json.loads(v.main(["--platform", "greenhouse", "--json"]) or "{}") \
        if False else None
    # main prints; capture via the summary file path instead
    import io, contextlib
    buf = io.StringIO()
    _collapse_setup(tmp_path, monkeypatch, lambda s: False)
    with contextlib.redirect_stdout(buf):
        v.main(["--platform", "greenhouse", "--json"])
    assert json.loads(buf.getvalue())["platforms"]["greenhouse"]["endpoint_down"]


def test_a_bamboohr_board_that_demands_a_login_is_not_live(monkeypatch):
    """401 has no rule in _decide, so it used to raise and leave the slug
    permanently unknown -- 358 of one run's 1,012 bamboohr probes, re-probed
    every run and unable to ever resolve.

    Measured before treating it as a closure: 40 such slugs returned 401 on two
    passes twenty seconds apart, identical both times, while fifteen
    confirmed-live and fifteen known-dead tenants answered 200. The 401 belongs
    to the tenant, not the client.
    """
    monkeypatch.setattr(v, "_request", lambda *a, **k: (401, b"", "x"))
    assert v.live_bamboohr("acme") is False


def test_a_bamboohr_outage_is_still_unknown(monkeypatch):
    """Resolving 401 must not turn every other failure into a closure."""
    monkeypatch.setattr(v, "_request", lambda *a, **k: (503, b"", "x"))
    with pytest.raises(v.Unreachable):
        v.live_bamboohr("acme")


# --------------------------------------------------------------------------
# A long pass has to be distinguishable from a hung one
# --------------------------------------------------------------------------

def test_a_long_pass_reports_progress(monkeypatch):
    """Between its first and last line a platform used to emit nothing, so a
    forty-minute paylocity pass looked exactly like a hang. The counts on disk
    do not help either: results are written when the platform finishes."""
    clock = iter(range(0, 10_000, 30))  # 30s per call, so beats every other one
    lines = []
    v.validate_platform("greenhouse", [f"co{i}" for i in range(40)],
                        probe=lambda s: True, workers=1,
                        now=lambda: next(clock), log=lines.append)
    beats = [l for l in lines if "probed," in l]
    assert beats, "a long pass said nothing until it finished"
    assert "/s after" in beats[0]


def test_a_short_pass_stays_quiet(monkeypatch):
    """The heartbeat is for runs long enough to be ambiguous. A fast platform
    emitting one line per chunk would bury the summaries it matters to read."""
    clock = iter([0.0] * 500)
    lines = []
    v.validate_platform("greenhouse", [f"co{i}" for i in range(40)],
                        probe=lambda s: True, workers=1,
                        now=lambda: next(clock), log=lines.append)
    assert not [l for l in lines if "probed," in l]


# --------------------------------------------------------------------------
# Pacing for endpoints that refuse a normal rate
# --------------------------------------------------------------------------

def test_a_rate_limited_platform_is_paced(monkeypatch):
    """Workable answers 429 to anything quicker. An eight-worker pass over
    6,843 slugs came back 6,242 unknown -- 91% refused -- and left the address
    throttled for minutes, so even a serial retry was refused. Nothing was
    corrupted, since 429 is unreachable rather than dead, but the entire pass
    was wasted and the next would have been too."""
    slept = []
    monkeypatch.setattr(v.time, "sleep", slept.append)
    monkeypatch.setitem(v.DELAYS, "workable", 0.5)
    v.validate_platform("workable", ["a", "b", "c"], probe=lambda s: True,
                        workers=1, log=lambda m: None)
    assert slept == [0.5, 0.5, 0.5]


def test_platforms_that_answer_normally_are_not_slowed(monkeypatch):
    """Pacing every platform would turn a run that finishes in seconds into
    one that does not finish at all."""
    slept = []
    monkeypatch.setattr(v.time, "sleep", slept.append)
    v.validate_platform("greenhouse", ["a", "b", "c"], probe=lambda s: True,
                        workers=1, log=lambda m: None)
    assert slept == []


def test_workable_is_paced_by_default():
    """The pacing has to be the default, not something a caller remembers:
    the unpaced run is the one that gets the address throttled."""
    assert v.DELAYS.get("workable", 0) > 0
    # The pace is what makes it safe, so concurrency stays well under the
    # eight-worker unpaced run that triggered the throttle. Four workers at a
    # quarter-second measured 10/s with zero refusals.
    assert v.WORKERS["workable"] <= 4
