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
    (200, True), (404, False), (410, False), (403, False), (422, False),
])
def test_decide_definitive(status, expected):
    assert v._decide(status) is expected


@pytest.mark.parametrize("status", [None, 500, 502, 503, 429])
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
    (tmp_path / "ats_companies" / f"{platform}_companies.json").write_text(
        json.dumps(list(companies)))
    (tmp_path / "ats_harvest" / f"{platform}.json").write_text(json.dumps(list(harvest)))
    if dead is not None:
        (tmp_path / "dead_slugs" / f"{platform}.json").write_text(json.dumps(dead))
    monkeypatch.setattr(v, "COMPANY_DIR", tmp_path / "ats_companies")
    monkeypatch.setattr(v, "HARVEST_DIR", tmp_path / "ats_harvest")
    monkeypatch.setattr(v, "DEAD_DIR", tmp_path / "dead_slugs")


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
