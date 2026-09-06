"""Catching a platform that is validated but contributes nothing.

A scraper returning [] is indistinguishable from a platform with no openings, so
this failure is silent by construction. It was found by reading the archive by
source: bamboohr and paylocity were absent across a full week, against 13,755
and 9,325 confirmed-live companies. Two different causes, both worth catching --
bamboohr is switched off by a config flag that defaults to False, and paylocity
had no scraper at all while its companies were harvested and probed anyway.

Everything here is hermetic: the archive and the confirmed-live files are built
in tmp_path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_platform_yield as y  # noqa: E402


# ── the verdict ─────────────────────────────────────────────────────────────

def test_coverage_with_no_jobs_is_the_alarm():
    assert y.assess(live=13755, jobs=0, min_live=100) == y.SILENT


def test_coverage_with_jobs_is_fine():
    assert y.assess(live=13755, jobs=1, min_live=100) == y.OK


def test_a_platform_with_little_coverage_cannot_alarm():
    """Six confirmed-live companies having no openings this week proves
    nothing. Alarming on it would fire constantly and get ignored."""
    assert y.assess(live=6, jobs=0, min_live=100) == y.NO_COVERAGE


def test_the_boundary_is_inclusive_of_the_threshold():
    assert y.assess(live=100, jobs=0, min_live=100) == y.SILENT
    assert y.assess(live=99, jobs=0, min_live=100) == y.NO_COVERAGE


# ── counting ────────────────────────────────────────────────────────────────

def _jobs(records, platforms):
    return {p: c["jobs"] for p, c in y.field_coverage(records, platforms).items()}


def test_jobs_are_counted_by_source_platform():
    records = [
        {"_source_site": "greenhouse"}, {"_source_site": "greenhouse"},
        {"_source_site": "lever"},
    ]
    assert _jobs(records, ["greenhouse", "lever", "ashby"]) == {
        "greenhouse": 2, "lever": 1, "ashby": 0}


def test_the_watcher_send_path_is_not_counted():
    """Those records carry a presentation label ("LinkedIn"), not a platform
    key. Counting them would mask a dead ATS scraper behind unrelated sources."""
    records = [{"_source_site": "LinkedIn"}, {"_source_site": "Indeed/CAE"},
               {"_source_site": "greenhouse"}]
    assert _jobs(records, ["greenhouse"]) == {"greenhouse": 1}


def test_source_matching_is_case_insensitive():
    """The archive holds both "iCIMS" and "icims" depending on the writer."""
    records = [{"_source_site": "iCIMS"}, {"_source_site": "icims"}]
    assert _jobs(records, ["icims"]) == {"icims": 2}


def test_malformed_records_do_not_stop_the_count():
    records = [None, "junk", {"_source_site": None}, {}, {"_source_site": "lever"}]
    assert _jobs(records, ["lever"]) == {"lever": 1}


# ── reading the confirmed-live store ────────────────────────────────────────

def test_confirmed_live_reads_the_store(tmp_path):
    (tmp_path / "greenhouse.json").write_text(
        json.dumps({"a": "2026-09-01", "b": "2026-09-01"}), encoding="utf-8")
    assert y.confirmed_live("greenhouse", tmp_path) == 2


def test_a_missing_or_corrupt_store_reads_as_no_coverage(tmp_path):
    """It must not raise, and it must not claim coverage it cannot see --
    claiming coverage would turn a missing file into a false alarm."""
    assert y.confirmed_live("nosuchplatform", tmp_path) == 0
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert y.confirmed_live("broken", tmp_path) == 0


# ── end to end ──────────────────────────────────────────────────────────────

@pytest.fixture()
def archive(tmp_path, monkeypatch):
    """A one-day archive and a confirmed-live store, both in tmp_path."""
    monkeypatch.setattr(y, "CHECKED_DIR", tmp_path)
    (tmp_path / "greenhouse.json").write_text(
        json.dumps({f"c{i}": "2026-09-01" for i in range(500)}), encoding="utf-8")
    (tmp_path / "paylocity.json").write_text(
        json.dumps({f"c{i}": "2026-09-01" for i in range(500)}), encoding="utf-8")
    return tmp_path


def test_a_silent_platform_makes_the_run_fail(archive, monkeypatch, capsys):
    import services.jba.merge_data as merge

    monkeypatch.setattr(merge, "load_daily_log",
                        lambda d: [{"_source_site": "greenhouse"}])
    rc = y.main(["--days", "1", "--min-live", "100"])
    out = capsys.readouterr().out
    assert rc == 1, "a silent platform did not fail the run"
    assert "paylocity" in out.split("Silent:")[1]


def test_every_platform_yielding_makes_the_run_pass(archive, monkeypatch, capsys):
    import services.jba.merge_data as merge

    monkeypatch.setattr(merge, "load_daily_log", lambda d: [
        {"_source_site": "greenhouse"}, {"_source_site": "paylocity"}])
    assert y.main(["--days", "1", "--min-live", "100"]) == 0
    assert "Silent" not in capsys.readouterr().out


def test_an_unreadable_day_does_not_hide_the_rest(archive, monkeypatch, capsys):
    """One corrupt zip must not silence the whole report -- that would turn a
    read error into "every platform is fine"."""
    import services.jba.merge_data as merge

    def flaky(date_key):
        raise OSError("bad zip")

    monkeypatch.setattr(merge, "load_daily_log", flaky)
    rc = y.main(["--days", "1", "--min-live", "100"])
    assert rc == 1, "unreadable archive reported everything as healthy"


# ── description coverage ────────────────────────────────────────────────────
#
# job_service matches semantically against "description", so a platform
# supplying none is invisible to matching rather than merely sparse. Measured
# over one week: teamtailor, recruitee, rippling, jazzhr, workable and jobvite
# at 100%, while workday, greenhouse, ashby, icims, lever and smartrecruiters --
# 74% of all archived ATS jobs -- sit between 1% and 6%.

def _described(records, platforms):
    cov = y.field_coverage(records, platforms, ("description",))
    return {p: (c["jobs"], c["description"]) for p, c in cov.items()}


def test_description_coverage_is_counted_per_platform():
    records = [
        {"_source_site": "greenhouse", "description": "Build things."},
        {"_source_site": "greenhouse", "description": ""},
        {"_source_site": "greenhouse"},
        {"_source_site": "lever", "description": "Ship things."},
    ]
    assert _described(records, ["greenhouse", "lever", "ashby"]) == {
        "greenhouse": (3, 1), "lever": (1, 1), "ashby": (0, 0)}


def test_a_whitespace_only_description_does_not_count():
    """"   " is the same as absent to a matcher, and counting it would report
    a platform as covered while it supplies nothing readable."""
    records = [{"_source_site": "lever", "description": "   \n  "}]
    assert _described(records, ["lever"]) == {"lever": (1, 0)}


def test_the_total_counts_a_job_whether_or_not_it_has_a_description():
    """The total is the denominator of every percentage in the report, so a
    job missing the field must still be counted as a job."""
    records = [{"_source_site": "lever", "description": "x"},
               {"_source_site": "lever"}, {"_source_site": "LinkedIn"}]
    assert _described(records, ["lever"]) == {"lever": (2, 1)}


def test_description_pct_of_a_platform_with_no_jobs_is_zero():
    """Not a division by zero, and not 100% -- a platform with no jobs has no
    description coverage to report."""
    assert y.description_pct(0, 0) == 0.0
    assert y.description_pct(4, 1) == 25.0


def test_thin_descriptions_are_reported_but_do_not_fail_the_run(
        archive, monkeypatch, capsys):
    """A thin description is a quality problem; a silent platform is a broken
    one. Collapsing them would make the exit status useless for gating."""
    import services.jba.merge_data as merge

    monkeypatch.setattr(merge, "load_daily_log", lambda d: [
        {"_source_site": "greenhouse"},
        {"_source_site": "paylocity", "description": "Build things."}])
    rc = y.main(["--days", "1", "--min-live", "100"])
    out = capsys.readouterr().out
    assert rc == 0, "a thin description failed the run"
    assert "Thin description" in out
    assert "greenhouse" in out.split("Thin description")[1]


def test_a_silent_platform_is_not_also_reported_as_thin(
        archive, monkeypatch, capsys):
    """0% of nothing is not a description problem, and listing it twice buries
    the real finding."""
    import services.jba.merge_data as merge

    monkeypatch.setattr(merge, "load_daily_log",
                        lambda d: [{"_source_site": "greenhouse", "description": "x"}])
    y.main(["--days", "1", "--min-live", "100"])
    out = capsys.readouterr().out
    assert "paylocity" in out.split("Silent:")[1]
    assert "Thin description" not in out


def test_a_well_described_platform_is_not_flagged(archive, monkeypatch, capsys):
    import services.jba.merge_data as merge

    monkeypatch.setattr(merge, "load_daily_log", lambda d: [
        {"_source_site": "greenhouse", "description": "x"},
        {"_source_site": "paylocity", "description": "y"}])
    assert y.main(["--days", "1", "--min-live", "100"]) == 0
    assert "Thin descriptions" not in capsys.readouterr().out


# ── location and date coverage ──────────────────────────────────────────────
#
# Three fields, three different consequences when blank, and two of them fail in
# opposite directions -- which is why they are reported separately rather than
# as one "incomplete" number:
#
#   location    blank -> _matches_location returns False, so the job is DROPPED
#               from every location-scoped search. Measured: iCIMS at 2%.
#   date_posted blank -> _posting_age_ok returns True ("rows with no date_posted
#               always pass"), so the job is EXEMPT from the age filter and a
#               stale posting reads as fresh. Measured: Ashby at 4%.

def test_all_three_fields_are_counted_in_one_pass():
    records = [
        {"_source_site": "ashby", "description": "d", "location": "Toronto",
         "date_posted": "2026-09-01"},
        {"_source_site": "ashby", "location": "Berlin"},
    ]
    cov = y.field_coverage(records, ["ashby"])
    assert cov["ashby"] == {"jobs": 2, "description": 1, "location": 2,
                            "date_posted": 1}


def test_the_tracked_fields_are_the_ones_the_pipeline_acts_on():
    """Dropping one silently stops reporting a whole failure mode."""
    assert set(y.TRACKED_FIELDS) == {"description", "location", "date_posted"}
    for field in y.TRACKED_FIELDS:
        assert field in y.FIELD_CONSEQUENCE, f"{field} has no stated consequence"


def test_each_consequence_says_what_actually_happens():
    """The two directions must not read the same.

    Someone told only "location is thin" goes looking for missing jobs; someone
    told only "date is thin" needs to know stale jobs are being shown instead.

    This used to assert the word DROPPED, which is how the location text got to
    claim the opposite of what the code does: job_match._matches_channel_region
    ends `return country is None` and job_service.filter_rows_by_region keeps an
    unknown region explicitly for ATS rows. Pinning a word was a proxy for
    pinning a meaning, and the proxy held while the meaning was false. So the
    absence of that claim is now what is pinned.
    """
    location = y.FIELD_CONSEQUENCE["location"]
    assert "DROPPED" not in location, (
        "both region gates keep an unplaceable location; saying otherwise "
        "defends the floor with a consequence that does not happen"
    )
    assert "kept" in location
    assert "EXEMPT" in y.FIELD_CONSEQUENCE["date_posted"]
    assert location != y.FIELD_CONSEQUENCE["date_posted"]


def test_a_thin_location_is_reported_with_its_own_consequence(
        archive, monkeypatch, capsys):
    import services.jba.merge_data as merge

    monkeypatch.setattr(merge, "load_daily_log", lambda d: [
        {"_source_site": "greenhouse", "description": "d", "date_posted": "x"},
        {"_source_site": "paylocity", "description": "d", "location": "Toronto",
         "date_posted": "x"}])
    rc = y.main(["--days", "1", "--min-live", "100"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Thin location" in out and "greenhouse" in out.split("Thin location")[1]
    assert "kept by both region gates" in out


def test_a_thin_date_is_reported_as_an_exemption_not_a_loss(
        archive, monkeypatch, capsys):
    import services.jba.merge_data as merge

    monkeypatch.setattr(merge, "load_daily_log", lambda d: [
        {"_source_site": "greenhouse", "description": "d", "location": "Toronto"},
        {"_source_site": "paylocity", "description": "d", "location": "Toronto",
         "date_posted": "x"}])
    y.main(["--days", "1", "--min-live", "100"])
    out = capsys.readouterr().out
    assert "Thin date_posted" in out
    assert "EXEMPT" in out


def test_a_platform_complete_on_every_field_is_not_flagged(
        archive, monkeypatch, capsys):
    import services.jba.merge_data as merge

    full = {"description": "d", "location": "Toronto", "date_posted": "x"}
    monkeypatch.setattr(merge, "load_daily_log", lambda d: [
        dict(full, _source_site="greenhouse"),
        dict(full, _source_site="paylocity")])
    assert y.main(["--days", "1", "--min-live", "100"]) == 0
    assert "Thin" not in capsys.readouterr().out


# ── the unvalidated backlog ─────────────────────────────────────────────────
#
# The bot skips dead slugs but scrapes everything else, so a company that has
# never been resolved either way costs a request on every scrape cycle until a
# validation pass reaches it. Measured across the fleet: 19%, concentrated in
# platforms that simply had not been run -- workable 90% (its quota is spent),
# recruitee 73%, applicantpro 65%, jazzhr 45%.
#
# A backlog, not a fault: the fix is a run, not a code change, so it is
# reported and never fatal.

def test_unvalidated_counts_candidates_in_neither_store(tmp_path):
    (tmp_path / "checked").mkdir()
    (tmp_path / "dead").mkdir()
    (tmp_path / "checked" / "lever.json").write_text(
        json.dumps({"known-live": "2026-09-01"}), encoding="utf-8")
    (tmp_path / "dead" / "lever.json").write_text(
        json.dumps({"known-dead": "2026-09-01"}), encoding="utf-8")

    n = y.unvalidated("lever", ["known-live", "known-dead", "never-asked"],
                      checked_dir=tmp_path / "checked", dead_dir=tmp_path / "dead")
    assert n == 1


def test_a_dead_slug_counts_as_validated(tmp_path):
    """A dead mark is an answer. Counting it as outstanding would report the
    backlog as permanently unclearable, since dead slugs are never re-probed
    until their recheck window."""
    (tmp_path / "checked").mkdir()
    (tmp_path / "dead").mkdir()
    (tmp_path / "dead" / "lever.json").write_text(
        json.dumps({"gone": "2026-09-01"}), encoding="utf-8")
    assert y.unvalidated("lever", ["gone"], checked_dir=tmp_path / "checked",
                         dead_dir=tmp_path / "dead") == 0


def test_missing_stores_mean_everything_is_outstanding(tmp_path):
    """A platform that has never been validated has no files at all, and that
    is exactly the case worth reporting -- it must not read as zero."""
    (tmp_path / "checked").mkdir()
    (tmp_path / "dead").mkdir()
    assert y.unvalidated("nosuch", ["a", "b", "c"],
                         checked_dir=tmp_path / "checked",
                         dead_dir=tmp_path / "dead") == 3


def test_a_corrupt_store_does_not_hide_the_backlog(tmp_path):
    (tmp_path / "checked").mkdir()
    (tmp_path / "dead").mkdir()
    (tmp_path / "checked" / "lever.json").write_text("{bad", encoding="utf-8")
    assert y.unvalidated("lever", ["a"], checked_dir=tmp_path / "checked",
                         dead_dir=tmp_path / "dead") == 1


def test_both_store_shapes_are_read(tmp_path):
    """dead_slugs is a {slug: date} map; some stores are plain lists."""
    (tmp_path / "checked").mkdir()
    (tmp_path / "dead").mkdir()
    (tmp_path / "checked" / "lever.json").write_text(
        json.dumps(["a"]), encoding="utf-8")
    (tmp_path / "dead" / "lever.json").write_text(
        json.dumps({"b": "2026-09-01"}), encoding="utf-8")
    assert y.unvalidated("lever", ["a", "b", "c"],
                         checked_dir=tmp_path / "checked",
                         dead_dir=tmp_path / "dead") == 1


def test_the_backlog_is_reported_but_never_fatal(archive, monkeypatch, capsys):
    import services.jba.merge_data as merge

    monkeypatch.setattr(merge, "load_daily_log", lambda d: [
        {"_source_site": "greenhouse", "description": "d", "location": "L",
         "date_posted": "x"},
        {"_source_site": "paylocity", "description": "d", "location": "L",
         "date_posted": "x"}])
    monkeypatch.setattr(y, "DEAD_DIR", archive)
    monkeypatch.setattr(y, "_candidate_loader",
                        lambda: (lambda p: [f"{p}-never-asked"]))
    rc = y.main(["--days", "1", "--min-live", "100"])
    out = capsys.readouterr().out
    assert rc == 0, "a backlog failed the run"
    assert "Unvalidated:" in out
    assert "a run, not a change" in out


def test_no_backlog_prints_nothing(archive, monkeypatch, capsys):
    import services.jba.merge_data as merge

    monkeypatch.setattr(merge, "load_daily_log", lambda d: [
        {"_source_site": "greenhouse", "description": "d", "location": "L",
         "date_posted": "x"},
        {"_source_site": "paylocity", "description": "d", "location": "L",
         "date_posted": "x"}])
    monkeypatch.setattr(y, "_candidate_loader", lambda: (lambda p: []))
    y.main(["--days", "1", "--min-live", "100"])
    assert "Unvalidated:" not in capsys.readouterr().out


def test_an_unreadable_store_yields_no_keys_at_all(tmp_path):
    """Asserted on the loader, not through unvalidated().

    Going through unvalidated() only proves a *particular* candidate is still
    counted, so a mutant returning some other non-empty set survives. The
    property that matters is that a store which cannot be read contributes
    nothing, because anything it did contribute would be invented.
    """
    (tmp_path / "missing.json")  # never created
    assert y._load_keys(tmp_path / "missing.json") == set()

    (tmp_path / "corrupt.json").write_text("{not json", encoding="utf-8")
    assert y._load_keys(tmp_path / "corrupt.json") == set()

    (tmp_path / "wrong-shape.json").write_text("42", encoding="utf-8")
    assert y._load_keys(tmp_path / "wrong-shape.json") == set()

    (tmp_path / "good.json").write_text(
        json.dumps({"a": "2026-09-01"}), encoding="utf-8")
    assert y._load_keys(tmp_path / "good.json") == {"a"}
