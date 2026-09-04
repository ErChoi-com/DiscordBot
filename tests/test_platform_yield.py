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

def test_jobs_are_counted_by_source_platform():
    records = [
        {"_source_site": "greenhouse"}, {"_source_site": "greenhouse"},
        {"_source_site": "lever"},
    ]
    counts = y.count_by_platform(records, ["greenhouse", "lever", "ashby"])
    assert counts == {"greenhouse": 2, "lever": 1, "ashby": 0}


def test_the_watcher_send_path_is_not_counted():
    """Those records carry a presentation label ("LinkedIn"), not a platform
    key. Counting them would mask a dead ATS scraper behind unrelated sources."""
    records = [{"_source_site": "LinkedIn"}, {"_source_site": "Indeed/CAE"},
               {"_source_site": "greenhouse"}]
    assert y.count_by_platform(records, ["greenhouse"]) == {"greenhouse": 1}


def test_source_matching_is_case_insensitive():
    """The archive holds both "iCIMS" and "icims" depending on the writer."""
    records = [{"_source_site": "iCIMS"}, {"_source_site": "icims"}]
    assert y.count_by_platform(records, ["icims"]) == {"icims": 2}


def test_malformed_records_do_not_stop_the_count():
    records = [None, "junk", {"_source_site": None}, {}, {"_source_site": "lever"}]
    assert y.count_by_platform(records, ["lever"]) == {"lever": 1}


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
