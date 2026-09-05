"""The README's platform count has to match what actually ships.

It has now been stale twice in one day: "sixteen" survived BambooHR being
switched on, and would have survived Oracle Cloud Recruiting and Personio being
added too. A reader trusting a stale count builds against the wrong fleet, and
the count is exactly the kind of prose nothing else checks.

Deliberately string-only: this asserts what the document says, not what the
scraper registry holds, so it stays meaningful on a checkout where the two have
drifted apart.
"""
from __future__ import annotations

from pathlib import Path

import pytest

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


def test_the_stale_counts_are_gone():
    assert "sixteen job-board platforms" not in README
    assert "scrapes sixteen ATS" not in README


def test_the_current_count_is_stated():
    assert "eighteen job-board platforms" in README
    assert "scrapes eighteen ATS" in README


@pytest.mark.parametrize("name", [
    "Greenhouse", "Lever", "Ashby", "Workday", "iCIMS", "BambooHR", "Workable",
    "Breezy", "SmartRecruiters", "Recruitee", "TeamTailor", "Rippling",
    "JazzHR", "Jobvite", "ApplicantPro", "Paylocity",
    "Oracle Cloud Recruiting", "Personio",
])
def test_every_platform_is_named(name):
    assert name in README


def test_the_two_unusual_identifiers_are_explained():
    """Oracle is keyed by a host and Personio by a subdomain that may be all
    digits. Both surprised this codebase, and an unexplained identifier shape
    is what sends the next reader looking for a bug that is not there.
    """
    assert "eeho.fa.us2.oraclecloud.com" in README
    assert "siteNumber" in README
    assert "workzag-jobs" in README


def test_the_bamboohr_reason_is_the_measured_one():
    assert "is Cloudflare-gated and needs a" not in README
    assert "zero** challenge pages" in README
