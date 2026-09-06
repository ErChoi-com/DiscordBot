"""Glassdoor rows must carry the date the card already told us.

The card's age was parsed only to filter on it and then discarded, so every
Glassdoor row reached the pipeline with no date_posted at all. The job watcher
backfills a missing date with the day it is archiving
(src/watchers/manager.py), so the archive recorded a *fabricated* posting date
for every Glassdoor listing -- a job the card called "7d" old stored as posted
today. merge_data._collides keys repost detection on that field, and
_is_fresh_enough re-checks it on write, so a wrong value is not inert.

Glassdoor was the only source doing this. ZipRecruiter recovers real
timestamps from rollingPostedAtUtc.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_service
from services.job_service import glassdoor_card_age_hours


def _card(age_text: str):
    html = f"""
    <div class="jobCard">
      <a class="jobTitle" href="/job-listing/x-JV_IC1.htm">Software Intern</a>
      <span class="EmployerProfile_compactEmployerName">Acme</span>
      <div class="jobLocation">Toronto, ON</div>
      <div class="listing-age">{age_text}</div>
    </div>"""
    return BeautifulSoup(html, "html.parser").select_one(".jobCard")


def test_the_card_age_is_read_the_way_the_scraper_reads_it():
    """Pins the units the date conversion depends on."""
    assert glassdoor_card_age_hours(_card("24h")) == 24
    assert glassdoor_card_age_hours(_card("3d")) == 72
    assert glassdoor_card_age_hours(_card("30d+")) == 720
    assert glassdoor_card_age_hours(_card("Just Posted")) == 0
    assert glassdoor_card_age_hours(_card("Salary estimate")) is None


def _expected(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")


def test_a_dated_card_becomes_a_dated_row(monkeypatch):
    rows = _scrape(monkeypatch, ["24h", "3d", "Just Posted"], hours_old=168)
    assert [r.get("date_posted") for r in rows] == [
        _expected(24), _expected(72), _expected(0)
    ]


def test_an_undated_card_is_left_undated_rather_than_guessed(monkeypatch):
    """No date is honestly 'unknown' and is handled downstream. A guess is
    indistinguishable from a real date once it is in the archive."""
    rows = _scrape(monkeypatch, ["Salary estimate"], hours_old=168)
    assert len(rows) == 1
    assert not rows[0].get("date_posted")


def test_the_date_is_not_todays_date_for_an_old_posting(monkeypatch):
    """The actual regression: a 7-day-old listing must not be recorded as
    posted today, which is what the watcher's backfill did to every row."""
    rows = _scrape(monkeypatch, ["7d"], hours_old=168)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert rows[0]["date_posted"] != today
    assert rows[0]["date_posted"] == _expected(168)


def test_the_age_filter_still_applies(monkeypatch):
    """Recording the date must not stop the scraper dropping stale cards."""
    rows = _scrape(monkeypatch, ["2h", "10d"], hours_old=24)
    assert [r["date_posted"] for r in rows] == [_expected(2)]


def _scrape(monkeypatch, ages: list[str], hours_old: int):
    """Drive the real scrape_glassdoor_postings against a fabricated page."""
    cards = "".join(
        f"""<div class="jobCard">
              <a class="jobTitle" href="/job-listing/job{i}-JV_IC{i}.htm">Software Intern {i}</a>
              <span class="EmployerProfile_compactEmployerName">Acme {i}</span>
              <div class="jobLocation">Toronto, ON</div>
              <div class="listing-age">{age}</div>
            </div>"""
        for i, age in enumerate(ages)
    )
    page = f"<html><body>{cards}</body></html>"

    class _Resp:
        status_code = 200
        content = page.encode("utf-8")
        text = page
        apparent_encoding = "utf-8"
        encoding = "utf-8"

    class _Session:
        headers: dict = {}
        def get(self, *a, **k):
            return _Resp()
        def close(self):
            pass

    monkeypatch.setattr(job_service.requests, "Session", lambda: _Session())
    monkeypatch.setattr(
        job_service, "glassdoor_location_from_detail_page",
        lambda session, job_url, fallback: fallback,
    )
    monkeypatch.setattr(
        "services.jba.geo_db.resolve_glassdoor_location",
        lambda location, session: ("C2275123", "C"),
    )
    return job_service.scrape_glassdoor_postings(
        "software intern", "Toronto, ON", hours_old=hours_old,
        radius_miles=25, country_indeed="canada", results_wanted=20,
    )
