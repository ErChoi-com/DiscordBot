"""log_jobs() must apply the same dedup rule the scrape path applies.

There are two dedup layers on the write side -- the current week (jobs table)
and older weeks (zip archive index) -- and both used to suppress on the base key
alone, discarding date_posted. That silently contradicted archive_index.collides,
which the scrape path uses, and the contradiction was self-sustaining: a genuine
re-post passed the scrape filter, went out to Discord, was refused by log_jobs,
and therefore was never recorded anywhere that would stop the next cycle from
surfacing it again.

These tests drive the real log_jobs against a real SQLite DB and real zip
archives. Asserting on a mocked lookup would not have caught the original bug,
since the lookup itself returned the right data -- the caller threw it away.
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services.jba import archive_index, merge_data


def _iso(days_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _day(days_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _job(url: str, *, posted: str | None, site: str = "greenhouse") -> dict:
    job = {
        "title": "Engineer",
        "company": "acme",
        "location": "Toronto, ON",
        "job_url": url,
        "_source_site": site,
        "scraped_at": _iso(),
    }
    if posted is not None:
        job["date_posted"] = posted
    return job


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Real jobs.db + real archive dir, both isolated from the live data."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(merge_data, "_JOBS_DIR", jobs_dir)
    monkeypatch.setattr(merge_data, "_DB_PATH", jobs_dir / "jobs.db")
    monkeypatch.setattr(archive_index, "_JOBS_DIR", jobs_dir)
    monkeypatch.setattr(archive_index, "_INDEX_PATH", jobs_dir / "archive_index.db")
    # _get_conn caches per thread; a connection to a previous test's DB would
    # make every assertion here meaningless.
    monkeypatch.setattr(merge_data._local, "conn", None, raising=False)
    # Rollover is not under test and would move rows out from under it.
    monkeypatch.setattr(merge_data, "_archive_old_weeks", lambda *_: None)
    monkeypatch.setattr(merge_data, "_consolidate_old_months", lambda *_: None)
    return jobs_dir


def _archive(jobs_dir: Path, month: str, day: str, records: list[dict]) -> None:
    month_dir = jobs_dir / month
    month_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(month_dir / f"{month}.zip", "w") as zf:
        zf.writestr(f"{day}.json", json.dumps(records))
    archive_index.ensure_index()


def _logged(url: str) -> list[dict]:
    rows = merge_data._get_conn().execute(
        "SELECT data FROM jobs WHERE dedup_key = ?", (url,)
    ).fetchall()
    return [json.loads(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# The regression: a re-post must reach the archive
# ---------------------------------------------------------------------------

def test_repost_with_a_new_date_is_logged_despite_an_older_archived_sighting(store):
    """The bug: suppressed on base key alone, so this was rejected forever while
    the scrape path kept re-surfacing it every cycle."""
    old = _job("https://x/1", posted=_iso(100))
    old["scraped_at"] = _iso(100)
    _archive(store, _day(100)[:7], _day(100), [old])

    fresh = _job("https://x/1", posted=_iso(1))
    assert merge_data.log_jobs([fresh], date_str=_day()) == 1
    assert len(_logged("https://x/1")) == 1


def test_identical_archived_listing_is_still_rejected(store):
    posted = _iso(3)
    archived = _job("https://x/1", posted=posted)
    archived["scraped_at"] = _iso(3)
    _archive(store, _day(3)[:7], _day(3), [archived])

    assert merge_data.log_jobs([_job("https://x/1", posted=posted)], date_str=_day()) == 0
    assert _logged("https://x/1") == []


def test_undated_incoming_does_not_slip_past_an_archived_sighting(store):
    """bamboohr is a dateless platform, so it clears the freshness gate; the
    collision rule is what must stop it."""
    archived = _job("https://x/1", posted=_iso(3), site="bamboohr")
    archived["scraped_at"] = _iso(3)
    _archive(store, _day(3)[:7], _day(3), [archived])

    assert merge_data.log_jobs([_job("https://x/1", posted=None, site="bamboohr")], date_str=_day()) == 0


# ---------------------------------------------------------------------------
# The current-week layer (jobs table), which had the same defect
# ---------------------------------------------------------------------------

def test_repost_is_logged_when_the_prior_sighting_is_in_the_current_week(store):
    assert merge_data.log_jobs([_job("https://x/1", posted=_iso(6))], date_str=_day(2)) == 1
    assert merge_data.log_jobs([_job("https://x/1", posted=_iso(1))], date_str=_day()) == 1
    assert len(_logged("https://x/1")) == 2


def test_same_date_repeat_on_a_later_day_is_rejected(store):
    posted = _iso(4)
    assert merge_data.log_jobs([_job("https://x/1", posted=posted)], date_str=_day(2)) == 1
    assert merge_data.log_jobs([_job("https://x/1", posted=posted)], date_str=_day()) == 0
    assert len(_logged("https://x/1")) == 1


def test_duplicate_identities_within_one_batch_are_collapsed(store):
    posted = _iso(2)
    batch = [_job("https://x/1", posted=posted), _job("https://x/1", posted=posted)]
    assert merge_data.log_jobs(batch, date_str=_day()) == 1


# ---------------------------------------------------------------------------
# The two paths must agree -- the property the bug violated
# ---------------------------------------------------------------------------

def test_scrape_filter_and_log_jobs_reach_the_same_verdict(store):
    """Anything filter_new_listings keeps, log_jobs must write. Otherwise the
    kept listing is announced but never recorded, and recurs every cycle."""
    # One pinned value: _iso() is microsecond-precise, so calling it twice would
    # yield two genuinely different dates and the "duplicate" would not be one.
    original_date = _iso(100)
    archived = _job("https://x/1", posted=original_date)
    archived["scraped_at"] = _iso(100)
    _archive(store, _day(100)[:7], _day(100), [archived])

    incoming = [
        _job("https://x/1", posted=original_date),   # exact duplicate -> both drop
        _job("https://x/1", posted=_iso(1)),     # re-post         -> both keep
        _job("https://x/2", posted=_iso(1)),     # unseen          -> both keep
    ]
    kept, _ = archive_index.filter_new_listings(incoming)
    written = merge_data.log_jobs(kept, date_str=_day())

    assert [j["job_url"] for j in kept] == ["https://x/1", "https://x/2"]
    assert written == len(kept)
