"""ATS duplicate elimination: same job + same date_posted, within ~4 months.

The rule under test:
  * same base job AND same date_posted, inside the window -> duplicate, dropped
  * same base job, DIFFERENT date_posted                  -> new posting, kept
  * incoming listing with NO date_posted                  -> collides with any
    prior sighting of that job in the window
  * outside the window                                    -> free to reappear

Builds REAL zip archives in the production layout and a REAL SQLite index rather
than mocking the lookup: a mock would not catch a wrong dedup key, a broken
window comparison, or the asymmetry of the undated rule.
"""
from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from services import ats_service
from services.jba import archive_index, merge_data


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _job(url: str, *, posted: str | None = "2026-07-01", seen_days_ago: int = 1,
         company: str = "acme", site: str = "greenhouse", title: str = "Engineer") -> dict:
    job = {
        "title": title,
        "company": company,
        "location": "Toronto, ON",
        "job_url": url,
        "_source_site": site,
        "_source_sites": [site],
        "scraped_at": _iso(seen_days_ago),
    }
    if posted is not None:
        job["date_posted"] = posted
    return job


def _write_archive(jobs_dir: Path, month: str, records: list[dict],
                   seen_urls: list[dict] | None = None) -> Path:
    month_dir = jobs_dir / month
    month_dir.mkdir(parents=True, exist_ok=True)
    path = month_dir / f"{month}.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{month}-01.json", json.dumps(records))
        if seen_urls is not None:
            zf.writestr("seen_urls.json", json.dumps(seen_urls))
    return path


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """Isolated jobs dir + index, with the live jobs.db absent by default."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(archive_index, "_JOBS_DIR", jobs_dir)
    monkeypatch.setattr(archive_index, "_INDEX_PATH", jobs_dir / "archive_index.db")
    monkeypatch.setattr(merge_data, "_DB_PATH", jobs_dir / "absent-jobs.db")
    return jobs_dir


# ---------------------------------------------------------------------------
# The core rule
# ---------------------------------------------------------------------------

def test_same_job_same_date_inside_window_is_dropped(archive):
    _write_archive(archive, "2026-06", [_job("https://x/1", posted="2026-06-01", seen_days_ago=10)])
    archive_index.ensure_index()

    kept, dropped = archive_index.filter_new_listings([_job("https://x/1", posted="2026-06-01")])

    assert (dropped, kept) == (1, [])


def test_same_job_different_date_is_a_new_posting(archive):
    """A re-post carrying a different date_posted is genuinely new."""
    _write_archive(archive, "2026-06", [_job("https://x/1", posted="2026-06-01", seen_days_ago=10)])
    archive_index.ensure_index()

    fresh = _job("https://x/1", posted="2026-08-01")
    kept, dropped = archive_index.filter_new_listings([fresh])

    assert dropped == 0
    assert kept == [fresh]


def test_incoming_listing_without_a_date_collides_with_any_prior_sighting(archive):
    """An undated listing carries no evidence of being new, so it must not slip
    past a prior sighting of the same job."""
    _write_archive(archive, "2026-06", [_job("https://x/1", posted="2026-06-01", seen_days_ago=10)])
    archive_index.ensure_index()

    undated = _job("https://x/1", posted=None)
    kept, dropped = archive_index.filter_new_listings([undated])

    assert (dropped, kept) == (1, [])


def test_undated_sighting_does_not_suppress_a_dated_listing(archive):
    """Roughly 10% of archived job records carry no date_posted (Ashby and
    BambooHR never supply one). Those undated sightings must not match any date,
    or every re-post from those platforms would be suppressed."""
    _write_archive(archive, "2026-06", [_job("https://x/1", posted=None, seen_days_ago=10)])
    archive_index.ensure_index()

    dated = _job("https://x/1", posted="2026-08-01")
    kept, dropped = archive_index.filter_new_listings([dated])

    assert dropped == 0, "an undated archive sighting suppressed a dated re-post"
    assert kept == [dated]


def test_undated_incoming_still_collides_with_an_undated_sighting(archive):
    _write_archive(archive, "2026-06", [_job("https://x/1", posted=None, seen_days_ago=10)])
    archive_index.ensure_index()

    kept, dropped = archive_index.filter_new_listings([_job("https://x/1", posted=None)])
    assert (dropped, kept) == (1, [])


def test_seen_urls_entries_are_ignored(archive):
    """seen_urls.json carries no date_posted, so indexing it would put every
    entry in the undated bucket where it can never take part in the date rule.
    The job records are the source of truth."""
    _write_archive(
        archive, "2026-06", [],
        seen_urls=[{"url": "https://x/only-in-seen-urls", "first_seen": _iso(10)}],
    )
    archive_index.ensure_index()

    assert archive_index.recent_sightings(["https://x/only-in-seen-urls"]) == {}


# ---------------------------------------------------------------------------
# The ~4 month window
# ---------------------------------------------------------------------------

def test_sighting_older_than_the_window_does_not_suppress(archive):
    """Past ~4 months the job is free to appear again."""
    stale = archive_index.DEDUP_WINDOW_DAYS + 30
    _write_archive(archive, "2026-01", [_job("https://x/1", posted="2026-01-01", seen_days_ago=stale)])
    archive_index.ensure_index()

    again = _job("https://x/1", posted="2026-01-01")
    kept, dropped = archive_index.filter_new_listings([again])

    assert dropped == 0, "a sighting older than the window still suppressed the job"
    assert kept == [again]


def test_sighting_just_inside_the_window_does_suppress(archive):
    """Boundary companion to the test above -- together they pin the edge."""
    recent = archive_index.DEDUP_WINDOW_DAYS - 5
    _write_archive(archive, "2026-04", [_job("https://x/1", posted="2026-04-01", seen_days_ago=recent)])
    archive_index.ensure_index()

    kept, dropped = archive_index.filter_new_listings([_job("https://x/1", posted="2026-04-01")])
    assert (dropped, kept) == (1, [])


def test_window_cutoff_is_about_four_months():
    cutoff = datetime.fromisoformat(archive_index.window_cutoff())
    days = (datetime.now(timezone.utc) - cutoff).days
    assert 118 <= days <= 126, f"window is {days} days, expected ~4 months"


def test_sighting_with_unknown_first_seen_is_not_treated_as_recent(archive):
    """Without a sighting date there is no evidence it falls inside the window;
    suppressing on that basis would hide the job forever."""
    archive_index.ensure_index()
    conn = archive_index._connect()
    try:
        # first_seen unknown: neither the record nor its day file supplied one.
        archive_index._upsert(conn, [("https://x/1", "2026-06-01", "")])
        conn.commit()
    finally:
        conn.close()

    kept, dropped = archive_index.filter_new_listings([_job("https://x/1", posted="2026-06-01")])
    assert dropped == 0, "a dateless sighting was treated as inside the window"


# ---------------------------------------------------------------------------
# collides(): the rule in isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("posted,seen,expected", [
    ("2026-06-01", set(),                        False),  # nothing seen
    ("2026-06-01", {"2026-06-01"},               True),   # same date
    ("2026-06-01", {"2026-07-01"},               False),  # different date -> new
    ("",           {"2026-06-01"},               True),   # undated incoming
    ("",           {""},                         True),   # undated both sides
    ("2026-06-01", {""},                         False),  # undated sighting only
    ("2026-06-01", {"2026-07-01", "2026-06-01"}, True),   # matches one of many
])
def test_collision_rule(posted, seen, expected):
    assert archive_index.collides(posted, seen) is expected


# ---------------------------------------------------------------------------
# Within a single scrape
# ---------------------------------------------------------------------------

def test_within_batch_same_job_same_date_keeps_the_oldest(archive):
    archive_index.ensure_index()
    batch = [
        _job("https://x/d", posted="2026-06-01", seen_days_ago=1, title="newest"),
        _job("https://x/d", posted="2026-06-01", seen_days_ago=30, title="OLDEST"),
    ]
    kept, dropped = archive_index.filter_new_listings(batch)

    assert dropped == 1
    assert kept[0]["title"] == "OLDEST"


def test_within_batch_different_dates_are_both_kept(archive):
    archive_index.ensure_index()
    batch = [
        _job("https://x/d", posted="2026-06-01"),
        _job("https://x/d", posted="2026-08-01"),
    ]
    kept, dropped = archive_index.filter_new_listings(batch)

    assert dropped == 0
    assert {j["date_posted"] for j in kept} == {"2026-06-01", "2026-08-01"}


def test_within_batch_undated_collides_with_a_dated_sibling(archive):
    archive_index.ensure_index()
    batch = [_job("https://x/d", posted="2026-06-01"), _job("https://x/d", posted=None)]

    kept, dropped = archive_index.filter_new_listings(batch)
    assert dropped == 1
    assert kept[0]["date_posted"] == "2026-06-01"


def test_unkeyable_listings_are_kept(archive):
    archive_index.ensure_index()
    kept, dropped = archive_index.filter_new_listings([_job("")])
    assert dropped == 0 and len(kept) == 1


# ---------------------------------------------------------------------------
# Index mechanics
# ---------------------------------------------------------------------------

def test_workday_duplicates_collapse_on_job_id(archive):
    """Workday URLs differ by tenant path while naming the same job id."""
    base = {
        "title": "Eng", "company": "acme", "location": "T",
        "job_url": "https://acme.wd1.myworkdayjobs.com/en-US/careers/jobs/12345",
        "date_posted": "2026-06-01", "scraped_at": _iso(10), "_source_site": "workday",
    }
    _write_archive(archive, "2026-06", [base])
    archive_index.ensure_index()

    variant = dict(base, job_url="https://acme.wd1.myworkdayjobs.com/en-US/External/jobs/12345")
    kept, dropped = archive_index.filter_new_listings([variant])

    assert (dropped, kept) == (1, [])


def test_reindex_is_skipped_when_nothing_changed(archive):
    _write_archive(archive, "2026-06", [_job("https://x/1")])
    assert archive_index.ensure_index() == 1
    assert archive_index.ensure_index() == 0, "unchanged archive was re-parsed"


def test_changed_archive_is_reindexed(archive):
    path = _write_archive(archive, "2026-06", [_job("https://x/1", posted="2026-06-01", seen_days_ago=5)])
    archive_index.ensure_index()

    _write_archive(archive, "2026-06", [
        _job("https://x/1", posted="2026-06-01", seen_days_ago=5),
        _job("https://x/2", posted="2026-06-02", seen_days_ago=5),
    ])
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10))

    # A check made in the last minute stands (see _CHECK_INTERVAL_S); this
    # test is about the staleness detection behind it, so let the minute pass.
    archive_index._last_checked.clear()
    assert archive_index.ensure_index() == 1
    kept, dropped = archive_index.filter_new_listings([_job("https://x/2", posted="2026-06-02")])
    assert dropped == 1


def test_stale_schema_is_rebuilt_not_trusted(archive):
    """An index from an older layout cannot answer the current query; silently
    keeping it would mean wrong dedup decisions rather than a visible failure."""
    _write_archive(archive, "2026-06", [_job("https://x/1", posted="2026-06-01", seen_days_ago=5)])
    archive_index.ensure_index()

    conn = sqlite3.connect(archive_index._INDEX_PATH)
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    archive_index.ensure_index(force=True)   # must notice the mismatch and rebuild
    kept, dropped = archive_index.filter_new_listings([_job("https://x/1", posted="2026-06-01")])
    assert dropped == 1, "stale-schema index was trusted instead of rebuilt"


def test_corrupt_archive_does_not_abort_the_build(archive):
    (archive / "2026-05").mkdir(parents=True)
    (archive / "2026-05" / "2026-05.zip").write_bytes(b"not a zip")
    _write_archive(archive, "2026-06", [_job("https://x/good", posted="2026-06-01", seen_days_ago=5)])

    archive_index.ensure_index()
    kept, dropped = archive_index.filter_new_listings([_job("https://x/good", posted="2026-06-01")])
    assert dropped == 1


def test_missing_index_fails_open(archive):
    jobs = [_job("https://x/1"), _job("https://x/2")]
    assert archive_index.filter_new_listings(jobs) == (jobs, 0)


def test_index_connection_is_read_only(archive):
    """mode=ro is load-bearing: a read-write connect would let a read create the
    index file."""
    _write_archive(archive, "2026-06", [_job("https://x/1")])
    archive_index.ensure_index()

    conn = archive_index._connect(readonly=True)
    assert conn is not None
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO archive_seen VALUES ('x', '', '')")
            conn.commit()
    finally:
        conn.close()


def test_readonly_open_never_creates_the_index(archive, monkeypatch):
    absent = archive / "no-such-index.db"
    monkeypatch.setattr(archive_index, "_INDEX_PATH", absent)

    assert archive_index._connect(readonly=True) is None
    assert not absent.exists(), "a read created the index database"


def test_lookup_handles_more_keys_than_the_sql_batch_size(archive):
    """recent_sightings batches at 500; a real ATS sweep returns thousands, and a
    broken batch loop would silently miss every duplicate past the first 500."""
    urls = [f"https://x/bulk/{i}" for i in range(1300)]
    _write_archive(archive, "2026-06",
                   [_job(u, posted="2026-06-01", seen_days_ago=5) for u in urls])
    archive_index.ensure_index()

    found = archive_index.recent_sightings(urls)
    assert len(found) == 1300, f"only {len(found)} of 1300 keys resolved"

    kept, dropped = archive_index.filter_new_listings(
        [_job(u, posted="2026-06-01") for u in urls]
    )
    assert (dropped, kept) == (1300, [])


def test_oldest_first_seen_wins_across_archives(archive):
    """The window is measured from the true first sighting, so the earliest one
    must survive regardless of ingest order."""
    _write_archive(archive, "2026-07", [_job("https://x/1", posted="2026-06-01", seen_days_ago=5)])
    _write_archive(archive, "2026-06", [_job("https://x/1", posted="2026-06-01", seen_days_ago=200)])
    archive_index.ensure_index()

    conn = sqlite3.connect(archive_index._INDEX_PATH)
    stored = conn.execute(
        "SELECT first_seen FROM archive_seen WHERE base_key = 'https://x/1'"
    ).fetchone()[0]
    conn.close()

    assert stored < _iso(150), "a later sighting overwrote the original one"


def test_unknown_first_seen_never_clobbers_a_known_one(archive):
    archive_index.ensure_index()
    conn = archive_index._connect()
    try:
        archive_index._upsert(conn, [("k", "2026-06-01", "2026-06-01T00:00:00Z")])
        conn.commit()
        archive_index._upsert(conn, [("k", "2026-06-01", "")])
        conn.commit()
        stored = conn.execute("SELECT first_seen FROM archive_seen WHERE base_key='k'").fetchone()[0]
    finally:
        conn.close()

    assert stored == "2026-06-01T00:00:00Z"


def test_current_week_from_jobs_db_participates_in_the_date_rule(archive, tmp_path, monkeypatch):
    """The zips hold only ARCHIVED weeks; the current week lives in jobs.db. It
    is read from the `jobs` table, not `seen_urls`, so its date_posted is known
    and can distinguish a re-post from a duplicate."""
    db = tmp_path / "jobs.db"
    conn = sqlite3.connect(db)
    merge_data._init_tables(conn)
    conn.execute(
        "INSERT INTO jobs (date_key, dedup_key, scraped_at, data) VALUES (?, ?, ?, ?)",
        ("2026-08-05", "https://x/week", _iso(1),
         json.dumps({"job_url": "https://x/week", "date_posted": "2026-08-01"})),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(merge_data, "_DB_PATH", db)

    archive_index.ensure_index()

    _, dup_dropped = archive_index.filter_new_listings([_job("https://x/week", posted="2026-08-01")])
    assert dup_dropped == 1, "current-week duplicate was not caught"

    _, repost_dropped = archive_index.filter_new_listings([_job("https://x/week", posted="2026-08-20")])
    assert repost_dropped == 0, "a dated re-post was wrongly suppressed"


# ---------------------------------------------------------------------------
# Wiring into the ATS scraper
# ---------------------------------------------------------------------------

def _stub_scraper(rows):
    def _scraper(slug, keywords, location, max_jobs):
        return list(rows)
    return _scraper


def test_scrape_ats_platform_eliminates_duplicates(archive, monkeypatch):
    _write_archive(archive, "2026-06", [_job("https://x/old", posted="2026-06-01", seen_days_ago=10)])
    archive_index.ensure_index()

    monkeypatch.setitem(
        ats_service._SCRAPERS, ats_service.GREENHOUSE,
        _stub_scraper([
            _job("https://x/old", posted="2026-06-01"),   # duplicate
            _job("https://x/old", posted="2026-08-01"),   # re-post, keep
            _job("https://x/new", posted="2026-08-01"),   # novel, keep
        ]),
    )
    monkeypatch.setattr(ats_service, "flush_dead_slugs", lambda: None)

    rows = ats_service.scrape_ats_platform(
        ats_service.GREENHOUSE, "engineer", "Toronto", company_slugs=["acme"],
    )
    assert [(r["job_url"], r["date_posted"]) for r in rows] == [
        ("https://x/old", "2026-08-01"),
        ("https://x/new", "2026-08-01"),
    ]


def test_scrape_returns_everything_when_dedup_disabled(archive, monkeypatch):
    _write_archive(archive, "2026-06", [_job("https://x/old", posted="2026-06-01", seen_days_ago=10)])
    archive_index.ensure_index()
    monkeypatch.setattr(ats_service, "ATS_ARCHIVE_DEDUP_ENABLED", False)

    monkeypatch.setitem(
        ats_service._SCRAPERS, ats_service.GREENHOUSE,
        _stub_scraper([_job("https://x/old", posted="2026-06-01"), _job("https://x/new")]),
    )
    monkeypatch.setattr(ats_service, "flush_dead_slugs", lambda: None)

    rows = ats_service.scrape_ats_platform(
        ats_service.GREENHOUSE, "engineer", "Toronto", company_slugs=["acme"],
    )
    assert len(rows) == 2


def test_scrape_survives_a_broken_archive(archive, monkeypatch):
    """An ATS scrape must never fail because the job archive is unavailable."""
    def _boom(_jobs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(archive_index, "filter_new_listings", _boom)
    monkeypatch.setitem(
        ats_service._SCRAPERS, ats_service.GREENHOUSE, _stub_scraper([_job("https://x/1")]),
    )
    monkeypatch.setattr(ats_service, "flush_dead_slugs", lambda: None)

    rows = ats_service.scrape_ats_platform(
        ats_service.GREENHOUSE, "engineer", "Toronto", company_slugs=["acme"],
    )
    assert len(rows) == 1, "scrape lost its results when the archive failed"
