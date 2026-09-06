"""ATS listings are deduplicated on the provider's job id, not the URL.

Measured on the stored archive before this rule existed: 169,406 of 880,911
records (19.2%) were repeat sightings of a job already recorded -- one posting
under several URLs, and one posting logged again every time its date_posted
moved. Both causes are covered here, through the real log_jobs write path.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.jba import merge_data


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_data, "_JOBS_DIR", tmp_path)
    monkeypatch.setattr(merge_data, "_DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(merge_data._local, "conn", None, raising=False)
    monkeypatch.setattr(merge_data, "_last_archive_week", None, raising=False)
    # The archive layer is exercised in test_archive_index; keep this file on
    # the write path alone.
    monkeypatch.setattr(merge_data, "_already_archived", lambda jobs: {})
    monkeypatch.setattr(merge_data, "_archive_old_weeks", lambda today: None)
    monkeypatch.setattr(merge_data, "_consolidate_old_months", lambda today: None)
    yield
    conn = getattr(merge_data._local, "conn", None)
    if conn is not None:
        conn.close()
        merge_data._local.conn = None


def _recent(days_ago: float) -> str:
    """A date_posted inside log_jobs' 7-day write-freshness window.

    Relative to wall-clock now on purpose: _is_fresh_enough compares against the
    real clock, so hard-coded dates would silently turn every log_jobs call into
    a no-op once they aged past the window.
    """
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _job(url: str, *, title="Engineer", company="acme", site="greenhouse", posted=None):
    return {
        "title": title, "company": company, "job_url": url,
        "date_posted": _recent(1) if posted is None else posted,
        "_source_site": site,
        "scraped_at": "2026-08-20T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# Identity extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,why", [
    ("https://boards.greenhouse.io/acme/jobs/4724387005",
     "https://job-boards.greenhouse.io/acme/jobs/4724387005",
     "greenhouse host migration"),
    ("https://boards.eu.greenhouse.io/acme/jobs/4609619101?gh_jid=4609619101",
     "https://job-boards.eu.greenhouse.io/acme/jobs/4609619101",
     "gh_jid param vs bare path"),
    ("https://stripe.com/jobs/search?gh_jid=8012497",
     "https://boards.greenhouse.io/stripe/jobs/8012497",
     "company-hosted embed vs greenhouse board"),
    ("https://careers-acme.icims.com/jobs/4866/biling-administrative-assistant/job",
     "https://careers-acme.icims.com/jobs/4866/billing-administrative-assistant/job",
     "icims title slug is mutable"),
    ("https://careers-acme.icims.com/jobs/4866/x/job?in_iframe=1",
     "https://careers-acme.icims.com/jobs/4866/x/job",
     "icims query string"),
])
def test_urls_of_one_posting_share_an_identity(a, b, why):
    assert merge_data._dedup_key(_job(a)) == merge_data._dedup_key(_job(b)), why
    assert merge_data._dedup_key(_job(a)).startswith("ats:")


@pytest.mark.parametrize("url,expected", [
    ("https://clunegc.wd12.myworkdayjobs.com/clunegc/job/Chicago-IL/Senior-Specialist_JR10108",
     "ats:workday:clunegc.wd12.myworkdayjobs.com:JR10108"),
    ("https://msk.wd108.myworkdayjobs.com/primary/job/New-York-NY/Line-Cook_100019283",
     "ats:workday:msk.wd108.myworkdayjobs.com:100019283"),
    ("https://jobs.ashbyhq.com/abby-care/f555c8a4-7e9a-4971-b4ed-065f269846ec",
     "ats:ashby:f555c8a4-7e9a-4971-b4ed-065f269846ec"),
    ("https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
     "ats:lever:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
    ("https://acme.bamboohr.com/careers/4242", "ats:bamboohr:acme.bamboohr.com:4242"),
])
def test_provider_ids_are_extracted(url, expected):
    assert merge_data._dedup_key(_job(url)) == expected


def test_per_tenant_ids_stay_host_qualified():
    """icims/workday/bamboohr ids repeat across tenants; dropping the host would
    merge unrelated jobs from different companies."""
    a = merge_data._dedup_key(_job("https://careers-one.icims.com/jobs/4801/x/job"))
    b = merge_data._dedup_key(_job("https://careers-two.icims.com/jobs/4801/x/job"))
    assert a != b


def test_unrecognised_urls_keep_the_url_identity():
    """None from the extractor must mean 'behave exactly as before', never
    'collapse into some catch-all key'."""
    for url in [
        "https://www.linkedin.com/jobs/view/4425828853",
        "https://example.com/careers/engineer",
        "https://careers-acme.icims.com/connect",          # no /jobs/<id>
        "https://jobs.ashbyhq.com/acme/not-a-uuid",
        "https://acme.wd1.myworkdayjobs.com/site/job/City/Title_2",  # too few digits
    ]:
        key = merge_data._dedup_key(_job(url, site="linkedin"))
        assert key == url, f"{url} was rewritten to {key}"


# ---------------------------------------------------------------------------
# The write path
# ---------------------------------------------------------------------------

def test_ats_repost_with_a_new_date_is_not_logged_twice():
    """The 8.6% cause: Greenhouse date_posted is updated_at, which bumps on any
    edit. Same id therefore means same posting, whatever the date says."""
    url = "https://boards.greenhouse.io/acme/jobs/4976140008"
    assert merge_data.log_jobs([_job(url, posted=_recent(3))], "2026-08-20") == 1
    added = merge_data.log_jobs([_job(url, posted=_recent(1))], "2026-08-21")

    assert added == 0, "an edited ATS listing was logged as a new posting"


def test_same_job_under_two_urls_is_logged_once():
    """The 1.3% cause: one posting served from several URLs."""
    assert merge_data.log_jobs(
        [_job("https://boards.greenhouse.io/acme/jobs/123")], "2026-08-20") == 1
    added = merge_data.log_jobs(
        [_job("https://job-boards.greenhouse.io/acme/jobs/123")], "2026-08-21")

    assert added == 0


def test_a_batch_carrying_both_variants_logs_one():
    added = merge_data.log_jobs([
        _job("https://boards.greenhouse.io/acme/jobs/999"),
        _job("https://job-boards.greenhouse.io/acme/jobs/999"),
        _job("https://acme.com/careers?gh_jid=999"),
    ], "2026-08-20")

    assert added == 1


def test_non_ats_reposts_are_still_kept():
    """The date rule must survive where a URL really is reused across postings.
    Without this the change would silently suppress genuine LinkedIn re-posts."""
    url = "https://www.linkedin.com/jobs/view/4425828853"
    first = _job(url, site="linkedin", posted=_recent(5))
    second = _job(url, site="linkedin", posted=_recent(1))

    assert merge_data.log_jobs([first], "2026-08-20") == 1
    assert merge_data.log_jobs([second], "2026-08-21") == 1, (
        "a genuine re-post on a URL-keyed source was suppressed"
    )


def test_distinct_ats_jobs_are_all_kept():
    """Negative case: the rule must not over-merge."""
    added = merge_data.log_jobs([
        _job("https://boards.greenhouse.io/acme/jobs/1"),
        _job("https://boards.greenhouse.io/acme/jobs/2"),
        _job("https://careers-acme.icims.com/jobs/1/x/job"),
        _job("https://careers-other.icims.com/jobs/1/x/job"),
    ], "2026-08-20")

    assert added == 4


# ---------------------------------------------------------------------------
# Migration of an existing jobs.db
# ---------------------------------------------------------------------------

def _raw_insert(conn, date_key, key, scraped_at, record):
    conn.execute(
        "INSERT INTO jobs (date_key, dedup_key, scraped_at, data) VALUES (?,?,?,?)",
        (date_key, key, scraped_at, json.dumps(record)),
    )


def test_existing_rows_are_rekeyed_so_the_live_week_is_not_relogged(tmp_path):
    """Without the migration, every still-open job in the current week is filed
    under its old URL key, can never be matched again, and gets logged a second
    time on the next cycle."""
    conn = merge_data._get_conn()
    rec = _job("https://boards.greenhouse.io/acme/jobs/555")
    _raw_insert(conn, "2026-08-20", rec["job_url"], "2026-08-20T00:00:00Z", rec)
    conn.execute("PRAGMA user_version = 0")
    conn.commit()

    merge_data._migrate_dedup_keys(conn)

    keys = [r[0] for r in conn.execute("SELECT dedup_key FROM jobs")]
    assert keys == ["ats:greenhouse:555"]
    assert merge_data.log_jobs([rec], "2026-08-21") == 0, "the live week was re-logged"


def test_migration_merges_collisions_keeping_the_earliest_sighting(tmp_path):
    """Several URL-keyed rows collapsing onto one id key is the point of the
    migration, not an error -- and the first sighting must be the survivor."""
    conn = merge_data._get_conn()
    early = _job("https://boards.greenhouse.io/acme/jobs/777", title="Engineer")
    late = _job("https://job-boards.greenhouse.io/acme/jobs/777", title="Engineer II")
    _raw_insert(conn, "2026-08-20", late["job_url"], "2026-08-20T12:00:00Z", late)
    _raw_insert(conn, "2026-08-20", early["job_url"], "2026-08-20T01:00:00Z", early)
    conn.execute("PRAGMA user_version = 0")
    conn.commit()

    merge_data._migrate_dedup_keys(conn)

    rows = conn.execute("SELECT dedup_key, scraped_at, data FROM jobs").fetchall()
    assert len(rows) == 1, f"collision was not merged: {rows}"
    assert rows[0][0] == "ats:greenhouse:777"
    assert rows[0][1] == "2026-08-20T01:00:00Z", "the later sighting survived"
    assert json.loads(rows[0][2])["title"] == "Engineer"


def test_migration_runs_once():
    conn = merge_data._get_conn()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == merge_data._DEDUP_KEY_SCHEMA

    rec = _job("https://boards.greenhouse.io/acme/jobs/888")
    _raw_insert(conn, "2026-08-20", rec["job_url"], "2026-08-20T00:00:00Z", rec)
    conn.commit()

    merge_data._migrate_dedup_keys(conn)   # already at target version

    keys = [r[0] for r in conn.execute("SELECT dedup_key FROM jobs")]
    assert keys == [rec["job_url"]], "migration re-ran and rewrote a fresh row"


def test_unreadable_payload_keeps_its_key_and_is_not_dropped():
    """A row whose JSON cannot be parsed cannot be re-derived; losing it would
    lose the sighting."""
    conn = merge_data._get_conn()
    conn.execute(
        "INSERT INTO jobs (date_key, dedup_key, scraped_at, data) VALUES (?,?,?,?)",
        ("2026-08-20", "https://example.com/x", "2026-08-20T00:00:00Z", "{not json"),
    )
    conn.execute("PRAGMA user_version = 0")
    conn.commit()

    merge_data._migrate_dedup_keys(conn)

    assert [r[0] for r in conn.execute("SELECT dedup_key FROM jobs")] == ["https://example.com/x"]


def test_migration_failure_leaves_the_table_untouched(monkeypatch):
    """A half-migrated key table would be worse than an unmigrated one."""
    conn = merge_data._get_conn()
    rec = _job("https://boards.greenhouse.io/acme/jobs/999")
    _raw_insert(conn, "2026-08-20", rec["job_url"], "2026-08-20T00:00:00Z", rec)
    conn.execute("PRAGMA user_version = 0")
    conn.commit()

    def boom(job):
        raise RuntimeError("keying blew up")

    monkeypatch.setattr(merge_data, "_dedup_key", boom)
    merge_data._migrate_dedup_keys(conn)

    assert [r[0] for r in conn.execute("SELECT dedup_key FROM jobs")] == [rec["job_url"]]
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0, "version advanced despite failure"
