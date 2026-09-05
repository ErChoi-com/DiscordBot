"""The description cache, which committed job_match already imports.

`services.job_match` calls into this module at three points, and the module was
never added to the repository -- so a checkout could not import job_match at
all, and tests/test_job_match.py failed collection before running anything.

The behaviour worth pinning is the TTL asymmetry. A description that fetched
fine does not go stale (SUCCESS_TTL_DAYS is None), because a posting's body does
not change once written; a failure does (FAILURE_TTL_DAYS = 3), because a 403 or
a timeout says something about the moment, not about the posting. Caching a
failure forever would turn one bad afternoon into a permanently empty
description.

Hermetic: every test drives a temporary database, never data/jba/jobs.db.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.jba import description_cache as D  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point the module at a scratch database and clear its connection cache.

    Both are required: patching only _DB_PATH leaves a thread-local connection
    open on the real archive, which is the partially-restored-global shape that
    makes a leak surface in whatever test runs next.
    """
    monkeypatch.setattr(D, "_DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(D, "_local", type(D._local)())
    yield


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


# ── keys ─────────────────────────────────────────────────────────────────────

def test_the_same_url_yields_the_same_key():
    assert D.url_key("https://x.com/jobs/1") == D.url_key("https://x.com/jobs/1")


def test_different_urls_yield_different_keys():
    assert D.url_key("https://x.com/jobs/1") != D.url_key("https://x.com/jobs/2")


def test_an_empty_url_does_not_raise():
    D.url_key("")


# ── round trip ───────────────────────────────────────────────────────────────

def test_a_stored_description_comes_back():
    D.store_many([("https://x.com/1", "the body", D.STATUS_OK)])
    got = D.lookup(["https://x.com/1"])

    assert D.url_key("https://x.com/1") in got
    status, body = got[D.url_key("https://x.com/1")]
    assert status == D.STATUS_OK
    assert body == "the body"


def test_an_unknown_url_is_simply_absent():
    assert D.lookup(["https://x.com/never-seen"]) == {}


def test_looking_up_nothing_is_not_an_error():
    assert D.lookup([]) == {}


def test_storing_nothing_is_not_an_error():
    D.store_many([])


def test_a_later_store_replaces_an_earlier_one():
    D.store_many([("https://x.com/1", "first", D.STATUS_OK)])
    D.store_many([("https://x.com/1", "second", D.STATUS_OK)])

    assert D.lookup(["https://x.com/1"])[D.url_key("https://x.com/1")][1] == "second"


def test_a_lookup_larger_than_one_chunk_returns_everything():
    """Lookups are chunked to stay inside SQLite's parameter limit; a fleet
    query is far larger than one chunk, so a chunking bug would silently drop
    the tail rather than raise.
    """
    urls = [f"https://x.com/{i}" for i in range(D._LOOKUP_CHUNK * 2 + 7)]
    D.store_many([(u, "body", D.STATUS_OK) for u in urls])

    assert len(D.lookup(urls)) == len(urls)


# ── the TTL asymmetry, which is the point of the module ──────────────────────

def test_a_successful_fetch_does_not_expire():
    """A posting's body does not change once written, so re-fetching it buys
    nothing. SUCCESS_TTL_DAYS is None to say exactly that.
    """
    assert D.SUCCESS_TTL_DAYS is None
    assert D._expired(D.STATUS_OK, _iso(3650), datetime.now(timezone.utc)) is False


@pytest.mark.parametrize("status", [D.STATUS_BLOCKED, D.STATUS_ERROR, D.STATUS_EMPTY, D.STATUS_DEAD])
def test_a_failure_expires_so_one_bad_afternoon_is_not_permanent(status):
    """403s, timeouts and empty bodies describe the moment, not the posting.
    Caching them forever would make a transient block a permanent blank.

    The ages are literals bracketing the real threshold, not FAILURE_TTL_DAYS
    arithmetic. Written the self-referential way this test moved with the
    constant: mutation set the TTL to 100000 days and it still passed, because
    it was asking "is TTL+1 past TTL" rather than "is four days past three".
    """
    assert D.FAILURE_TTL_DAYS == 3, "threshold moved -- the literals below must move with it"
    now = datetime.now(timezone.utc)

    assert D._expired(status, _iso(4), now) is True     # past the 3-day threshold
    assert D._expired(status, _iso(2), now) is False    # inside it


@pytest.mark.parametrize("status", [D.STATUS_BLOCKED, D.STATUS_ERROR, D.STATUS_EMPTY, D.STATUS_DEAD])
def test_a_recent_failure_is_still_honoured(status):
    """Retrying a 403 on every pass is how a rate limit becomes a ban."""
    now = datetime.now(timezone.utc)
    assert D._expired(status, _iso(0.1), now) is False


def test_an_unparseable_timestamp_counts_as_expired():
    """Corrupt beats stale: a row whose date cannot be read should be refetched
    rather than trusted indefinitely.
    """
    assert D._expired(D.STATUS_ERROR, "not-a-date", datetime.now(timezone.utc)) is True


# ── failure modes must degrade, not raise ────────────────────────────────────

def test_a_missing_directory_is_created_rather_than_failing(monkeypatch, tmp_path):
    """A first run has no data/jba/jobs at all. _connect mkdirs its parents, so
    the cache works on a fresh checkout instead of silently degrading on every
    lookup for the life of the process.
    """
    target = tmp_path / "not-made-yet" / "jobs" / "jobs.db"
    monkeypatch.setattr(D, "_DB_PATH", target)
    monkeypatch.setattr(D, "_local", type(D._local)())

    D.store_many([("https://x.com/1", "body", D.STATUS_OK)])
    assert target.exists()
    assert D.lookup(["https://x.com/1"])[D.url_key("https://x.com/1")][1] == "body"


def test_an_unopenable_store_degrades_instead_of_raising(monkeypatch):
    """The cache is an optimisation, and an optimisation that can take the
    command down is a bug -- _connect's own docstring says so. Callers must get
    an empty result and go fetch the slow way.
    """
    monkeypatch.setattr(D, "_local", type(D._local)())
    monkeypatch.setattr(D, "_connect", lambda: None)

    D.store_many([("https://x.com/1", "body", D.STATUS_OK)])
    assert D.lookup(["https://x.com/1"]) == {}
