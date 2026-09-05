"""Persistent store for job descriptions fetched by the ranking commands.

The archive keeps what a search-results page exposes -- title, company,
location -- so under 4% of records carry a description. `.bestjobs` therefore
fetches the real posting text for its strongest candidates before judging them,
which is the slowest and most fragile stage of the command: real requests to
boards that rate-limit, block, and 404.

Until this module existed every one of those fetches was thrown away when the
command returned. Running `.bestjobs` twice in a row fetched the same postings
twice; two profiles ranking the same archive fetched them twice again. A
description belongs to the *job*, not to whoever asked about it, so one fetch
should serve every later run and every profile.

Where this lives
----------------
A table inside the existing data/jba/jobs/jobs.db rather than a database of its
own, and deliberately not inside archive_index.db: that index rebuilds itself
by DROPping its tables whenever its schema version changes
(archive_index._create_schema), which would silently discard weeks of
accumulated network work. Nothing here ever drops a table.

A side table rather than a field written back into each job record, because the
weekly rotation exports rows to a zip and deletes them from `jobs`
(merge_data._archive_completed_weeks). Records already rotated out cannot be
updated in place, so a description attached to a row would only ever persist
for postings from the current week. A table keyed on the posting URL persists
for all of them.

Failures are cached too
-----------------------
A posting that 404s or blocks the client costs exactly as much time as one that
succeeds, and there are boards -- LinkedIn and Indeed especially -- that
reliably refuse. Recording only successes would mean re-attempting the same
refusals on every run, forever, spending the fetch budget to rediscover what is
already known. Failures therefore get rows too, with a short TTL so a transient
block does not become permanent.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "jba" / "jobs" / "jobs.db"

#: A posting's text does not change in any way that matters to ranking, so a
#: successful fetch is kept indefinitely.
SUCCESS_TTL_DAYS: int | None = None

#: Failures expire, because they are frequently about the moment rather than
#: the posting: a rate-limit, a slow board, a transient 5xx. Long enough to
#: stop the same run-to-run retries, short enough that a board which starts
#: answering again is picked back up within days.
FAILURE_TTL_DAYS: int = 3

#: SQLite caps how many host parameters one statement may carry (999 on older
#: builds); lookups are chunked well inside it.
_LOOKUP_CHUNK: int = 400

#: Statuses. Anything that is not OK is a failure and expires.
STATUS_OK = "ok"
STATUS_EMPTY = "empty"      # fetched fine, posting carried no usable body
STATUS_DEAD = "dead"        # 404 / posting withdrawn
STATUS_BLOCKED = "blocked"  # 403 / bot wall / rate limit
STATUS_ERROR = "error"      # anything else, including timeouts

_lock = threading.Lock()
_local = threading.local()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _connect() -> sqlite3.Connection | None:
    """Thread-local connection, or None when the store is unusable.

    Every entry point tolerates None. A ranking that cannot reach its cache
    should run slower, never fail: this is an optimisation, and an optimisation
    that can take the command down is a bug.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    try:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        _ensure_schema(conn)
    except sqlite3.Error:
        return None
    _local.conn = conn
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the table if absent. Never drops: the rows here represent
    network work that cannot be cheaply recreated."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS job_descriptions (
            url_key    TEXT PRIMARY KEY,
            body       TEXT NOT NULL,   -- '' for every non-ok status
            status     TEXT NOT NULL,
            fetched_at TEXT NOT NULL    -- ISO 8601, UTC
        );
        CREATE INDEX IF NOT EXISTS idx_job_descriptions_status
            ON job_descriptions (status);
        """
    )
    conn.commit()


def url_key(url: str) -> str:
    """The identity a description is stored under.

    Canonicalised the same way the ranker dedupes postings, so a job the ranker
    considers one job is one row here too.
    """
    from services.job_service import canonicalize_job_link

    raw = str(url or "").strip()
    if not raw:
        return ""
    return (canonicalize_job_link(raw) or raw).lower()


def _expired(status: str, fetched_at: str, now: datetime) -> bool:
    ttl = SUCCESS_TTL_DAYS if status == STATUS_OK else FAILURE_TTL_DAYS
    if ttl is None:
        return False
    try:
        stamp = datetime.fromisoformat(fetched_at)
    except ValueError:
        return True
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp < now - timedelta(days=ttl)


def lookup(urls: Iterable[str]) -> dict[str, tuple[str, str]]:
    """``{url_key: (status, body)}`` for every cached, unexpired posting.

    A url absent from the result has never been fetched, or its entry has
    aged out -- both mean "fetch it". Expired rows are simply not returned;
    the next successful fetch overwrites them.
    """
    keys = [key for key in dict.fromkeys(url_key(url) for url in urls) if key]
    if not keys:
        return {}
    conn = _connect()
    if conn is None:
        return {}

    now = _now()
    found: dict[str, tuple[str, str]] = {}
    try:
        for start in range(0, len(keys), _LOOKUP_CHUNK):
            chunk = keys[start : start + _LOOKUP_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT url_key, status, body, fetched_at FROM job_descriptions "
                f"WHERE url_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for key, status, body, fetched_at in rows:
                if not _expired(status, fetched_at, now):
                    found[key] = (status, body or "")
    except sqlite3.Error:
        return found
    return found


def store(url: str, body: str, status: str = STATUS_OK) -> bool:
    """Record one fetch outcome. Returns whether it was written."""
    return store_many([(url, body, status)]) == 1


def store_many(entries: Iterable[tuple[str, str, str]]) -> int:
    """Record several outcomes as one transaction. Returns rows written.

    Writes go through a lock because the scraper writes to this same database
    file from its own threads; SQLite would serialise them anyway, and holding
    the lock keeps the whole batch in a single transaction.
    """
    rows = [
        (key, body if status == STATUS_OK else "", status, _now().isoformat())
        for key, body, status in (
            (url_key(url), body or "", status) for url, body, status in entries
        )
        if key
    ]
    if not rows:
        return 0
    conn = _connect()
    if conn is None:
        return 0
    try:
        with _lock:
            conn.executemany(
                "INSERT INTO job_descriptions (url_key, body, status, fetched_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(url_key) DO UPDATE SET "
                "  body = excluded.body, status = excluded.status, "
                "  fetched_at = excluded.fetched_at",
                rows,
            )
            conn.commit()
    except sqlite3.Error:
        return 0
    return len(rows)


def stats() -> dict[str, int]:
    """Row counts by status, for the health command and for tests."""
    conn = _connect()
    if conn is None:
        return {}
    try:
        return {
            str(status): int(count)
            for status, count in conn.execute(
                "SELECT status, COUNT(*) FROM job_descriptions GROUP BY status"
            )
        }
    except sqlite3.Error:
        return {}
