"""Index of archived jobs, for ATS duplicate elimination inside a time window.

The rule this implements, in full:

  * The same job must not appear twice within DEDUP_WINDOW_DAYS (~4 months).
  * "The same job" means the same base identity (job URL, or Workday
    company+job-id) AND the same date_posted. A re-post carrying a *different*
    date_posted is a genuinely new posting and is kept.
  * A listing with NO date_posted is treated as matching every listing of that
    job inside the window -- it carries no evidence that it is a new posting, so
    it is not allowed to masquerade as one.
  * Outside the window the job is free to appear again.

That history lives in data/jba/jobs/**/*.zip -- roughly half a million records.

Why an on-disk index rather than reading the zips per scrape:

  * Parsing is cheap (~0.4s for a 400k-entry month) but the resulting key set is
    ~70MB of Python strings PER MONTH. Holding the archive resident would cost
    more memory than the rest of the bot combined, and the small-host scaling in
    services.capacity exists precisely to avoid that.
  * Queries become indexed batch lookups with O(1) resident memory.

The index is derived data: delete it and it rebuilds. It is refreshed only when
a source archive's size or mtime changes, so a normal scrape pays a few stat()
calls. The full history is always indexed; the window is applied at query time,
so changing DEDUP_WINDOW_DAYS needs no rebuild.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from services.jba.merge_data import _collides, _dedup_key, _identity, _posted_key

_JOBS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "jba" / "jobs"
_INDEX_PATH = _JOBS_DIR / "archive_index.db"

# ~4 months. Beyond this a job may legitimately be surfaced again.
DEDUP_WINDOW_DAYS: int = 122

# Bumped whenever the table layout OR the keying rule changes; a mismatch drops
# and rebuilds. v3: merge_data._dedup_key now keys ATS listings on the provider
# job id, so every base_key stored under v2 is stale.
_SCHEMA_VERSION = 3

# Empty string, not NULL, marks "no date". SQLite permits multiple NULLs in a
# non-INTEGER PRIMARY KEY, so a NULL date_posted would defeat the ON CONFLICT
# upsert and let duplicate rows accumulate.
_NO_DATE = ""

_build_lock = threading.Lock()
_BATCH = 1000


def _now() -> datetime:
    return datetime.now(timezone.utc)


def window_cutoff(now: datetime | None = None) -> str:
    """ISO timestamp marking the start of the dedup window."""
    return ((now or _now()) - timedelta(days=DEDUP_WINDOW_DAYS)).isoformat()


def _schema_matches(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.Error:
        return False
    return bool(row) and str(row[0]) == str(_SCHEMA_VERSION)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS archive_seen;
        DROP TABLE IF EXISTS sources;
        DROP TABLE IF EXISTS meta;

        CREATE TABLE archive_seen (
            base_key    TEXT NOT NULL,
            date_posted TEXT NOT NULL,   -- '' means the listing carried no date
            first_seen  TEXT NOT NULL,   -- '' means unknown
            PRIMARY KEY (base_key, date_posted)
        );
        CREATE INDEX idx_archive_base ON archive_seen (base_key);

        CREATE TABLE sources (
            name  TEXT PRIMARY KEY,
            size  INTEGER NOT NULL,
            mtime REAL    NOT NULL
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(_SCHEMA_VERSION),)
    )
    conn.commit()


def _connect(readonly: bool = False) -> sqlite3.Connection | None:
    if readonly:
        if not _INDEX_PATH.exists():
            return None
        try:
            return sqlite3.connect(f"file:{_INDEX_PATH}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error:
            return None

    _JOBS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_INDEX_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    if not _schema_matches(conn):
        # An index built by an older layout cannot answer the current query, and
        # silently keeping it would mean wrong dedup decisions rather than a
        # visible failure. Rebuilding is cheap relative to being wrong.
        _create_schema(conn)
    return conn


def archive_files() -> list[Path]:
    """Every stored archive, oldest first."""
    if not _JOBS_DIR.exists():
        return []
    return sorted(_JOBS_DIR.glob("*/*.zip"))


def _stale_sources(conn: sqlite3.Connection) -> list[Path]:
    """Archives not yet indexed, or changed since they were."""
    recorded = {
        name: (size, mtime)
        for name, size, mtime in conn.execute("SELECT name, size, mtime FROM sources")
    }
    stale: list[Path] = []
    for path in archive_files():
        try:
            stat = path.stat()
        except OSError:
            continue
        prior = recorded.get(path.name)
        if prior is None or prior[0] != stat.st_size or abs(prior[1] - stat.st_mtime) > 1e-6:
            stale.append(path)
    return stale


# Normalization and the collision rule live in merge_data and are re-exported
# here. They used to be defined in both modules; the copies drifted, and the
# write path spent that whole time rejecting every re-post. One definition.
_clean = _posted_key
job_identity = _identity


def _iter_archive_entries(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (base_key, date_posted, first_seen) from one archive's job records.

    Only the per-day job files are read -- the same records the `jobs` table
    holds -- because they carry date_posted, which the whole rule turns on.

    seen_urls.json is deliberately ignored. It stores only a URL and a
    first_seen, so every entry would land in the undated bucket and could never
    participate in a date comparison. Measured against this archive it also adds
    almost nothing: 954 URLs out of ~674k appear there and nowhere in the job
    files, and those 954 have no date either.
    """
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        return

    with archive:
        names = archive.namelist()

        for name in names:
            if name == "seen_urls.json" or not name.endswith(".json"):
                continue
            try:
                records = json.loads(archive.read(name))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(records, list):
                continue
            # Entry name is the day (YYYY-MM-DD.json): the fallback sighting date
            # when a record carries no scraped_at of its own.
            day = name[:-5]
            for record in records:
                if not isinstance(record, dict):
                    continue
                base, posted = job_identity(record)
                if not base:
                    continue
                first_seen = _clean(record.get("scraped_at")) or _clean(day)
                yield base, posted, first_seen


def _upsert(conn: sqlite3.Connection, rows: list[tuple[str, str, str]]) -> None:
    """Insert sightings, keeping the EARLIEST first_seen for each identity.

    A known date always beats '' (unknown); between two known dates the lesser
    wins. That keeps the window measured from the true first sighting.
    """
    conn.executemany(
        """
        INSERT INTO archive_seen (base_key, date_posted, first_seen) VALUES (?, ?, ?)
        ON CONFLICT(base_key, date_posted) DO UPDATE SET first_seen =
            CASE
                WHEN excluded.first_seen = '' THEN archive_seen.first_seen
                WHEN archive_seen.first_seen = '' THEN excluded.first_seen
                WHEN excluded.first_seen < archive_seen.first_seen THEN excluded.first_seen
                ELSE archive_seen.first_seen
            END
        """,
        rows,
    )


# When each index path was last checked for stale archives. The check opens
# a connection and stats every zip under the build lock, which is fine a few
# times a cycle and is not fine once per board: the enrichment prefilter made
# it that, and a py-spy dump then found 38-68 threads inside this function at
# every sample while the Discord gateway fell 42s behind. Archives change
# about weekly, so a check made in the last minute is still the answer.
_CHECK_INTERVAL_S = 60.0
_last_checked: dict[Path, float] = {}


def ensure_index(force: bool = False) -> int:
    """Build or refresh the index. Returns the number of archives ingested.

    Cheap when called often: after a check, calls in the next _CHECK_INTERVAL_S
    return without touching the lock, the connection or the filesystem.
    `force` always rebuilds.
    """
    if not force and time.monotonic() - _last_checked.get(_INDEX_PATH, -1e12) < _CHECK_INTERVAL_S:
        return 0
    with _build_lock:
        if not force and time.monotonic() - _last_checked.get(_INDEX_PATH, -1e12) < _CHECK_INTERVAL_S:
            return 0
        conn = _connect()
        if conn is None:
            return 0
        try:
            sources = archive_files() if force else _stale_sources(conn)
            _last_checked[_INDEX_PATH] = time.monotonic()
            if not sources:
                return 0

            for path in sources:
                batch: list[tuple[str, str, str]] = []
                for row in _iter_archive_entries(path):
                    batch.append(row)
                    if len(batch) >= _BATCH:
                        _upsert(conn, batch)
                        batch.clear()
                if batch:
                    _upsert(conn, batch)

                try:
                    stat = path.stat()
                except OSError:
                    continue
                conn.execute(
                    "INSERT INTO sources (name, size, mtime) VALUES (?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET size = excluded.size, mtime = excluded.mtime",
                    (path.name, stat.st_size, stat.st_mtime),
                )
                conn.commit()
            return len(sources)
        finally:
            conn.close()


def recent_sightings(
    base_keys: Iterable[str], cutoff: str | None = None
) -> dict[str, set[str]]:
    """base_key -> set of date_posted values sighted inside the window.

    A sighting with an unknown first_seen ('') is excluded: without a date there
    is no evidence it falls in the window, and suppressing on that basis would
    hide jobs indefinitely. Fails open (empty dict) when the index is missing.
    """
    unique = [k for k in dict.fromkeys(base_keys) if k]
    if not unique:
        return {}

    since = cutoff if cutoff is not None else window_cutoff()
    conn = _connect(readonly=True)
    if conn is None:
        return {}

    found: dict[str, set[str]] = {}
    try:
        for i in range(0, len(unique), 500):
            chunk = unique[i:i + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT base_key, date_posted FROM archive_seen "
                f"WHERE base_key IN ({placeholders}) AND first_seen != '' AND first_seen >= ?",
                (*chunk, since),
            )
            for base, posted in rows:
                found.setdefault(base, set()).add(posted)
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return found


def collides(posted: str, seen_dates: set[str]) -> bool:
    """Is a listing dated `posted` a duplicate of any sighting in `seen_dates`?

    Delegates to merge_data._collides, where the rule is defined. Asymmetric,
    deliberately -- it keys on the INCOMING listing:

      * No date_posted  -> collides with ANY prior sighting of this job in the
        window. It carries no evidence of being a new posting, so it is not
        allowed to masquerade as one.
      * Has date_posted -> collides only with a sighting carrying the SAME date.
        A different date is a genuine re-post and is kept.

    Note what this deliberately does NOT do: an undated *sighting* does not
    suppress a dated incoming listing. seen_urls.json records only a URL and a
    first_seen -- no date_posted -- so every one of its ~400k entries lands in
    the undated bucket. Treating those as matching any date would suppress
    essentially every re-post and defeat the date rule entirely.
    """
    return _collides(posted, seen_dates)


def _job_stamp(job: dict[str, Any]) -> tuple[str, str]:
    """Sort key for age: (date_posted, scraped_at), undated sorting last.

    Two copies of one job in the same scrape normally share a date_posted, so
    that field alone ties and the first-encountered copy would win by accident.
    scraped_at breaks the tie on the real sighting time, which is what "keep the
    oldest" is actually about.
    """
    return (
        str(job.get("date_posted") or "9999"),
        str(job.get("scraped_at") or "9999"),
    )


def filter_new_listings(
    jobs: list[dict[str, Any]], cutoff: str | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Drop listings already sighted for the same job+date inside the window.

    Returns (kept, dropped), preserving input order. Listings with no derivable
    base key are always kept -- an unkeyable record is not evidence of anything.
    """
    if not jobs:
        return jobs, 0

    identities = [job_identity(job) for job in jobs]
    bases = [base for base, _ in identities if base]

    archived = recent_sightings(bases, cutoff)
    for base, dates in _current_week_sightings(bases).items():
        archived.setdefault(base, set()).update(dates)

    kept: list[dict[str, Any]] = []
    batch_dates: dict[str, set[str]] = {}
    # Oldest copy of each identity wins when a single scrape carries duplicates.
    preferred = _preferred_within_batch(jobs, identities)

    for index, job in enumerate(jobs):
        base, posted = identities[index]
        if not base:
            kept.append(job)
            continue
        if collides(posted, archived.get(base, set())):
            continue
        if collides(posted, batch_dates.get(base, set())):
            continue
        if preferred.get((base, posted)) != index:
            continue
        batch_dates.setdefault(base, set()).add(posted)
        kept.append(job)

    return kept, len(jobs) - len(kept)


def _preferred_within_batch(
    jobs: list[dict[str, Any]], identities: list[tuple[str, str]]
) -> dict[tuple[str, str], int]:
    """Index of the oldest copy of each (base_key, date_posted) in one scrape."""
    best: dict[tuple[str, str], int] = {}
    for index, job in enumerate(jobs):
        identity = identities[index]
        if not identity[0]:
            continue
        current = best.get(identity)
        if current is None or _job_stamp(job) < _job_stamp(jobs[current]):
            best[identity] = index
    return best


def _current_week_sightings(bases: list[str]) -> dict[str, set[str]]:
    """base_key -> date_posted values recorded in the live jobs.db.

    merge_data exports the *previous* week to a zip and purges it from the DB, so
    the current week exists only there. Consulting the zip index alone would miss
    everything scraped since the last rollover -- the most likely duplicates.

    Reads the `jobs` table rather than `seen_urls`, because seen_urls stores only
    a URL and a first_seen. Without date_posted every current-week sighting would
    land in the undated bucket and could not participate in the date rule at all.
    The whole current week is inside the window by definition, so no cutoff
    applies here.
    """
    unique = [b for b in dict.fromkeys(bases) if b]
    if not unique:
        return {}

    from services.jba import merge_data

    if not merge_data._DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{merge_data._DB_PATH}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return {}

    found: dict[str, set[str]] = {}
    try:
        for i in range(0, len(unique), 500):
            chunk = unique[i:i + 500]
            placeholders = ",".join("?" for _ in chunk)
            for base, payload in conn.execute(
                f"SELECT dedup_key, data FROM jobs WHERE dedup_key IN ({placeholders})", chunk
            ):
                try:
                    record = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    record = {}
                posted = _clean(record.get("date_posted") if isinstance(record, dict) else "")
                found.setdefault(base, set()).add(posted)
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return found
