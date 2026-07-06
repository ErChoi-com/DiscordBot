"""Date-based job log storage backed by SQLite + weekly/monthly zip archives.

Current week's data lives in data/jba/jobs/jobs.db for fast lookups.
When a new week begins, the previous week's jobs are exported to a
weekly zip and purged from the DB.  When a new month begins, all
weekly zips are consolidated into a single YYYY-MM.zip containing
per-day JSON entries + seen_urls.json, and the result is git-committed.
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_JOBS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "jba" / "jobs"
_DB_PATH = _JOBS_DIR / "jobs.db"
_file_lock = threading.Lock()
_local = threading.local()

_last_archive_week: str | None = None
_last_consolidate_month: str | None = None


# ---------------------------------------------------------------------------
# Week helpers — weeks are 1-indexed by day-of-month:
#   w1 = days 1-7, w2 = 8-14, w3 = 15-21, w4 = 22-end
# ---------------------------------------------------------------------------

def _week_of_month(date_str: str) -> int:
    day = int(date_str[8:10])
    return min((day - 1) // 7 + 1, 4)


def _week_label(date_str: str) -> str:
    month = date_str[:7]
    return f"{month}-w{_week_of_month(date_str)}"


def _week_zip_path(date_str: str) -> Path:
    month = date_str[:7]
    return _JOBS_DIR / month / f"{_week_label(date_str)}.zip"


# ---------------------------------------------------------------------------
# DB connection + schema
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        _JOBS_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
        _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS seen_urls (
            url       TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date_key   TEXT    NOT NULL,
            dedup_key  TEXT    NOT NULL,
            scraped_at TEXT    NOT NULL,
            data       TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_date_key  ON jobs (date_key);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_dedup_date ON jobs (date_key, dedup_key);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Weekly archival — export old weeks from DB to zip, then purge
# ---------------------------------------------------------------------------

def _archive_old_weeks(today: str) -> None:
    """Export any DB rows from previous weeks into zip files and delete them."""
    global _last_archive_week
    current_week = _week_label(today)

    if _last_archive_week == current_week:
        return
    _last_archive_week = current_week

    conn = _get_conn()
    date_keys = [
        r[0] for r in conn.execute("SELECT DISTINCT date_key FROM jobs ORDER BY date_key")
    ]

    weeks_to_archive: dict[str, list[str]] = {}
    for dk in date_keys:
        wl = _week_label(dk)
        if wl < current_week:
            weeks_to_archive.setdefault(wl, []).append(dk)

    if not weeks_to_archive:
        return

    for wl, dates in weeks_to_archive.items():
        month = wl[:7]
        zpath = _JOBS_DIR / month / f"{wl}.zip"
        zpath.parent.mkdir(parents=True, exist_ok=True)

        existing_entries: set[str] = set()
        if zpath.exists():
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    existing_entries = set(zf.namelist())
            except Exception:
                pass

        mode = "a" if zpath.exists() else "w"
        try:
            with zipfile.ZipFile(zpath, mode, zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for dk in sorted(dates):
                    entry = f"{dk}.json"
                    if entry in existing_entries:
                        continue
                    rows = conn.execute(
                        "SELECT data FROM jobs WHERE date_key = ? ORDER BY scraped_at",
                        (dk,),
                    ).fetchall()
                    if not rows:
                        continue
                    jobs = [json.loads(r[0]) for r in rows]
                    zf.writestr(entry, json.dumps(jobs, indent=1, default=str))

            conn.execute(
                "DELETE FROM jobs WHERE date_key IN ({})".format(
                    ",".join("?" for _ in dates)
                ),
                dates,
            )
            conn.commit()
            print(f"[jba-log] Archived {wl}: {len(dates)} days -> {zpath.name}")
        except Exception as exc:
            _last_archive_week = None
            print(f"[jba-log] Failed to archive {wl}: {exc}")


# ---------------------------------------------------------------------------
# Monthly consolidation — merge weekly zips + seen_urls into one archive
# ---------------------------------------------------------------------------

def _consolidate_old_months(today: str) -> None:
    """If *today* is in a new month, consolidate the previous month's weekly
    zips into a single YYYY-MM.zip with per-day entries + seen_urls.json,
    then git-commit the result."""
    global _last_consolidate_month
    current_month = today[:7]

    if _last_consolidate_month == current_month:
        return
    _last_consolidate_month = current_month

    if not _JOBS_DIR.exists():
        return

    for month_dir in sorted(_JOBS_DIR.iterdir()):
        if not month_dir.is_dir():
            continue
        month = month_dir.name
        if len(month) != 7 or month >= current_month:
            continue

        monthly_zip = month_dir / f"{month}.zip"
        if monthly_zip.exists():
            continue

        conn = _get_conn()
        old_month_keys = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT date_key FROM jobs WHERE date_key LIKE ?",
                (f"{month}%",),
            )
        ]
        if old_month_keys:
            _archive_old_weeks(today)

        weekly_zips = sorted(month_dir.glob(f"{month}-w*.zip"))
        if not weekly_zips:
            continue

        # Collect per-day entries from weekly zips (already stored as YYYY-MM-DD.json)
        day_entries: dict[str, bytes] = {}
        for wz in weekly_zips:
            try:
                with zipfile.ZipFile(wz, "r") as zf:
                    for name in sorted(zf.namelist()):
                        if name.endswith(".json"):
                            day_entries[name] = zf.read(name)
            except Exception as exc:
                print(f"[jba-log] Failed to read {wz.name}: {exc}")
                _last_consolidate_month = None
                return

        seen_urls = [
            {"url": r[0], "first_seen": r[1]}
            for r in conn.execute("SELECT url, first_seen FROM seen_urls ORDER BY url")
        ]

        job_count = 0
        try:
            with zipfile.ZipFile(monthly_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for name in sorted(day_entries):
                    zf.writestr(name, day_entries[name])
                    try:
                        job_count += len(json.loads(day_entries[name]))
                    except Exception:
                        pass
                zf.writestr("seen_urls.json", json.dumps(seen_urls, default=str))

            for wz in weekly_zips:
                wz.unlink()

            conn.execute("VACUUM")
            _git_commit_monthly(monthly_zip, month, job_count, len(seen_urls))
            print(f"[jba-log] Consolidated {month}: {job_count:,} jobs + "
                  f"{len(seen_urls):,} seen URLs -> {monthly_zip.name}")
        except Exception as exc:
            _last_consolidate_month = None
            print(f"[jba-log] Failed to consolidate {month}: {exc}")


def _git_commit_monthly(zip_path: Path, month: str, job_count: int, url_count: int) -> None:
    """Stage and commit a monthly archive zip."""
    repo_root = _JOBS_DIR.parent.parent.parent
    rel_path = zip_path.relative_to(repo_root)
    try:
        subprocess.run(
            ["git", "add", "-f", str(rel_path)],
            cwd=str(repo_root), capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m",
             f"archive: {month} job data ({job_count:,} jobs, {url_count:,} seen URLs)"],
            cwd=str(repo_root), capture_output=True, timeout=30,
        )
    except Exception as exc:
        print(f"[jba-log] Git commit failed for {month}: {exc}")


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def _dedup_key(job: dict[str, Any]) -> str:
    url = job.get("job_url") or job.get("url") or job.get("link") or ""
    if job.get("_source_site") == "workday" or job.get("ats") == "Workday":
        match = re.search(r"/jobs/(\d+)", url)
        if match:
            company = job.get("company", "")
            return f"workday:{company}:{match.group(1)}"
    return url


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


_WRITE_FRESHNESS_SECONDS = 7 * 86_400  # 7 days
_DATELESS_PLATFORMS = frozenset({"bamboohr", "ashby"})
_SEEN_URL_TTL_DAYS = 30
_last_seen_purge_date: str | None = None


def _is_fresh_enough(job: dict[str, Any], now_iso: str) -> bool:
    posted = str(job.get("date_posted") or "").strip()
    if not posted:
        return job.get("_source_site") in _DATELESS_PLATFORMS
    try:
        posted_dt = datetime.fromisoformat(posted.replace("Z", "+00:00"))
        if posted_dt.tzinfo is None:
            posted_dt = posted_dt.replace(tzinfo=timezone.utc)
        now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        age = (now_dt - posted_dt).total_seconds()
        return 0 <= age <= _WRITE_FRESHNESS_SECONDS
    except (ValueError, TypeError):
        return False


def _purge_old_seen_urls(conn: sqlite3.Connection, today: str) -> int:
    """Stamp 'migrated' entries with today so they age out in TTL days, then
    delete entries older than TTL. Runs at most once per day. Returns count removed."""
    global _last_seen_purge_date
    if _last_seen_purge_date == today:
        return 0
    _last_seen_purge_date = today
    # Give migrated entries a real date so the TTL clock starts now
    stamped = conn.execute(
        "UPDATE seen_urls SET first_seen = ? WHERE first_seen = 'migrated'",
        (today,),
    ).rowcount
    if stamped:
        print(f"[jba-log] Stamped {stamped:,} migrated seen_urls entries with {today} (retry in {_SEEN_URL_TTL_DAYS}d)")
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=_SEEN_URL_TTL_DAYS)
    ).strftime("%Y-%m-%d")
    removed = conn.execute(
        "DELETE FROM seen_urls WHERE first_seen < ?",
        (cutoff,),
    ).rowcount
    if stamped or removed:
        conn.commit()
    if removed:
        print(f"[jba-log] Purged {removed:,} seen_urls entries older than {_SEEN_URL_TTL_DAYS}d")
    return removed


def _bulk_check_seen(conn: sqlite3.Connection, keys: list[str]) -> set[str]:
    """Return the subset of *keys* that are already in seen_urls."""
    if not keys:
        return set()
    seen: set[str] = set()
    batch_size = 500
    for i in range(0, len(keys), batch_size):
        batch = keys[i:i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT url FROM seen_urls WHERE url IN ({placeholders})", batch
        ).fetchall()
        seen.update(r[0] for r in rows)
    return seen


def _mark_seen(conn: sqlite3.Connection, keys: list[str], now: str) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO seen_urls (url, first_seen) VALUES (?, ?)",
        [(k, now) for k in keys],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_jobs(jobs: list[dict[str, Any]], date_str: str | None = None) -> int:
    """Merge *jobs* into today's (or *date_str*'s) daily log.

    Deduplicates by job URL globally via seen_urls.
    Archives previous weeks' data to zip before writing.
    Returns the number of **new** entries added.
    """
    if not jobs:
        return 0

    date_str = date_str or _today_str()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    with _file_lock:
        conn = _get_conn()
        _archive_old_weeks(date_str)
        _consolidate_old_months(date_str)
        _purge_old_seen_urls(conn, date_str)

        existing_keys: set[str] = {
            row[0]
            for row in conn.execute(
                "SELECT dedup_key FROM jobs WHERE date_key = ?", (date_str,)
            )
        }

        candidates: list[tuple[str, dict[str, Any]]] = []
        for job in jobs:
            key = _dedup_key(job)
            if not key or key in existing_keys:
                continue
            if not _is_fresh_enough(job, now):
                continue
            candidates.append((key, job))

        if not candidates:
            return 0

        candidate_keys = [k for k, _ in candidates]
        already_seen = _bulk_check_seen(conn, candidate_keys)

        new_rows: list[tuple[str, str, str, str]] = []
        new_keys: list[str] = []
        for key, job in candidates:
            if key in already_seen:
                continue
            job.setdefault("scraped_at", now)
            new_rows.append((date_str, key, job.get("scraped_at", now), json.dumps(job, default=str)))
            new_keys.append(key)
            existing_keys.add(key)

        if new_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO jobs (date_key, dedup_key, scraped_at, data) VALUES (?, ?, ?, ?)",
                new_rows,
            )
            _mark_seen(conn, new_keys, now)
            conn.commit()

    added = len(new_rows)
    if added:
        total = _get_conn().execute(
            "SELECT COUNT(*) FROM jobs WHERE date_key = ?", (date_str,)
        ).fetchone()[0]
        print(f"[jba-log] {date_str}: +{added} new jobs ({total} total)")
    return added


def load_daily_log(date_str: str) -> list[dict[str, Any]]:
    """Load a specific day's job log (DB -> weekly zip -> monthly zip)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT data FROM jobs WHERE date_key = ? ORDER BY scraped_at",
        (date_str,),
    ).fetchall()
    if rows:
        result = []
        for (blob,) in rows:
            try:
                result.append(json.loads(blob))
            except Exception:
                pass
        return result

    # Try weekly zip (per-day entries)
    zpath = _week_zip_path(date_str)
    if zpath.exists():
        entry = f"{date_str}.json"
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                if entry in zf.namelist():
                    data = json.loads(zf.read(entry))
                    return data if isinstance(data, list) else []
        except Exception:
            pass

    # Try monthly zip
    month = date_str[:7]
    monthly = _JOBS_DIR / month / f"{month}.zip"
    if monthly.exists():
        entry = f"{date_str}.json"
        try:
            with zipfile.ZipFile(monthly, "r") as zf:
                names = zf.namelist()
                # New format: per-day entries
                if entry in names:
                    data = json.loads(zf.read(entry))
                    return data if isinstance(data, list) else []
                # Legacy format: single jobs.json with all jobs
                if "jobs.json" in names:
                    all_jobs = json.loads(zf.read("jobs.json"))
                    if isinstance(all_jobs, list):
                        return [
                            j for j in all_jobs
                            if str(j.get("scraped_at", ""))[:10] == date_str
                        ]
        except Exception:
            pass

    return []


def daily_log_count(date_str: str | None = None) -> int:
    date_str = date_str or _today_str()
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE date_key = ?", (date_str,),
    ).fetchone()
    count = row[0] if row else 0
    if count > 0:
        return count
    return len(load_daily_log(date_str))


def list_log_dates() -> list[str]:
    """Return available log dates (DB + weekly/monthly zip archives), most recent first."""
    conn = _get_conn()
    dates: set[str] = {
        r[0] for r in conn.execute("SELECT DISTINCT date_key FROM jobs")
    }

    if _JOBS_DIR.exists():
        for month_dir in _JOBS_DIR.iterdir():
            if not month_dir.is_dir():
                continue
            for f in month_dir.iterdir():
                if f.suffix != ".zip":
                    continue
                try:
                    with zipfile.ZipFile(f, "r") as zf:
                        names = zf.namelist()
                        has_per_day = any(
                            len(n.removesuffix(".json")) == 10
                            for n in names if n.endswith(".json") and n != "seen_urls.json"
                        )
                        if has_per_day:
                            for name in names:
                                stem = name.removesuffix(".json")
                                if len(stem) == 10:
                                    dates.add(stem)
                        elif "jobs.json" in names:
                            # Legacy format: parse scraped_at dates
                            all_jobs = json.loads(zf.read("jobs.json"))
                            if isinstance(all_jobs, list):
                                for j in all_jobs:
                                    sa = str(j.get("scraped_at", ""))[:10]
                                    if len(sa) == 10:
                                        dates.add(sa)
                except Exception:
                    pass

    return sorted(dates, reverse=True)


def seen_url_count() -> int:
    """Return total number of seen URLs."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM seen_urls").fetchone()
    return row[0] if row else 0
