"""Date-based job log storage backed by SQLite + weekly/monthly zip archives.

Current week's data lives in data/jba/jobs/jobs.db for fast lookups.
When a new week begins, the previous week's jobs are exported to a
weekly zip and purged from the DB.  When a new month begins, all
weekly zips are consolidated into a single YYYY-MM.zip containing
per-day JSON entries, and the result is git-committed.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

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
        CREATE TABLE IF NOT EXISTS jobs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date_key   TEXT    NOT NULL,
            dedup_key  TEXT    NOT NULL,
            scraped_at TEXT    NOT NULL,
            data       TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_date_key  ON jobs (date_key);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_dedup_date ON jobs (date_key, dedup_key);

        -- Dedup lookups are by key alone ("logged on ANY day?"). The composite
        -- index above is keyed on date_key first, so it cannot serve them.
        CREATE INDEX IF NOT EXISTS idx_jobs_dedup_key ON jobs (dedup_key);

        -- seen_urls is gone: it stored only a URL and a first_seen, so it could
        -- not answer anything about date_posted, and its 30-day TTL made it a
        -- narrower memory than the job records themselves. The jobs table (and
        -- the archives built from it) are now the single source of truth.
        DROP TABLE IF EXISTS seen_urls;
    """)
    conn.commit()
    _migrate_dedup_keys(conn)


# Bumped when _dedup_key starts producing different keys for the same job.
# Stored in the DB file via PRAGMA user_version, so the migration runs once per
# database rather than once per thread-local connection.
_DEDUP_KEY_SCHEMA: int = 1


def _migrate_dedup_keys(conn: sqlite3.Connection) -> None:
    """Rewrite stored dedup_keys after a change to the keying rule.

    Without this, the day _dedup_key starts returning an ATS id key every job
    still open in the current week looks brand new -- its stored row is filed
    under the old URL key and can never be matched again -- so the whole live
    week would be logged a second time. The archive index self-heals via its own
    schema version; this table does not, because it is not derived data.

    Collisions are expected and are the point: several URL-keyed rows for one
    job collapse onto a single id key. The earliest sighting survives, matching
    the first-sighting-wins rule used everywhere else.
    """
    if conn.execute("PRAGMA user_version").fetchone()[0] >= _DEDUP_KEY_SCHEMA:
        return
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.Error:
        return
    try:
        # Re-check inside the write lock: another process may have just run it.
        if conn.execute("PRAGMA user_version").fetchone()[0] >= _DEDUP_KEY_SCHEMA:
            conn.execute("ROLLBACK")
            return

        winners: dict[tuple[str, str], tuple[str, int]] = {}
        losers: list[int] = []
        updates: list[tuple[str, int]] = []

        for row_id, date_key, old_key, scraped_at, payload in conn.execute(
            "SELECT id, date_key, dedup_key, scraped_at, data FROM jobs"
        ).fetchall():
            try:
                record = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                record = None
            # An unreadable payload keeps its existing key: it cannot be
            # re-derived, and dropping it would lose the sighting entirely.
            new_key = _dedup_key(record) if isinstance(record, dict) else ""
            new_key = new_key or old_key

            slot = (date_key, new_key)
            best = winners.get(slot)
            stamp = (str(scraped_at or "~"), row_id)
            if best is None:
                winners[slot] = (stamp, row_id)
            elif stamp < best[0]:
                losers.append(best[1])
                winners[slot] = (stamp, row_id)
            else:
                losers.append(row_id)
                continue
            if new_key != old_key:
                updates.append((new_key, row_id))

        # Losers first: the UNIQUE(date_key, dedup_key) index would otherwise
        # reject the very updates that create the collision.
        if losers:
            conn.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in losers])
        live = {row_id for _, row_id in winners.values()}
        if updates:
            conn.executemany(
                "UPDATE jobs SET dedup_key = ? WHERE id = ?",
                [(k, i) for k, i in updates if i in live],
            )
        conn.execute(f"PRAGMA user_version = {_DEDUP_KEY_SCHEMA}")
        conn.execute("COMMIT")
        if updates or losers:
            print(f"[jba-log] dedup keys migrated: {len(updates)} rekeyed, {len(losers)} merged")
    except Exception as exc:
        conn.execute("ROLLBACK")
        print(f"[jba-log] dedup key migration failed, left unchanged: {exc}")


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
# Monthly consolidation — merge weekly zips into one archive
# ---------------------------------------------------------------------------

def _consolidate_old_months(today: str) -> None:
    """If *today* is in a new month, consolidate the previous month's weekly
    zips into a single YYYY-MM.zip with per-day entries, then git-commit
    the result."""
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

        job_count = 0
        try:
            # Per-day job records only. seen_urls.json is no longer written: it
            # duplicated the URLs already present in those records while
            # carrying no date_posted, so it could not answer any dedup question
            # the job records cannot answer better.
            with zipfile.ZipFile(monthly_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for name in sorted(day_entries):
                    zf.writestr(name, day_entries[name])
                    try:
                        job_count += len(json.loads(day_entries[name]))
                    except Exception:
                        pass

            for wz in weekly_zips:
                wz.unlink()

            conn.execute("VACUUM")
            committed = _git_commit_monthly(monthly_zip, month, job_count)
            state = "committed" if committed else "on disk, uncommitted"
            print(f"[jba-log] Consolidated {month}: {job_count:,} jobs -> "
                  f"{monthly_zip.name} ({state})")
        except Exception as exc:
            _last_consolidate_month = None
            print(f"[jba-log] Failed to consolidate {month}: {exc}")


_TRUTHY = {"1", "true", "yes"}


def _archive_commit_enabled() -> bool:
    return (os.getenv("JBA_ARCHIVE_GIT_COMMIT") or "").strip().lower() in _TRUTHY


def _archive_push_enabled() -> bool:
    """Pushing is gated separately from committing, and stays off by default.

    Committing touches only the local tree; pushing publishes to a shared remote
    and needs credentials the service account may not have. Someone who wants
    local archive commits has not thereby asked for automatic publication.
    """
    return (os.getenv("JBA_ARCHIVE_GIT_PUSH") or "").strip().lower() in _TRUTHY


def _repo_root() -> Path:
    return _JOBS_DIR.parent.parent.parent


def _git_identity() -> list[str]:
    """Identity via -c, not GIT_AUTHOR_*.

    The env-var route needs all four of AUTHOR/COMMITTER name and email, and
    setting only the author pair still fails on the committer. Under systemd the
    service account usually has no ~/.gitconfig, so without this `git commit`
    exits 128 with "Please tell me who you are".
    """
    name = os.getenv("JBA_GIT_AUTHOR_NAME", "discordbot")
    email = os.getenv("JBA_GIT_AUTHOR_EMAIL", "discordbot@localhost")
    return ["-c", f"user.name={name}", "-c", f"user.email={email}"]


def _run_git(argv: list[str], *, context: str, timeout: int = 30) -> tuple[int, str]:
    """Run a git command in the repo root. Returns (returncode, output).

    Return codes are checked and stderr surfaced by every caller. An earlier
    version did neither: capture_output swallowed git's complaint and
    `except Exception` never fired, because a non-zero exit is not an exception.
    """
    try:
        result = subprocess.run(
            ["git", *argv], cwd=str(_repo_root()),
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception as exc:
        print(f"[jba-log] git {argv[0]} failed for {context}: {exc}")
        return 1, str(exc)
    output = (result.stderr or result.stdout or "").strip()
    if result.returncode != 0:
        print(f"[jba-log] git {argv[0]} failed for {context} "
              f"(rc={result.returncode}): {output}")
    return result.returncode, output


_ARCHIVE_PATHSPEC = "data/jba/jobs"

_last_archive_commit_day: str | None = None


def _current_branch() -> str | None:
    """Checked-out branch name, or None on a detached HEAD.

    Pushing from a detached HEAD has no sensible target -- `git push origin HEAD`
    would either be rejected or, worse, land on whatever the remote's default
    happens to be. Refuse instead of guessing.
    """
    rc, out = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], context="branch lookup")
    if rc != 0 or not out or out == "HEAD":
        return None
    return out


def _archive_changes_staged() -> bool:
    """Whether staging produced anything to commit under the archive pathspec."""
    rc, out = _run_git(
        ["diff", "--cached", "--name-only", "--", _ARCHIVE_PATHSPEC],
        context="staged archive check",
    )
    return rc == 0 and bool(out)


def commit_archives(*, push: bool | None = None) -> bool:
    """Commit (and optionally push) every job archive zip. Returns True if a
    commit was made.

    Scoped to data/jba/jobs throughout -- `git add -A -- <pathspec>` and a
    pathspec-scoped `git commit`. The bot's working tree routinely holds
    unrelated edits, and an unscoped commit here would sweep them into an
    archive commit.

    No -f on the add, ever. The zips are already un-ignored by the
    `!data/jba/jobs/*/*.zip` negation, so -f buys nothing -- and applied to a
    directory pathspec it overrides .gitignore for everything underneath,
    which force-adds jobs.db (137MB), archive_index.db (294MB) and their WAL
    sidecars. GitHub's 100MB limit rejects that push; a self-hosted remote
    would simply accept it.

    Covers weekly zips as well as the consolidated monthly ones. Only the
    monthly zip was ever committed before, so a week's archive sat untracked
    until month-end consolidation folded it in -- and if the host was lost in
    between, that data was gone.

    Push failures are reported and left alone rather than retried with a
    rebase: --autostash on a tree carrying the operator's own uncommitted work
    is not a risk worth taking unattended. The next run tries again.
    """
    if not _archive_commit_enabled():
        return False

    if _run_git(["rev-parse", "--git-dir"], context="repo check")[0] != 0:
        return False

    if _run_git(["add", "-A", "--", _ARCHIVE_PATHSPEC], context="archive staging")[0] != 0:
        return False

    if not _archive_changes_staged():
        return False

    rc, _ = _run_git(
        [*_git_identity(), "commit", "-m", _archive_commit_message(), "--", _ARCHIVE_PATHSPEC],
        context="archive commit",
    )
    if rc != 0:
        return False
    print("[jba-log] Committed job archives.")

    if push is None:
        push = _archive_push_enabled()
    if push:
        _push_archives()
    return True


def _archive_commit_message() -> str:
    """Summarise the staged archive change, e.g. 'archive: 2026-08 job data'."""
    _, out = _run_git(
        ["diff", "--cached", "--name-only", "--", _ARCHIVE_PATHSPEC],
        context="archive commit message",
    )
    months = sorted({
        Path(line).parent.name
        for line in out.splitlines()
        if line.strip()
    })
    if not months:
        return "archive: job data"
    if len(months) == 1:
        return f"archive: {months[0]} job data"
    return f"archive: job data ({months[0]}..{months[-1]})"


def _resolve_remote() -> str | None:
    """The remote to push to: JBA_GIT_REMOTE, else the only one, else origin.

    Not hardcoded to "origin": this repo's remote is named DiscordBot, and a
    hardcoded default would have made the push fail on the one machine the
    feature exists for. A single remote is unambiguous whatever it is called;
    with several, "origin" is the sane convention and an explicit setting is
    available for the rest.
    """
    explicit = (os.getenv("JBA_GIT_REMOTE") or "").strip()
    if explicit:
        return explicit
    rc, out = _run_git(["remote"], context="remote lookup")
    if rc != 0:
        return None
    remotes = [line.strip() for line in out.splitlines() if line.strip()]
    if not remotes:
        return None
    if len(remotes) == 1:
        return remotes[0]
    return "origin" if "origin" in remotes else remotes[0]


def _push_archives() -> bool:
    branch = _current_branch()
    if branch is None:
        print("[jba-log] Detached HEAD; skipping archive push.")
        return False
    remote = _resolve_remote()
    if remote is None:
        print("[jba-log] No git remote configured; archive commit stays local.")
        return False
    rc, out = _run_git(["push", remote, branch], context="archive push", timeout=180)
    if rc != 0:
        print(f"[jba-log] Archive push rejected; the commit is local and the "
              f"next run will retry. ({out.splitlines()[-1] if out else 'no detail'})")
        return False
    print(f"[jba-log] Pushed job archives to {remote}/{branch}.")
    return True


def commit_archives_daily() -> bool:
    """Commit archives at most once per UTC day. Returns True if it committed.

    The current week lives in jobs.db, which is gitignored, so most days there
    is nothing new to commit and this is a cheap no-op; a weekly rollover or a
    monthly consolidation is what actually produces a committable zip. Running
    daily means the archive reaches the remote within a day of being written
    rather than waiting for month-end.
    """
    global _last_archive_commit_day
    if not _archive_commit_enabled():
        return False
    today = _today_str()
    if _last_archive_commit_day == today:
        return False
    _last_archive_commit_day = today
    try:
        return commit_archives()
    except Exception as exc:
        # Never let archive bookkeeping take down the caller's loop.
        print(f"[jba-log] Daily archive commit failed: {exc}")
        return False


def _git_commit_monthly(zip_path: Path, month: str, job_count: int) -> bool:
    """Commit a freshly consolidated monthly archive. Returns whether it did.

    Opt-in via JBA_ARCHIVE_GIT_COMMIT. A service account with a writable .git is
    a liability, and a local auto-commit collides with pull-based deploys, so
    this stays off unless explicitly asked for.

    Consolidation deletes the weekly zips it folded in, so this stages the whole
    archive pathspec rather than just the new zip -- committing the addition
    while leaving those deletions unstaged would leave the tree permanently
    dirty. The commit message keeps the month and job count, which the generic
    daily path cannot know.
    """
    if not _archive_commit_enabled():
        return False

    if _run_git(["add", "-A", "--", _ARCHIVE_PATHSPEC], context=month)[0] != 0:
        return False
    if not _archive_changes_staged():
        return False

    rc, _ = _run_git(
        [*_git_identity(), "commit",
         "-m", f"archive: {month} job data ({job_count:,} jobs)",
         "--", _ARCHIVE_PATHSPEC],
        context=month,
    )
    if rc != 0:
        return False

    if _archive_push_enabled():
        _push_archives()
    return True


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ATS job identity
# ---------------------------------------------------------------------------
#
# On an ATS board the URL is not the identity -- the provider's job id is. Two
# things follow, and both were getting this wrong before:
#
#   * One posting appears under several URLs. Greenhouse serves the same job
#     from boards.greenhouse.io, job-boards.greenhouse.io and the company's own
#     domain; iCIMS puts the MUTABLE title slug in the path, so a retitled job
#     changes URL. Measured on the stored archive: 5,282 iCIMS ids under two or
#     more slugs, 353 Greenhouse ids under two or more URLs.
#   * date_posted is not evidence of a re-post. Greenhouse reports updated_at,
#     which bumps on any edit; a genuine re-post gets a NEW id and therefore a
#     new key. Measured: 62,503 ids carrying multiple dates, all but one of them
#     inside the dedup window.
#
# Each extractor returns None unless it is certain. None is the safe answer --
# the caller falls back to the URL key and the record is treated exactly as it
# was before.

_RE_GH_HOST = re.compile(r"^(?:job-)?boards(?:\.eu)?\.greenhouse\.io$", re.I)
_RE_GH_PATH = re.compile(r"/(?:embed/job_app|[^/]+/jobs)/(\d+)")
_RE_ICIMS_HOST = re.compile(r"\.icims\.com$", re.I)
_RE_ICIMS_PATH = re.compile(r"/jobs/(\d+)(?:/|$)")
_RE_UUID_PATH = re.compile(
    r"^/([^/]+)/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
# Workday paths end in /job/{Location}/{Title-Slug}_{REQID}: JR10108, R-12345,
# 100019283. Requiring >=3 digits stops a title ending in "_2" being read as one.
_RE_WORKDAY_REQ = re.compile(r"_([A-Za-z]{0,4}-?\d{3,}[\w-]*)$")
_RE_WORKDAY_NUMERIC = re.compile(r"/(\d{4,})(?:$|\?)")
_RE_BAMBOO_PATH = re.compile(r"^/careers/(\d+)")


def _ats_job_id(job: dict[str, Any]) -> str | None:
    """Provider-assigned job id for an ATS listing, or None when not certain."""
    url = str(job.get("job_url") or job.get("url") or job.get("link") or "").strip()
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    # Greenhouse: gh_jid is globally unique and appears on greenhouse-hosted
    # boards AND company-hosted embeds, so it must NOT be host-qualified -- that
    # is precisely what unifies the several URLs one posting is served under.
    try:
        gh_jid = (parse_qs(parsed.query).get("gh_jid") or [""])[0]
    except ValueError:
        gh_jid = ""
    if gh_jid.isdigit():
        return f"greenhouse:{gh_jid}"
    if _RE_GH_HOST.match(host):
        match = _RE_GH_PATH.search(path)
        if match:
            return f"greenhouse:{match.group(1)}"

    # iCIMS ids are per-tenant (job 4801 exists on every board), so the host
    # stays in the key. The mutable title slug after the id is dropped.
    if _RE_ICIMS_HOST.search(host):
        match = _RE_ICIMS_PATH.search(path)
        if match:
            return f"icims:{host}:{match.group(1)}"

    if host.endswith("jobs.ashbyhq.com"):
        match = _RE_UUID_PATH.match(path)
        if match:
            return f"ashby:{match.group(2).lower()}"

    if host.endswith("jobs.lever.co"):
        match = _RE_UUID_PATH.match(path)
        if match:
            return f"lever:{match.group(2).lower()}"

    if host.endswith(".bamboohr.com"):
        match = _RE_BAMBOO_PATH.match(path)
        if match:
            return f"bamboohr:{host}:{match.group(1)}"

    if "myworkdayjobs.com" in host:
        match = _RE_WORKDAY_REQ.search(path.rstrip("/").rsplit("/", 1)[-1])
        if match:
            return f"workday:{host}:{match.group(1).upper()}"
        match = _RE_WORKDAY_NUMERIC.search(path)
        if match:
            return f"workday:{host}:{match.group(1)}"

    return None


def _dedup_key(job: dict[str, Any]) -> str:
    ats_id = _ats_job_id(job)
    if ats_id:
        return f"ats:{ats_id}"

    url = job.get("job_url") or job.get("url") or job.get("link") or ""
    # Fallback for records tagged Workday whose URL is not a myworkdayjobs host
    # (proxied or shortened links). Kept verbatim so their existing keys stand.
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


# seen_urls.json entries carried over from the pre-timestamp era record this
# instead of a date. It means "seen, date unknown" -- never a real ordering.
_MIGRATED_SENTINEL = "migrated"

# Marks a key produced from a provider job id rather than a URL. See _identity.
_ATS_KEY_PREFIX = "ats:"


def _posted_key(value: Any) -> str:
    """Normalized date_posted. '' means the listing carried no usable date."""
    text = str(value or "").strip()
    return "" if not text or text == _MIGRATED_SENTINEL else text


def _identity(job: dict[str, Any]) -> tuple[str, str]:
    """(base_key, date_posted) for a listing. date_posted is '' when absent.

    An ATS-keyed listing always reports '' for the date, which makes _collides
    treat it as colliding with ANY prior sighting of the same id. That is not a
    special case bolted on: '' means "carries no evidence of being a new
    posting", and on an ATS board the date genuinely carries none, because a
    real re-post is issued a new job id and therefore a different base_key. The
    date rule below stays in force for every other source, where a URL really is
    reused across postings.
    """
    key = _dedup_key(job) or ""
    if key.startswith(_ATS_KEY_PREFIX):
        return key, ""
    return key, _posted_key(job.get("date_posted"))


def _collides(posted: str, seen_dates: set[str]) -> bool:
    """Is a listing dated *posted* a duplicate of any sighting in *seen_dates*?

    Asymmetric, deliberately, and keyed on the INCOMING listing:

      * No date_posted  -> collides with ANY prior sighting of this job. It
        carries no evidence of being a new posting, so it is not allowed to
        masquerade as one.
      * Has date_posted -> collides only with a sighting carrying the SAME date.
        A different date is a genuine re-post and is kept.

    Defined here rather than in archive_index because log_jobs must apply the
    rule even when the archive index is unavailable, and because two copies of
    this rule is exactly how the write path drifted out of step with the scrape
    path in the first place. archive_index.collides delegates to this.
    """
    if not seen_dates:
        return False
    if posted == "":
        return True
    return posted in seen_dates


def _merge_sightings(into: dict[str, set[str]], other: dict[str, set[str]]) -> dict[str, set[str]]:
    for base, dates in other.items():
        into.setdefault(base, set()).update(dates)
    return into


def _already_logged(conn: sqlite3.Connection, keys: list[str]) -> dict[str, set[str]]:
    """base_key -> date_posted values already in the jobs table, on ANY day.

    Replaces the old seen_urls lookup. The jobs table holds only the current
    week -- previous weeks are exported to zips and purged -- so this covers the
    same ground seen_urls did for recent data, without a second table to keep in
    step. Older weeks are covered by the archive index (see _already_archived).

    Returns sightings rather than a bare key set: suppressing on the key alone
    would drop a genuine re-post (same URL, new date_posted), and because the
    drop happens before the insert nothing would ever record the re-post -- so
    the scrape path, which does apply the date rule, would surface it again on
    every cycle. The date must survive to the collision check.
    """
    if not keys:
        return {}
    found: dict[str, set[str]] = {}
    for i in range(0, len(keys), 500):
        batch = keys[i:i + 500]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT dedup_key, data FROM jobs WHERE dedup_key IN ({placeholders})", batch
        )
        for base, payload in rows:
            try:
                record = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                record = {}
            # Through _identity, not the raw date_posted field: an ATS-keyed
            # row must report '' here too, or the stored date would come back
            # and re-enable the date rule for exactly the rows it must not
            # apply to.
            posted = _identity(record)[1] if isinstance(record, dict) else ""
            found.setdefault(base, set()).add(posted)
    return found


def _already_archived(jobs: list[dict[str, Any]]) -> dict[str, set[str]]:
    """base_key -> date_posted values sighted in the zip archives, in-window.

    Imported locally: archive_index imports _dedup_key from this module, so a
    module-level import would be circular. Failures are swallowed -- the archive
    is an optimisation and must never break logging.

    Returns recent_sightings unchanged. It previously returned `set(sightings)`,
    which keeps the dict's KEYS and discards the dates, collapsing the date rule
    into "seen this URL at all" and permanently rejecting every re-post.
    """
    try:
        from services.jba import archive_index

        archive_index.ensure_index()
        return archive_index.recent_sightings(_dedup_key(job) for job in jobs)
    except Exception:
        return {}


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_jobs(jobs: list[dict[str, Any]], date_str: str | None = None) -> int:
    """Merge *jobs* into today's (or *date_str*'s) daily log.

    Deduplicates by dedup key against the jobs table (current week) and the zip
    archives (older weeks), so a job is logged once rather than re-appearing
    each day it stays open.
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

        # Key-only, deliberately: UNIQUE(date_key, dedup_key) means one row per
        # key per day regardless of date_posted, so a same-day repeat cannot be
        # stored even if the date rule would call it new.
        existing_keys: set[str] = {
            row[0]
            for row in conn.execute(
                "SELECT dedup_key FROM jobs WHERE date_key = ?", (date_str,)
            )
        }

        candidates: list[tuple[str, str, dict[str, Any]]] = []
        for job in jobs:
            key, posted = _identity(job)
            if not key or key in existing_keys:
                continue
            if not _is_fresh_enough(job, now):
                continue
            candidates.append((key, posted, job))

        if not candidates:
            return 0

        candidate_keys = [k for k, _, _ in candidates]
        # Current week (jobs table) + older weeks (zip archives), as
        # base_key -> {date_posted}. Both layers apply the same date rule, so a
        # genuine re-post survives to be written; suppressing on the key alone
        # would drop it here forever while the scrape path kept re-surfacing it.
        sightings = _already_logged(conn, candidate_keys)
        _merge_sightings(sightings, _already_archived([job for _, _, job in candidates]))

        new_rows: list[tuple[str, str, str, str]] = []
        for key, posted, job in candidates:
            if _collides(posted, sightings.get(key, set())):
                continue
            job.setdefault("scraped_at", now)
            new_rows.append((date_str, key, job.get("scraped_at", now), json.dumps(job, default=str)))
            existing_keys.add(key)
            # Two copies of one identity in a single batch: the first wins.
            sightings.setdefault(key, set()).add(posted)

        if new_rows:
            # The jobs row IS the record of having seen this key; there is no
            # longer a separate table to mark.
            conn.executemany(
                "INSERT OR IGNORE INTO jobs (date_key, dedup_key, scraped_at, data) VALUES (?, ?, ?, ?)",
                new_rows,
            )
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
