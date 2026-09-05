"""Which company boards have ever posted a Canadian job, learned from the archive.

The ATS fleet is collected globally -- neither the harvester nor the scraper has
any notion of country -- while a channel is usually scoped to one. Measured over
a 3-day window, 1.9% of archived ATS rows were Canadian against roughly 29% US.
So the great majority of every cycle's request budget buys postings that cannot
survive the channel's own region gate.

The archive already knows which companies are worth asking first. Every ATS row
it holds carries the platform slug in `company` and a free-text `location`, and
months of them are sitting in data/jba/jobs/**. A company that has ever posted a
Canadian job is a property of the company, not of whichever channel scraped it,
so this signal is safe to share across an archive that many channels read.

**This orders, it never filters.** `rank()` is a stable partition: Canada-yielding
slugs move to the front, everything else keeps its relative order behind them,
and nothing is dropped. On a cycle that completes -- the common case, 73-98% --
the ordering changes nothing at all, because every slug is asked either way. It
only bites on a cycle that is cut off, where the tail is cancelled: there, the
companies most likely to yield a relevant job are now inside the part that ran.
No filter is relaxed and no extra request is made.

Derived data, like archive_index: delete the cache and it rebuilds.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

_JBA_DIR = Path(__file__).resolve().parents[3] / "data" / "jba"
_JOBS_DIR = _JBA_DIR / "jobs"
_CACHE_PATH = _JOBS_DIR / "geo_priority.json"
_DB_PATH = _JOBS_DIR / "jobs.db"

# Full province and territory names, plus the two-letter codes boards publish.
# Codes are matched only as whole segments and only alongside a Canadian
# country tail, because two-letter codes collide with English words and with US
# state codes -- see _is_canadian.
_PROVINCE_NAMES: frozenset[str] = frozenset({
    "alberta", "british columbia", "manitoba", "new brunswick",
    "newfoundland and labrador", "newfoundland", "labrador", "nova scotia",
    "ontario", "prince edward island", "quebec", "québec", "saskatchewan",
    "northwest territories", "nunavut", "yukon",
})
_PROVINCE_CODES: frozenset[str] = frozenset({
    "ab", "bc", "mb", "nb", "nl", "ns", "nt", "nu", "on", "pe", "qc", "sk", "yt",
})
# "ca" is the tail on both "Toronto, ON, CA" and "San Francisco, CA". Only the
# first is Canada, and the difference is whether a province sits beside it.
_CANADA_TAILS: frozenset[str] = frozenset({"canada", "ca", "can"})

_WS = re.compile(r"\s+")


def _segments(location: str) -> list[str]:
    return [_WS.sub(" ", part.strip().casefold()) for part in str(location or "").split(",")]


def _is_canadian(location: str) -> bool:
    """Whether a free-text location names somewhere in Canada.

    The whole difficulty is `CA`. "Toronto, ON, CA" is Canada and "San
    Francisco, CA" is California, and both end in the same two letters -- so a
    bare tail is never enough on its own. It counts only when a province sits
    beside it, or when the word Canada is spelled out.

    Deliberately conservative: this decides which companies are asked *first*,
    so a false negative costs a little priority and a false positive spends the
    budget on the wrong boards.
    """
    parts = [p for p in _segments(location) if p]
    if not parts:
        return False
    if any(p == "canada" for p in parts):
        return True
    if any(p in _PROVINCE_NAMES for p in parts):
        return True
    # A country-ish tail only counts with a province code beside it.
    if parts[-1] in _CANADA_TAILS and any(p in _PROVINCE_CODES for p in parts[:-1]):
        return True
    return False


def _iter_rows(path: Path) -> Iterator[tuple[str, str]]:
    """(company, location) for every ATS-shaped record in one archive zip."""
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        return
    with archive:
        for name in archive.namelist():
            if name == "seen_urls.json" or not name.endswith(".json"):
                continue
            try:
                records = json.loads(archive.read(name))
            except (json.JSONDecodeError, OSError, ValueError):
                continue
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                # ATS rows carry _source_site; channel-posted rows carry `site`.
                if not record.get("_source_site"):
                    continue
                company = str(record.get("company") or "").strip()
                if company:
                    yield company, str(record.get("location") or "")


def _iter_db_rows(path: Path) -> Iterator[tuple[str, str]]:
    """Same, for the current week's live jobs.db."""
    if not path.exists():
        return
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return
    try:
        for (blob,) in conn.execute("SELECT data FROM jobs"):
            try:
                record = json.loads(blob)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(record, dict) or not record.get("_source_site"):
                continue
            company = str(record.get("company") or "").strip()
            if company:
                yield company, str(record.get("location") or "")
    except sqlite3.Error:
        return
    finally:
        conn.close()


def _source_stats() -> dict[str, list[float]]:
    stats: dict[str, list[float]] = {}
    paths = sorted(_JOBS_DIR.glob("*/*.zip")) if _JOBS_DIR.exists() else []
    for path in list(paths) + [_DB_PATH]:
        try:
            st = path.stat()
        except OSError:
            continue
        stats[path.name] = [st.st_size, st.st_mtime]
    return stats


def load_cache(path: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads(Path(path or _CACHE_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"canada_slugs": [], "sources": {}}
    if not isinstance(data, dict) or not isinstance(data.get("canada_slugs"), list):
        return {"canada_slugs": [], "sources": {}}
    return data


def save_cache(cache: dict[str, Any], path: Path | None = None) -> None:
    p = Path(path or _CACHE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def build(rows: Iterable[tuple[str, str]]) -> set[str]:
    """Company slugs with at least one Canadian posting. Pure, so it is testable
    without an archive on disk."""
    found: set[str] = set()
    for company, location in rows:
        slug = str(company or "").strip().casefold()
        if slug and _is_canadian(location):
            found.add(slug)
    return found


def refresh(force: bool = False, cache_path: Path | None = None) -> int:
    """Rebuild the cache if any archive changed. Returns the slug count.

    Rebuilds on a source's size or mtime moving, the same staleness test
    archive_index uses -- a full rescan of every zip on every scrape cycle would
    cost far more than the ordering it buys.
    """
    cache = load_cache(cache_path)
    stats = _source_stats()
    if not force and cache.get("sources") == stats and cache.get("canada_slugs"):
        return len(cache["canada_slugs"])

    rows: list[tuple[str, str]] = []
    if _JOBS_DIR.exists():
        for zip_path in sorted(_JOBS_DIR.glob("*/*.zip")):
            rows.extend(_iter_rows(zip_path))
    rows.extend(_iter_db_rows(_DB_PATH))

    slugs = build(rows)
    save_cache({"canada_slugs": sorted(slugs), "sources": stats}, cache_path)
    return len(slugs)


def canada_slugs(cache_path: Path | None = None) -> frozenset[str]:
    return frozenset(load_cache(cache_path).get("canada_slugs") or ())


def match_keys(slug: str) -> tuple[str, ...]:
    """The forms a fleet slug might be recorded under in the archive.

    Most platforms store the slug verbatim, but Workday's is a
    `tenant|host|site` triple while its rows record only the tenant -- so
    `ahri|wd3|ahri1` in the fleet is `ahri` in the archive and an exact match
    finds nothing. Measured before this existed: workday and paylocity both
    reported 0 Canada-yielding companies out of fleets of 14,990 and 26,160,
    which is what exposed it.
    """
    text = str(slug or "").strip().casefold()
    if not text:
        return ()
    if "|" in text:
        return (text, text.split("|", 1)[0])
    return (text,)


def rank(slugs: list[str], preferred: Iterable[str] | None = None) -> list[str]:
    """Canada-yielding slugs first, everything else after, order preserved.

    A stable partition and nothing more. Every input slug appears exactly once
    in the output, so a caller that submits the result submits the same work it
    would have anyway -- only the order it is queued in changes, which is what
    decides who survives a cycle that gets cut off.
    """
    if preferred is None:
        preferred = canada_slugs()
    wanted = {str(s).strip().casefold() for s in preferred}
    if not wanted:
        # A fast path, not a behavioural branch: with nothing preferred the
        # loop below would put every slug in `back` and return the same order.
        # Mutation confirms no test can distinguish them -- it is here to skip
        # a 26,000-element pass on a checkout with no archive yet.
        return list(slugs)
    front: list[str] = []
    back: list[str] = []
    for slug in slugs:
        (front if any(k in wanted for k in match_keys(slug)) else back).append(slug)
    return front + back
