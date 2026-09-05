"""Which company boards have posted in a given country, learned from the archive.

The ATS fleet is collected globally -- neither the harvester nor the scraper has
any notion of country -- while a channel is usually scoped to one. Measured over
a 3-day window, 1.9% of archived ATS rows were Canadian against roughly 29% US.
So the great majority of every cycle's request budget buys postings that cannot
survive the channel's own region gate.

The archive already knows which boards are worth asking first. Every ATS row it
holds carries the platform slug in `company` and a free-text `location`, and
months of them sit in data/jba/jobs/**. Which countries a company posts in is a
property of the company, not of whichever channel scraped it, so the signal is
safe to share across an archive that many channels read.

**Country-general on purpose.** The immediate need is Canada, but nothing here is
Canada-shaped: the index is keyed by country, so a channel scoped to Germany or
the US gets the same treatment by passing a different code. Hardcoding one
country would have made the next region a rewrite rather than an argument.

**This orders, it never filters.** `rank()` is a stable partition: preferred slugs
move to the front, everything else keeps its relative order behind them, and
nothing is dropped. On a cycle that completes -- the common case, 73-98% -- the
ordering changes nothing at all, because every slug is asked either way. It only
bites on a cycle cut off at the budget, where the tail is cancelled: there, the
boards most likely to yield a relevant job are now inside the part that ran.

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

DEFAULT_COUNTRY = "CA"

# Full names and codes for the subdivisions of the two countries whose codes
# actually collide. Everywhere else a country tail is unambiguous on its own.
_CA_SUBDIVISION_NAMES: frozenset[str] = frozenset({
    "alberta", "british columbia", "manitoba", "new brunswick",
    "newfoundland and labrador", "newfoundland", "labrador", "nova scotia",
    "ontario", "prince edward island", "quebec", "québec", "saskatchewan",
    "northwest territories", "nunavut", "yukon",
})
_CA_SUBDIVISION_CODES: frozenset[str] = frozenset({
    "ab", "bc", "mb", "nb", "nl", "ns", "nt", "nu", "on", "pe", "qc", "sk", "yt",
})
_US_SUBDIVISION_CODES: frozenset[str] = frozenset({
    "al", "ak", "az", "ar", "co", "ct", "de", "fl", "ga", "hi", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo",
    "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or",
    "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi",
    "wy", "dc",
})

# Spelled-out country names worth recognising; anything else falls through to
# its two-letter tail, which is what boards overwhelmingly publish.
_COUNTRY_NAMES: dict[str, str] = {
    "canada": "CA",
    "united states": "US", "united states of america": "US", "usa": "US",
    "united kingdom": "GB", "great britain": "GB",
    "germany": "DE", "france": "FR", "netherlands": "NL", "sweden": "SE",
    "australia": "AU", "india": "IN", "ireland": "IE", "spain": "ES",
    "italy": "IT", "poland": "PL", "brazil": "BR", "mexico": "MX",
    "japan": "JP", "singapore": "SG", "switzerland": "CH", "norway": "NO",
    "denmark": "DK", "finland": "FI", "belgium": "BE", "portugal": "PT",
}
# Two-letter codes that are BOTH a US state and a country. "Toronto, ON, CA" is
# Canada and "San Francisco, CA" is California; "Berlin, DE" is Germany and
# "Dover, DE" is Delaware. Structurally identical, so neither can be decided
# from the code alone -- only from what sits beside it. Reading DE as Delaware
# is how a first cut mislabelled every German posting as American.
_AMBIGUOUS_CODES: frozenset[str] = frozenset({
    "ca",  # California / Canada
    "de",  # Delaware / Germany
    "in",  # Indiana / India
    "la",  # Louisiana / Laos
    "md",  # Maryland / Moldova
    "mt",  # Montana / Malta
    "ne",  # Nebraska / Niger
    "pa",  # Pennsylvania / Panama
    "ga",  # Georgia the state / Georgia the country
    "id",  # Idaho / Indonesia
    "al",  # Alabama / Albania
    "ar",  # Arkansas / Argentina
    "mo",  # Missouri / Macau
    "ms",  # Mississippi / Montserrat
    "sc",  # South Carolina / Seychelles
    "va",  # Virginia / Vatican City
})

# Codes some boards use in place of the ISO two-letter form.
_COUNTRY_ALIASES: dict[str, str] = {"can": "CA", "usa": "US", "uk": "GB", "gbr": "GB"}

_WS = re.compile(r"\s+")


def _segments(location: str) -> list[str]:
    return [_WS.sub(" ", part.strip().casefold()) for part in str(location or "").split(",")]


def country_of(location: str) -> str:
    """The ISO-ish country code a free-text location names, or "" if unclear.

    The whole difficulty is `CA`. "Toronto, ON, CA" is Canada and "San
    Francisco, CA" is California, and both end in the same two letters -- so a
    bare two-letter tail can never decide on its own. It is resolved by what
    sits beside it: a Canadian province means Canada, a US state means the US,
    and neither means unknown.

    Returning "" rather than guessing is deliberate. This decides which boards
    are asked *first*, so an unknown costs a little ordering while a wrong
    answer spends the priority slots on the wrong country -- worse than not
    prioritising at all.
    """
    parts = [p for p in _segments(location) if p]
    if not parts:
        return ""

    for part in parts:
        if part in _COUNTRY_NAMES:
            return _COUNTRY_NAMES[part]

    tail = parts[-1]
    if tail in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[tail]

    if tail in _AMBIGUOUS_CODES:
        # Decide from the subdivision beside it, never from the code alone.
        others = parts[:-1]
        if any(p in _CA_SUBDIVISION_CODES or p in _CA_SUBDIVISION_NAMES for p in others):
            return "CA"
        if any(p in _US_SUBDIVISION_CODES for p in others):
            return "US"
        # Undecidable without a gazetteer. "" costs a little ordering; guessing
        # would spend the priority slots on the wrong country, which is worse
        # than not prioritising at all.
        return ""

    if any(p in _CA_SUBDIVISION_NAMES for p in parts):
        return "CA"

    if len(tail) == 2 and tail.isalpha():
        # An unambiguous US state code written without its country ("Austin, TX").
        if tail in _US_SUBDIVISION_CODES:
            return "US"
        return tail.upper()

    return ""


def _is_canadian(location: str) -> bool:
    """Kept as the country-specific reading of country_of, for readability at
    call sites that genuinely only care about Canada."""
    return country_of(location) == "CA"


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
    """The country index, or an empty one.

    An unreadable or malformed cache reads as empty rather than raising. It is
    derived data: losing it costs one rebuild, while refusing to start would
    take the scrape down for an optimisation that is allowed to be absent.
    """
    empty: dict[str, Any] = {"by_country": {}, "sources": {}}
    try:
        data = json.loads(Path(path or _CACHE_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(data, dict) or not isinstance(data.get("by_country"), dict):
        return empty
    return data


def save_cache(cache: dict[str, Any], path: Path | None = None) -> None:
    """Write atomically -- a rebuild killed mid-write must not leave a half
    file that the next run discards and rebuilds from scratch again."""
    p = Path(path or _CACHE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def build(rows: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """{country: {company slugs seen posting there}}. Pure, so it is testable
    without an archive on disk."""
    found: dict[str, set[str]] = {}
    for company, location in rows:
        slug = str(company or "").strip().casefold()
        if not slug:
            continue
        code = country_of(location)
        if code:
            found.setdefault(code, set()).add(slug)
    return found


def refresh(force: bool = False, cache_path: Path | None = None) -> dict[str, int]:
    """Rebuild the index if any archive changed. Returns {country: slug count}.

    Rebuilds on a source's size or mtime moving, the same staleness test
    archive_index uses -- rescanning every zip on every scrape cycle would cost
    far more than the ordering it buys.
    """
    cache = load_cache(cache_path)
    stats = _source_stats()
    if not force and cache.get("sources") == stats and cache.get("by_country"):
        return {k: len(v) for k, v in cache["by_country"].items()}

    rows: list[tuple[str, str]] = []
    if _JOBS_DIR.exists():
        for zip_path in sorted(_JOBS_DIR.glob("*/*.zip")):
            rows.extend(_iter_rows(zip_path))
    rows.extend(_iter_db_rows(_DB_PATH))

    by_country = build(rows)
    save_cache(
        {"by_country": {k: sorted(v) for k, v in sorted(by_country.items())}, "sources": stats},
        cache_path,
    )
    return {k: len(v) for k, v in by_country.items()}


def slugs_for(country: str = DEFAULT_COUNTRY, cache_path: Path | None = None) -> frozenset[str]:
    """Company slugs known to post in `country`."""
    by_country = load_cache(cache_path).get("by_country") or {}
    return frozenset(by_country.get(str(country or "").strip().upper()) or ())


def countries(cache_path: Path | None = None) -> dict[str, int]:
    """Every country the archive has seen, with how many boards post there.
    Useful for deciding whether a channel's region is worth prioritising at
    all before wiring it in."""
    by_country = load_cache(cache_path).get("by_country") or {}
    return {k: len(v) for k, v in sorted(by_country.items(), key=lambda kv: -len(kv[1]))}


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


def rank(
    slugs: list[str],
    preferred: Iterable[str] | None = None,
    country: str = DEFAULT_COUNTRY,
) -> list[str]:
    """Boards that post in `country` first, everything else after, order kept.

    A stable partition and nothing more. Every input slug appears exactly once
    in the output, so a caller that submits the result submits the same work it
    would have anyway -- only the order it is queued in changes, which is what
    decides who survives a cycle that gets cut off.

    `preferred` overrides the country lookup, which is what makes this testable
    without an archive and lets a caller supply its own ordering rule.
    """
    front, back = partition(slugs, preferred, country)
    return front + back


def partition(
    slugs: list[str],
    preferred: Iterable[str] | None = None,
    country: str = DEFAULT_COUNTRY,
) -> tuple[list[str], list[str]]:
    """The same stable partition `rank` performs, with the boundary still visible.

    `rank` concatenates the two groups, so a caller that wants to treat them
    differently -- reordering only the part a cut-off cycle never reaches, while
    leaving the preferred head where it is -- cannot recover the boundary
    without redoing the matching. Returning it keeps one definition of
    "preferred" rather than a second, drifting copy at the call site.

    `front + back` is always exactly the input, in the input's relative order
    within each group.
    """
    if preferred is None:
        preferred = slugs_for(country)
    wanted = {str(s).strip().casefold() for s in preferred}
    if not wanted:
        # A fast path, not a behavioural branch: with nothing preferred the
        # loop below would put every slug in `back` and return the same order.
        # Mutation confirms no test can distinguish them -- it is here to skip
        # a 26,000-element pass on a checkout with no archive yet.
        return [], list(slugs)
    front: list[str] = []
    back: list[str] = []
    for slug in slugs:
        (front if any(k in wanted for k in match_keys(slug)) else back).append(slug)
    return front, back
