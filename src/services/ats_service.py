"""ATS platform scrapers for Greenhouse, Lever, Ashby, Workday, and iCIMS.

Fetches job listings directly from company career board APIs rather than
public aggregators. Based on: https://github.com/Feashliaa/job-board-aggregator
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import xml.etree.ElementTree as ET
from datetime import date, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import datetime as _dt
import html
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

import requests

from services import capacity
from services import job_level
from services.net_util import retry_backoff_delay

GREENHOUSE = "greenhouse"
LEVER = "lever"
ASHBY = "ashby"
WORKDAY = "workday"
ICIMS = "icims"
BAMBOOHR = "bamboohr"
WORKABLE = "workable"
BREEZY = "breezy"
SMARTRECRUITERS = "smartrecruiters"
RECRUITEE = "recruitee"
TEAMTAILOR = "teamtailor"
RIPPLING = "rippling"
JAZZHR = "jazzhr"
JOBVITE = "jobvite"
APPLICANTPRO = "applicantpro"
PAYLOCITY = "paylocity"
ORACLE = "oracle"
PERSONIO = "personio"

# Oracle pages by offset and sorts newest-first, so the scrape can stop as soon
# as a page opens older than the archive's write window. Sized against a
# measured tenant: 2,276 requisitions, of which the first ~150 fall inside
# seven days -- one or two pages instead of twelve.
ORACLE_PAGE_SIZE = 200
ORACLE_FRESH_DAYS = 7
# A hard stop as well as the date stop: a tenant whose clock or PostedDate
# field misbehaves must not be able to walk thousands of pages.
ORACLE_MAX_PAGES = 8

ATS_PLATFORMS: tuple[str, ...] = (GREENHOUSE, LEVER, ASHBY, WORKDAY, ICIMS, BAMBOOHR,
                                  WORKABLE, BREEZY, SMARTRECRUITERS, RECRUITEE,
                                  TEAMTAILOR, RIPPLING, JAZZHR, JOBVITE,
                                  APPLICANTPRO, PAYLOCITY, ORACLE, PERSONIO)

PLATFORM_WORKERS: dict[str, int] = {
    GREENHOUSE: 30,
    LEVER: 30,
    ASHBY: 20,
    WORKDAY: 50,
    ICIMS: 30,
    BAMBOOHR: 30,
    # These have no upstream company list, so every slug they scrape comes from
    # the local harvest.
    #
    # The numbers below are deliberately smaller than the six above, because
    # what matters is the total rather than any one platform. watchers/manager
    # fans every platform out at once, so these counts sum: at 20 apiece the
    # fleet went from 190 concurrent HTTP threads to 312. The comment on the
    # scrape fan-out already records that 24 threads on a one-core host was
    # considered too many, and it was written when there were six platforms.
    #
    # The multiplier matters more than the sum. A platform that enriches opens
    # a nested pool per company, so its real ceiling is workers x 5 -- at 20
    # workers that is 100 threads from one platform, and six of these enrich.
    # Halving the enriching ones costs wall-clock on a job that already runs in
    # minutes and buys back most of the increase.
    #
    # Workable and Recruitee are lower again for a different reason: they meter
    # a quota, and eight unpaced workers is what got this address throttled
    # during validation -- 91% of a workable pass came back 429. Four measured
    # clean, so the scraper does not get to be greedier than the validator was.
    WORKABLE: 4,
    RECRUITEE: 4,
    # Enriches per row (schema.org or a detail API), so this is multiplied.
    BREEZY: 8,
    SMARTRECRUITERS: 8,
    RIPPLING: 8,
    # Rendered boards: a page per slug, and a second page per row to enrich.
    JAZZHR: 8,
    JOBVITE: 8,
    APPLICANTPRO: 6,
    # One page per company and everything comes from it, so no nested pool.
    PAYLOCITY: 8,
    # Carries title, location, date and description in one listing response,
    # so it never opens a nested pool and can afford the higher count.
    TEAMTAILOR: 16,
    # New platforms start low. These are politeness ceilings as much as
    # throughput settings, and this repo's rule is that they scale down and
    # never up -- so a first guess should be one that never needs walking back.
    # Oracle tenants are enterprise Fusion instances shared with payroll and
    # finance, which is a further reason not to lean on them.
    ORACLE: 8,
    PERSONIO: 8,
}

# Hard stop on SmartRecruiters paging, so a board that keeps answering cannot
# turn one company into an unbounded crawl.
_SMARTRECRUITERS_MAX_OFFSET: int = 1000

# Jobvite's real locations are only on the posting pages, so a location search
# has to consider more candidates than it will keep. Mirrors the iCIMS numbers.
_JOBVITE_LOCATION_OVERFETCH: int = 4
_JOBVITE_MAX_CANDIDATES: int = 500

DEAD_SLUG_TTL_DAYS: int = 90

# How long a dead mark suppresses a slug before it is probed once more. Without
# this, _is_dead short-circuits every scrape before any request is made, so a
# board that 404d during a one-hour outage could not prove it was back until the
# 90-day TTL evicted it. A failed re-probe re-dates the mark, so a permanently
# dead board costs one request per company per this many days.
DEAD_SLUG_RECHECK_DAYS: int = 7

# Re-probes are spread over this many extra days by a per-slug offset. Without
# it every slug marked before the window expires on the same day: the ~19,800
# currently-dead companies would all be re-probed in a single cycle, and having
# re-dated together they would re-synchronise and do it again every
# DEAD_SLUG_RECHECK_DAYS forever. The offset is derived from the slug, so it is
# stable across restarts and needs nothing stored.
DEAD_SLUG_RECHECK_SPREAD_DAYS: int = 7

# Drop ATS listings already present in the stored job archive (the full history
# in data/jba/jobs/**/*.zip, via services.jba.archive_index). Off switches back
# to the previous behaviour, where the only dedup is the per-channel one
# downstream.
ATS_ARCHIVE_DEDUP_ENABLED: bool = True

REQUEST_TIMEOUT: int = 30

# Lever's US and EU regions are separate deployments and a company exists in
# exactly one. Ordered US-first because that is where most of the list lives.
_LEVER_API_HOSTS: tuple[str, ...] = ("api.lever.co", "api.eu.lever.co")

USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:147.0) Gecko/20100101 Firefox/147.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0",
]


def _make_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h: dict[str, str] = {"User-Agent": random.choice(USER_AGENTS), "Accept": "application/json"}
    if extra:
        h.update(extra)
    return h


_HTTP_LOCAL = threading.local()


class _PreloadedTLSAdapter(requests.adapters.HTTPAdapter):
    """An HTTPS adapter that verifies with one CA store loaded once.

    requests hands urllib3 the CA-bundle *path* for every connection
    (cert_verify sets conn.ca_certs), and urllib3 then creates a fresh
    SSLContext and parses the whole bundle inside ssl_wrap_socket -- per
    connection, not per Session. A pooled Session only helps hosts we
    reconnect to, and most ATS platforms are one host per company. The
    py-spy dump's 166 threads in ssl_wrap_socket were this line.

    So the pool is given one context, verified against the same default
    bundle, and the per-connection path is withheld when verification is the
    default. An explicit bundle (verify="/path", or REQUESTS_CA_BUNDLE, which
    requests turns into a path) still reaches urllib3 unchanged.
    """

    _context: Any = None
    _context_lock = threading.Lock()

    @classmethod
    def context(cls) -> Any:
        with cls._context_lock:
            if cls._context is None:
                from urllib3.util.ssl_ import create_urllib3_context

                ctx = create_urllib3_context()
                ctx.load_verify_locations(requests.utils.DEFAULT_CA_BUNDLE_PATH)
                cls._context = ctx
            return cls._context

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):  # type: ignore[override]
        pool_kwargs.setdefault("ssl_context", self.context())
        return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    def cert_verify(self, conn, url, verify, cert):  # type: ignore[override]
        super().cert_verify(conn, url, verify, cert)
        if verify is True:
            # Verification stays on (cert_reqs is still CERT_REQUIRED); the
            # store that does it is the preloaded context's, not a reload.
            conn.ca_certs = None
            conn.ca_cert_dir = None


def _http() -> requests.Session:
    """The calling thread's requests.Session, created on first use.

    A bare requests.get builds a Session, an adapter and an SSLContext -- and
    reloads the CA bundle -- for every call, then throws them away. A py-spy
    dump of the live bot found 166 of 541 threads inside ssl_wrap_socket for
    exactly that reason. One Session per worker thread keeps the context and
    the pooled connections for the life of the thread; per thread rather than
    one shared Session so no lock sits in front of every fetch and so a
    thread's connection pool is only ever used by that thread.
    """
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.mount("https://", _PreloadedTLSAdapter())
        _configure_from_environment(session)
        _HTTP_LOCAL.session = session
    return session


def _configure_from_environment(session: requests.Session) -> None:
    """Read the proxy and CA environment once, for the life of the Session.

    With trust_env on, requests consults the environment on every request:
    on Windows that is a registry walk for proxy settings plus a ~/.netrc
    lookup per call, both of which showed up in the live profile beside the
    TLS work. Nothing about the environment changes between two fetches from
    the same worker thread, so it is captured here instead. What is kept:
    HTTP(S)_PROXY / ALL_PROXY, and REQUESTS_CA_BUNDLE or CURL_CA_BUNDLE. What
    is given up: per-URL NO_PROXY matching and netrc credentials, neither of
    which any ATS vendor is reached through.
    """
    session.trust_env = False
    proxies = requests.utils.getproxies()
    if proxies:
        session.proxies.update(proxies)
    bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if bundle:
        session.verify = bundle


def _http_get(url: str, **kwargs: Any) -> requests.Response:
    """GET through the thread's Session. The one seam tests stub for HTTP."""
    return _http().get(url, **kwargs)


def _http_post(url: str, **kwargs: Any) -> requests.Response:
    return _http().post(url, **kwargs)


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_JBA_DIR = _DATA_DIR / "ats_companies"
_DEAD_SLUG_DIR = _DATA_DIR / "dead_slugs"

# Companies discovered by scripts/harvest_ats.py, unioned on top of the
# upstream lists. It has to be a separate directory: sync_ats_companies.py
# rewrites every file in _JBA_DIR wholesale from the upstream repo, so anything
# harvested into there is erased on the next sync.
_HARVEST_DIR = _DATA_DIR / "ats_harvest"

# The same harvest as run by .github/workflows/ats-harvest.yml and published to
# the ats-harvest branch, kept apart from the local one for exactly the reason
# above: sync_ats_companies.py owns this directory and replaces it wholesale,
# so a local sweep written into it would be erased on the next sync.
#
# They are not redundant. Measured 2026-09-05, the local files hold 14,278
# slugs the published ones do not -- Wayback discoveries that CI deliberately
# stopped sweeping for (commit d2e68f5) -- while CI holds 243 the local files
# lack. Replaying the current extraction rules over the local files removes
# nothing, so neither set is stale junk, and unioning is the only merge that
# cannot shrink a fleet the bot already scrapes.
_HARVEST_CI_DIR = _DATA_DIR / "ats_harvest_ci"

def _harvest_sources() -> tuple[tuple[str, Path], ...]:
    """Every harvest directory, resolved when it is read rather than at import.

    A tuple frozen at module scope would capture these paths once, so a test
    that redirects `_HARVEST_DIR` would keep reading the real one -- silently,
    and with the real 119k-slug fleet, which is exactly how three isolation
    tests broke when the second source was added. Reading the globals here
    keeps one seam for all of them.

    Order decides nothing but which duplicate is seen first; the merge is a
    set union.
    """
    return (
        ("local harvest", _HARVEST_DIR),
        ("published harvest", _HARVEST_CI_DIR),
    )

_HARVEST_FILES: dict[str, str] = {
    platform: f"{platform}.json" for platform in ATS_PLATFORMS
}

_PLATFORM_FILES: dict[str, str] = {
    # Upstream ships this one under a different name and as {guid, name, jobs}
    # objects; the harvest file is a flat list, which is what is read here.
    PAYLOCITY: "paylocity_companies_clean.json",
    GREENHOUSE: "greenhouse_companies.json",
    LEVER: "lever_companies.json",
    ASHBY: "ashby_companies.json",
    WORKDAY: "workday_companies.json",
    ICIMS: "icims_companies.json",
    BAMBOOHR: "bamboohr_companies.json",
}

_company_cache: dict[str, list[str]] | None = None
_dead_slugs: dict[str, set[str]] = {}           # platform -> set of dead slugs (fast lookup)
_dead_slug_dates: dict[str, dict[str, str]] = {}  # platform -> {slug: iso_date_marked}
_dead_slugs_dirty: set[str] = set()

# Every one of the dicts above is read and mutated from inside the per-slug
# thread pools (30-50 workers per platform), and watchers/manager.py fans all
# ~6 platforms out concurrently, so flush_dead_slugs can run while another
# platform is still marking. Without this lock two threads that both cold-load
# the same platform each parse the file and the second assignment discards the
# first one's marks, and the dirty-set clear at the end of a flush can drop a
# platform marked while the save loop was running. Reentrant because
# _mark_dead/_mark_alive call _load_dead_slugs, and flush calls _save.
_dead_slug_lock = threading.RLock()


def reload_company_lists() -> None:
    global _company_cache
    _company_cache = None


def _company_entry(entry: Any) -> str:
    """One company list entry as a slug the scrapers can actually request.

    Upstream ships paylocity as {guid, name, jobs} objects rather than strings.
    str() on a dict yields "{'guid': '73b6...', 'name': '10 Design', 'jobs': 0}",
    which was then requested as a board id: 10,252 of paylocity's 26,160
    entries -- 39% of its fan-out every cycle -- spent on URLs that cannot
    resolve, and dead-marked under that garbage key when they 404. The comment
    on _PLATFORM_FILES already noted upstream's shape but said the flat harvest
    file "is what is read here", which was not true: both files are unioned.
    """
    if isinstance(entry, dict):
        for key in ("guid", "slug", "id", "company", "name"):
            value = str(entry.get(key) or "").strip()
            if value:
                return value
        return ""
    return str(entry or "").strip()


_company_cache_lock = threading.Lock()


def load_company_lists() -> dict[str, list[str]]:
    global _company_cache
    if _company_cache is not None:
        return _company_cache
    # One loader at a time, and a second check inside: the ATS cycle hands
    # every platform to its own thread at once, and on a cold cache each of
    # them parsed all 135k slugs from disk independently -- five identical
    # "+5333 companies from local harvest" lines interleaved in the log, five
    # copies of the fleet in memory until the last writer won.
    with _company_cache_lock:
        if _company_cache is not None:
            return _company_cache
        _company_cache = _load_company_lists_uncached()
        return _company_cache


def _load_company_lists_uncached() -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    for platform, filename in _PLATFORM_FILES.items():
        path = _JBA_DIR / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                cache[platform] = [s for s in (_company_entry(e) for e in data) if s]
        except Exception as exc:
            print(f"[ats] Failed to load {filename}: {exc}")

    upstream_loaded = bool(cache)

    # Union in every harvest source. Order matters only in that an entry
    # already present wins; the merge is a set union, so a harvest file that is
    # missing, empty or corrupt costs nothing but the companies it would have
    # added -- it can never shrink a list that is already loaded.
    for label, directory in _harvest_sources():
        for platform, filename in _HARVEST_FILES.items():
            path = directory / filename
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[ats] Failed to load {label} {filename}: {exc}")
                continue
            if not isinstance(data, list):
                continue
            existing = cache.get(platform, [])
            seen = set(existing)
            added: list[str] = []
            for raw in data:
                slug = _company_entry(raw)
                if slug and slug not in seen:
                    seen.add(slug)
                    added.append(slug)
            if added:
                cache[platform] = existing + added
                print(f"[ats] {platform}: +{len(added)} companies from {label}")

    if not upstream_loaded:
        # Loud on purpose, and keyed on upstream rather than on the merged
        # result. data/ats_companies is gitignored (it carries its own git
        # history), so on a fresh deployment this directory is absent and every
        # platform silently scrapes zero companies -- indistinguishable from
        # "no jobs matched". Now that the harvest is unioned in, a missing
        # upstream no longer leaves the cache empty, so checking `cache` here
        # would hide exactly the failure this warning exists to surface.
        detail = (f" Only the local harvest is loaded ({sum(len(v) for v in cache.values())} "
                  "companies)." if cache else " ATS scraping will return nothing.")
        print(
            f"[ats] WARNING: no company lists found under {_JBA_DIR}.{detail}"
            " Run: python sync_ats_companies.py"
        )

    return cache


def _load_dead_slugs(platform: str) -> set[str]:
    with _dead_slug_lock:
        return _load_dead_slugs_locked(platform)


def _load_dead_slugs_locked(platform: str) -> set[str]:
    if platform in _dead_slugs:
        return _dead_slugs[platform]
    path = _DEAD_SLUG_DIR / f"{platform}.json"
    today = time.strftime("%Y-%m-%d")
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Legacy format — assign today's date so TTL starts now
                dates: dict[str, str] = {slug: today for slug in data}
            elif isinstance(data, dict):
                dates = data
            else:
                dates = {}
        except Exception:
            dates = {}
    else:
        dates = {}
    _dead_slug_dates[platform] = dates
    _dead_slugs[platform] = set(dates.keys())
    return _dead_slugs[platform]


def _save_dead_slugs(platform: str) -> None:
    """Write a platform's dead-slug map atomically.

    write_text truncates in place, so an interrupted or overlapping write left a
    half-written file that the next _load_dead_slugs parse would discard
    wholesale -- silently resurrecting every dead slug on that platform. tmp +
    os.replace makes a reader see either the old file or the new one.
    """
    dates = _dead_slug_dates.get(platform, {})
    _DEAD_SLUG_DIR.mkdir(parents=True, exist_ok=True)
    target = _DEAD_SLUG_DIR / f"{platform}.json"
    tmp = target.with_suffix(f".json.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(dates, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _purge_expired_dead_slugs(platform: str) -> int:
    """Remove dead slug entries older than DEAD_SLUG_TTL_DAYS. Returns count removed."""
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=DEAD_SLUG_TTL_DAYS)).isoformat()
    dates = _dead_slug_dates.get(platform)
    if not dates:
        return 0
    expired = [slug for slug, marked in dates.items() if marked < cutoff]
    # flush_dead_slugs drives this from _dead_slug_dates, which can name a
    # platform whose fast-lookup set was never populated. Indexing _dead_slugs
    # directly would raise there and abort the whole flush, losing the saves
    # queued behind it.
    lookup = _dead_slugs.setdefault(platform, set())
    for slug in expired:
        del dates[slug]
        lookup.discard(slug)
    if expired:
        _dead_slugs_dirty.add(platform)
    return len(expired)


def flush_dead_slugs() -> None:
    # Purge every loaded platform, not just the dirty ones. Purging only dirty
    # platforms meant the TTL never fired for a platform that recorded no *new*
    # deaths this run, so its expired slugs were skipped forever and those
    # companies were never re-probed. A purge marks its platform dirty, so the
    # save loop below still picks it up.
    with _dead_slug_lock:
        for platform in list(_dead_slug_dates):
            _purge_expired_dead_slugs(platform)
        # Take the dirty set and clear it in the same critical section. Saving
        # first and clearing afterwards let a mark recorded mid-flush be wiped
        # without ever being written.
        pending = sorted(_dead_slugs_dirty)
        _dead_slugs_dirty.clear()
        for platform in pending:
            _save_dead_slugs(platform)


def _recheck_days_for(slug: str) -> int:
    """Per-slug re-probe age, spread across the recheck window. See the
    DEAD_SLUG_RECHECK_SPREAD_DAYS comment for why this is not a constant."""
    if DEAD_SLUG_RECHECK_SPREAD_DAYS <= 0:
        return DEAD_SLUG_RECHECK_DAYS
    digest = hashlib.blake2b(slug.encode("utf-8", "replace"), digest_size=4).digest()
    offset = int.from_bytes(digest, "big") % DEAD_SLUG_RECHECK_SPREAD_DAYS
    return DEAD_SLUG_RECHECK_DAYS + offset


def _is_dead(platform: str, slug: str) -> bool:
    """True when *slug* should be skipped without a request.

    A mark older than this slug's recheck age reports False so it gets one
    probe: either it 200s and _mark_alive clears it, or it 404s again and
    _mark_dead re-dates the mark for another window.
    """
    from datetime import date, timedelta

    with _dead_slug_lock:
        if slug not in _load_dead_slugs_locked(platform):
            return False
        marked = _dead_slug_dates.get(platform, {}).get(slug)
        if not marked:
            return True
        cutoff = (date.today() - timedelta(days=_recheck_days_for(slug))).isoformat()
        return marked > cutoff


def _mark_dead(platform: str, slug: str) -> None:
    with _dead_slug_lock:
        _load_dead_slugs_locked(platform).add(slug)
        _dead_slug_dates.setdefault(platform, {})[slug] = time.strftime("%Y-%m-%d")
        _dead_slugs_dirty.add(platform)


def _mark_alive(platform: str, slug: str) -> None:
    """Clear a dead mark after a successful fetch.

    Without this, recovery depends entirely on DEAD_SLUG_TTL_DAYS: a company
    whose board 404s during a one-hour outage stays skipped for three months.
    A slug can only be marked dead by a 404/410, so a 200 is direct evidence
    the mark is stale.
    """
    with _dead_slug_lock:
        lookup = _dead_slugs.get(platform) or set()
        dates = _dead_slug_dates.get(platform) or {}
        if slug not in lookup and slug not in dates:
            return
        lookup.discard(slug)
        dates.pop(slug, None)
        _dead_slugs_dirty.add(platform)


# orphan-ok: test-isolation reset for module-global state. The dead-slug maps
# outlive any one test, so a suite without this leaks suppressions from one
# test into the next; production never wants it, because forgetting which slugs
# are dead is the whole cost this cache exists to avoid.
def clear_dead_slugs() -> None:
    with _dead_slug_lock:
        _dead_slugs.clear()
        _dead_slug_dates.clear()
        _dead_slugs_dirty.clear()


def _matches_keywords(title: str, keywords: str) -> bool:
    if not keywords or not keywords.strip():
        return True
    title_lower = title.lower()
    tokens = re.findall(r'"([^"]+)"|(\S+)', keywords.lower())
    for quoted, unquoted in tokens:
        token = quoted or unquoted
        if token and re.search(r'\b' + re.escape(token) + r'\b', title_lower):
            return True
    return False


_REMOTE_PATTERN = re.compile(
    r"remote\s*[-–—:]\s*(.+)", re.IGNORECASE,
)
_REMOTE_SUFFIX_PATTERN = re.compile(
    r"(.+?)\s*[-–—]\s*remote", re.IGNORECASE,
)
_PAREN_REMOTE_PATTERN = re.compile(
    r"(.+?)\s*\(remote\)", re.IGNORECASE,
)

_COUNTRY_ALIASES: dict[str, str] = {
    "uk": "GB", "u.k.": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
    "u.s.": "US", "u.s.a.": "US", "usa": "US", "uae": "AE",
}

# ── GeoNames 632k-city lookup (built offline from cities500.txt) ─────────────

def _load_geo_lookup() -> tuple[dict, dict, dict]:
    from services.jba.geo_db import load_geo_lookup_from_db
    return load_geo_lookup_from_db()

_geo_cities: dict = {}
_geo_admin1_name: dict = {}
_iso_country_codes: set[str] = set()
_ca_us_subdiv_codes: dict[str, str] = {}
_geo_loaded = False
# pycountry's tables come from an installed package and cannot fail the way
# geo.db can, so they are tracked separately -- otherwise a retry of the db
# load would walk pycountry's ~5k subdivisions again for nothing.
_countries_loaded = False
_geo_failures = 0
_geo_next_retry = 0.0
# Long enough that a genuinely missing geo.db costs one cheap failed connect
# every couple of minutes rather than one per lookup, short enough that a
# process which started during a momentary lock is matching on cities again
# within a scrape cycle instead of never.
GEO_RETRY_INTERVAL_S = 120.0
# _ensure_geo_loaded is first reached from inside the per-slug thread pools, so
# without this every worker on a cold process starts its own 632k-row load, and
# because _geo_loaded only flips at the end, threads could read a half-populated
# _geo_cities and miss cities that were about to exist.
_geo_lock = threading.Lock()

def _ensure_geo_loaded() -> None:
    if _geo_loaded:
        return
    with _geo_lock:
        if _geo_loaded:
            return
        global _geo_next_retry
        now = time.monotonic()
        if _geo_failures and now < _geo_next_retry:
            return
        _geo_next_retry = now + GEO_RETRY_INTERVAL_S
        _load_geo_into_globals()


def _load_geo_into_globals() -> None:
    global _geo_cities, _geo_admin1_name, _geo_loaded, _countries_loaded
    global _iso_country_codes, _ca_us_subdiv_codes, _geo_failures
    import pycountry

    try:
        _geo_cities, _, _geo_admin1_name = _load_geo_lookup()
    except Exception as exc:
        # data/geo.db is gitignored (78 MB), so a fresh deployment has none.
        # sqlite3.connect() creates an empty file rather than failing, so the
        # error surfaces as "no such table: geo_cities" on first lookup and
        # would otherwise propagate out of every ATS location filter.
        # Degrade to country-only matching instead of taking the scrape down.
        #
        # Degrade, but do NOT latch. _geo_loaded used to be set here too, so a
        # single failure disabled city matching for the life of the process --
        # and the cheapest way to get one is a momentary lock at startup, which
        # is transient and was being treated as permanent. Measured: the bot
        # lost city matching for a whole run to a five-second contention that
        # had cleared minutes later.
        _geo_cities, _geo_admin1_name = {}, {}
        if not _geo_failures:
            print(
                f"[ats] geo.db unavailable ({exc}). City/region matching off for "
                f"now; retrying every {int(GEO_RETRY_INTERVAL_S)}s. If this "
                "persists, run: python scripts/build_geo_db.py"
            )
        _geo_failures += 1
    else:
        if _geo_failures:
            print(
                f"[ats] geo.db recovered after {_geo_failures} failed attempt(s); "
                f"city/region matching is back on ({len(_geo_cities):,} cities)"
            )
        _geo_failures = 0
        _geo_loaded = True

    if not _countries_loaded:
        _iso_country_codes.update(c.alpha_2 for c in pycountry.countries)
        for _sub in pycountry.subdivisions:
            if _sub.country_code in ("CA", "US"):
                _short = _sub.code.split("-", 1)[1]
                if _short not in _ca_us_subdiv_codes:
                    _ca_us_subdiv_codes[_short] = _sub.country_code
        _countries_loaded = True

_country_lookup_cache: dict[str, str | None] = {}


def _lookup_city(name: str, neighbor_hint: str | None = None) -> str | None:
    """Resolve a city name to a country code using the GeoNames 632k index.

    *neighbor_hint* is typically a province/state code from an adjacent
    comma-part (e.g. "MA", "ON").  When multiple countries share the city
    name, we pick the entry whose admin1 or country code matches the hint.
    When a hint is provided but nothing matches, returns None (don't guess).
    """
    _ensure_geo_loaded()
    entries = _geo_cities.get(name.lower().strip())
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0][0]
    if neighbor_hint:
        hint_upper = neighbor_hint.strip().upper()
        for cc, a1 in entries:
            if a1 == hint_upper:
                return cc
        for cc, a1 in entries:
            if cc == hint_upper:
                return cc
        return None
    return entries[0][0]


def _lookup_subdivision(code: str) -> str | None:
    """Resolve a 2-letter code to CA/US parent country.

    Skips codes that are also ISO country codes to avoid ambiguity.
    """
    _ensure_geo_loaded()
    upper = code.upper().strip()
    if len(upper) != 2:
        return None
    if upper in _iso_country_codes:
        return None
    return _ca_us_subdiv_codes.get(upper)


def _lookup_subdivision_name(name: str) -> str | None:
    """Resolve a full subdivision name ('Ontario', 'California') to country."""
    _ensure_geo_loaded()
    entries = _geo_admin1_name.get(name.lower().strip())
    if not entries:
        return None
    for cc, _short in entries:
        if cc in ("CA", "US"):
            return cc
    return None


def _lookup_country(text: str) -> str | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    key = cleaned.lower()
    alias = _COUNTRY_ALIASES.get(key)
    if alias:
        return alias
    if key in _country_lookup_cache:
        return _country_lookup_cache[key]
    try:
        import pycountry
        c = pycountry.countries.lookup(cleaned)
        _country_lookup_cache[key] = c.alpha_2
        return c.alpha_2
    except LookupError:
        pass
    _country_lookup_cache[key] = None
    return None


def _infer_country(location: str) -> str | None:
    """Resolve a freeform location string to a 2-letter country code."""
    _ensure_geo_loaded()
    text = location.strip()
    if not text:
        return None
    lower = text.lower()

    m = _REMOTE_PATTERN.search(lower)
    if not m:
        m = _REMOTE_SUFFIX_PATTERN.search(lower)
    if not m:
        m = _PAREN_REMOTE_PATTERN.search(lower)
    if m:
        qualifier = m.group(1).strip().rstrip(".")
        qualifier = re.sub(r"\s*:\s*select locations.*", "", qualifier, flags=re.IGNORECASE)
        qualifier = re.sub(r"\(.*?\)", "", qualifier).strip()
        if qualifier:
            resolved = _infer_country(qualifier)
            if resolved:
                return resolved

    # Split on both commas and semicolons.
    parts = [p.strip().rstrip(".") for p in re.split(r"[,;]", text)]

    # Pass 1: explicit country names (skip bare 2-letter codes).
    for part in reversed(parts):
        cleaned = part.strip()
        if not cleaned or len(cleaned) <= 2:
            continue
        cc = _lookup_country(cleaned)
        if cc:
            return cc

    # Pass 1.5: "XX - City" dash-separated format (e.g. "AL - Birmingham").
    _dash_pat = re.compile(r"^([A-Za-z]{2})\s+[-–—]\s+(.+)")
    for part in parts:
        dm = _dash_pat.match(part.strip())
        if not dm:
            continue
        code = dm.group(1).upper()
        city_raw = dm.group(2).strip()
        sc = _lookup_subdivision(code)
        if sc:
            return sc
        if city_raw:
            # Try full string, then progressively shorter prefixes
            # to handle "Phoenix Metropolitan Area" → "Phoenix".
            city_words = city_raw.split()
            for n in range(len(city_words), 0, -1):
                candidate = " ".join(city_words[:n])
                city_cc = _lookup_city(candidate, neighbor_hint=code)
                if city_cc:
                    return city_cc
                if candidate.lower() in _geo_cities:
                    subdiv_cc = _ca_us_subdiv_codes.get(code)
                    if subdiv_cc:
                        return subdiv_cc

    # Pass 2: 2-letter codes with context.
    for i, part in enumerate(parts):
        stripped = part.strip()
        words = stripped.split()
        if not words:
            continue
        first = words[0]
        if len(first) != 2 or not first.isalpha():
            continue
        # Skip if remaining words form a known city name (e.g. "St Paul"
        # where "St" is not a code).  Allow through if:
        #   - part is just the code ("ON", "CA")
        #   - followed by postal/zip chars ("ON M5R 3L2", "CA 91764")
        #   - followed by a country name ("TX United States")
        if len(words) > 1:
            rest = " ".join(words[1:])
            rest_lower = rest.lower()
            if rest_lower in _geo_cities:
                continue
            has_digit = any(c.isdigit() for c in rest)
            if not (has_digit or _lookup_country(rest)):
                continue
        code = first.upper()
        sc = _lookup_subdivision(code)
        if sc:
            return sc
        city_part = parts[i - 1].strip() if i > 0 else ""
        if city_part:
            city_cc = _lookup_city(city_part, neighbor_hint=code)
            if city_cc:
                return city_cc
            if city_part.lower() in _geo_cities:
                subdiv_cc = _ca_us_subdiv_codes.get(code)
                if subdiv_cc:
                    return subdiv_cc
        # Only fall back to country lookup for codes that are NOT also
        # CA/US subdivision codes (those are inherently ambiguous without
        # city context — e.g. "CA" could be California or Canada).
        if code not in _ca_us_subdiv_codes:
            cc = _lookup_country(code)
            if cc:
                return cc

    # Pass 3: subdivision names, including within multi-word parts
    # (e.g. "Northern California" → "California" → US).
    for part in parts:
        cleaned = part.strip()
        sc = _lookup_subdivision_name(cleaned)
        if sc:
            return sc
        for word in cleaned.split():
            sc = _lookup_subdivision_name(word)
            if sc:
                return sc

    # Pass 4: bare city name (first comma-part)
    first_part = parts[0].strip()
    if first_part:
        city_cc = _lookup_city(first_part)
        if city_cc:
            return city_cc

    if "remote" in lower and not m:
        return None

    return None


# A "Remote" posting whose text names no country at all is dropped when the
# search names one. Kept deliberately: the alternative (treat unqualified remote
# as matching every country) floods a Canada-scoped channel with US-only remote
# roles, which is the failure this filter exists to prevent. Flip to True to
# take the opposite trade.
UNQUALIFIED_REMOTE_MATCHES_ANY_COUNTRY: bool = False


# Region names as a searcher types them, mapped to the codes boards actually
# publish. iCIMS, Workday and Greenhouse all emit "Winder, GA, US" style
# locations, so a search for "Georgia" tokenised to ["georgia"] found nothing
# in that string and silently dropped every job in the state -- the same
# failure shape as an empty location, and just as invisible.
_REGION_NAME_TO_CODE: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
    "alberta": "AB", "british columbia": "BC", "manitoba": "MB",
    "new brunswick": "NB", "newfoundland and labrador": "NL",
    "nova scotia": "NS", "ontario": "ON", "prince edward island": "PE",
    "quebec": "QC", "saskatchewan": "SK", "northwest territories": "NT",
    "nunavut": "NU", "yukon": "YT",
}


def _region_code_for(search_location: str) -> str | None:
    """The two-letter code for a spelled-out region, or None.

    Only an exact match on the whole search string counts. Matching a code
    inside a longer phrase would make "New York City" resolve to NY and then
    match every job in Buffalo, which is not what the searcher asked for.
    """
    return _REGION_NAME_TO_CODE.get(" ".join(search_location.lower().split()))


def _matches_region_code(job_location: str, code: str) -> bool:
    """Whether a job location carries *code* as its region field.

    Deliberately case-sensitive against the original string. Lowercasing first
    makes "IN" (Indiana) match the ordinary English word in "Remote in US",
    and "OR" match "Sales or Marketing" -- two-letter codes are too short to
    be safe once case is thrown away. Boards publish these codes uppercase.
    """
    return re.search(r"(?<![A-Za-z])" + code + r"(?![A-Za-z])", job_location) is not None


def _locality_overlap(job_location: str, search_location: str) -> bool:
    """Whether a job location names the same place the search did.

    Word-boundary token match, plus a region-code pass so that a search for a
    spelled-out region matches the code boards actually publish -- "Ontario"
    has to match "Toronto, ON, CA", which token matching alone cannot do.
    """
    job_lower = job_location.lower()
    search_tokens = [t.strip() for t in search_location.lower().strip().replace(",", " ").split()
                     if len(t.strip()) > 2]
    for token in search_tokens:
        if re.search(r'\b' + re.escape(token) + r'\b', job_lower):
            return True
    code = _region_code_for(search_location)
    return bool(code and _matches_region_code(job_location, code))


def _is_bare_country(text: str) -> bool:
    """Whether the whole string is just a country name ("Canada", "United States")."""
    return bool(_lookup_country(text.strip()))


def _matches_location(job_location: str, search_location: str) -> bool:
    if not search_location or not search_location.strip():
        return True
    job_loc = job_location.strip()
    if not job_loc:
        return False

    search_country = _infer_country(search_location)
    job_country = _infer_country(job_loc)

    is_remote = "remote" in job_loc.lower()

    if search_country and job_country:
        if job_country != search_country:
            return False
        # Same country -- but "resolves to a country" is not the same as "is a
        # country". _infer_country maps cities and regions to their country too,
        # so comparing only the codes made every search inside one country match
        # everything in it: a search for Vancouver returned Toronto jobs, and a
        # search for Boston returned Austin jobs. Toronto looked correct purely
        # because most postings a Canadian search saw were already Canadian.
        if _is_bare_country(search_location):
            # The search asked for the country and nothing more.
            return True
        if is_remote or _is_bare_country(job_loc):
            # The posting is country-wide or remote, so there is no locality to
            # contradict the search. Keeping it is the non-destructive read.
            return True
        return _locality_overlap(job_loc, search_location)

    if is_remote and not job_country:
        return UNQUALIFIED_REMOTE_MATCHES_ANY_COUNTRY or search_country is None

    if not search_country:
        return _locality_overlap(job_loc, search_location)

    return False


def _fanout_budget(count: int, workers: int) -> float:
    """Overall wall-clock budget for a fan-out of *count* fetches over *workers*.

    Derived from REQUEST_TIMEOUT rather than a standalone constant: each fetch is
    already capped at REQUEST_TIMEOUT, and a pool of `workers` drains `count`
    fetches in ceil(count/workers) waves, so that product is the real worst case.
    A flat 120s would cut off a legitimately large (but healthy) iCIMS batch,
    while leaving a big batch of *stalled* fetches running for far longer.
    """
    waves = -(-max(count, 1) // max(workers, 1))
    return REQUEST_TIMEOUT * (waves + 1)


def fanout_capacity(budget_s: float, workers: int) -> int:
    """How many fetches `workers` are guaranteed to drain inside *budget_s*.

    The inverse of _fanout_budget, and the only place the two are related:
    _fanout_budget says how long a count would take in the worst case, this
    says how large a count that same worst case lets a budget afford. It is
    deliberately pessimistic -- every fetch spending its whole REQUEST_TIMEOUT
    -- because it is used as a floor for the first cycle on a platform that
    has not reported yet, and a floor that could overrun is not a floor.
    Never less than one wave: a budget too small for a single round still
    gets to ask one.
    """
    waves = int(max(budget_s, 0) // REQUEST_TIMEOUT) - 1
    return max(1, workers) * max(1, waves)


def fanout_workers(platform: str, fleet_size: int) -> int:
    """The pool width scrape_ats_platform actually runs *platform* at.

    One definition, read by the scrape and by whoever sizes the fleet handed
    to it, so the two cannot drift: a caller that assumed the nominal
    PLATFORM_WORKERS while the pool had been scaled down for the host would
    hand over more than the pool can drain. capacity.workers scales the tuned
    ceiling down on small hardware and never up, since these values are also
    politeness limits per platform. minimum=2, not 4: the manager fans several
    platforms out concurrently, so this floor is paid once per platform.
    """
    return max(1, min(
        capacity.workers(PLATFORM_WORKERS.get(platform, 10), minimum=2),
        max(int(fleet_size), 1),
    ))


def _collect_results(futures: dict, timeout: float | None) -> dict:
    """Drain a {future: key} map into {key: result}, bounded by *timeout*.

    The previous shape -- `for f in as_completed(futures): f.result(timeout=T)`
    -- never bounded anything: as_completed only yields futures that are already
    done, so result() returns instantly and T was dead code. The only real limit
    on a stalled slug was REQUEST_TIMEOUT times the retry count, and the pool's
    context manager still waited for every worker on exit. Putting the timeout on
    as_completed makes it bite, and partial results collected before it fires are
    kept rather than thrown away.

    Scope: this bounds how long we *wait*, and cancels fetches still queued --
    the bulk of them whenever count >> workers. Fetches already in flight cannot
    be cancelled, so the pool's context manager still waits out the longest one
    on exit; that residual is bounded by REQUEST_TIMEOUT plus its retries. The
    alternative (shutdown(wait=False) outside the context manager) would leave
    live threads writing into a dead scrape, which is worse.
    """
    collected: dict = {}
    try:
        for future in as_completed(futures, timeout=timeout):
            try:
                collected[futures[future]] = future.result()
            except Exception:
                continue
    except TimeoutError:
        pending = [f for f in futures if not f.done()]
        for future in pending:
            future.cancel()
        print(f"[ats] timed out waiting on {len(pending)} of {len(futures)} fetches")
    return collected


# ── Greenhouse ──────────────────────────────────────────────────────────────

def _scrape_greenhouse(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(GREENHOUSE, slug):
        return []
    # content=true returns the posting body. Without it the board API sends
    # title and location only, and job_service matches semantically against
    # "description" -- so every Greenhouse job was being scored on its title
    # alone, invisible to matching rather than merely sparse.
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
    headers = _make_headers()
    resp = None
    for attempt in range(3):
        try:
            resp = _http_get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            return []
        if resp.status_code == 200:
            _mark_alive(GREENHOUSE, slug)
            break
        if resp.status_code in (404, 410):
            _mark_dead(GREENHOUSE, slug)
            return []
        if resp.status_code in (429, 503, 502) and attempt < 2:
            time.sleep(retry_backoff_delay(attempt))
            headers["User-Agent"] = random.choice(USER_AGENTS)
            continue
        return []
    if resp is None or resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for job in data.get("jobs") or []:
        title = str(job.get("title") or "").strip()
        if not title:
            continue
        loc = str((job.get("location") or {}).get("name") or "").strip()
        job_url = str(job.get("absolute_url") or "").strip()
        if not job_url:
            continue
        if not _matches_keywords(title, keywords):
            continue
        if not _matches_location(loc, location):
            continue
        rows.append({
            "title": title,
            "company": slug,
            "location": loc,
            "job_url": job_url,
            # first_published, not updated_at: updated_at moves on any edit, so
            # an eighteen-month-old requisition that had its salary band tweaked
            # yesterday reads as posted yesterday and passes a 24-hour filter.
            "date_posted": _normalise_posted(
                job.get("first_published") or job.get("updated_at")),
            "description": _html_to_text(job.get("content")),
            "_source_site": GREENHOUSE,
            "_source_sites": [GREENHOUSE],
        })
        if len(rows) >= max_jobs:
            break
    return rows


# ── Lever ───────────────────────────────────────────────────────────────────

def _scrape_lever(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(LEVER, slug):
        return []
    # Lever runs a separate EU region, and a company lives in exactly one of
    # them: amicustherapeutics and amo are 200 on api.eu.lever.co and 404 on
    # api.lever.co. Querying only the US host means every EU-hosted company is
    # marked dead on first contact and never scraped again, so the boards on
    # jobs.eu.lever.co are unreachable no matter how many of them we harvest.
    headers = _make_headers()
    resp = None
    for host in _LEVER_API_HOSTS:
        url = f"https://{host}/v0/postings/{slug}"
        for attempt in range(3):
            try:
                resp = _http_get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            except Exception:
                return []
            if resp.status_code == 200:
                _mark_alive(LEVER, slug)
                break
            if resp.status_code in (404, 410):
                break            # wrong region; try the next host
            if resp.status_code in (429, 503, 502) and attempt < 2:
                time.sleep(retry_backoff_delay(attempt))
                headers["User-Agent"] = random.choice(USER_AGENTS)
                continue
            return []
        if resp is not None and resp.status_code == 200:
            break
    else:
        # Dead only once every region has 404d. Marking after the first miss is
        # what would make the entire EU population look permanently gone.
        _mark_dead(LEVER, slug)
        return []
    if resp is None or resp.status_code != 200:
        return []
    try:
        postings = resp.json()
    except Exception:
        return []

    if not isinstance(postings, list):
        return []

    rows: list[dict[str, Any]] = []
    for posting in postings:
        title = str(posting.get("text") or "").strip()
        if not title:
            continue
        categories = posting.get("categories") or {}
        loc = str(categories.get("location") or "").strip()
        job_url = str(posting.get("hostedUrl") or "").strip()
        apply_url = str(posting.get("applyUrl") or "").strip()
        if not job_url:
            continue
        if not _matches_keywords(title, keywords):
            continue
        if not _matches_location(loc, location):
            continue
        created_ms = posting.get("createdAt")
        date_posted = ""
        if isinstance(created_ms, (int, float)) and created_ms > 0:
            from datetime import datetime, timezone
            date_posted = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc).isoformat()
        row: dict[str, Any] = {
            "title": title,
            "company": slug,
            "location": loc,
            "job_url": job_url,
            "date_posted": date_posted,
            # descriptionPlain is already plain text; description is HTML. Both
            # ship in the same listing response, so this costs no extra request.
            "description": (str(posting.get("descriptionPlain") or "").strip()
                            or _html_to_text(posting.get("description"))),
            # Verified live: categories.commitment carries "Full-time",
            # "Part-time", "Internship". It is literally the intern flag and was
            # being parsed past on every Lever posting.
            "employment_type": str(
                (posting.get("categories") or {}).get("commitment") or ""),
            "_source_site": LEVER,
            "_source_sites": [LEVER],
        }
        if apply_url:
            row["apply_link"] = apply_url
        rows.append(row)
        if len(rows) >= max_jobs:
            break
    return rows


# ── Ashby ───────────────────────────────────────────────────────────────────

_ASHBY_GRAPHQL_URL = "https://jobs.ashbyhq.com/api/non-user-graphql?op=ApiJobBoardWithTeams"
_ASHBY_QUERY = """
query ApiJobBoardWithTeams($organizationHostedJobsPageName: String!) {
  jobBoard: jobBoardWithTeams(
    organizationHostedJobsPageName: $organizationHostedJobsPageName
  ) {
    jobPostings {
      id
      title
      locationName
      employmentType
    }
  }
}
"""


def _scrape_ashby(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    """Ashby boards, via the public posting API rather than the board GraphQL.

    The GraphQL endpoint returns brief objects -- id, title, locationName,
    employmentType -- and nothing else. It has no description and no published
    date, so every Ashby job was scored on its title alone and was permanently
    exempt from the age filter. Asking GraphQL for those fields does not work
    either: they do not exist on JobPostingBriefsWithIdsAndTeamId, and adding
    them makes the whole query error, which returns zero jobs for all ~5,000
    Ashby companies while looking exactly like "no openings".

    The posting API carries what the board query lacks -- descriptionPlain,
    publishedAt, a real jobUrl, isRemote, compensation -- and it 404s an
    unknown org, which GraphQL never does. That last part matters beyond this
    function: it is why dead-marking could not work here before.

    It is also the endpoint validate_ats_slugs probes, so a company the
    validator calls live is now one this can actually read.
    """
    if _is_dead(ASHBY, slug):
        return []
    data = _fetch_board_json(
        ASHBY, slug,
        f"https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true")
    if not isinstance(data, dict):
        return []

    def shape(job):
        loc = str(job.get("location") or "").strip()
        if job.get("isRemote") and "remote" not in loc.lower():
            loc = _join_location(loc, "Remote")
        return (job.get("title"), loc,
                job.get("jobUrl") or job.get("applyUrl"),
                job.get("publishedAt"),
                job.get("descriptionPlain") or _html_to_text(job.get("descriptionHtml")),
                job.get("employmentType"))

    # isListed False means the posting exists but is not on the public board.
    jobs = [j for j in (data.get("jobs") or [])
            if not isinstance(j, dict) or j.get("isListed", True)]
    return _board_rows(ASHBY, slug, jobs, keywords, location, max_jobs, shape)


# ── Workday ─────────────────────────────────────────────────────────────────

def _parse_workday_slug(slug: str) -> tuple[str, str, str] | None:
    parts = slug.split("|")
    if len(parts) != 3:
        return None
    company, wd_num, site_id = (p.strip() for p in parts)
    if not company or not wd_num or not site_id:
        return None
    return company, wd_num, site_id


def _fetch_workday_detail(detail_url: str, headers: dict[str, str]) -> dict[str, str]:
    """Posting date and body for one Workday job.

    Was _fetch_workday_date, returning only startDate. The same response
    already carries jobDescription, so the description was being fetched and
    thrown away on every job -- and Workday is the largest platform here.

    startDate is a localised string ("08/01/2025", "1 Aug 2025"), not ISO.
    _posting_age_ok answers True when the parse raises, so an unnormalised date
    does not drop the row -- it exempts the row from the age filter. Every
    Workday job was passing "newer than N hours" regardless of age.
    """
    out = {"date_posted": "", "description": ""}
    try:
        resp = _http_get(detail_url, headers=headers, timeout=REQUEST_TIMEOUT)
    except Exception:
        return out
    if resp.status_code != 200:
        return out
    try:
        info = resp.json().get("jobPostingInfo") or {}
    except Exception:
        return out
    if not isinstance(info, dict):
        return out
    out["date_posted"] = _normalise_posted(info.get("startDate"))
    out["description"] = _html_to_text(info.get("jobDescription"))
    return out


def _fetch_workday_date(detail_url: str, headers: dict[str, str]) -> str:
    try:
        resp = _http_get(detail_url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return ""
        info = resp.json().get("jobPostingInfo") or {}
        return str(info.get("startDate") or "").strip()
    except Exception:
        return ""


def _scrape_workday(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(WORKDAY, slug):
        return []
    parsed = _parse_workday_slug(slug)
    if not parsed:
        return []
    company, wd_num, site_id = parsed

    base_url = f"https://{company}.{wd_num}.myworkdayjobs.com"
    api_url = f"{base_url}/wday/cxs/{company}/{site_id}/jobs"

    headers = _make_headers({
        "Content-Type": "application/json",
        "Origin": base_url,
        "Referer": f"{base_url}/{site_id}",
    })

    candidates: list[dict[str, Any]] = []
    offset = 0
    limit = 20
    retries = 0
    max_retries = 2
    observed_total = None

    while True:
        payload: dict[str, Any] = {
            "appliedFacets": {},
            "limit": limit,
            "offset": offset,
            "searchText": keywords or "",
        }

        try:
            resp = _http_post(api_url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            break

        if resp.status_code in (404, 410, 422):
            if offset == 0:
                _mark_dead(WORKDAY, slug)
            break
        if resp.status_code == 200 and offset == 0:
            _mark_alive(WORKDAY, slug)
        if resp.status_code != 200:
            if retries < max_retries:
                retries += 1
                time.sleep(random.uniform(2.0, 4.0))
                continue
            break

        try:
            data = resp.json()
        except Exception:
            break

        jobs = data.get("jobPostings") or []
        total = data.get("total", 0)

        if observed_total is None:
            observed_total = total
        elif total != observed_total:
            break

        if not jobs:
            break

        for posting in jobs:
            title = str(posting.get("title") or "").strip()
            if not title:
                continue
            loc = str(posting.get("locationsText") or "").strip()
            external_path = str(posting.get("externalPath") or "").strip()
            if not external_path:
                continue
            job_url = f"{base_url}/{site_id}{external_path}"
            if not _matches_keywords(title, keywords):
                continue
            if not _matches_location(loc, location):
                continue
            detail_url = f"{base_url}/wday/cxs/{company}/{site_id}{external_path}"
            candidates.append({
                "title": title,
                "company": company,
                "location": loc,
                "job_url": job_url,
                "detail_url": detail_url,
                "_source_site": WORKDAY,
                "_source_sites": [WORKDAY],
            })
            if len(candidates) >= max_jobs:
                break

        if len(candidates) >= max_jobs:
            break
        offset += limit
        if offset >= total:
            break

        time.sleep(random.uniform(0.3, 1.0))

    if not candidates:
        return []

    # Nested pool: bounded (accepted exception to scheduler-visible concurrency;
    # the caller holds one PriorityWorkScheduler slot for this whole fan-out).
    #
    # This scraper used to submit _fetch_workday_detail straight to the nested
    # pool, bypassing _enrich_gate -- the per-platform Semaphore that caps
    # in-flight requests against one vendor at PLATFORM_WORKERS. A py-spy dump
    # of the live bot found 70 threads inside _fetch_workday_detail at once:
    # outer scheduler pool x this nested pool, unbounded, all hitting Workday.
    # Routing each submission through the gate keeps that fan-out at
    # PLATFORM_WORKERS[workday] in flight, same as _enrich_rows.
    detail_workers = capacity.workers(5, minimum=2)
    gate = _enrich_gate(WORKDAY)
    # As for icims: the archive check keys on the job URL, which the search
    # response already gave us, so it runs before the detail fetch, not after.
    candidates = _needs_enrichment(candidates)

    def _gated_detail(detail_url: str, hdrs: dict[str, str]) -> dict[str, str]:
        with gate:
            return _fetch_workday_detail(detail_url, hdrs)

    with ThreadPoolExecutor(max_workers=detail_workers) as pool:
        futures = {pool.submit(_gated_detail, row["detail_url"], headers): row["detail_url"] for row in candidates}
        url_to_detail: dict[str, dict[str, str]] = _collect_results(
            futures, capacity.timeout(_fanout_budget(len(candidates), detail_workers))
        )

    rows: list[dict[str, Any]] = []
    for row in candidates:
        detail = url_to_detail.get(row.pop("detail_url")) or {}
        # Guarded, not assigned unconditionally: a detail fetch that failed
        # must leave what the listing gave rather than blanking it. The same
        # rule the iCIMS path already follows, and for the same reason -- an
        # empty date exempts a row from the age filter and an empty description
        # removes it from semantic matching.
        if detail.get("date_posted"):
            row["date_posted"] = detail["date_posted"]
        if detail.get("description"):
            row["description"] = detail["description"]
        row.setdefault("date_posted", "")
        row.setdefault("description", "")
        rows.append(row)

    return rows


# ── iCIMS ──────────────────────────────────────────────────────────────────

_ICIMS_SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# See the candidate_cap comment in _scrape_icims.
_ICIMS_LOCATION_OVERFETCH = 4
_ICIMS_MAX_CANDIDATES = 500

# iCIMS is the one platform whose location is knowable only from a second
# request, so a rate-limited page leaves a listing genuinely unclassified. False
# drops it (never post a job we cannot confirm is in the right country); True
# posts it unfiltered. Dropping is the default because the alternative fills a
# location-scoped channel with jobs from the wrong continent, but the count is
# always logged so this never looks like "no jobs matched".
ICIMS_KEEP_UNRESOLVED_LOCATION: bool = False


def _icims_iframe_url(job_url: str) -> str:
    """Point a job URL at the variant that actually carries the JSON-LD.

    The plain job URL now returns a shell whose only ld+json block is
    WebPage/BreadcrumbList/WebSite -- no JobPosting at all. The same URL with
    `in_iframe=1` (the parameter iCIMS's own portal uses to load job content
    into an iframe) returns the rendered posting, JSON-LD included. Without
    this, every iCIMS listing came back with an empty location, which matches
    no location filter, so all 8,747 companies dropped out of any scoped search
    while looking exactly like "no jobs matched".

    Existing query parameters are preserved; the value is forced to 1 so a URL
    that already carries the parameter does not end up with two of them.
    """
    parsed = urlparse(job_url)
    query = parse_qs(parsed.query or "")
    query["in_iframe"] = ["1"]
    return parsed._replace(
        query=urlencode({k: v[0] for k, v in query.items()})
    ).geturl()


def _fetch_icims_metadata(job_url: str) -> dict[str, str]:
    """Fetch an iCIMS job page and extract title, location, datePosted from JSON-LD.

    "resolved" reports whether the page was actually parsed. The caller cannot
    tell an unlocated posting from a failed fetch otherwise, and it needs to:
    location is only ever known from this page, so a transient failure would
    otherwise be silently indistinguishable from "this job is in the wrong
    country". Retried once because these pages rate-limit readily and a single
    miss costs the listing entirely.
    """
    result: dict[str, str] = {"title": "", "location": "", "date_posted": "",
                              "description": "", "employment_type": "",
                              "resolved": ""}
    # This is an HTML page with an embedded JSON-LD <script>; asking for JSON
    # invites a 406 or a body that has no <script> block to scrape.
    hdrs = _make_headers({"Accept": "text/html,application/xhtml+xml"})
    fetch_url = _icims_iframe_url(job_url)
    resp = None
    for attempt in range(2):
        try:
            resp = _http_get(fetch_url, headers=hdrs, timeout=REQUEST_TIMEOUT)
        except Exception:
            resp = None
        if resp is not None and resp.status_code == 200:
            break
        if attempt == 0:
            time.sleep(retry_backoff_delay(attempt))
            hdrs["User-Agent"] = random.choice(USER_AGENTS)
    if resp is None or resp.status_code != 200:
        return result
    try:
        html = resp.text
        for match in re.finditer(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.DOTALL):
            try:
                ld = _find_jobposting(json.loads(match.group(1)))
                if ld is None:
                    continue
                result["title"] = str(ld.get("title") or "").strip()
                result["date_posted"] = _normalise_posted(ld.get("datePosted"))
                # Already parsed out of the page this function fetches, so the
                # description was being discarded for free.
                result["description"] = _html_to_text(ld.get("description"))
                emp = ld.get("employmentType")
                if isinstance(emp, list):
                    emp = " ".join(str(e) for e in emp)
                result["employment_type"] = str(emp or "")
                job_loc = ld.get("jobLocation")
                if isinstance(job_loc, list):
                    job_loc = job_loc[0] if job_loc else None
                if job_loc:
                    addr = job_loc.get("address") or {}
                    parts = [
                        str(addr.get("addressLocality") or "").strip(),
                        str(addr.get("addressRegion") or "").strip(),
                        str(addr.get("addressCountry") or "").strip(),
                    ]
                    # iCIMS fills unknown address fields with the literal string
                    # "UNAVAILABLE" rather than omitting them, so a job with no
                    # city was being shown to users as
                    # "UNAVAILABLE, UNAVAILABLE, US" and, worse, was matching a
                    # keyword search for that word. Dropping the placeholders
                    # leaves the parts that are actually known.
                    result["location"] = ", ".join(
                        p for p in parts if p and p.upper() != "UNAVAILABLE")
                result["resolved"] = "1"
                return result
            except (json.JSONDecodeError, AttributeError, StopIteration):
                continue
        # Page fetched and parsed, but it carried no JobPosting block.
        result["resolved"] = "1"
    except Exception:
        pass
    return result


def _scrape_icims(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(ICIMS, slug):
        return []
    # iCIMS runs two mutually exclusive host conventions and the stored slug
    # does not say which one a company uses. Measured over a 120-slug sample:
    # 46 resolved only as careers-<slug>.icims.com, 34 only as
    # <slug>.icims.com, and *zero* resolved both ways. Prepending
    # unconditionally -- which is what this did -- therefore cannot reach about
    # 42% of live iCIMS boards no matter how long it retries. Try both forms
    # before concluding a company is gone.
    hosts = [slug] if slug.startswith("careers-") else [f"careers-{slug}", slug]
    headers = _make_headers({"Accept": "application/xml"})
    resp = None
    sitemap_url = ""
    for host in hosts:
        sitemap_url = f"https://{host}.icims.com/sitemap.xml"
        for attempt in range(3):
            try:
                resp = _http_get(sitemap_url, headers=headers, timeout=REQUEST_TIMEOUT)
            except Exception:
                return []
            if resp.status_code == 200:
                _mark_alive(ICIMS, slug)
                break
            if resp.status_code in (404, 410):
                break          # wrong host form; fall through to the next one
            if resp.status_code in (429, 503, 502) and attempt < 2:
                time.sleep(retry_backoff_delay(attempt))
                headers["User-Agent"] = random.choice(USER_AGENTS)
                continue
            return []
        if resp is not None and resp.status_code == 200:
            break
    else:
        # Only dead once *every* host form 404s. Marking dead after the first
        # miss is what made the prefixed entries look 100% dead.
        _mark_dead(ICIMS, slug)
        return []
    if resp is None or resp.status_code != 200:
        return []
    try:
        root = ET.fromstring(resp.content)
    except Exception:
        return []

    # Unlike every other platform, iCIMS filters by location *after* this loop
    # (location is only known from the per-job page fetched below), so capping
    # candidates at max_jobs here guaranteed fewer than results_wanted rows
    # whenever a location filter was set. Over-collect so the post-filter cap
    # has something to work with, bounded so a nationwide sitemap does not turn
    # into tens of thousands of page fetches.
    candidate_cap = max_jobs
    if location and location.strip():
        candidate_cap = min(max_jobs * _ICIMS_LOCATION_OVERFETCH, _ICIMS_MAX_CANDIDATES)

    candidates: list[dict[str, Any]] = []
    for url_el in root.findall(".//s:url", _ICIMS_SITEMAP_NS):
        loc_el = url_el.find("s:loc", _ICIMS_SITEMAP_NS)
        if loc_el is None:
            continue
        job_url = (loc_el.text or "").strip()
        if not job_url or "/jobs/" not in job_url or job_url.endswith("/jobs/intro"):
            continue
        path = job_url.split("/jobs/")[-1]
        parts = path.split("/")
        if len(parts) < 2:
            continue
        title = unquote(parts[1]).replace("-", " ").strip().title()
        if not title:
            continue
        if not _matches_keywords(title, keywords):
            continue
        lastmod_el = url_el.find("s:lastmod", _ICIMS_SITEMAP_NS)
        date_posted = (lastmod_el.text or "").strip() if lastmod_el is not None else ""
        candidates.append({
            "title": title,
            "company": slug,
            "location": "",
            "job_url": job_url,
            "date_posted": date_posted,
            "_source_site": ICIMS,
            "_source_sites": [ICIMS],
        })
        if len(candidates) >= candidate_cap:
            break

    if not candidates:
        return []

    # Nested pool: bounded (accepted exception to scheduler-visible concurrency;
    # the caller holds one PriorityWorkScheduler slot for this whole fan-out).
    #
    # This scraper used to submit _fetch_icims_metadata straight to the nested
    # pool, bypassing _enrich_gate -- the per-platform Semaphore that caps
    # in-flight requests against one vendor at PLATFORM_WORKERS. A py-spy dump
    # of the live bot found 215 threads inside _fetch_icims_metadata at once:
    # outer scheduler pool x this nested pool, unbounded, all hitting iCIMS.
    # Routing each submission through the gate keeps that fan-out at
    # PLATFORM_WORKERS[icims] in flight, same as _enrich_rows.
    meta_workers = capacity.workers(5, minimum=2)
    gate = _enrich_gate(ICIMS)
    # Postings the archive already holds would be dropped after their page
    # was fetched; skip the fetch. On a board asked every cycle that is
    # nearly every posting, and the page fetch is what made an icims board
    # cost more than its whole budget's worth of other platforms' boards.
    candidates = _needs_enrichment(candidates)

    def _gated_meta(url: str) -> dict[str, str]:
        with gate:
            return _fetch_icims_metadata(url)

    with ThreadPoolExecutor(max_workers=meta_workers) as pool:
        futures = {pool.submit(_gated_meta, row["job_url"]): row["job_url"] for row in candidates}
        url_to_meta: dict[str, dict[str, str]] = _collect_results(
            futures, capacity.timeout(_fanout_budget(len(candidates), meta_workers))
        )

    rows: list[dict[str, Any]] = []
    unresolved = 0
    for row in candidates:
        meta = url_to_meta.get(row["job_url"], {})
        if meta.get("title"):
            row["title"] = meta["title"]
        if meta.get("date_posted"):
            row["date_posted"] = meta["date_posted"]
        if meta.get("description"):
            row["description"] = meta["description"]
        # Guarded like title and date_posted above. The old unconditional write
        # meant a failed metadata fetch blanked the location, and an empty job
        # location never matches a non-empty search -- so a rate-limited or
        # timed-out fetch dropped the listing while looking exactly like "no
        # jobs matched".
        if meta.get("location"):
            row["location"] = meta["location"]
        resolved = bool(meta.get("resolved"))
        if not resolved:
            unresolved += 1
        if resolved or not ICIMS_KEEP_UNRESOLVED_LOCATION:
            if not _matches_location(row["location"], location):
                continue
        rows.append(row)
        if len(rows) >= max_jobs:
            break

    if unresolved:
        # Loud on purpose: these are listings we could not classify, not
        # listings that failed the filter.
        print(
            f"[ats] icims/{slug}: {unresolved} of {len(candidates)} job page(s) "
            "unreadable; their location could not be resolved"
        )
    return rows


# ── BambooHR ──────────────────────────────────────────────────────────────

def _fetch_bamboohr_detail(job_url: str) -> dict[str, str]:
    """Read one BambooHR posting's detail record.

    `/careers/list` returns title, id and location and nothing else, so the
    description and the posting date are only knowable from
    `/careers/{id}/detail`, which answers `result.jobOpening`. Costs one request
    per job, hence the _enrich_rows pool.

    Returns {} on any failure so _enrich_rows leaves the listing's own values
    alone rather than blanking them.
    """
    try:
        resp = _http_get(
            f"{job_url.rstrip('/')}/detail",
            headers=_make_headers(),
            timeout=REQUEST_TIMEOUT,
        )
    except Exception:
        return {}
    if resp.status_code != 200:
        return {}
    try:
        opening = ((resp.json() or {}).get("result") or {}).get("jobOpening") or {}
    except Exception:
        return {}
    if not isinstance(opening, dict):
        return {}
    loc_data = opening.get("location")
    loc = ""
    if isinstance(loc_data, dict):
        loc = ", ".join(
            p for p in (
                str(loc_data.get("city") or "").strip(),
                str(loc_data.get("state") or "").strip(),
            ) if p
        )
    elif loc_data:
        loc = str(loc_data).strip()
    return {
        "date_posted": str(opening.get("datePosted") or "").strip(),
        "description": _html_to_text(opening.get("description")),
        "location": loc,
    }


def _scrape_bamboohr(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(BAMBOOHR, slug):
        return []
    time.sleep(random.uniform(0.5, 2.0))
    url = f"https://{slug}.bamboohr.com/careers/list"
    headers = _make_headers()
    resp = None
    for attempt in range(3):
        try:
            resp = _http_get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.SSLError:
            if attempt < 2:
                time.sleep(retry_backoff_delay(attempt))
                continue
            return []
        except Exception:
            return []
        if resp.status_code == 200:
            break
        if resp.status_code in (404, 410):
            _mark_dead(BAMBOOHR, slug)
            return []
        if resp.status_code in (429, 503, 502) and attempt < 2:
            backoff = retry_backoff_delay(attempt)
            time.sleep(backoff)
            headers["User-Agent"] = random.choice(USER_AGENTS)
            continue
        return []
    if resp is None or resp.status_code != 200:
        return []
    # An unknown tenant answers 200 and redirects to www.bamboohr.com, so the
    # tell is the response leaving the tenant's own host -- the same rule
    # live_bamboohr uses in the validator.
    #
    # Content type alone was the old test, and it marks dead far too readily. A
    # Cloudflare interstitial, a captcha, a maintenance page or any HTML error
    # is a 200 with text/html from the tenant's own host, and every one of them
    # suppressed a real company for the whole TTL. This platform holds 21,290
    # slugs, sleeps up to 2s per slug and runs many workers, so an anti-bot
    # response is not a remote possibility -- and a run of them would mark a
    # large slice of the list dead at once, silently.
    if f"{slug}.bamboohr.com" not in (resp.url or ""):
        _mark_dead(BAMBOOHR, slug)
        return []
    if "application/json" not in resp.headers.get("Content-Type", ""):
        # On the tenant's host but not serving the board: unreadable now, which
        # says nothing about whether the company exists. No mark either way.
        return []
    _mark_alive(BAMBOOHR, slug)
    try:
        data = resp.json()
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for job in data.get("result") or []:
        title = str(job.get("jobOpeningName") or "").strip()
        if not title:
            continue
        loc_data = job.get("location") or {}
        if isinstance(loc_data, dict):
            city = str(loc_data.get("city") or "").strip()
            state = str(loc_data.get("state") or "").strip()
            loc = ", ".join(p for p in [city, state] if p)
        else:
            loc = str(loc_data).strip()
        job_id = job.get("id")
        if not job_id:
            continue
        job_url = f"https://{slug}.bamboohr.com/careers/{job_id}"
        if not _matches_keywords(title, keywords):
            continue
        rows.append({
            "title": title,
            "company": slug,
            "location": loc,
            "job_url": job_url,
            "date_posted": "",
            "_source_site": BAMBOOHR,
            "_source_sites": [BAMBOOHR],
        })
        if len(rows) >= _enrich_budget(max_jobs, location):
            break
    # The list endpoint carries no description and no date at all: every
    # BambooHR row was dateless (so exempt from the age filter entirely) and
    # empty of the field SEMANTIC_MATCH_TARGET reads. Location is filtered after
    # this, not before -- the list endpoint's location can be blank, and "" never
    # matches a non-empty search, so filtering first drops real jobs and reports
    # it as "no jobs matched".
    _enrich_rows(rows, _fetch_bamboohr_detail)
    return _location_pass(rows, location, max_jobs)


# ── Platform dispatch ───────────────────────────────────────────────────────

# -- JSON-board platforms -----------------------------------------------------
#
# workable, breezy, smartrecruiters, recruitee, teamtailor and rippling each
# publish an unauthenticated JSON endpoint listing a company's open roles, so
# they share one fetch and differ only in how their rows are shaped. The older
# six scrapers each carry their own copy of the retry block below; adding six
# more copies would be six more places for a fix to miss.

_POSTED_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y", "%d %b %Y")


def _find_jobposting(raw: Any) -> dict[str, Any] | None:
    """Pull the JobPosting node out of a parsed ld+json block.

    Three shapes are in the wild and all three are load-bearing here:

      * the bare object, which is what most boards emit;
      * a list of objects, where an Organization block often comes first --
        jazzhr does this, and taking the first entry finds no fields at all;
      * an @graph wrapper, which is what iCIMS switched to. That one is the
        reason this helper exists: the old check tested @type on the outer
        object, found "@context"/"@graph" instead, and returned a result whose
        title, location, date and description were all empty. Every iCIMS job
        silently fell back to a title reconstructed from its URL and no
        location -- across 8,747 companies, presented as normal output.

    Returns None when there is no JobPosting, so a caller can tell "parsed but
    not a posting" from "parsed and empty".
    """
    if isinstance(raw, dict) and isinstance(raw.get("@graph"), list):
        raw = raw["@graph"]
    if isinstance(raw, list):
        for node in raw:
            found = _find_jobposting(node)
            if found is not None:
                return found
        return None
    if isinstance(raw, dict) and raw.get("@type") == "JobPosting":
        return raw
    return None


def _html_to_text(raw: Any) -> str:
    """Collapse an HTML posting body to text.

    The matcher normalises descriptions itself, but it is handed whatever the
    scraper stored, and several of these APIs return escaped HTML entities
    rather than markup -- Greenhouse's content field arrives as &lt;p&gt;
    rather than <p>. Unescaping first means the stored text reads as prose
    instead of as entity soup, whichever consumer looks at it.
    """
    text = str(raw or "")
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_posted(value: Any) -> str:
    """Return date_posted in a form datetime.fromisoformat can read.

    _posting_age_ok parses this field to honour "only jobs newer than N hours",
    and on a ValueError it returns True -- so an unparseable date does not drop
    the row, it exempts it from the filter entirely. Two of these boards were
    doing exactly that: recruitee sends "2026-08-19 13:16:05 UTC" and
    applicantpro "Sep 01, 2026", and every one of their jobs passed an age
    filter regardless of age. Silent, and in the direction that shows stale
    postings rather than hiding fresh ones.

    Anything already ISO is returned untouched. Anything unrecognised is
    returned as-is rather than blanked: an unreadable date is still worth
    showing a reader, and blanking it would also exempt the row.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    probe = text.replace("Z", "+00:00")
    try:
        _dt.datetime.fromisoformat(probe)
        return text
    except (ValueError, TypeError):
        pass
    # "2026-08-19 13:16:05 UTC" -- a space separator and a named zone.
    trimmed = re.sub(r"\s+(UTC|GMT)$", "+00:00", text)
    if trimmed != text:
        try:
            _dt.datetime.fromisoformat(trimmed.replace(" ", "T", 1))
            return trimmed.replace(" ", "T", 1)
        except (ValueError, TypeError):
            pass
    for fmt in _POSTED_FORMATS:
        try:
            return _dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return text


# ── per-platform refusal breaker ────────────────────────────────────────────
#
# A host that is refusing this IP outright answers every request the same way,
# and the fan-out has no way to notice: each slug is handled independently, so
# a wholesale block looks like N separate unlucky boards.
#
# Measured on workable 2026-09-05: 18 of 18 sampled requests answered 429 with
# no 200 at all. The retry path treats 429 as worth retrying -- correctly, for
# a transient limit -- so each of its 9,268 non-dead slugs costs three requests
# and roughly 4.5s of backoff sleep. Across its 4 workers that is about 2.9
# hours of sleeping against a 600s per-platform budget: workable burned its
# entire cycle allowance every cycle, returned nothing, and the time came out
# of the same wall clock the other platforms were sharing.
#
# So the breaker is not only about wasted requests. Stopping early hands the
# budget back to platforms that answer, and stops hammering a host that has
# already said no -- the politeness argument and the coverage argument point
# the same way.
#
# Deliberately NOT dead-marking anything: 403/429 is the host declining, and
# says nothing about whether a company exists. Marking on refusal would
# suppress live boards for the whole TTL, which is the mistake
# _fetch_board_json's docstring already warns about.
_REFUSAL_STATUSES = frozenset({403, 429})

# How many consecutive refusals, with no success anywhere in the cycle, before
# a platform is treated as blocked rather than unlucky. Large enough that a
# burst of rate-limiting on a healthy platform rides through -- a run of 25 with
# not one 200 is not a burst -- and small enough that a blocked platform is cut
# off in its first few seconds instead of its 600th.
_REFUSAL_TRIP = 25

_refusal_streak: dict[str, int] = {}
_platform_answered: set[str] = set()
_open_breakers: set[str] = set()

# Its own lock rather than _dead_slug_lock: this is read on the hot path of
# every request, and sharing the dead-slug lock would serialise the fan-out
# behind dead-mark bookkeeping.
_breaker_lock = threading.Lock()


def reset_refusal_breaker(platform: str) -> None:
    """Forget what a previous cycle learned about this platform.

    Called at the top of every fan-out, so a platform that was blocked an hour
    ago gets a clean try rather than staying suppressed until a restart. The
    breaker is a within-cycle economy, never a persistent dead mark.
    """
    with _breaker_lock:
        _refusal_streak.pop(platform, None)
        _platform_answered.discard(platform)
        _open_breakers.discard(platform)


def platform_is_refusing(platform: str) -> bool:
    """True once this platform has refused enough in a row to stop asking."""
    with _breaker_lock:
        return platform in _open_breakers


def refusal_report() -> dict[str, int]:
    """Open breakers and the streak that opened them, for health reporting."""
    with _breaker_lock:
        return {p: _refusal_streak.get(p, 0) for p in _open_breakers}


def _note_board_answered(platform: str) -> None:
    """A 200. Clears the streak, and permanently arms this platform's cycle.

    Once anything has answered, a later run of refusals is rate-limiting rather
    than a block, and rate-limiting is what the retry path is for.
    """
    with _breaker_lock:
        _platform_answered.add(platform)
        _refusal_streak[platform] = 0


def _note_board_refused(platform: str) -> bool:
    """A 403/429. Returns True if this refusal opened the breaker."""
    with _breaker_lock:
        if platform in _platform_answered or platform in _open_breakers:
            return False
        streak = _refusal_streak.get(platform, 0) + 1
        _refusal_streak[platform] = streak
        if streak < _REFUSAL_TRIP:
            return False
        _open_breakers.add(platform)
        return True


def _fetch_board_json(platform: str, slug: str, url: str) -> Any:
    """GET a board's JSON, handling retries and dead-marking. None on failure.

    Marks dead only on 404/410, a definite "no such board". A 403 or 429 is the
    host declining to answer and says nothing about whether the company exists,
    so it returns None and leaves the mark alone -- reading those as dead would
    suppress a live board for the whole TTL.
    """
    # Cheapest possible check, before a socket is opened: once a platform has
    # refused this many requests in a row with nothing answering, the next
    # 9,000 will be refused too.
    if platform_is_refusing(platform):
        return None

    headers = _make_headers()
    for attempt in range(3):
        try:
            resp = _http_get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            return None
        if resp.status_code == 200:
            _note_board_answered(platform)
            _mark_alive(platform, slug)
            try:
                return resp.json()
            except Exception:
                return None
        if resp.status_code in (404, 410):
            # A real answer about a real board, so it also proves the platform
            # is talking to us -- otherwise a fleet whose sampled head is all
            # deleted boards could never arm the breaker's "answered" flag.
            _note_board_answered(platform)
            _mark_dead(platform, slug)
            return None
        if resp.status_code in _REFUSAL_STATUSES and _note_board_refused(platform):
            print(f"[ats] {platform}: {_REFUSAL_TRIP} consecutive refusals "
                  f"(HTTP {resp.status_code}) and no successful request — "
                  "stopping this platform for the rest of the cycle")
            return None
        # Retrying a refusal is right for a transient limit and pointless once
        # the breaker is open, so re-check rather than sleeping into a wall.
        if (resp.status_code in (429, 503, 502) and attempt < 2
                and not platform_is_refusing(platform)):
            time.sleep(retry_backoff_delay(attempt))
            headers["User-Agent"] = random.choice(USER_AGENTS)
            continue
        return None
    return None


def _board_rows(platform: str, slug: str, jobs: Any, keywords: str, location: str,
                max_jobs: int, shape: Any) -> list[dict[str, Any]]:
    """Apply the shared filters to one board's jobs.

    `shape` reads a row's title, location, url and date out of that platform's
    own field names; everything after that is identical across platforms.
    """
    rows: list[dict[str, Any]] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        try:
            shaped = shape(job)
            # A platform that carries the description in its listing returns a
            # fifth element; the rest return four and enrich later, if at all.
            title, loc, job_url, posted = shaped[:4]
            description = shaped[4] if len(shaped) > 4 else ""
            # Sixth element: the platform's own employment-type field. Only
            # some boards publish one, so it stays optional rather than forcing
            # every shape() to return a placeholder.
            employment_type = shaped[5] if len(shaped) > 5 else ""
        except Exception:
            # One malformed row must not cost the rest of the board.
            continue
        title = str(title or "").strip()
        job_url = str(job_url or "").strip()
        if not title or not job_url:
            continue
        loc = str(loc or "").strip()
        if not _matches_keywords(title, keywords):
            continue
        if not _matches_location(loc, location):
            continue
        rows.append({
            "title": title,
            "company": slug,
            "location": loc,
            "job_url": job_url,
            "date_posted": _normalise_posted(posted),
            # job_service matches semantically against this field, so a board
            # that supplies none is invisible to matching rather than merely
            # sparse.
            "description": str(description or ""),
            # job_level.classify reads this: "Internship" here is the only
            # signal on a posting titled "Software Developer (Winter 2027)".
            "employment_type": str(employment_type or ""),
            "_source_site": platform,
            "_source_sites": [platform],
        })
        if len(rows) >= max_jobs:
            break
    return rows


def _join_location(*parts: Any) -> str:
    """Join the location fragments a platform supplies, dropping the blanks."""
    return ", ".join(str(p).strip() for p in parts if str(p or "").strip())


def _scrape_workable(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(WORKABLE, slug):
        return []
    data = _fetch_board_json(
        WORKABLE, slug,
        f"https://apply.workable.com/api/v1/widget/accounts/{slug}?details=true")
    if not isinstance(data, dict):
        return []
    return _board_rows(
        WORKABLE, slug, data.get("jobs"), keywords, location, max_jobs,
        lambda j: (j.get("title"),
                   _join_location(j.get("city"), j.get("state"), j.get("country")),
                   j.get("url") or j.get("shortlink"),
                   j.get("published_on") or j.get("created_at"),
                   j.get("description"),
                   j.get("employment_type")))


def _scrape_breezy(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(BREEZY, slug):
        return []
    data = _fetch_board_json(BREEZY, slug, f"https://{slug}.breezy.hr/json")
    if not isinstance(data, list):
        return []

    def shape(job):
        loc = job.get("location") or {}
        if not isinstance(loc, dict):
            loc = {}
        city = loc.get("city") or ""
        country = loc.get("country") or ""
        if isinstance(country, dict):
            country = country.get("name") or ""
        kind = job.get("type") or {}
        return (job.get("name"), _join_location(city, country),
                job.get("url"), job.get("published_date"), "",
                kind.get("name") if isinstance(kind, dict) else kind)

    # Location filtering is deferred: a Breezy listing that omits its city
    # fields yields "", which matches no search, and the posting page carries
    # the real location.
    rows = _board_rows(BREEZY, slug, data, keywords, "",
                       _enrich_budget(max_jobs, location), shape)
    # The board feed carries no description either, and the same page has one.
    _enrich_rows(rows, _fetch_jobposting_meta)
    return _location_pass(rows, location, max_jobs)


def _scrape_smartrecruiters(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(SMARTRECRUITERS, slug):
        return []
    # This API caps a page at 100 and large boards run far past it -- accorhotel
    # advertises 6,149. Keeping only the first page would silently drop the rest,
    # and since the caller cannot tell a truncated board from a small one, that
    # loss would never surface. Page until the caller has what it asked for.
    postings: list[Any] = []
    offset = 0
    while offset < _SMARTRECRUITERS_MAX_OFFSET:
        page = _fetch_board_json(
            SMARTRECRUITERS, slug,
            f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
            f"?limit=100&offset={offset}")
        if not isinstance(page, dict):
            break
        content = page.get("content")
        if not isinstance(content, list) or not content:
            break
        postings.extend(content)
        offset += len(content)
        # Stop as soon as enough rows exist to satisfy max_jobs even if every
        # later one were filtered out; keywords and location can only shrink
        # the set, so more pages cannot help past that point.
        if len(postings) >= max_jobs or offset >= int(page.get("totalFound") or 0):
            break
    if not postings:
        return []

    def shape(job):
        loc = job.get("location") or {}
        if not isinstance(loc, dict):
            loc = {}
        # The API carries no browser URL, so it is rebuilt from the company
        # identifier and posting id, which is the form the board itself links.
        job_id = str(job.get("id") or "").strip()
        url = f"https://jobs.smartrecruiters.com/{slug}/{job_id}" if job_id else ""
        employment = job.get("typeOfEmployment") or {}
        return (job.get("name"),
                _join_location(loc.get("city"), loc.get("region"), loc.get("country")),
                url, job.get("releasedDate"), "",
                employment.get("label") if isinstance(employment, dict) else employment)

    rows = _board_rows(SMARTRECRUITERS, slug, postings, keywords,
                       location, max_jobs, shape)
    # The postings list carries no description; the per-posting endpoint returns
    # it split across jobAd sections.
    _enrich_rows(rows, _fetch_smartrecruiters_detail)
    return rows


def _scrape_recruitee(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(RECRUITEE, slug):
        return []
    data = _fetch_board_json(RECRUITEE, slug, f"https://{slug}.recruitee.com/api/offers/")
    if not isinstance(data, dict):
        return []
    return _board_rows(
        RECRUITEE, slug, data.get("offers"), keywords, location, max_jobs,
        lambda j: (j.get("title"),
                   _join_location(j.get("city"), j.get("country_code")),
                   j.get("careers_url") or j.get("careers_apply_url"),
                   j.get("published_at") or j.get("created_at"),
                   " ".join(x for x in (j.get("description"),
                                        j.get("requirements")) if x),
                   # "internship", "fulltime_permanent", "parttime_minijob".
                   j.get("employment_type_code")))


def _scrape_teamtailor(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(TEAMTAILOR, slug):
        return []
    # A JSON Feed document, so the jobs are under "items".
    data = _fetch_board_json(TEAMTAILOR, slug, f"https://{slug}.teamtailor.com/jobs.json")
    if not isinstance(data, dict):
        return []

    def shape(job):
        # The feed carries a schema.org JobPosting alongside each item, and the
        # location only exists in there -- reading the item alone yields a blank
        # for every job, which reads downstream as "unlocated" rather than as a
        # mapping that was never wired up.
        posting = job.get("_jobposting") or {}
        if not isinstance(posting, dict):
            posting = {}
        places = posting.get("jobLocation") or []
        if isinstance(places, dict):
            places = [places]
        addr = {}
        if places and isinstance(places[0], dict):
            addr = places[0].get("address") or {}
            if not isinstance(addr, dict):
                addr = {}
        loc = _join_location(addr.get("addressLocality"),
                             addr.get("addressRegion"),
                             addr.get("addressCountry"))
        return (job.get("title"), loc, job.get("url"),
                job.get("date_published") or posting.get("datePosted"),
                job.get("content_html") or posting.get("description"))

    return _board_rows(TEAMTAILOR, slug, data.get("items"), keywords, location,
                       max_jobs, shape)


def _scrape_rippling(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(RIPPLING, slug):
        return []
    data = _fetch_board_json(
        RIPPLING, slug,
        f"https://api.rippling.com/platform/api/ats/v1/board/{slug}/jobs")
    if not isinstance(data, list):
        return []

    def shape(job):
        work = job.get("workLocation") or {}
        loc = work.get("label") if isinstance(work, dict) else ""
        # This feed carries no posting date. The archive records first sighting
        # anyway, so an empty date is the honest value rather than today's.
        return (job.get("name"), loc, job.get("url"), "")

    rows = _board_rows(RIPPLING, slug, data, keywords, location, max_jobs, shape)
    # The feed has no date field at all; it only exists on the posting page, so
    # it is fetched for the rows that already survived filtering.
    _enrich_rows(rows, _fetch_rippling_created)
    return rows


# -- Rendered-board platforms -------------------------------------------------
#
# jazzhr, jobvite and applicantpro publish no job-listing API, so these read the
# board the way a browser would. That is more fragile than the JSON platforms
# above -- a markup change breaks extraction where a field rename would not --
# so each parser is pinned by a test against the real markup, and each returns
# nothing rather than guessing when the shape it expects is absent.

_LD_BLOCK = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I)
_NEXT_DATA = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def _ld_address(place: Any) -> str:
    """One schema.org Place rendered as a location string."""
    if not isinstance(place, dict):
        return ""
    addr = place.get("address")
    if not isinstance(addr, dict):
        return ""
    return _join_location(addr.get("addressLocality"),
                          addr.get("addressRegion"),
                          addr.get("addressCountry"))


def _fetch_jobposting_meta(job_url: str) -> dict[str, str]:
    """Read datePosted and jobLocation from a job page's schema.org JobPosting.

    Several of these boards list a job without either field -- jobvite renders
    "2 Locations" where a city belongs, and jazzhr carries no date at all -- and
    the posting page is where the real values live.

    "resolved" reports whether the page was actually parsed, so the caller can
    tell an unlocated posting from a fetch that failed. That distinction decides
    whether a row survives a location filter, and getting it wrong drops real
    jobs: an empty location never matches a non-empty search.
    """
    meta: dict[str, str] = {"date_posted": "", "location": "", "description": "",
                            "resolved": ""}
    headers = _make_headers({"Accept": "text/html,application/xhtml+xml"})
    resp = None
    for attempt in range(2):
        try:
            resp = _http_get(job_url, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            resp = None
        if resp is not None and resp.status_code == 200:
            break
        if attempt == 0:
            time.sleep(retry_backoff_delay(attempt))
            headers["User-Agent"] = random.choice(USER_AGENTS)
    if resp is None or resp.status_code != 200:
        return meta

    for raw in _LD_BLOCK.findall(resp.text or ""):
        try:
            data = json.loads(raw.strip())
        except Exception:
            continue
        data = _find_jobposting(data)
        if data is None:
            # Not a posting block. jazzhr emits an Organization block before the
            # JobPosting one, and iCIMS wraps its posting in @graph -- both of
            # which the shared finder handles.
            continue
        meta["resolved"] = "1"
        meta["date_posted"] = str(data.get("datePosted") or "")
        meta["description"] = str(data.get("description") or "")
        # schema.org publishes this as INTERN/FULL_TIME/PART_TIME, or as a list.
        emp = data.get("employmentType")
        if isinstance(emp, list):
            emp = " ".join(str(e) for e in emp)
        meta["employment_type"] = str(emp or "")
        places = data.get("jobLocation")
        if isinstance(places, dict):
            places = [places]
        if isinstance(places, list):
            # A multi-site posting is why the listing said "2 Locations". Keep
            # every one: the filter matches a search against this string, so
            # dropping the extras would discard jobs in the searched city.
            seen: list[str] = []
            for place in places:
                text = _ld_address(place)
                if text and text not in seen:
                    seen.append(text)
            meta["location"] = "; ".join(seen)
        return meta
    return meta


def _fetch_rippling_created(job_url: str) -> dict[str, str]:
    """Posting date and description for one Rippling job.

    The board feed carries neither, and the page publishes no schema.org block,
    so both come out of the __NEXT_DATA__ payload the page hydrates from.

    Parsed as JSON rather than pattern-matched. A regex over the raw payload
    looks equivalent and is not: the same document carries UI labels under the
    same key names, and matching "createdOn" loosely returns the literal string
    "Created on" as a posting date. Walking to props.pageProps.apiData.jobPost
    reads the one that belongs to this job.
    """
    meta: dict[str, str] = {"date_posted": "", "location": "", "description": "",
                            "resolved": ""}
    headers = _make_headers({"Accept": "text/html,application/xhtml+xml"})
    try:
        resp = _http_get(job_url, headers=headers, timeout=REQUEST_TIMEOUT)
    except Exception:
        return meta
    if resp.status_code != 200:
        return meta
    blob = _NEXT_DATA.search(resp.text or "")
    if not blob:
        return meta
    try:
        payload = json.loads(blob.group(1))
        post = (payload["props"]["pageProps"]["apiData"]["jobPost"]) or {}
    except Exception:
        return meta
    if not isinstance(post, dict):
        return meta

    created = post.get("createdOn")
    if isinstance(created, str) and created.strip():
        meta["date_posted"] = created.strip()
        meta["resolved"] = "1"

    # The description is a dict of named sections (company blurb, the role, and
    # so on) rather than one string, so the sections are joined -- taking any
    # single one would hand the matcher boilerplate instead of the job.
    described = post.get("description")
    if isinstance(described, dict):
        described = " ".join(v for v in described.values() if isinstance(v, str))
    if isinstance(described, str) and described.strip():
        meta["description"] = described.strip()
        meta["resolved"] = "1"
    return meta


def _fetch_smartrecruiters_detail(job_url: str) -> dict[str, str]:
    """Description for one SmartRecruiters posting.

    The browser URL is jobs.smartrecruiters.com/<company>/<id>, and the API
    that answers with the text is api.smartrecruiters.com, so the id is taken
    back out of the URL the listing built.
    """
    meta: dict[str, str] = {"date_posted": "", "location": "", "description": "",
                            "resolved": ""}
    parts = [p for p in job_url.split("/") if p]
    if len(parts) < 2:
        return meta
    company, posting_id = parts[-2], parts[-1]
    try:
        resp = _http_get(
            f"https://api.smartrecruiters.com/v1/companies/{company}/postings/{posting_id}",
            headers=_make_headers(), timeout=REQUEST_TIMEOUT)
    except Exception:
        return meta
    if resp.status_code != 200:
        return meta
    try:
        sections = (resp.json().get("jobAd") or {}).get("sections") or {}
    except Exception:
        return meta
    if not isinstance(sections, dict):
        return meta
    text = " ".join(
        str((sections.get(name) or {}).get("text") or "")
        for name in ("jobDescription", "qualifications", "additionalInformation")
        if isinstance(sections.get(name), dict))
    meta["description"] = text.strip()
    meta["resolved"] = "1" if text.strip() else ""
    return meta


_ENRICH_GATES: dict[str, threading.Semaphore] = {}
_ENRICH_GATES_LOCK = threading.Lock()


def _enrich_gate(platform: str) -> threading.Semaphore:
    """The number of enrichment requests allowed in flight against one vendor.

    PLATFORM_WORKERS is described throughout this file as a measured politeness
    ceiling, to be scaled down and never up. It was not the ceiling.
    _scrape_ats_platform fans out that many slugs at once, and every one of
    those slugs then opened its own enrichment pool of five on top. The pools
    nest, so the real rate against a vendor was the product: thirty slugs times
    five enrichers is a hundred and fifty concurrent requests to bamboohr.com.

    Measured consequence, not a theory. bamboohr's /careers/{id}/detail returns
    a posting date and a full description reliably when asked one at a time --
    three of three probed, about 2.9s each -- and 4 of 212 archived rows came
    back with either. The same 4 for both fields, so enrichment succeeded four
    times and was refused the rest. icims and oracle, the other platforms that
    must enrich, sit at 26% and 18%. greenhouse and lever report 100% because
    their list endpoints carry the fields and they never enrich at all, which
    is what places the damage exactly where the extra requests are.

    One gate per platform, held for the life of the process, because what needs
    bounding is the total in flight against a host -- and that is the one thing
    a per-call pool size cannot express, however small it is made.
    """
    with _ENRICH_GATES_LOCK:
        gate = _ENRICH_GATES.get(platform)
        if gate is None:
            gate = threading.Semaphore(
                capacity.workers(PLATFORM_WORKERS.get(platform, 10), minimum=2))
            _ENRICH_GATES[platform] = gate
        return gate


def _enrich_rows(rows: list[dict[str, Any]], fetch: Any) -> None:
    """Fill date_posted (and location, when blank) from each row's job page.

    Writes are guarded: a failed fetch must leave what the listing already gave
    us rather than blanking it. An unconditional write is how a rate-limited
    batch would erase good locations and take the jobs with them.

    The platform is read off the rows rather than passed in. Every caller here
    builds rows through _board_rows or stamps _source_site itself, so it is
    already present at all seven call sites -- and taking it from the data
    removes the one way a caller could gate a vendor behind another vendor's
    ceiling.
    """
    if not rows:
        return
    gate = _enrich_gate(str(rows[0].get("_source_site") or ""))
    # Rows the archive already holds keep their listing values and are
    # dropped later by _drop_already_archived; fetching their pages first
    # bought nothing.
    rows = _needs_enrichment(rows)
    if not rows:
        return

    def _gated(url: str) -> dict[str, str]:
        with gate:
            return fetch(url)

    meta_workers = capacity.workers(5, minimum=2)
    with ThreadPoolExecutor(max_workers=meta_workers) as pool:
        futures = {pool.submit(_gated, row["job_url"]): row["job_url"] for row in rows}
        by_url: dict[str, dict[str, str]] = _collect_results(
            futures, capacity.timeout(_fanout_budget(len(rows), meta_workers)))
    for row in rows:
        meta = by_url.get(row["job_url"]) or {}
        if meta.get("date_posted"):
            row["date_posted"] = _normalise_posted(meta["date_posted"])
        if meta.get("location"):
            row["location"] = meta["location"]
        if meta.get("description") and not row.get("description"):
            row["description"] = meta["description"]


def _location_pass(rows: list[dict[str, Any]], location: str,
                   max_jobs: int) -> list[dict[str, Any]]:
    """Apply the location filter after enrichment, and cap the result.

    A listing that omits its location field yields "", and _matches_location
    returns False for "" against any non-empty search -- so filtering before
    the posting page has been read drops real jobs in the searched city and
    presents it as "no jobs matched". The fix is ordering, not a new rule: read
    the location first, then filter, which is what iCIMS and Jobvite already do.
    """
    if not location.strip():
        return rows[:max_jobs]
    kept: list[dict[str, Any]] = []
    for row in rows:
        if not _matches_location(row.get("location", ""), location):
            continue
        kept.append(row)
        if len(kept) >= max_jobs:
            break
    return kept


def _enrich_budget(max_jobs: int, location: str) -> int:
    """How many candidates to gather before enriching.

    Only over-fetch when a location search is active, since that is the only
    case where enrichment changes which rows survive. Mirrors the iCIMS numbers.
    """
    if not location.strip():
        return max_jobs
    return min(max_jobs * _JOBVITE_LOCATION_OVERFETCH, _JOBVITE_MAX_CANDIDATES)


def _fetch_board_html(platform: str, slug: str, url: str) -> str:
    """GET a rendered board page. Empty string on failure.

    Same dead-marking rule as the JSON boards: 404/410 marks, everything else
    is a failure to read rather than evidence the company is gone.
    """
    headers = _make_headers({"Accept": "text/html,application/xhtml+xml"})
    for attempt in range(3):
        try:
            resp = _http_get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            return ""
        if resp.status_code == 200:
            _mark_alive(platform, slug)
            return resp.text or ""
        if resp.status_code in (404, 410):
            _mark_dead(platform, slug)
            return ""
        if resp.status_code in (429, 503, 502) and attempt < 2:
            time.sleep(retry_backoff_delay(attempt))
            headers["User-Agent"] = random.choice(USER_AGENTS)
            continue
        return ""
    return ""


def _strip_tags(fragment: str) -> str:
    """Visible text of an HTML fragment, whitespace collapsed."""
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    text = (text.replace("&amp;", "&").replace("&#039;", "'")
                .replace("&quot;", '"').replace("&nbsp;", " ")
                .replace("&lt;", "<").replace("&gt;", ">"))
    return re.sub(r"\s+", " ", text).strip()


_JAZZHR_ITEM = re.compile(r'<li class="list-group-item">(.*?)</li>\s*</ul>\s*</li>', re.S)
_JAZZHR_LINK = re.compile(r'<a\s+href="([^"]+/apply/[^"]+)"[^>]*>(.*?)</a>', re.S)
_JAZZHR_LOC = re.compile(r'fa-map-marker[^>]*></i>\s*([^<]*)', re.S)


def _scrape_jazzhr(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(JAZZHR, slug):
        return []
    html = _fetch_board_html(JAZZHR, slug, f"https://{slug}.applytojob.com/apply")
    if not html:
        return []
    rows: list[dict[str, Any]] = []
    for block in _JAZZHR_ITEM.findall(html):
        link = _JAZZHR_LINK.search(block)
        if not link:
            continue
        job_url = link.group(1).strip()
        title = _strip_tags(link.group(2))
        loc_match = _JAZZHR_LOC.search(block)
        loc = _strip_tags(loc_match.group(1)) if loc_match else ""
        if not title or not job_url:
            continue
        if not _matches_keywords(title, keywords):
            continue
        rows.append({
            "title": title, "company": slug, "location": loc,
            "job_url": job_url, "date_posted": "",
            "_source_site": JAZZHR, "_source_sites": [JAZZHR],
        })
        if len(rows) >= _enrich_budget(max_jobs, location):
            break
    # The listing usually carries a location, but only when the row includes a
    # map-marker element -- rows without one yield "", which matches no search.
    # So the page is read first and the location filter applied after, or a job
    # in the searched city disappears because its listing lacked an icon.
    _enrich_rows(rows, _fetch_jobposting_meta)
    return _location_pass(rows, location, max_jobs)


_JOBVITE_ROW = re.compile(
    r'<td class="jv-job-list-name">\s*<a href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<td class="jv-job-list-location">(.*?)</td>', re.S)


def _scrape_jobvite(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(JOBVITE, slug):
        return []
    html = _fetch_board_html(JOBVITE, slug, f"https://jobs.jobvite.com/{slug}")
    if not html:
        return []
    # The listing renders "2 Locations" for any multi-site posting, which is not
    # a place and matches no search, so a real job in the searched city is
    # dropped before anything can look at it. The location has to come from the
    # posting page, which means candidates are gathered on title alone and
    # filtered on location only after enrichment -- the same order iCIMS uses,
    # and for the same reason.
    candidate_cap = max_jobs
    if location.strip():
        candidate_cap = min(max_jobs * _JOBVITE_LOCATION_OVERFETCH,
                            _JOBVITE_MAX_CANDIDATES)
    candidates: list[dict[str, Any]] = []
    for href, raw_title, raw_loc in _JOBVITE_ROW.findall(html):
        title = _strip_tags(raw_title)
        loc = _strip_tags(raw_loc)
        if not title:
            continue
        job_url = href.strip()
        if job_url.startswith("/"):
            job_url = f"https://jobs.jobvite.com{job_url}"
        if not _matches_keywords(title, keywords):
            continue
        candidates.append({
            "title": title, "company": slug, "location": loc,
            "job_url": job_url, "date_posted": "",
            "_source_site": JOBVITE, "_source_sites": [JOBVITE],
        })
        if len(candidates) >= candidate_cap:
            break
    if not candidates:
        return []
    _enrich_rows(candidates, _fetch_jobposting_meta)

    rows: list[dict[str, Any]] = []
    for row in candidates:
        if not _matches_location(row["location"], location):
            continue
        rows.append(row)
        if len(rows) >= max_jobs:
            break
    return rows


_APPLICANTPRO_DOMAIN = re.compile(r"domainId\s*:\s*(\d+)")


def _scrape_applicantpro(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    """Two requests: the board page carries the domain id its own script needs,
    and the listing endpoint will not answer without it.

    The endpoint also rejects a call with no getParams -- it returns a PHP type
    error rather than an empty result -- so a minimal one is always sent.
    """
    if _is_dead(APPLICANTPRO, slug):
        return []
    html = _fetch_board_html(APPLICANTPRO, slug, f"https://{slug}.applicantpro.com/jobs/")
    if not html:
        return []
    found = _APPLICANTPRO_DOMAIN.search(html)
    if not found:
        # A disabled or nonexistent board serves a bare sentence with no script
        # block. Not evidence of a 404, so it is not marked dead here.
        return []
    params = quote(json.dumps({"isInternal": 0, "showLocation": 1}))
    data = _fetch_board_json(
        APPLICANTPRO, slug,
        f"https://{slug}.applicantpro.com/core/jobs/{found.group(1)}?getParams={params}")
    if not isinstance(data, dict):
        return []
    jobs = (data.get("data") or {}).get("jobs")

    def shape(job):
        job_id = str(job.get("id") or "").strip()
        sub = str(job.get("subdomain") or slug).strip()
        url = f"https://{sub}.applicantpro.com/jobs/{job_id}" if job_id else ""
        return (job.get("title"),
                _join_location(job.get("city"), job.get("abbreviation")),
                url, job.get("startDateRef"))

    rows = _board_rows(APPLICANTPRO, slug, jobs, keywords, "",
                       _enrich_budget(max_jobs, location), shape)
    # The listing date is "Sep 01, 2026", which no ISO parser reads; the posting
    # page carries a real datePosted, the description, and the location for rows
    # whose listing omitted the city -- which is why the filter waits until now.
    _enrich_rows(rows, _fetch_jobposting_meta)
    return _location_pass(rows, location, max_jobs)


_PAYLOCITY_PAGEDATA = re.compile(
    r"window\.pageData\s*=\s*(\{.*?\});?\s*</script>", re.S)


def _paylocity_location(value: Any) -> str:
    """A readable location from Paylocity's location field, which may be an object.

    It ships either a plain string or a dict carrying City/State/Country among
    a dozen internal ids. Stringifying the dict produced a location no search
    could ever match, so the useful fields are read out instead. A dict whose
    place fields are all null still yields its country, which is worth more
    than nothing and far more than the raw blob.
    """
    if isinstance(value, dict):
        # _join_location already drops blanks, so nulls need no handling here.
        return _join_location(*(value.get(k) for k in ("City", "State", "Country")))
    return str(value or "").strip()


def _scrape_paylocity(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    """Paylocity boards, keyed by the company GUID the harvester collects.

    This platform was harvested (15,908 companies) and validated weekly, and
    then nothing could read it: paylocity was absent from ATS_PLATFORMS and had
    no scraper, so every one of those requests bought nothing. That is the
    largest single block of wasted work in the pipeline.

    The board renders client-side, but it does not need an API: the whole jobs
    array is embedded in the initial HTML as window.pageData, with the title,
    location, an ISO PublishedDate and the full description already present. So
    one request per company yields complete rows, and no per-job enrichment is
    needed at all.
    """
    if _is_dead(PAYLOCITY, slug):
        return []
    html_text = _fetch_board_html(
        PAYLOCITY, slug,
        f"https://recruiting.paylocity.com/recruiting/jobs/All/{slug}")
    if not html_text:
        return []
    found = _PAYLOCITY_PAGEDATA.search(html_text)
    if not found:
        # A board that renders without pageData is one this cannot read. Not a
        # 404, so it is not marked dead -- unreadable is not the same as gone.
        return []
    try:
        page = json.loads(found.group(1))
    except Exception:
        return []
    jobs = page.get("Jobs") if isinstance(page, dict) else None

    def shape(job):
        job_id = str(job.get("JobId") or "").strip()
        url = (f"https://recruiting.paylocity.com/recruiting/jobs/Details/{job_id}"
               if job_id else "")
        # Paylocity returns this as an object, not a string, so the previous
        # `or ""` fallthrough handed a dict straight to _join_location and the
        # archive recorded locations like "{'LocationId': 3092273, ...,
        # 'Country': 'USA', ...}, Remote". Every paylocity row was therefore
        # unmatchable by _matches_location -- not filtered out, just never
        # matching any search a channel could express.
        loc = _paylocity_location(job.get("LocationName") or job.get("JobLocation"))
        if job.get("IsRemote") and "remote" not in str(loc).lower():
            loc = _join_location(loc, "Remote")
        return (job.get("JobTitle"), loc, url, job.get("PublishedDate"),
                _html_to_text(job.get("Description")))

    return _board_rows(PAYLOCITY, slug, jobs, keywords, location, max_jobs, shape)


def _scrape_oracle(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    """Oracle Cloud Recruiting (Fusion), the successor to Taleo.

    The slug is the whole Fusion host -- `eeho.fa.us2.oraclecloud.com` -- not a
    company label. The pod, region and instance are all encoded in it and no
    single part identifies the customer. `siteNumber` is in the finder because
    the parameter is required, not because it selects: CX_1, CX_2, CX_3 and
    CX_45001 all returned the same TotalJobsCount when measured, so the host
    addresses the tenant's whole job set rather than one site's slice.

    Paging stops at the freshness edge rather than at a page count. The feed
    sorts by posting date descending, and merge_data only archives postings
    inside seven days, so once a page opens with an older date every remaining
    page is older still. On the tenant measured that turns 2,276 jobs into one
    or two requests instead of twelve -- which is what makes a platform with
    thousands of requisitions per tenant affordable at all.

    The finder's arguments are positional and comma-separated INSIDE it.
    Putting `limit` or `sortBy` in the query string instead returns 200 with
    zero rows, which reads exactly like an empty tenant.
    """
    if _is_dead(ORACLE, slug):
        return []

    wanted = max_jobs if max_jobs > 0 else 10_000
    cutoff = date.today() - timedelta(days=ORACLE_FRESH_DAYS)
    rows: list[dict[str, Any]] = []
    offset = 0

    for _ in range(ORACLE_MAX_PAGES):
        finder = (f"findReqs;siteNumber=CX_1,limit={ORACLE_PAGE_SIZE},"
                  f"offset={offset},sortBy=POSTING_DATES_DESC")
        data = _fetch_board_json(
            ORACLE, slug,
            f"https://{slug}/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
            f"?onlyData=true&expand=requisitionList.secondaryLocations&finder={finder}")
        if not isinstance(data, dict):
            break
        items = data.get("items") or []
        page = (items[0].get("requisitionList") or []) if items else []
        if not page:
            break

        rows.extend(_board_rows(
            ORACLE, slug, page, keywords, location, wanted,
            lambda j: (j.get("Title"),
                       _oracle_location(j),
                       _oracle_url(slug, j),
                       j.get("PostedDate"),
                       j.get("ShortDescriptionStr") or j.get("ExternalResponsibilitiesStr"),
                       j.get("JobType") or j.get("ContractType"))))

        # The page that crosses the cutoff is still kept -- it is only the
        # pages after it that cannot contain anything archivable.
        oldest = str(page[-1].get("PostedDate") or "")
        if oldest and oldest < cutoff.isoformat():
            break
        if len(page) < ORACLE_PAGE_SIZE or len(rows) >= wanted:
            break
        offset += ORACLE_PAGE_SIZE

    return rows[:wanted]


def _oracle_location(job: dict[str, Any]) -> str:
    """Oracle already knows the ISO country, so do not re-derive it from prose.

    `PrimaryLocation` is a display string ("NOIDA, UTTAR PRADESH, India") while
    `PrimaryLocationCountry` is a plain ISO code. Appending the code makes
    geo_priority.country_of decide on the code rather than parsing the tail of
    a localised string, which is where the CA/DE ambiguities come from.
    """
    display = str(job.get("PrimaryLocation") or "").strip()
    code = str(job.get("PrimaryLocationCountry") or "").strip()
    if code and display and not display.upper().endswith(code.upper()):
        return f"{display}, {code}"
    return display or code


def _oracle_url(slug: str, job: dict[str, Any]) -> str:
    job_id = str(job.get("Id") or "").strip()
    if not job_id:
        return ""
    return (f"https://{slug}/hcmUI/CandidateExperience/en/sites/CX_1/job/{job_id}")


def _scrape_personio(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    """Personio's per-tenant XML feed.

    XML rather than JSON, which is why this does not go through
    _fetch_board_json: that helper parses JSON and would read every feed as a
    failure. The dead-marking and refusal accounting it provides are replicated
    here rather than skipped, or Personio would be the one platform whose 404s
    never suppressed a slug.

    The envelope is <workzag-jobs> (Personio's original product name) holding
    <position> elements. A live tenant with no open roles still returns the
    envelope, so its presence -- not the position count -- is what proves the
    board exists.
    """
    if _is_dead(PERSONIO, slug):
        return []
    if platform_is_refusing(PERSONIO):
        return []

    url = f"https://{slug}.jobs.personio.de/xml?language=en"
    try:
        resp = _http_get(url, headers=_make_headers(), timeout=REQUEST_TIMEOUT)
    except Exception:
        return []

    if resp.status_code in (404, 410):
        _note_board_answered(PERSONIO)
        _mark_dead(PERSONIO, slug)
        return []
    if resp.status_code in _REFUSAL_STATUSES:
        if _note_board_refused(PERSONIO):
            print(f"[ats] {PERSONIO}: {_REFUSAL_TRIP} consecutive refusals "
                  f"(HTTP {resp.status_code}) and no successful request — "
                  "stopping this platform for the rest of the cycle")
        return []
    if resp.status_code != 200:
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []
    _note_board_answered(PERSONIO)
    _mark_alive(PERSONIO, slug)

    def _text(node: Any, tag: str) -> str:
        found = node.find(tag)
        return (found.text or "").strip() if found is not None and found.text else ""

    positions = []
    for pos in root.findall(".//position"):
        offices = [_text(pos, "office")]
        offices += [(o.text or "").strip()
                    for o in pos.findall("./additionalOffices/office") if o.text]
        positions.append({
            "title": _text(pos, "name"),
            "office": ", ".join(o for o in offices if o),
            "id": _text(pos, "id"),
            "posted": _text(pos, "createdAt") or _text(pos, "occupationCategory"),
            "type": _text(pos, "employmentType") or _text(pos, "schedule"),
        })

    return _board_rows(
        PERSONIO, slug, positions, keywords, location,
        max_jobs if max_jobs > 0 else 10_000,
        lambda j: (j.get("title"),
                   j.get("office"),
                   f"https://{slug}.jobs.personio.de/job/{j.get('id')}" if j.get("id") else "",
                   j.get("posted"),
                   "",
                   j.get("type")))


_SCRAPERS: dict[str, Any] = {
    GREENHOUSE: _scrape_greenhouse,
    LEVER: _scrape_lever,
    ASHBY: _scrape_ashby,
    WORKDAY: _scrape_workday,
    ICIMS: _scrape_icims,
    BAMBOOHR: _scrape_bamboohr,
    WORKABLE: _scrape_workable,
    BREEZY: _scrape_breezy,
    SMARTRECRUITERS: _scrape_smartrecruiters,
    RECRUITEE: _scrape_recruitee,
    TEAMTAILOR: _scrape_teamtailor,
    RIPPLING: _scrape_rippling,
    JAZZHR: _scrape_jazzhr,
    JOBVITE: _scrape_jobvite,
    APPLICANTPRO: _scrape_applicantpro,
    PAYLOCITY: _scrape_paylocity,
    ORACLE: _scrape_oracle,
    PERSONIO: _scrape_personio,
}


def _needs_enrichment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`rows` minus the postings the archive already holds.

    Enrichment is a page fetch per posting, and _drop_already_archived then
    discards every row the archive has seen -- which, on a board asked every
    cycle, is nearly all of them. Paying for the fetch first and dropping the
    row after is the cost that put icims at a sitemap plus hundreds of pages
    per board. The identity the archive keys on (the ATS job id, or the URL)
    is on the row before enrichment, so the same check can run before it.

    Same failure posture as _drop_already_archived: an unavailable archive
    means everything is enriched, never that anything is skipped.
    """
    if not rows or not ATS_ARCHIVE_DEDUP_ENABLED:
        return rows
    try:
        from services.jba import archive_index

        archive_index.ensure_index()
        kept, _dropped = archive_index.filter_new_listings(rows)
    except Exception:
        return rows
    return kept


def _drop_already_archived(platform: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop listings already present anywhere in the stored job archive.

    Checks the full history in data/jba/jobs/**/*.zip (~500k records), not just
    the current week: merge_data's own seen_urls table is purged on a 30-day TTL,
    so it cannot answer "have we ever seen this". The per-channel dedup
    downstream is narrower still -- one channel, capped at
    DEDUP_SEEN_LINKS_CAP entries.

    When a duplicate is found the ARCHIVED copy is preserved and the freshly
    scraped one dropped, so the recorded first-sighting date stays the true one.

    Import is local and failures are swallowed: the archive is an optimisation,
    and an ATS scrape must never fail because the job log is unavailable.
    """
    if not rows or not ATS_ARCHIVE_DEDUP_ENABLED:
        return rows
    try:
        from services.jba import archive_index

        archive_index.ensure_index()
        kept, dropped = archive_index.filter_new_listings(rows)
    except Exception as exc:
        print(f"[ats] archive dedup unavailable for {platform}: {exc}")
        return rows

    if dropped:
        print(f"[ats] {platform}: dropped {dropped} duplicate listing(s) already in the archive")
    return kept


def _stamp_levels(rows: list[dict[str, Any]],
                  levels: list[str] | tuple[str, ...] | None) -> list[dict[str, Any]]:
    """Attach a seniority level to every row, and drop rows outside *levels*.

    Done here rather than in each scraper for two reasons: it is one place
    instead of sixteen, and every row has been through its platform's own
    enrichment by this point, so employment_type and description are populated
    -- classify() reads both, and doing this earlier would classify on a title
    alone and silently lose the co-op/intern distinction.

    The level is stamped even when no filter is set, so callers that rank rather
    than filter can read it. job_level had a full classifier and 76 passing
    tests but no caller anywhere in src/ -- the same shape as the fields that
    were parsed and thrown away.
    """
    kept: list[dict[str, Any]] = []
    for row in rows:
        verdict = job_level.classify(
            str(row.get("title") or ""),
            str(row.get("description") or ""),
            str(row.get("employment_type") or ""),
        )
        row["level"] = verdict.level
        if verdict.term:
            row["level_term"] = f"{verdict.term[0]} {verdict.term[1]}".strip()
        if job_level.allowed(verdict, levels):
            kept.append(row)
    return kept


# Per-platform {submitted, completed} for the most recent company fan-out.
# Read by the scrape loop to record real coverage; a plain dict rather than
# state on the tracker so this module keeps no dependency on services.health.
LAST_FANOUT: dict[str, dict[str, int]] = {}


def scrape_ats_platform(
    platform: str,
    keywords: str,
    location: str,
    results_wanted: int = 20,
    company_slugs: list[str] | None = None,
    levels: list[str] | tuple[str, ...] | None = None,
    max_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Scrape one ATS platform's fleet.

    *max_seconds* caps how long the fan-out will WAIT for its fetches. Without
    it the wait is _fanout_budget(), which is derived from the size of the
    fleet -- and the fleet grows every time the harvest runs. Measured against
    the live lists: 2.2h for lever, 5.9h for bamboohr, 20.5h for workable,
    against a caller that allows 600s. See the fan-out below.
    """
    scraper = _SCRAPERS.get(platform)
    if not scraper:
        return []

    # A new cycle deserves a clean judgement: a platform blocked six hours ago
    # may well answer now, and the breaker must never outlive the fan-out that
    # opened it.
    reset_refusal_breaker(platform)

    if company_slugs is None:
        company_slugs = load_company_lists().get(platform, [])

    if not company_slugs:
        return []

    max_per_company = results_wanted if results_wanted > 0 else 10_000
    # See fanout_workers for why the width is computed there and nowhere else.
    workers = fanout_workers(platform, len(company_slugs))
    all_rows: list[dict[str, Any]] = []

    # Nested pool: bounded (accepted exception to scheduler-visible concurrency;
    # the caller holds one PriorityWorkScheduler slot for this whole fan-out).
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scraper, slug, keywords, location, max_per_company): slug
            for slug in company_slugs
        }
        # _fanout_budget answers "how long would asking everyone take", which
        # is the right question only if we are allowed that long. We are not:
        # the caller bounds a platform at ATS_PLATFORM_TIMEOUT_S, and the two
        # numbers had no relationship at all. Measured against the live fleet
        # lists, the fan-out budget ran from 13x that bound (lever, 2.2h) to
        # 123x (workable, 20.5h) -- and the observed thread lifetimes matched
        # the budgets, not the bound: breezy ran 18,020s against a budget of
        # 20,610s.
        #
        # The outer bound could not correct it, because asyncio.wait_for
        # cancels the await and not the thread: the awaiting side returned []
        # at 600s while the thread kept its scheduler worker for hours. That is
        # what starved the job watchers, and it is also why "it worked earlier
        # today" -- the budget grows with the fleet, so every harvest makes it
        # worse.
        #
        # Capped after capacity.timeout, not before: that call STRETCHES a
        # budget for slower hardware, so capping first would let the stretch
        # walk straight back past the cap.
        budget = capacity.timeout(_fanout_budget(len(company_slugs), workers))
        if max_seconds is not None:
            budget = min(budget, max(1.0, float(max_seconds)))
        per_slug = _collect_results(futures, budget)
        # How much of the fleet this cycle actually reached. _collect_results
        # only printed the cancelled count, and reasoning about coverage from
        # that alone is how "43-100% of companies reached" was once misread as
        # "2% reached". The complement is the number worth recording.
        LAST_FANOUT[platform] = {"submitted": len(futures), "completed": len(per_slug)}
    for rows in per_slug.values():
        all_rows.extend(rows)

    flush_dead_slugs()
    # Level filtering runs after the per-company cap, so a company whose first
    # `max_per_company` rows are all senior contributes nothing here. That is
    # the same trade the location filter already makes, and widening it means
    # fetching every posting from every company.
    return _stamp_levels(_drop_already_archived(platform, all_rows), levels)
