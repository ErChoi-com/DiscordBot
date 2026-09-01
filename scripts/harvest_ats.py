"""Harvest ATS company slugs from the Common Crawl URL index.

The company lists under ``data/ats_companies`` come from
``Feashliaa/job-board-aggregator`` (see ``sync_ats_companies.py``).  That repo
builds them with a Common Crawl harvester it does not publish -- its README
describes the pipeline in one paragraph and ships none of the code.  So the
lists only grow when that maintainer decides to regenerate them, and a company
that opened a board last week is invisible to us until then.

This is that harvester, rebuilt from the public CDX API.  It is deliberately
*additive*: it never removes a slug and never rewrites the upstream lists.  It
emits a separate set of files that ``ats_service.load_company_lists`` unions on
top of upstream, so a bad harvest degrades to "some extra slugs that 404" --
which the existing dead-slug machinery (TTL + staggered re-probe) already
disposes of -- rather than to a broken company list.

Why the CDX API and not the WARC/WAT files: the index answers a domain-prefix
query with a URL list directly, over plain HTTP, with no key and no S3 egress.
Harvesting one crawl across every platform is ~15 requests.  Reading the WAT
files for the same information would be terabytes.

Usage:
    python scripts/harvest_ats.py                  # latest crawl
    python scripts/harvest_ats.py --crawl CC-MAIN-2026-30
    python scripts/harvest_ats.py --crawls 3       # 3 most recent
    python scripts/harvest_ats.py --platform greenhouse --dry-run
    python scripts/harvest_ats.py --out data/ats_harvest

Exit codes:
    0  harvest completed (possibly with zero new slugs)
    1  harvest produced nothing usable -- every platform failed
    2  harness error (bad arguments, unwritable output)
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]

COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
INDEX_URL_TEMPLATE = "https://index.commoncrawl.org/{crawl}-index"

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "ats_harvest"

# The CDX API is a free public service with no key and no published quota.  A
# courteous fixed delay between requests matters more than raw speed here: a
# full sweep is ~15 requests per crawl, so even a second apiece is nothing, and
# being the client that hammers it is how free endpoints acquire quotas.
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT = 180
MAX_RETRIES = 4

# How many pages to walk when the page count is unavailable. Well above the
# largest real count seen (Greenhouse and Workday peak at 5), so it bounds a
# runaway rather than truncating a genuine result set.
BLIND_PAGE_LIMIT = 25

USER_AGENT = (
    "job-board-harvester/1.0 "
    "(+https://github.com/Feashliaa/job-board-aggregator; ATS company discovery)"
)

# Path segments that appear in the slug position but are not companies.  These
# are cheap to drop here and expensive to keep: every one costs a live probe per
# scrape cycle until the dead-slug TTL evicts it.
RESERVED_SEGMENTS = frozenset({
    "embed", "jobs", "job", "error", "404", "500", "api", "static", "assets",
    "favicon.ico", "robots.txt", "sitemap.xml", "images", "img", "css", "js",
    "search", "login", "signup", "about", "privacy", "terms", "blog", "www",
    "index.html", "apply", "applications", "boards", "confirmation",
})

# Workday tenant hosts are wd1..wd105 plus impl/impl-wd* staging hosts.  The
# staging ones serve nothing public, so they are dropped rather than harvested
# into a list that would probe them forever.
_WORKDAY_HOST_RE = re.compile(r"^wd\d+$")


def _looks_like_company(slug: str) -> bool:
    """Structural plausibility check for a harvested slug.

    Not a quality filter -- the live probe is the arbiter of whether a board
    exists.  This only rejects shapes that cannot be a company identifier, so
    the output file does not fill with tracking hashes.  The Greenhouse index
    alone yields entries like ``1945bce8d3924ece9421ba8630f57b0c`` and
    ``5364856uhdfnvbkldfnbhrpkdfgbdvtyhro``.
    """
    if not slug or len(slug) < 2 or len(slug) > 100:
        return False
    if slug in RESERVED_SEGMENTS:
        return False
    # "&" and "+" belong here: harrison&star is a live Greenhouse board and
    # harrisonstar is not, so the character is part of the identifier rather
    # than something to normalise away. They are safe because the extractors
    # take a single path segment ([^/?#]+), so neither can be query-string
    # spill. Rare -- one in ~700 segments -- but a silent loss of real
    # companies.
    if not re.fullmatch(r"[a-z0-9][a-z0-9._&+-]*", slug):
        return False
    # A bare 32/40-char hex string is a session or tracking id, never a company.
    if len(slug) >= 32 and re.fullmatch(r"[0-9a-f]+", slug):
        return False
    # Long unseparated strings that are almost all consonants are machine
    # generated. A presence test is not enough -- the real Greenhouse capture
    # "5364856uhdfnvbkldfnbhrpkdfgbdvtyhro" has vowels, just 7% of them, while a
    # genuine long slug like "advocateslawcareers" runs about 37%.
    if len(slug) >= 20 and "-" not in slug and "_" not in slug and "." not in slug:
        letters = [c for c in slug if c.isalpha()]
        if letters:
            vowels = sum(1 for c in letters if c in "aeiou")
            if vowels / len(letters) < 0.15:
                return False
    return True


@dataclass(frozen=True)
class Platform:
    """One ATS platform's index queries and slug extraction rule."""

    name: str
    queries: tuple[tuple[str, str], ...]  # (url, matchType) pairs
    extract: Callable[[str], str | None]


def _path_segment_extractor(host_suffix: str) -> Callable[[str], str | None]:
    """Slug is the first path segment: ``https://<host>/<slug>/...``."""

    pattern = re.compile(
        r"^https?://([^/]*" + re.escape(host_suffix) + r")/([^/?#]+)",
        re.IGNORECASE,
    )

    def extract(url: str) -> str | None:
        m = pattern.match(url)
        if not m:
            return None
        slug = urllib.parse.unquote(m.group(2)).strip().lower()
        return slug if _looks_like_company(slug) else None

    return extract


def _subdomain_extractor(
    domain: str, strip_prefix: str | None = None
) -> Callable[[str], str | None]:
    """Slug is the leftmost subdomain label: ``https://<slug>.<domain>/...``."""

    pattern = re.compile(
        r"^https?://([a-z0-9][a-z0-9-]*)\." + re.escape(domain) + r"(?:[:/]|$)",
        re.IGNORECASE,
    )

    def extract(url: str) -> str | None:
        m = pattern.match(url)
        if not m:
            return None
        slug = m.group(1).strip().lower()
        if slug in ("www", "api", "static", "cdn", "help", "support", "app"):
            return None
        if strip_prefix and slug.startswith(strip_prefix):
            # ats_service builds the host as f"careers-{slug}.icims.com", so the
            # canonical stored form is the label *without* that prefix.  Upstream
            # stores both forms; the prefixed 3,255 are 100% dead precisely
            # because they round-trip to careers-careers-<x>.icims.com.
            slug = slug[len(strip_prefix):]
        return slug if _looks_like_company(slug) else None

    return extract


_WORKDAY_RE = re.compile(
    r"^https?://([a-z0-9][a-z0-9-]*)\.([a-z0-9-]+)\.myworkdayjobs\.com/(.+)$",
    re.IGNORECASE,
)

# A locale sits in front of the site segment often enough that ignoring it is
# not an option: ``en-US`` is the single most common first path segment on the
# domain (3,279 of one page's captures), and bare ``en``/``es`` occur too.
# Treating either as the site name mints entries like ``3m|wd1|en`` -- upstream
# has none, because a locale is not a board.
_WORKDAY_LOCALE_RE = re.compile(r"^[a-z]{2}(?:[-_][a-z]{2})?$", re.IGNORECASE)

# The workday *site* position is not the same namespace as a path slug: ``jobs``
# is a legitimate site name for 33 upstream tenants, so the general
# RESERVED_SEGMENTS set would throw away real boards here.  Only reject the
# things that are never a site.
# Segments that sit where a site name would but never name a board. "wday" is
# the API path prefix (/wday/cxs/...), the rest are static-asset and CDN roots.
# These surfaced once a deeper collapse depth exposed more URL variety per
# tenant: a 200-slug sample of newly harvested Workday entries was only 23%
# live, and the misses were almost entirely these.
_WORKDAY_SITE_REJECT = frozenset({
    "robots.txt", "sitemap.xml", "favicon.ico", "index.html",
    "wday", "assets", "cdn-cgi", "static", "refreshfacet", "images", "img",
    "css", "js", "fonts", "media", "api", "wday-assets",
})

# Workday's other public domain, with the host and tenant the other way round:
# myworkdayjobs.com is <tenant>.<wdN>.myworkdayjobs.com, while myworkdaysite.com
# is <wdN>.myworkdaysite.com/[locale/]recruiting/<tenant>/<site>/...
#
# Worth extracting because the resulting triple works against the
# myworkdayjobs.com API that ats_service already calls -- verified live for
# whitecase|wd1|external and woodcountyhospital|wd503|jobs, neither of which
# appears anywhere in the upstream Workday list. Some myworkdaysite URLs have no
# "recruiting" segment at all (e.g. /de-CH/jobs/job/...) and name no tenant, so
# they are skipped rather than guessed at.
_WORKDAY_SITE_DOMAIN_RE = re.compile(
    r"^https?://([a-z0-9][a-z0-9-]*)\.myworkdaysite\.com/(.+)$", re.IGNORECASE,
)


def _workday_triple(tenant: str, host: str, site: str) -> str | None:
    """Validate and format the ``tenant|host|site`` triple."""
    tenant, host, site = tenant.lower(), host.lower(), site.lower()
    if not _WORKDAY_HOST_RE.fullmatch(host):
        # impl-wd103, dr-wd108 and friends are staging tenants; they never serve
        # a real board.
        return None
    if not _looks_like_company(tenant):
        return None
    if not site or site in _WORKDAY_SITE_REJECT:
        return None
    # No dot: a site segment never has one, while ads.txt, app-ads.txt and
    # shared-vendors.min.js all do. Verified against upstream -- zero of its
    # 12,884 Workday entries carry a dot in the site position -- so this cannot
    # reject a shape the bot is known to scrape.
    if len(site) > 100 or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", site):
        return None
    return f"{tenant}|{host}|{site}"


def _extract_workday_site(url: str) -> str | None:
    """``<wdN>.myworkdaysite.com/[locale/]recruiting/<tenant>/<site>/...``"""
    m = _WORKDAY_SITE_DOMAIN_RE.match(url)
    if not m:
        return None
    host = m.group(1)
    segments = [urllib.parse.unquote(x).strip()
                for x in m.group(2).split("?")[0].split("#")[0].split("/") if x]
    # Skip any leading locale, then require the literal "recruiting" marker --
    # without it the URL names no tenant and there is nothing to harvest.
    while segments and _WORKDAY_LOCALE_RE.fullmatch(segments[0]):
        segments.pop(0)
    if len(segments) < 3 or segments[0].lower() != "recruiting":
        return None
    return _workday_triple(segments[1], host, segments[2])


def _extract_workday(url: str) -> str | None:
    """Workday needs three parts, stored as ``tenant|host|site``.

    That storage shape is not ours -- it is upstream's, and ats_service:918
    rebuilds ``https://{company}.{wd_num}.myworkdayjobs.com`` from it, so the
    harvester has to emit exactly the same triple.
    """
    m = _WORKDAY_RE.match(url)
    if not m:
        return _extract_workday_site(url)
    tenant, host = m.group(1), m.group(2)

    # Walk past any leading locale segments to the first real one.
    site = None
    for raw in m.group(3).split("?")[0].split("#")[0].split("/"):
        segment = urllib.parse.unquote(raw).strip()
        if not segment:
            continue
        if _WORKDAY_LOCALE_RE.fullmatch(segment):
            continue
        site = segment
        break
    if not site:
        return None
    return _workday_triple(tenant, host, site)


PLATFORMS: tuple[Platform, ...] = (
    Platform(
        "greenhouse",
        # Greenhouse served boards on boards.greenhouse.io for years and moved to
        # job-boards.greenhouse.io; old crawls carry the former, new ones the
        # latter, and neither host alone covers the company set.
        (("boards.greenhouse.io/*", "prefix"),
         ("job-boards.greenhouse.io/*", "prefix"),
         # The EU board host carries companies the US hosts do not (abbyy and
         # others); it was in the Wayback queries but missing here.
         ("job-boards.eu.greenhouse.io/*", "prefix")),
        _path_segment_extractor("greenhouse.io"),
    ),
    Platform(
        "lever",
        # jobs.eu.lever.co is a separate region with its own companies, and it
        # matters disproportionately: Lever blocks CCBot on jobs.lever.co, so
        # the EU host is the only Lever board host Common Crawl still carries.
        (("jobs.lever.co/*", "prefix"),
         ("jobs.eu.lever.co/*", "prefix")),
        _path_segment_extractor("lever.co"),
    ),
    Platform(
        "ashby",
        (("jobs.ashbyhq.com/*", "prefix"),),
        _path_segment_extractor("ashbyhq.com"),
    ),
    Platform(
        # A host query returns zero pages here: every tenant is its own
        # subdomain, so the query has to match the registered domain.
        "workday",
        (("myworkdayjobs.com", "domain"),
         ("myworkdaysite.com", "domain")),
        _extract_workday,
    ),
    Platform(
        "icims",
        (("icims.com", "domain"),),
        _subdomain_extractor("icims.com", strip_prefix="careers-"),
    ),
    Platform(
        "bamboohr",
        (("bamboohr.com", "domain"),),
        _subdomain_extractor("bamboohr.com"),
    ),
)

PLATFORM_BY_NAME = {p.name: p for p in PLATFORMS}


class HarvestError(RuntimeError):
    """A CDX request failed in a way retrying will not fix."""


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read()


Fetcher = Callable[[str], bytes]


def cdx_request(
    url: str,
    *,
    fetch: Fetcher | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes | None:
    """GET a CDX URL, retrying transient failures.

    Returns ``None`` for "the index has nothing here", which the API signals as
    a 404 carrying a JSON ``message`` body.  A 404 is therefore not necessarily
    an error -- treating it as one would abort a sweep the first time a platform
    happened to be absent from an older crawl.
    """
    # Resolved here, not as a default argument: a default binds _http_get at
    # import time, which silently defeats monkeypatching it -- the hermetic test
    # suite would reach the live index and nobody would notice until CI ran
    # offline.
    if fetch is None:
        fetch = _http_get
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fetch(url)
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:
                pass
            if exc.code == 404:
                if b"No Captures found" in body:
                    return None
                raise HarvestError(f"index not found: {url}") from exc
            # 503 is what the index returns when it is shedding load, and it is
            # common enough during a large sweep that giving up on the first one
            # would make full runs unreliable.
            if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                last_exc = exc
                sleep(2.0 * (2 ** attempt))
                continue
            raise HarvestError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException) as exc:
            # http.client.IncompleteRead is the one that actually bites: the
            # index serves these pages chunked, and a large one (the Greenhouse
            # pages run ~13k records) is truncated mid-transfer often enough to
            # hit it on a normal run. It is an HTTPException, not an OSError, so
            # without it listed here the whole harvest dies on a partial read
            # that retrying would have fixed. Never return the partial body:
            # iter_urls would parse it happily and we would publish a page's
            # worth of companies as if it were complete.
            if attempt < MAX_RETRIES - 1:
                last_exc = exc
                sleep(2.0 * (2 ** attempt))
                continue
            raise HarvestError(f"request failed for {url}: {exc}") from exc
    raise HarvestError(f"exhausted retries for {url}: {last_exc}")


def build_query_url(
    crawl: str, url: str, match_type: str, *, page: int | None = None,
    num_pages: bool = False,
) -> str:
    params = {
        "url": url,
        "output": "json",
        "fl": "url",
        # Redirects and error captures still name a real company in the path, but
        # a 200 is the strongest evidence the board existed at crawl time, and
        # filtering server-side cuts the payload we transfer by roughly half.
        "filter": "status:200",
    }
    if match_type != "prefix":
        params["matchType"] = match_type
    if num_pages:
        params["showNumPages"] = "true"
        params.pop("filter", None)
        params.pop("fl", None)
    elif page is not None:
        params["page"] = str(page)
    return INDEX_URL_TEMPLATE.format(crawl=crawl) + "?" + urllib.parse.urlencode(params)


def page_count(
    crawl: str, url: str, match_type: str, *, fetch: Fetcher | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    raw = cdx_request(
        build_query_url(crawl, url, match_type, num_pages=True), fetch=fetch, sleep=sleep
    )
    if raw is None:
        return 0
    try:
        return int(json.loads(raw).get("pages", 0))
    except (ValueError, AttributeError, TypeError) as exc:
        raise HarvestError(f"unparseable page count for {url}: {exc}") from exc


def iter_urls(raw: bytes) -> Iterator[str]:
    """Yield the ``url`` field of each JSON line, skipping unparseable ones.

    The index streams one JSON object per line and a single malformed line
    (truncated transfer, stray byte) must not discard the other 13,000.
    """
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        value = record.get("url") if isinstance(record, dict) else None
        if isinstance(value, str):
            yield value


@dataclass
class PlatformResult:
    platform: str
    slugs: set[str] = field(default_factory=set)
    pages_fetched: int = 0
    records_seen: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.slugs) or not self.errors


def harvest_platform(
    platform: Platform,
    crawl: str,
    *,
    fetch: Fetcher | None = None,
    sleep: Callable[[float], None] = time.sleep,
    delay: float = REQUEST_DELAY_SECONDS,
    max_pages: int | None = None,
    log: Callable[[str], None] = print,
) -> PlatformResult:
    """Sweep every page of every query for one platform in one crawl."""
    result = PlatformResult(platform.name)

    for query_url, match_type in platform.queries:
        blind = False
        try:
            pages = page_count(crawl, query_url, match_type, fetch=fetch, sleep=sleep)
        except HarvestError as exc:
            # Losing the count must not cost us the platform. showNumPages on a
            # big domain is the most expensive query we make -- bamboohr.com
            # spans the whole marketing site -- and it times out on a loaded
            # index while the paged fetches for the same query still succeed.
            # Walk pages until one comes back empty instead of giving up.
            result.errors.append(f"{query_url}: page count failed, walking blind: {exc}")
            log(f"[harvest]   {query_url}: page count failed ({exc}); walking blind")
            pages, blind = BLIND_PAGE_LIMIT, True

        if pages == 0:
            log(f"[harvest]   {query_url}: no captures in {crawl}")
            continue

        limit = pages if max_pages is None else min(pages, max_pages)
        log(f"[harvest]   {query_url}: "
            f"{'unknown' if blind else pages} page(s), fetching up to {limit}")

        for page in range(limit):
            sleep(delay)
            try:
                raw = cdx_request(
                    build_query_url(crawl, query_url, match_type, page=page),
                    fetch=fetch, sleep=sleep,
                )
            except HarvestError as exc:
                result.errors.append(f"{query_url} page {page}: {exc}")
                log(f"[harvest]   {query_url} page {page}: {exc}")
                continue
            if raw is None:
                # Walking blind, an empty page is the end of the result set --
                # that is the only stop signal available without a count.
                if blind:
                    log(f"[harvest]   {query_url}: page {page} empty, stopping")
                    break
                continue
            result.pages_fetched += 1
            for url in iter_urls(raw):
                result.records_seen += 1
                slug = platform.extract(url)
                if slug:
                    result.slugs.add(slug)

    return result


def latest_crawls(
    count: int, *, fetch: Fetcher | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """The N most recent crawl ids, newest first.

    collinfo.json is ordered newest-first, but relying on that order alone would
    silently harvest the wrong crawls if it ever changed, so the ids are sorted
    on their own year/week.
    """
    raw = cdx_request(COLLINFO_URL, fetch=fetch, sleep=sleep)
    if raw is None:
        raise HarvestError("collinfo.json returned no data")
    try:
        entries = json.loads(raw)
    except ValueError as exc:
        raise HarvestError(f"unparseable collinfo.json: {exc}") from exc
    ids = [
        e["id"] for e in entries
        if isinstance(e, dict) and isinstance(e.get("id"), str)
        and re.fullmatch(r"CC-MAIN-\d{4}-\d{2}", e["id"])
    ]
    if not ids:
        raise HarvestError("collinfo.json listed no usable crawl ids")
    ids.sort(key=lambda cid: tuple(int(x) for x in cid.split("-")[2:]), reverse=True)
    return ids[:count]


# ---------------------------------------------------------------------------
# Wayback Machine index
# ---------------------------------------------------------------------------
#
# A second, independent archive of the same web. It matters for two reasons.
#
# Coverage: Lever blocks Common Crawl outright -- jobs.lever.co/robots.txt
# carries "User-agent: CCBot" / "Disallow: /" -- so Lever vanished from CC after
# CC-MAIN-2025-38 and is not coming back. Wayback is still permitted there
# ("User-agent: *" / "Allow: /") and is the only live source for Lever.
#
# Volume: a Wayback sweep of Greenhouse alone returned 11,591 slugs against
# 5,446 from six Common Crawl snapshots.

WAYBACK_URL = "http://web.archive.org/cdx/search/cdx"

WAYBACK_QUERIES: dict[str, tuple[str, ...]] = {
    "greenhouse": ("boards.greenhouse.io/*", "job-boards.greenhouse.io/*",
                   "job-boards.eu.greenhouse.io/*"),
    "lever": ("jobs.lever.co/*", "jobs.eu.lever.co/*"),
    "ashby": ("jobs.ashbyhq.com/*",),
    "workday": ("*.myworkdayjobs.com/*", "*.myworkdaysite.com/*"),
    "icims": ("*.icims.com/*",),
    "bamboohr": ("*.bamboohr.com/*",),
}

# Rows per request. Wayback truncates large responses reliably, so this stays
# small enough that most pages arrive whole.
WAYBACK_PAGE_ROWS = 2000
WAYBACK_MAX_PAGES = 60
WAYBACK_SINCE = "2022"
# Give up on a query after this many consecutive pages that add nothing new.
WAYBACK_STALL_LIMIT = 3

# Wall-clock ceiling for a single query+depth sweep. Wayback can slow to the
# point where one query would otherwise consume the entire CI job: at a 180s
# request timeout with retries, sixty pages is theoretically hours. The sweep is
# cumulative across runs, so cutting a slow query short costs nothing but a
# little progress this week.
WAYBACK_QUERY_BUDGET_SECONDS = 420.0


# Explicit collapse depths for queries whose identifying part is not where the
# derivation below assumes. The heuristic reads the slug from the path, but a
# wildcard-subdomain query puts it in the *host* portion of the sort key, ahead
# of the ")". Workday is the extreme case: its key is
# `com,myworkdayjobs,<wdN-host>,<tenant>)/...`, so the derived depth of 23 cuts
# inside the host and collapses every tenant beneath it. Measured on a live
# sweep: depth 23 returned 7 rows and zero companies, depth 40 returned 1,500
# rows and 643 companies.
WAYBACK_EXPLICIT_DEPTHS: dict[str, tuple[int, ...]] = {
    "*.myworkdayjobs.com/*": (32, 40),
    "*.myworkdaysite.com/*": (30, 38),
}


def wayback_collapse_depths(query: str) -> tuple[int, ...]:
    """Collapse depths to sweep for a query, shallow first.

    ``collapse=urlkey:N`` groups captures sharing the first N characters of the
    sort key (``co,lever,jobs)/acme/...``). Without it a sweep drowns: 200,000
    rows of jobs.lever.co yielded 853 companies, because consecutive rows are
    all the same handful of employers. With it, 3,000 rows yielded 2,055 -- two
    orders of magnitude better.

    The key prefix is the reversed host plus ")/", so the slug starts at
    len(host)+2. Sweeping a shallow and a deeper offset and unioning gets both
    breadth (aggressive collapse reaches more companies per row) and the slugs
    that a shallow collapse merges because they share leading characters.

    This derivation only holds when the slug is in the path. Queries whose slug
    lives in the host are listed in WAYBACK_EXPLICIT_DEPTHS instead.
    """
    explicit = WAYBACK_EXPLICIT_DEPTHS.get(query)
    if explicit:
        return explicit
    host = query.split("/")[0].lstrip("*.")
    base = len(host) + 2
    return base + 4, base + 9


def _wayback_fetch_lines(
    url: str, *, fetch: Fetcher | None = None,
    sleep: Callable[[float], None] | None = None, tries: int = 3,
) -> tuple[list[str], bool]:
    """Fetch CDX text output, salvaging complete lines from a truncated body.

    Returns (lines, complete). Wayback truncates often enough that discarding
    partial responses loses most of a sweep. Text output is one record per line,
    so every line before the cut is still usable and only the half-written last
    one is dropped.

    This is deliberately the opposite of the Common Crawl path, which rejects
    partial bodies. There a short read is indistinguishable from a complete page
    and would silently under-harvest; here the truncation is explicit, so
    salvaging is safe and discarding is what loses data.
    """
    # Both resolved here rather than as default arguments: a default binds the
    # module-level function at import, which defeats patching and lets the
    # hermetic suite reach the network and really sleep.
    if fetch is None:
        fetch = _http_get
    if sleep is None:
        sleep = time.sleep
    for attempt in range(tries):
        try:
            return fetch(url).decode("utf-8", "replace").splitlines(), True
        except http.client.IncompleteRead as exc:
            lines = exc.partial.decode("utf-8", "replace").splitlines()
            if len(lines) > 1:
                return lines[:-1], False
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException):
            if attempt < tries - 1:
                sleep(4.0 * (attempt + 1))
    return [], False


def harvest_wayback_query(
    query: str, extract: Callable[[str], str | None], depth: int, *,
    fetch: Fetcher | None = None, sleep: Callable[[float], None] | None = None,
    delay: float = 1.5, max_pages: int = WAYBACK_MAX_PAGES,
    rows: int = WAYBACK_PAGE_ROWS, since: str = WAYBACK_SINCE,
    budget_seconds: float = WAYBACK_QUERY_BUDGET_SECONDS,
    now: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> tuple[set[str], int]:
    """Page one Wayback query at one collapse depth."""
    if sleep is None:
        sleep = time.sleep
    started = now()
    slugs: set[str] = set()
    resume: str | None = None
    seen_rows = stalls = 0

    for page in range(max_pages):
        if page and now() - started > budget_seconds:
            log(f"[harvest]   wayback {query}@{depth}: budget spent after "
                f"{page} page(s), stopping")
            break
        params = {
            "url": query, "fl": "original,urlkey", "collapse": f"urlkey:{depth}",
            "limit": str(rows), "from": since, "showResumeKey": "true",
        }
        if resume:
            params["resumeKey"] = resume
        lines, complete = _wayback_fetch_lines(
            WAYBACK_URL + "?" + urllib.parse.urlencode(params),
            fetch=fetch, sleep=sleep,
        )
        if not lines:
            break

        next_resume: str | None = None
        urls: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) == 1 and not parts[0].startswith("http"):
                next_resume = parts[0]      # trailing resume key
            else:
                urls.append(parts[0])

        seen_rows += len(urls)
        before = len(slugs)
        for url in urls:
            slug = extract(url)
            if slug:
                slugs.add(slug)
        gained = len(slugs) - before

        # A truncated page loses its trailing resume key. Continue from the last
        # urlkey actually seen rather than abandoning the rest of the sweep.
        if next_resume is None and not complete and urls:
            tail = [ln for ln in lines if ln.strip()][-1].split(" ")
            if len(tail) > 1:
                next_resume = tail[-1]

        if not next_resume:
            break
        stalls = stalls + 1 if gained == 0 else 0
        if stalls >= WAYBACK_STALL_LIMIT:
            log(f"[harvest]   wayback {query}@{depth}: "
                f"{stalls} pages with nothing new, stopping")
            break
        resume = next_resume
        sleep(delay)

    return slugs, seen_rows


def harvest_platform_wayback(
    platform: Platform, *, fetch: Fetcher | None = None,
    sleep: Callable[[float], None] | None = None, delay: float = 1.5,
    max_pages: int = WAYBACK_MAX_PAGES, log: Callable[[str], None] = print,
) -> PlatformResult:
    """Sweep every Wayback query and collapse depth for one platform."""
    result = PlatformResult(platform.name)
    for query in WAYBACK_QUERIES.get(platform.name, ()):
        for depth in wayback_collapse_depths(query):
            try:
                slugs, rows = harvest_wayback_query(
                    query, platform.extract, depth, fetch=fetch, sleep=sleep,
                    delay=delay, max_pages=max_pages, log=log,
                )
            except Exception as exc:        # noqa: BLE001 - one query, not the sweep
                result.errors.append(f"wayback {query}@{depth}: {exc}")
                log(f"[harvest]   wayback {query}@{depth} failed: {exc}")
                continue
            result.slugs |= slugs
            result.records_seen += rows
            result.pages_fetched += 1
            log(f"[harvest]   wayback {query}@{depth}: {len(slugs)} slugs "
                f"from {rows} rows (running total {len(result.slugs)})")
    return result


def load_existing(path: Path) -> set[str]:
    """Read a previously written harvest file; absent or corrupt reads as empty."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return set()
    if not isinstance(data, list):
        return set()
    return {str(s).strip() for s in data if str(s).strip()}


def write_slugs(path: Path, slugs: Iterable[str]) -> None:
    """Write the slug list atomically, sorted.

    Sorted because the file is committed: unsorted output would produce a diff
    of the whole file on every run and make "what did this harvest add"
    unreadable.  Atomic because a truncated write is read back as empty by
    load_existing, which would silently discard every slug ever harvested.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(sorted(slugs), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--crawl", action="append", dest="crawls_explicit",
                        help="crawl id to harvest (repeatable), e.g. CC-MAIN-2026-34")
    parser.add_argument("--crawls", type=int, default=1,
                        help="harvest the N most recent crawls (default 1)")
    parser.add_argument("--platform", action="append", dest="platforms",
                        choices=sorted(PLATFORM_BY_NAME), help="limit to a platform")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"output directory (default {DEFAULT_OUT_DIR})")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="cap pages per query (for smoke tests)")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS,
                        help="seconds between requests")
    parser.add_argument("--dry-run", action="store_true",
                        help="harvest and report, write nothing")
    parser.add_argument("--json", action="store_true", help="summary as JSON to stdout")
    parser.add_argument("--index", action="append", dest="indexes",
                        choices=["commoncrawl", "wayback"],
                        help="which archive(s) to sweep (repeatable; "
                             "default: both)")
    args = parser.parse_args(argv)

    indexes = args.indexes or ["commoncrawl", "wayback"]

    if args.crawls < 1:
        print("--crawls must be at least 1", file=sys.stderr)
        return 2

    log: Callable[[str], None] = (
        (lambda msg: print(msg, file=sys.stderr)) if args.json else print
    )

    crawls: list[str] = []
    if "commoncrawl" in indexes:
        try:
            crawls = args.crawls_explicit or latest_crawls(args.crawls)
        except HarvestError as exc:
            # Losing Common Crawl must not cancel the Wayback sweep: they are
            # independent archives, and Wayback is the only source for Lever.
            print(f"[harvest] could not determine crawls: {exc}", file=sys.stderr)
            if "wayback" not in indexes:
                return 1
            indexes = [i for i in indexes if i != "commoncrawl"]

    platforms = [PLATFORM_BY_NAME[n] for n in args.platforms] if args.platforms \
        else list(PLATFORMS)

    log(f"[harvest] indexes: {', '.join(indexes)}")
    log(f"[harvest] crawls: {', '.join(crawls) or '(none)'}")
    log(f"[harvest] platforms: {', '.join(p.name for p in platforms)}")

    t_start = time.monotonic()
    summary: dict[str, dict[str, object]] = {}
    any_success = False

    for platform in platforms:
        existing = load_existing(args.out / f"{platform.name}.json")
        found: set[str] = set()
        errors: list[str] = []
        pages = records = 0

        if "commoncrawl" in indexes:
            for crawl in crawls:
                log(f"[harvest] {platform.name} @ {crawl}")
                result = harvest_platform(
                    platform, crawl, delay=args.delay, max_pages=args.max_pages,
                    log=log,
                )
                found |= result.slugs
                errors.extend(result.errors)
                pages += result.pages_fetched
                records += result.records_seen

        if "wayback" in indexes:
            log(f"[harvest] {platform.name} @ wayback")
            result = harvest_platform_wayback(
                platform, delay=max(args.delay, 1.0),
                max_pages=args.max_pages or WAYBACK_MAX_PAGES, log=log,
            )
            found |= result.slugs
            errors.extend(result.errors)
            pages += result.pages_fetched
            records += result.records_seen

        new = found - existing
        merged = existing | found
        summary[platform.name] = {
            "harvested": len(found), "new": len(new), "total": len(merged),
            "indexes": indexes,
            "pages": pages, "records": records, "errors": errors,
        }
        log(f"[harvest] {platform.name}: {len(found)} harvested, "
            f"{len(new)} new, {len(merged)} total ({records} records, {pages} pages)")

        if found:
            any_success = True
        if found and not args.dry_run:
            write_slugs(args.out / f"{platform.name}.json", merged)

    elapsed = time.monotonic() - t_start
    log(f"[harvest] done in {elapsed:.1f}s")

    if args.json:
        print(json.dumps({
            "indexes": indexes, "crawls": crawls, "elapsed_seconds": round(elapsed, 1),
            "dry_run": args.dry_run, "platforms": summary,
        }, indent=2))

    if not any_success:
        print("[harvest] every platform yielded nothing - treating as failure",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
