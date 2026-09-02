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
import contextlib
import gzip
import http.client
import os
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
    # Well-known root files that land in the slug position on every host.
    "ads.txt", "app-ads.txt", "security.txt", "humans.txt", "manifest.json",
})

# Workday tenant hosts are wd1..wd105 plus impl/impl-wd* staging hosts.  The
# staging ones serve nothing public, so they are dropped rather than harvested
# into a list that would probe them forever.
_WORKDAY_HOST_RE = re.compile(r"^wd\d+$")

# Infrastructure subdomains that are never a customer tenant. Numbered variants
# are included because hosts like www4.icims.com exist and were otherwise
# harvested as companies.
_INFRA_SUBDOMAIN_RE = re.compile(
    r"^(?:www|api|static|cdn|help|support|app|mail|smtp|ns|mx|ftp|dev|test"
    r"|staging|stage|demo|admin|assets|media|img|images|cms|vpn|portal)\d*$"
)


def _normalise_slug(slug: str) -> str:
    """Trim trailing punctuation a URL picked up from surrounding text.

    Archived URLs are frequently captured with a sentence's full stop or a
    stray dash glued on, giving "camber." and "inherent." -- both of which are
    live Ashby boards under their bare name, and "dnb." likewise on Lever.
    Rejecting them as malformed loses real companies; trimming recovers them.
    Only trailing characters are touched, so affinity.co is untouched.
    """
    return slug.rstrip(".-_")


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
    # A dotted slug is usually a company using its own domain -- affinity.co,
    # akasa.com, alignment.org and adept.ai are all live boards -- so dots
    # cannot simply be rejected. What separates them from version strings is
    # what follows the final dot: a TLD is two or more letters, while "2.5",
    # "2021b-49.2", "2022ae-13.3" and "u.s.a" all end in digits or a single
    # letter. (Checking for a two-letter run anywhere is not enough: the "ae"
    # in 2022ae-13.3 is itself a valid TLD.)
    if "." in slug and not re.fullmatch(r"[a-z]{2,}", slug.rsplit(".", 1)[-1]):
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
        slug = _normalise_slug(urllib.parse.unquote(m.group(2)).strip().lower())
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
        slug = _normalise_slug(m.group(1).strip().lower())
        # Numbered variants matter as much as the bare names: www4.icims.com is
        # real infrastructure and was being harvested as a company called
        # "www4".
        if _INFRA_SUBDOMAIN_RE.fullmatch(slug):
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
    # Path components that land in the site position, each probed and found
    # entirely dead across every tenant carrying them:
    #   job      62 entries, 0/30 live   (the /job/<id>/<title> component;
    #                                     upstream has 10 and they are dead too)
    #   details  25 entries, 0/25 live
    #   login    22 entries, 0/22 live
    # For contrast "external" runs 69% live and "search" 83.9%, so this list
    # stays evidence-led rather than intuition-led: "search" looks just as much
    # like a path component and is a real site for 26 tenants.
    "job", "details", "login",
})

# A truncated locale, e.g. "en-" from a URL cut mid-segment. 24 such entries
# were harvested and none resolved.
_WORKDAY_PARTIAL_LOCALE_RE = re.compile(r"^[a-z]{2}-$")

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
    if _WORKDAY_HOST_RE.fullmatch(tenant):
        # A host label sitting in the tenant slot, which happens when the URL
        # has no tenant subdomain at all: wd1.myworkdayjobs.com/... reads as
        # tenant "wd1". The result cannot resolve -- there is no
        # wd1.wd1.myworkdayjobs.com -- and it is not a hypothetical: 6,055 of
        # upstream's 12,884 Workday entries (47%) have this shape, and a
        # 50-slug sample of that cohort was entirely dead.
        return None
    if not site or site in _WORKDAY_SITE_REJECT:
        return None
    if _WORKDAY_PARTIAL_LOCALE_RE.fullmatch(site):
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

    # Skip at most ONE leading locale, then take the next segment.
    #
    # Consuming every locale-shaped segment loses real boards, because site
    # names look like locales too: abinbev|wd1|py and abinbev|wd1|hn-es are
    # both live, and both would be skipped on the way to whatever followed
    # (typically "job", which is not a board at all). A URL carries at most one
    # locale prefix, so one is all that should ever be skipped.
    segments = [
        seg for seg in (
            urllib.parse.unquote(raw).strip()
            for raw in m.group(3).split("?")[0].split("#")[0].split("/")
        ) if seg
    ]
    if segments and _WORKDAY_LOCALE_RE.fullmatch(segments[0]) and len(segments) > 1:
        segments = segments[1:]
    if not segments:
        return None
    return _workday_triple(tenant, host, segments[0])


# Paylocity identifies a company by GUID rather than a name slug, and the board
# URL carries it: recruiting.paylocity.com/recruiting/jobs/All/<guid>/<Name>.
# Upstream ships these as {guid, name, jobs} objects; we store the bare guid so
# the harvest file stays a flat list like every other platform.
_PAYLOCITY_RE = re.compile(
    r"^https?://recruiting\.paylocity\.com/recruiting/jobs/all/"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)


def _extract_paylocity(url: str) -> str | None:
    m = _PAYLOCITY_RE.match(url)
    return m.group(1).lower() if m else None


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
    Platform(
        "paylocity",
        (("recruiting.paylocity.com/*", "prefix"),),
        _extract_paylocity,
    ),
    # Workable and Breezy are new here rather than replacements. The seven
    # platforms above are saturated: re-sweeping the newest crawl for each
    # returned exactly zero companies not already held, so no amount of
    # additional crawling produces another one. New companies have to come from
    # hosts nobody has swept yet.
    #
    # Both were measured against CC-MAIN-2026-34 before being added: workable
    # yields 3,158 companies and breezy 2,063, and a sample of 25 each probed
    # 100% and 96% live. That is far above the harvested pools for the older
    # platforms, which run 36-52% because they are thick with expired archival
    # boards -- these hosts are recent enough that most of what they carry is
    # still hiring.
    Platform(
        "workable",
        (("apply.workable.com/*", "prefix"),),
        _path_segment_extractor("workable.com"),
    ),
    Platform(
        "breezy",
        (("breezy.hr", "domain"),),
        _subdomain_extractor("breezy.hr"),
    ),
    # A second wave, all measured against CC-MAIN-2026-34 and all probing 100%
    # live on a 20-slug sample. Recency is why: these hosts are young enough
    # that the index carries mostly current boards, where the older platforms'
    # harvested pools are 36-52% live because they reach back through years of
    # expired ones.
    Platform(
        "smartrecruiters",
        (("jobs.smartrecruiters.com/*", "prefix"),),
        _path_segment_extractor("smartrecruiters.com"),
    ),
    Platform(
        "rippling",
        (("ats.rippling.com/*", "prefix"),),
        _path_segment_extractor("rippling.com"),
    ),
    Platform(
        "teamtailor",
        (("teamtailor.com", "domain"),),
        _subdomain_extractor("teamtailor.com"),
    ),
    Platform(
        "jazzhr",
        (("applytojob.com", "domain"),),
        _subdomain_extractor("applytojob.com"),
    ),
    Platform(
        "recruitee",
        (("recruitee.com", "domain"),),
        _subdomain_extractor("recruitee.com"),
    ),
    Platform(
        "jobvite",
        (("jobs.jobvite.com/*", "prefix"),),
        _path_segment_extractor("jobvite.com"),
    ),
    Platform(
        "applicantpro",
        (("applicantpro.com", "domain"),),
        _subdomain_extractor("applicantpro.com"),
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
        # status is needed because the filtering moved client-side.
        "fl": "url,status",
        # No server-side status filter. It used to pin this to status:200,
        # which silently dropped every company whose captures were all
        # redirects -- 911 of them in one Greenhouse crawl, 38.6% of which are
        # live boards. Filtering client-side through capture_is_usable costs
        # more transfer and keeps the two index paths in exact agreement.
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
        if not isinstance(record, dict):
            continue
        value = record.get("url")
        # Same predicate the bulk reader uses, so the two paths cannot diverge
        # on what counts as a usable capture.
        if isinstance(value, str) and capture_is_usable(str(record.get("status", ""))):
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


# Cache of the crawl listing, written beside the harvest output so it travels
# with the published branch and is seeded back on the next run. The underscore
# keeps it out of the per-platform globs.
CRAWL_CACHE_NAME = "_crawls.json"


def _parse_collinfo(raw: bytes) -> list[str]:
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
    # collinfo.json is ordered newest-first, but relying on that order alone
    # would silently harvest the wrong crawls if it ever changed, so the ids are
    # sorted on their own year/week.
    ids.sort(key=lambda cid: tuple(int(x) for x in cid.split("-")[2:]), reverse=True)
    return ids


def latest_crawls(
    count: int, *, fetch: Fetcher | None = None,
    sleep: Callable[[float], None] = time.sleep,
    cache_path: Path | None = None,
    discover_years: Iterable[int] | None = None,
) -> list[str]:
    """The N most recent crawl ids, newest first, with two fallbacks.

    collinfo.json is genuinely unreliable -- two requests seconds apart returned
    200 in 0.23s and then timed out. Losing it means losing the entire Common
    Crawl half of a run, and the only symptom is one line on stderr, so a
    flaky minute quietly halves a week's harvest.

    A stale list is a good enough substitute: crawls are published roughly
    monthly and the ids are immutable once minted, so yesterday's list still
    names real crawls. The worst case is missing the newest one for a week.
    """
    try:
        raw = cdx_request(COLLINFO_URL, fetch=fetch, sleep=sleep)
        if raw is None:
            raise HarvestError("collinfo.json returned no data")
        ids = _parse_collinfo(raw)
    except HarvestError:
        cached = _load_crawl_cache(cache_path)
        if cached:
            return cached[:count]
        # Last resort: ask the bulk host which crawls exist. collinfo.json is
        # the only part of a bulk run that needs the query service at all, so
        # falling back here removes the last dependency on it.
        if discover_years:
            discovered = discover_crawls_bulk(discover_years)
            if discovered:
                _save_crawl_cache(cache_path, discovered)
                return discovered[:count]
        raise
    _save_crawl_cache(cache_path, ids)
    return ids[:count]


def _crawl_exists(crawl: str, *, head: Callable[[str], int] | None = None) -> bool:
    if head is None:
        head = _http_status
    try:
        return head(_cluster_url(crawl)) in (200, 206)
    except Exception:  # noqa: BLE001 - a probe failure is "unknown", not "absent"
        return False


def _http_status(url: str) -> int:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-10"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def discover_crawls_bulk(
    years: Iterable[int], *, head: Callable[[str], int] | None = None,
    workers: int = 8,
) -> list[str]:
    """Find which crawls exist by asking the bulk host, not the API.

    collinfo.json is the only thing that ever needed index.commoncrawl.org for
    a bulk run, and it is exactly the piece that has been unavailable. Crawl
    ids are CC-MAIN-<year>-<week>, so probing cluster.idx for each candidate
    settles the question directly: a real crawl answers 206 and a nonexistent
    one 404.

    Probing every week of a year is 52 cheap range requests and finds the
    roughly monthly schedule without having to know it.
    """
    candidates = [f"CC-MAIN-{year}-{week:02d}"
                  for year in years for week in range(1, 53)]
    found: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for crawl, exists in zip(candidates,
                                 pool.map(lambda c: _crawl_exists(c, head=head),
                                          candidates)):
            if exists:
                found.append(crawl)
    found.sort(key=lambda cid: tuple(int(x) for x in cid.split("-")[2:]),
               reverse=True)
    return found


def _load_crawl_cache(cache_path: Path | None) -> list[str]:
    if cache_path is None or not cache_path.exists():
        return []
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [c for c in data
            if isinstance(c, str) and re.fullmatch(r"CC-MAIN-\d{4}-\d{2}", c)]


def _save_crawl_cache(cache_path: Path | None, ids: list[str]) -> None:
    """Merge into the cache rather than replacing it.

    Discovery is usually scoped to a year or two, so a plain overwrite shrinks
    the cache to whatever that run happened to look for -- a 2026-only probe
    would cut 124 known crawls to 8. The cache exists for the case where
    collinfo.json is unreachable, so the loss stays invisible until the run
    that needed it. Crawl ids are CC-MAIN-YYYY-NN, which sorts newest-first in
    reverse, matching the order callers expect.
    """
    if cache_path is None:
        return
    try:
        ids = sorted(set(ids) | set(_load_crawl_cache(cache_path)), reverse=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(ids, indent=1) + "\n", encoding="utf-8")
        tmp.replace(cache_path)
    except OSError:
        # A cache we cannot write is not a reason to fail a harvest.
        pass


# ---------------------------------------------------------------------------
# Common Crawl bulk index (data.commoncrawl.org)
# ---------------------------------------------------------------------------
#
# The same index the CDX API serves, read directly from the published files
# instead of through the query service. This is the access path Common Crawl
# documents for bulk use, and it is better here on every axis that matters:
#
#   Availability. It is a different host. index.commoncrawl.org has been
#   returning 000 for hours at a stretch while data.commoncrawl.org answered
#   every request -- so the API being down no longer means no Common Crawl.
#
#   Cost. Locating the blocks for a host takes a binary search over
#   cluster.idx using HTTP Range requests -- about seven 200KB reads of a
#   103MB file -- and each block is then fetched by byte range. Measured on
#   CC-MAIN-2026-34: 21 blocks located in 2s, and 6 of them yielded 18,000 CDX
#   records and 1,033 distinct companies in 3s total. The API needed minutes
#   per page for the same data.
#
#   Agreement. It returns what the query service returns, which matters now
#   that it is the default path. On CC-MAIN-2026-34, measured against API
#   sweeps of the same crawl: Ashby 2,708 companies from 20,403 records
#   against the API's 2,708 from 20,403, and iCIMS 2,004 from 22,382 against
#   2,005 from 22,721. Both were roughly 11% short before the boundary blocks
#   either side of a range were included -- see bulk_find_blocks.
#
#   Politeness. Range reads of static files do not compete for the shared
#   query service, which is what the rate limit exists to protect.
#
# cluster.idx is sorted by SURT key, so the binary search is exact rather than
# heuristic: every capture for a host is contiguous.

CC_DATA_BASE = "https://data.commoncrawl.org"

# Bytes per probe while binary-searching cluster.idx. Large enough that a read
# always spans a line boundary, small enough that ~7 probes over a 103MB file
# cost little.
CLUSTER_PROBE_BYTES = 200_000


def _surt_prefix(query_url: str, match_type: str) -> str:
    """Translate a CDX query into the SURT prefix cluster.idx is sorted by.

    SURT reverses the host labels: job-boards.greenhouse.io becomes
    ``io,greenhouse,job-boards)/``. A domain match has to reach subdomains too,
    so it stops at the trailing comma -- ``com,myworkdayjobs,`` matches
    ``com,myworkdayjobs,acme)`` and every other tenant, which is exactly the
    set matchType=domain returns.
    """
    host = query_url.split("/")[0].lstrip("*.")
    reversed_host = ",".join(reversed(host.split(".")))
    if match_type == "domain":
        return reversed_host + ","
    return reversed_host + ")/"


def _retrying_range(
    fetch_range: RangeFetcher, url: str, start: int, end: int, *,
    sleep: Callable[[float], None], tries: int = MAX_RETRIES,
) -> bytes:
    """Range-fetch with backoff.

    data.commoncrawl.org throttles sustained reads: a 12-crawl sweep lost 11
    blocks to 503s, all on the platform with the most blocks to read. A block
    is roughly 170 companies, and unlike a paged API there is no later request
    that happens to cover the same ground -- a dropped block is simply a hole.
    """
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return fetch_range(url, start, end)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                last = exc
                sleep(2.0 * (2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException) as exc:
            if attempt < tries - 1:
                last = exc
                sleep(2.0 * (2 ** attempt))
                continue
            raise
    raise HarvestError(f"exhausted range retries for {url}: {last}")


def _http_get_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return response.read()


def _http_head_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return int(response.headers["Content-Length"])


RangeFetcher = Callable[[str, int, int], bytes]
SizeFetcher = Callable[[str], int]


def _cluster_url(crawl: str) -> str:
    return f"{CC_DATA_BASE}/cc-index/collections/{crawl}/indexes/cluster.idx"


def bulk_find_blocks(
    crawl: str, surt_prefix: str, *,
    fetch_range: RangeFetcher | None = None,
    fetch_size: SizeFetcher | None = None,
    probe_bytes: int = CLUSTER_PROBE_BYTES,
) -> list[tuple[str, int, int]]:
    """Locate the cdx shards holding a SURT prefix, as (name, offset, length).

    cluster.idx lines are ``<surt> <timestamp>\\t<cdx file>\\t<offset>\\t<length>``
    sorted by surt, so a binary search over byte offsets finds the region
    without reading the file. Each probe skips its first (probably partial)
    line before comparing.
    """
    if fetch_range is None:
        fetch_range = _http_get_range
    if fetch_size is None:
        fetch_size = _http_head_size

    url = _cluster_url(crawl)
    try:
        total = fetch_size(url)
    except Exception as exc:  # noqa: BLE001 - one crawl, not the sweep
        raise HarvestError(f"cluster.idx unavailable for {crawl}: {exc}") from exc

    low, high = 0, total
    while high - low > probe_bytes:
        mid = (low + high) // 2
        try:
            chunk = fetch_range(url, mid, mid + probe_bytes).decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            raise HarvestError(f"cluster.idx read failed for {crawl}: {exc}") from exc
        newline = chunk.find("\n")
        if newline < 0:
            break
        key = chunk[newline + 1:].split(" ", 1)[0]
        if key < surt_prefix:
            low = mid
        else:
            high = mid

    start = max(0, low - probe_bytes)
    span = min(total, high + probe_bytes * 3) - start
    try:
        window = fetch_range(url, start, start + span).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        raise HarvestError(f"cluster.idx read failed for {crawl}: {exc}") from exc

    blocks: list[tuple[str, int, int]] = []
    previous: tuple[str, int, int] | None = None
    # Skip the first line: a byte-range read almost always starts mid-line.
    for line in window.split("\n")[1:-1]:
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        key = fields[0].split(" ", 1)[0]
        try:
            entry = (fields[1], int(fields[2]), int(fields[3]))
        except ValueError:
            continue
        if key.startswith(surt_prefix):
            # cluster.idx names each block by its *first* key, so a prefix that
            # begins partway through a block leaves that block's key sorting
            # below it. Skipping it drops the start of the range. Taking the
            # preceding block costs one read, and iter_bulk_urls re-checks the
            # prefix so nothing foreign leaks in.
            if not blocks and previous is not None:
                blocks.append(previous)
            blocks.append(entry)
        elif blocks:
            # Sorted file: the first non-matching key after a run of matches
            # ends the region. Keep it too, for the mirror-image reason -- the
            # tail of the range can share that block.
            blocks.append(entry)
            break
        previous = entry
    return blocks


def bulk_fetch_block(crawl: str, name: str, offset: int, length: int, *,
                     fetch_range: RangeFetcher | None = None,
                     sleep: Callable[[float], None] | None = None) -> str:
    """Range-fetch one gzipped cdx shard and return its text."""
    if fetch_range is None:
        fetch_range = _http_get_range
    if sleep is None:
        sleep = time.sleep
    url = f"{CC_DATA_BASE}/cc-index/collections/{crawl}/indexes/{name}"
    raw = _retrying_range(fetch_range, url, offset, offset + length - 1,
                          sleep=sleep)
    try:
        return gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("utf-8", "replace")
    except (OSError, EOFError) as exc:
        raise HarvestError(f"undecodable cdx block {name}@{offset}: {exc}") from exc


def capture_is_usable(status: str) -> bool:
    """Whether a capture's HTTP status means its URL still names a company.

    2xx and 3xx count; 4xx and 5xx do not. A redirect is not evidence the board
    is gone -- Greenhouse answers 302 for tagged and retired job URLs while the
    company is very much still hiring. Measured on CC-MAIN-2026-34 Greenhouse:

        captures seen only with 2xx        3,017 slugs   97.1% live
        captures seen only with 3xx          911 slugs   38.6% live
        captures seen only with 4xx           26 slugs    0.0% live

    Filtering to 200 alone was dropping those 911 in a single crawl -- a third
    again on top of what it kept, at a live rate well above what sweeping older
    crawls yields. conviva is the case that surfaced it: eight captures, all
    302, a live board, and absent from our harvest entirely.

    The size of that is particular to Greenhouse, which redirects utm-tagged
    and retired job URLs heavily. Checked on the same crawl, BambooHR had two
    such slugs and Ashby none, so this rule is a large gain on one platform and
    a harmless no-op on the rest rather than a uniform uplift.

    And 911 is a per-crawl figure, not a cumulative one. Re-sweeping twelve
    crawls with the rule in place added 609 companies in total, 47.6% of them
    live -- greenhouse 342, icims 122, workday 113 -- because a capture that
    only redirected in one crawl usually returned 200 in another, so the
    accumulated harvest already had most of them. Roughly 290 live boards, not
    the ten thousand a naive multiplication suggests. The same saturation
    applies to every extraction fix measured on a single crawl.

    4xx stays out. Nothing in that bucket resolved, which is what the status
    says: the path was wrong.
    """
    return status.startswith("2") or status.startswith("3")


def iter_bulk_urls(text: str, surt_prefix: str) -> Iterator[str]:
    """Yield URLs from cdx shard lines matching the prefix.

    Each line is ``<surt> <timestamp> <json>``. Malformed lines are skipped
    rather than aborting the block, for the same reason iter_urls does it.
    """
    for line in text.split("\n"):
        if not line.startswith(surt_prefix):
            continue
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        try:
            record = json.loads(parts[2])
        except ValueError:
            continue
        url = record.get("url") if isinstance(record, dict) else None
        status = str(record.get("status", "")) if isinstance(record, dict) else ""
        if isinstance(url, str) and capture_is_usable(status):
            yield url


def harvest_platform_bulk(
    platform: Platform, crawl: str, *,
    fetch_range: RangeFetcher | None = None,
    fetch_size: SizeFetcher | None = None,
    sleep: Callable[[float], None] | None = None,
    delay: float = 0.0,
    max_blocks: int | None = None,
    log: Callable[[str], None] = print,
) -> PlatformResult:
    """Harvest one platform from one crawl via the bulk index."""
    if sleep is None:
        sleep = time.sleep
    result = PlatformResult(platform.name)

    for query_url, match_type in platform.queries:
        prefix = _surt_prefix(query_url, match_type)
        try:
            blocks = bulk_find_blocks(crawl, prefix, fetch_range=fetch_range,
                                      fetch_size=fetch_size)
        except HarvestError as exc:
            result.errors.append(f"bulk {query_url}: {exc}")
            log(f"[harvest]   bulk {query_url}: {exc}")
            continue

        if not blocks:
            log(f"[harvest]   bulk {query_url}: no blocks in {crawl}")
            continue

        limit = blocks if max_blocks is None else blocks[:max_blocks]
        log(f"[harvest]   bulk {query_url} ({prefix}): {len(blocks)} block(s), "
            f"reading {len(limit)}")

        for name, offset, length in limit:
            if delay:
                sleep(delay)
            try:
                text = bulk_fetch_block(crawl, name, offset, length,
                                        fetch_range=fetch_range, sleep=sleep)
            except HarvestError as exc:
                result.errors.append(f"bulk {name}@{offset}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - one block, not the sweep
                result.errors.append(f"bulk {name}@{offset}: {exc}")
                continue
            result.pages_fetched += 1
            for url in iter_bulk_urls(text, prefix):
                result.records_seen += 1
                slug = platform.extract(url)
                if slug:
                    result.slugs.add(slug)

    return result


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
    # No iCIMS query on purpose. Wayback sorts by urlkey, and "com,icims)"
    # (the www marketing site) sorts ahead of every "com,icims,<tenant>)", so a
    # wildcard sweep pages through tens of thousands of www asset URLs -- its
    # versioned JS bundles resist collapsing at any depth -- and hits the stall
    # cutoff before reaching a single tenant. A real run harvested 856 entries
    # and 0 new companies. Server-side filters to skip the bare host all time
    # out (504), and a path-scoped url= is ignored when the host is wildcarded.
    # Common Crawl covers iCIMS fine, so the query only burned a time budget.
    "icims": (),
    "paylocity": ("recruiting.paylocity.com/*",),
    "bamboohr": ("*.bamboohr.com/*",),
    # No Wayback queries for the platforms added from the bulk index, on the
    # same grounds the iCIMS entry above records. Measured on teamtailor: the
    # bulk index returns 2,057 companies in seconds, while Wayback ground out
    # 212 slugs in seven minutes and had not finished. Fifteen platforms doing
    # that turned a Collect step that takes minutes into one still running at
    # forty-five, which is how this was found -- the first real CI run.
    #
    # The yield is also the wrong kind. Wayback reaches back years, and this
    # workflow already records that archival slugs keep getting deader: a
    # 2022-23 sweep bought 145 live boards and 1,114 dead ones, and every dead
    # slug costs the bot a probe every recheck period, permanently.
    #
    # Lever remains the reason Wayback exists here at all: it blocks CCBot, so
    # the archive is its only source. Common Crawl covers these nine.
    "workable": (),
    "breezy": (),
    "smartrecruiters": (),
    "rippling": (),
    "teamtailor": (),
    "jazzhr": (),
    "recruitee": (),
    "jobvite": (),
    "applicantpro": (),
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
    # Paylocity puts a deep fixed path before the identifier --
    # `com,paylocity,recruiting)/recruiting/jobs/all/<guid>` -- so the derived
    # depth of 30 collapses every /recruiting/* URL into one group and yields
    # nothing. Measured over 600-900 rows: depth 46 returned 0 rows, 50 gave
    # 492 guids, 52 gave 775, 56 gave 378 and 60 gave 0.
    "recruiting.paylocity.com/*": (50, 52),
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


def prune_existing(out_dir: Path, platforms: Iterable[Platform], *,
                   log: Callable[[str], None] = print) -> dict[str, dict[str, int]]:
    """Re-apply the current extraction rules to already-harvested files.

    Harvest output is cumulative and published, so a slug written by an older
    revision of the filters stays there forever. Every tightening since the
    first sweep -- Workday asset paths, root.<uuid> debris, version strings,
    numbered infrastructure hosts -- had to be applied to the existing files by
    hand, which is both easy to forget and impossible to review.

    The check is the same one extraction uses, so pruning can never remove a
    slug a fresh harvest would keep. It runs offline: no index, no probing.
    Liveness is not its business -- a dead company is still a company, and the
    dead-slug machinery owns that decision.
    """
    report: dict[str, dict[str, int]] = {}
    for platform in platforms:
        path = out_dir / f"{platform.name}.json"
        before = load_existing(path)
        if not before:
            continue
        kept: set[str] = set()
        dropped: set[str] = set()
        rewritten: dict[str, str] = {}
        for slug in before:
            current = _current_identifier(platform, slug)
            if current is None:
                dropped.add(slug)
            elif current == slug:
                kept.add(slug)
            else:
                # The current rules would produce a different identifier from
                # the same URL -- "al-" became "al" once trailing punctuation
                # started being trimmed. Converging on it beats dropping: that
                # is what a fresh harvest of the same capture yields.
                rewritten[slug] = current
                kept.add(current)
        report[platform.name] = {
            "before": len(before), "kept": len(kept), "dropped": len(dropped),
            "rewritten": len(rewritten),
            "examples": sorted(dropped)[:5],
        }
        if dropped or rewritten:
            detail = []
            if dropped:
                detail.append(f"dropped {len(dropped)}: "
                              f"{', '.join(sorted(dropped)[:4])}")
            if rewritten:
                pairs = list(sorted(rewritten.items()))[:3]
                detail.append("rewrote " + str(len(rewritten)) + ": "
                              + ", ".join(f"{a}->{b}" for a, b in pairs))
            log(f"[prune] {platform.name}: {len(before)} -> {len(kept)} "
                f"({'; '.join(detail)})")
            write_slugs(path, kept)
        else:
            log(f"[prune] {platform.name}: {len(before)} clean")
    return report


def _current_identifier(platform: Platform, slug: str) -> str | None:
    """What the current extractor makes of a stored identifier, or None.

    Rebuilding a representative URL and re-extracting is the only check that
    stays honest as the rules change -- re-implementing them here would let the
    two drift apart, which is exactly the bug this exists to prevent.
    """
    probe = _IDENTIFIER_PROBES.get(platform.name)
    if probe is None:
        return slug
    url = probe(slug)
    if url is None:
        return None
    return platform.extract(url)


def _workday_probe_url(slug: str) -> str | None:
    parts = slug.split("|")
    if len(parts) != 3:
        return None
    tenant, host, site = parts
    # The site is the only path segment, deliberately. Anything after it would
    # let a locale-shaped site name be mistaken for a locale prefix and
    # skipped -- "/en-US/py/job/x" reads as site "job", and that rewrote four
    # live abinbev boards to a dead identifier. With one segment there is
    # nothing for the locale rule to skip to, so the round trip is exact for
    # every site name including py and hn-es.
    return f"https://{tenant}.{host}.myworkdayjobs.com/{site}"


_IDENTIFIER_PROBES: dict[str, Callable[[str], str | None]] = {
    "greenhouse": lambda s: f"https://job-boards.greenhouse.io/{s}/jobs/1",
    "lever": lambda s: f"https://jobs.lever.co/{s}/abc",
    "ashby": lambda s: f"https://jobs.ashbyhq.com/{s}/abc",
    "workable": lambda s: f"https://apply.workable.com/{s}/j/abc",
    "breezy": lambda s: f"https://{s}.breezy.hr/p/abc",
    "smartrecruiters": lambda s: f"https://jobs.smartrecruiters.com/{s}/abc",
    "rippling": lambda s: f"https://ats.rippling.com/{s}/jobs",
    "teamtailor": lambda s: f"https://{s}.teamtailor.com/jobs",
    "jazzhr": lambda s: f"https://{s}.applytojob.com/apply",
    "recruitee": lambda s: f"https://{s}.recruitee.com/o/abc",
    "jobvite": lambda s: f"https://jobs.jobvite.com/{s}/job/abc",
    "applicantpro": lambda s: f"https://{s}.applicantpro.com/jobs/",
    "icims": lambda s: f"https://{s}.icims.com/jobs/1",
    "bamboohr": lambda s: f"https://{s}.bamboohr.com/careers/list",
    "workday": _workday_probe_url,
    "paylocity": lambda s: (
        f"https://recruiting.paylocity.com/recruiting/jobs/All/{s}/X"),
}


LOCK_NAME = ".harvest.lock"
# A lock older than this is assumed to belong to a process that died without
# cleaning up. Long enough that a real 30-crawl sweep (23 minutes observed)
# never trips it, short enough that a crash does not block the next weekly run.
LOCK_STALE_SECONDS = 6 * 60 * 60


class HarvestLocked(RuntimeError):
    """Another harvest is already writing to this output directory."""


def _pid_alive(pid: int) -> bool:
    """Whether a process id is currently running.

    Used to decide whether a lock left on disk still belongs to anyone. Both
    unknown answers are resolved conservatively: a pid that cannot be checked
    counts as alive, so the caller falls back to the age rule rather than
    stealing a lock from a running harvest.

    A recycled pid can make a dead lock look held, which costs a wait. The
    opposite mistake -- deciding a running harvest is dead -- costs two
    processes writing the same files and silently dropping each other's finds,
    which is the failure this lock exists to prevent.
    """
    if pid <= 0:
        return True
    if os.name == "nt":
        import ctypes

        # SYNCHRONIZE. Enough to learn whether the process exists without
        # asking for rights that a foreign process would refuse.
        handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        # 5 is ERROR_ACCESS_DENIED: it exists, we simply may not open it.
        return ctypes.windll.kernel32.GetLastError() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


@contextlib.contextmanager
def output_lock(out_dir: Path, *, now: Callable[[], float] = time.time):
    """Refuse to run two harvests against the same output directory.

    Both write the same files, and the merge each performs is read-modify-write
    against what it loaded at its own start -- so the later writer silently
    discards whatever the earlier one added in between. Nothing errors and the
    totals just come out low.

    This is not hypothetical. Checking for a running harvest with
    `ps -W | grep harvest_ats` always returns nothing, because that output
    carries the executable path and not the script name, so the check reads as
    "clear" every time. On the strength of it I started a second sweep over a
    live one and had four processes racing the same files.

    A stale lock -- one left by a process that was killed -- is taken over
    rather than treated as fatal, since otherwise one crash blocks every run
    after it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / LOCK_NAME
    if lock.exists():
        try:
            age = now() - lock.stat().st_mtime
        except OSError:
            age = 0.0
        # The lock records the pid that took it, so ask whether that process is
        # still there rather than waiting out the age. Age alone meant a killed
        # run blocked every later one for six hours -- measured: a harvest
        # killed mid-sweep left a lock whose pid was already gone, and the next
        # run refused at 40 seconds old. Unattended, that turns one crash into
        # a week with no harvest.
        holder = 0
        try:
            holder = int((lock.read_text(encoding="utf-8") or "0").strip() or 0)
        except (OSError, ValueError):
            holder = 0
        abandoned = holder > 0 and not _pid_alive(holder)
        if age < LOCK_STALE_SECONDS and not abandoned:
            raise HarvestLocked(
                f"another harvest holds {lock} (age {age:.0f}s). "
                "Concurrent runs silently drop each other's finds; wait for it "
                "or delete the lock if you are certain it is stale."
            )
    try:
        lock.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        # A lock we cannot write is not a reason to refuse to harvest.
        yield
        return
    try:
        yield
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _checkpoint(args, platform_name: str, existing: set[str], found: set[str]) -> None:
    """Write what has been harvested so far, after each crawl.

    Output used to be written once a platform had finished every crawl. A
    12-crawl run that died partway through Greenhouse therefore lost the whole
    platform -- which is exactly what happened, silently, with no traceback:
    the process simply stopped and the file was untouched.

    Writing per crawl bounds the loss to one crawl instead of one platform. The
    write is atomic and the merge is a union, so a checkpoint can only ever add
    -- an interrupted run leaves a smaller harvest, never a corrupt one.
    """
    if args.dry_run or not found:
        return
    write_slugs(args.out / f"{platform_name}.json", existing | found)


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
    parser.add_argument("--discover-years", type=int, nargs="*", default=None,
                        metavar="YEAR",
                        help="if the crawl listing and its cache are both "
                             "unavailable, probe the bulk host for crawls in "
                             "these years")
    parser.add_argument("--prune", action="store_true",
                        help="re-apply current extraction rules to the existing "
                             "harvest files and exit; makes filter fixes "
                             "retroactive. Offline, no probing.")
    parser.add_argument("--index", action="append", dest="indexes",
                        choices=["commoncrawl", "ccbulk", "wayback"],
                        help="which archive(s) to sweep (repeatable; "
                             "default: both)")
    args = parser.parse_args(argv)

    # ccbulk before commoncrawl: it reads the same index from static files on a
    # different host, which is both far faster and available when the query
    # service is not. The API path stays as a fallback for the same crawl.
    indexes = args.indexes or ["ccbulk", "wayback"]

    if args.crawls < 1:
        print("--crawls must be at least 1", file=sys.stderr)
        return 2

    log: Callable[[str], None] = (
        (lambda msg: print(msg, file=sys.stderr)) if args.json else print
    )

    selected = [PLATFORM_BY_NAME[n] for n in args.platforms] if args.platforms         else list(PLATFORMS)

    # Taken before anything reads or writes the output directory, and before
    # crawl resolution in particular. Fetching the crawl listing can block for
    # minutes on a slow collinfo, and a lock acquired afterwards leaves
    # precisely that window unguarded -- which is when a second run is most
    # likely to be started, because nothing has been logged yet and the harvest
    # looks idle. Observed exactly that: a sweep sat in latest_crawls with no
    # lock and no output for over half a minute.
    #
    # It covers --prune too. Pruning is a read-modify-write over the same
    # files, so running it against a live harvest loses whichever side writes
    # first -- and the workflow runs prune immediately before a collection,
    # which is exactly where an overlapping manual run would land.
    try:
        lock_ctx = output_lock(args.out)
        lock_ctx.__enter__()
    except HarvestLocked as exc:
        print(f"[harvest] {exc}", file=sys.stderr)
        return 2
    try:
        if args.prune:
            report = prune_existing(args.out, selected, log=log)
            if args.json:
                print(json.dumps({"pruned": report}, indent=2))
            return 0

        crawls: list[str] = []
        if "commoncrawl" in indexes or "ccbulk" in indexes:
            try:
                crawls = args.crawls_explicit or latest_crawls(
                    args.crawls, cache_path=args.out / CRAWL_CACHE_NAME,
                    discover_years=args.discover_years)
            except HarvestError as exc:
                # Losing Common Crawl must not cancel the Wayback sweep: they
                # are independent archives, and Wayback is Lever's only source.
                print(f"[harvest] could not determine crawls: {exc}",
                      file=sys.stderr)
                if "wayback" not in indexes:
                    return 1
                indexes = [i for i in indexes
                           if i not in ("commoncrawl", "ccbulk")]

        platforms = [PLATFORM_BY_NAME[n] for n in args.platforms] \
            if args.platforms else list(PLATFORMS)

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

            if "ccbulk" in indexes:
                for crawl in crawls:
                    log(f"[harvest] {platform.name} @ {crawl} (bulk)")
                    result = harvest_platform_bulk(
                        platform, crawl, delay=args.delay if args.delay > 1 else 0.0,
                        max_blocks=args.max_pages, log=log,
                    )
                    found |= result.slugs
                    errors.extend(result.errors)
                    pages += result.pages_fetched
                    records += result.records_seen
                    _checkpoint(args, platform.name, existing, found)

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
                    _checkpoint(args, platform.name, existing, found)

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

    finally:
        lock_ctx.__exit__(None, None, None)

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
