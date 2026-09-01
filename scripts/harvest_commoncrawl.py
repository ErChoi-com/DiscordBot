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
    python scripts/harvest_commoncrawl.py                  # latest crawl
    python scripts/harvest_commoncrawl.py --crawl CC-MAIN-2026-30
    python scripts/harvest_commoncrawl.py --crawls 3       # 3 most recent
    python scripts/harvest_commoncrawl.py --platform greenhouse --dry-run
    python scripts/harvest_commoncrawl.py --out data/ats_harvest

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
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", slug):
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
_WORKDAY_SITE_REJECT = frozenset({
    "robots.txt", "sitemap.xml", "favicon.ico", "index.html",
})


def _extract_workday(url: str) -> str | None:
    """Workday needs three parts, stored as ``tenant|host|site``.

    That storage shape is not ours -- it is upstream's, and ats_service:918
    rebuilds ``https://{company}.{wd_num}.myworkdayjobs.com`` from it, so the
    harvester has to emit exactly the same triple.
    """
    m = _WORKDAY_RE.match(url)
    if not m:
        return None
    tenant, host = m.group(1).lower(), m.group(2).lower()
    if not _WORKDAY_HOST_RE.fullmatch(host):
        # impl-wd103 and friends are staging tenants; they never serve a real board.
        return None
    if not _looks_like_company(tenant):
        return None

    # Walk past any leading locale segments to the first real one.
    site = None
    for raw in m.group(3).split("?")[0].split("#")[0].split("/"):
        segment = urllib.parse.unquote(raw).strip()
        if not segment:
            continue
        if _WORKDAY_LOCALE_RE.fullmatch(segment):
            continue
        site = segment.lower()
        break
    if not site or site in _WORKDAY_SITE_REJECT:
        return None
    if len(site) > 100 or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", site):
        return None
    return f"{tenant}|{host}|{site}"


PLATFORMS: tuple[Platform, ...] = (
    Platform(
        "greenhouse",
        # Greenhouse served boards on boards.greenhouse.io for years and moved to
        # job-boards.greenhouse.io; old crawls carry the former, new ones the
        # latter, and neither host alone covers the company set.
        (("boards.greenhouse.io/*", "prefix"),
         ("job-boards.greenhouse.io/*", "prefix")),
        _path_segment_extractor("greenhouse.io"),
    ),
    Platform(
        "lever",
        (("jobs.lever.co/*", "prefix"),),
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
        (("myworkdayjobs.com", "domain"),),
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
        try:
            pages = page_count(crawl, query_url, match_type, fetch=fetch, sleep=sleep)
        except HarvestError as exc:
            # One dead query must not cost us the platform's other query -- the
            # two Greenhouse hosts are independent sources of companies.
            result.errors.append(f"{query_url}: {exc}")
            log(f"[harvest]   {query_url}: page count failed: {exc}")
            continue

        if pages == 0:
            log(f"[harvest]   {query_url}: no captures in {crawl}")
            continue

        limit = pages if max_pages is None else min(pages, max_pages)
        log(f"[harvest]   {query_url}: {pages} page(s), fetching {limit}")

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
    args = parser.parse_args(argv)

    if args.crawls < 1:
        print("--crawls must be at least 1", file=sys.stderr)
        return 2

    log: Callable[[str], None] = (
        (lambda msg: print(msg, file=sys.stderr)) if args.json else print
    )

    try:
        crawls = args.crawls_explicit or latest_crawls(args.crawls)
    except HarvestError as exc:
        print(f"[harvest] could not determine crawls: {exc}", file=sys.stderr)
        return 1

    platforms = [PLATFORM_BY_NAME[n] for n in args.platforms] if args.platforms \
        else list(PLATFORMS)

    log(f"[harvest] crawls: {', '.join(crawls)}")
    log(f"[harvest] platforms: {', '.join(p.name for p in platforms)}")

    t_start = time.monotonic()
    summary: dict[str, dict[str, object]] = {}
    any_success = False

    for platform in platforms:
        existing = load_existing(args.out / f"{platform.name}.json")
        found: set[str] = set()
        errors: list[str] = []
        pages = records = 0

        for crawl in crawls:
            log(f"[harvest] {platform.name} @ {crawl}")
            result = harvest_platform(
                platform, crawl, delay=args.delay, max_pages=args.max_pages, log=log,
            )
            found |= result.slugs
            errors.extend(result.errors)
            pages += result.pages_fetched
            records += result.records_seen

        new = found - existing
        merged = existing | found
        summary[platform.name] = {
            "harvested": len(found), "new": len(new), "total": len(merged),
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
            "crawls": crawls, "elapsed_seconds": round(elapsed, 1),
            "dry_run": args.dry_run, "platforms": summary,
        }, indent=2))

    if not any_success:
        print("[harvest] every platform yielded nothing - treating as failure",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
