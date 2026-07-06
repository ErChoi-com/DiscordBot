"""ATS platform scrapers for Greenhouse, Lever, Ashby, Workday, and iCIMS.

Fetches job listings directly from company career board APIs rather than
public aggregators. Based on: https://github.com/Feashliaa/job-board-aggregator
"""
from __future__ import annotations

import json
import random
import re
import xml.etree.ElementTree as ET
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests

GREENHOUSE = "greenhouse"
LEVER = "lever"
ASHBY = "ashby"
WORKDAY = "workday"
ICIMS = "icims"
BAMBOOHR = "bamboohr"

ATS_PLATFORMS: tuple[str, ...] = (GREENHOUSE, LEVER, ASHBY, WORKDAY, ICIMS, BAMBOOHR)

PLATFORM_WORKERS: dict[str, int] = {
    GREENHOUSE: 30,
    LEVER: 30,
    ASHBY: 20,
    WORKDAY: 50,
    ICIMS: 30,
    BAMBOOHR: 30,
}

DEAD_SLUG_TTL_DAYS: int = 90

REQUEST_TIMEOUT: int = 30
FUTURE_RESULT_TIMEOUT: int = 120

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


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_JBA_DIR = _DATA_DIR / "ats_companies"
_DEAD_SLUG_DIR = _DATA_DIR / "dead_slugs"

_PLATFORM_FILES: dict[str, str] = {
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


def reload_company_lists() -> None:
    global _company_cache
    _company_cache = None


def load_company_lists() -> dict[str, list[str]]:
    global _company_cache
    if _company_cache is not None:
        return _company_cache

    cache: dict[str, list[str]] = {}
    for platform, filename in _PLATFORM_FILES.items():
        path = _JBA_DIR / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                cache[platform] = [str(s).strip() for s in data if str(s).strip()]
        except Exception as exc:
            print(f"[ats] Failed to load {filename}: {exc}")

    _company_cache = cache
    return _company_cache


def _load_dead_slugs(platform: str) -> set[str]:
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
    dates = _dead_slug_dates.get(platform, {})
    _DEAD_SLUG_DIR.mkdir(parents=True, exist_ok=True)
    (_DEAD_SLUG_DIR / f"{platform}.json").write_text(
        json.dumps(dates, indent=2, sort_keys=True), encoding="utf-8"
    )


def _purge_expired_dead_slugs(platform: str) -> int:
    """Remove dead slug entries older than DEAD_SLUG_TTL_DAYS. Returns count removed."""
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=DEAD_SLUG_TTL_DAYS)).isoformat()
    dates = _dead_slug_dates.get(platform)
    if not dates:
        return 0
    expired = [slug for slug, marked in dates.items() if marked < cutoff]
    for slug in expired:
        del dates[slug]
        _dead_slugs[platform].discard(slug)
    if expired:
        _dead_slugs_dirty.add(platform)
    return len(expired)


def flush_dead_slugs() -> None:
    for platform in list(_dead_slugs_dirty):
        _purge_expired_dead_slugs(platform)
        _save_dead_slugs(platform)
    _dead_slugs_dirty.clear()


def _is_dead(platform: str, slug: str) -> bool:
    return slug in _load_dead_slugs(platform)


def _mark_dead(platform: str, slug: str) -> None:
    _load_dead_slugs(platform).add(slug)
    _dead_slug_dates.setdefault(platform, {})[slug] = time.strftime("%Y-%m-%d")
    _dead_slugs_dirty.add(platform)


def clear_dead_slugs() -> None:
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

def _ensure_geo_loaded() -> None:
    global _geo_cities, _geo_admin1_name, _geo_loaded
    global _iso_country_codes, _ca_us_subdiv_codes
    if _geo_loaded:
        return
    import pycountry
    _geo_cities, _, _geo_admin1_name = _load_geo_lookup()
    _iso_country_codes.update(c.alpha_2 for c in pycountry.countries)
    for _sub in pycountry.subdivisions:
        if _sub.country_code in ("CA", "US"):
            _short = _sub.code.split("-", 1)[1]
            if _short not in _ca_us_subdiv_codes:
                _ca_us_subdiv_codes[_short] = _sub.country_code
    _geo_loaded = True

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
        return job_country == search_country

    if is_remote and not job_country:
        return search_country is None

    if not search_country:
        search_lower = search_location.lower().strip()
        job_lower = job_loc.lower()
        search_tokens = [t.strip() for t in search_lower.replace(",", " ").split() if len(t.strip()) > 2]
        for token in search_tokens:
            if re.search(r'\b' + re.escape(token) + r'\b', job_lower):
                return True
        return False

    return False


# ── Greenhouse ──────────────────────────────────────────────────────────────

def _scrape_greenhouse(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(GREENHOUSE, slug):
        return []
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    headers = _make_headers()
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            return []
        if resp.status_code == 200:
            break
        if resp.status_code in (404, 410):
            _mark_dead(GREENHOUSE, slug)
            return []
        if resp.status_code in (429, 503, 502) and attempt < 2:
            time.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
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
            "date_posted": str(job.get("updated_at") or ""),
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
    url = f"https://api.lever.co/v0/postings/{slug}"
    headers = _make_headers()
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            return []
        if resp.status_code == 200:
            break
        if resp.status_code in (404, 410):
            _mark_dead(LEVER, slug)
            return []
        if resp.status_code in (429, 503, 502) and attempt < 2:
            time.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
            headers["User-Agent"] = random.choice(USER_AGENTS)
            continue
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
    if _is_dead(ASHBY, slug):
        return []
    time.sleep(random.uniform(0.5, 2.0))
    payload = {
        "operationName": "ApiJobBoardWithTeams",
        "variables": {"organizationHostedJobsPageName": slug},
        "query": _ASHBY_QUERY,
    }
    headers = _make_headers({"Content-Type": "application/json"})
    resp = None
    for attempt in range(3):
        try:
            resp = requests.post(_ASHBY_GRAPHQL_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            return []
        if resp.status_code == 200:
            break
        if resp.status_code in (404, 410):
            _mark_dead(ASHBY, slug)
            return []
        if resp.status_code in (429, 503, 502) and attempt < 2:
            backoff = (2 ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(backoff)
            headers["User-Agent"] = random.choice(USER_AGENTS)
            continue
        return []
    if resp is None or resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except Exception:
        return []

    board = (data.get("data") or {}).get("jobBoard")
    if not board:
        return []

    rows: list[dict[str, Any]] = []
    for posting in board.get("jobPostings") or []:
        title = str(posting.get("title") or "").strip()
        if not title:
            continue
        job_id = str(posting.get("id") or "").strip()
        loc = str(posting.get("locationName") or "").strip()
        if not job_id:
            continue
        job_url = f"https://jobs.ashbyhq.com/{slug}/{job_id}"
        if not _matches_keywords(title, keywords):
            continue
        if not _matches_location(loc, location):
            continue
        rows.append({
            "title": title,
            "company": slug,
            "location": loc,
            "job_url": job_url,
            "date_posted": "",
            "_source_site": ASHBY,
            "_source_sites": [ASHBY],
        })
        if len(rows) >= max_jobs:
            break
    return rows


# ── Workday ─────────────────────────────────────────────────────────────────

def _parse_workday_slug(slug: str) -> tuple[str, str, str] | None:
    parts = slug.split("|")
    if len(parts) != 3:
        return None
    company, wd_num, site_id = (p.strip() for p in parts)
    if not company or not wd_num or not site_id:
        return None
    return company, wd_num, site_id


def _fetch_workday_date(detail_url: str, headers: dict[str, str]) -> str:
    try:
        resp = requests.get(detail_url, headers=headers, timeout=REQUEST_TIMEOUT)
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
            resp = requests.post(api_url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            break

        if resp.status_code in (404, 410, 422):
            if offset == 0:
                _mark_dead(WORKDAY, slug)
            break
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

    url_to_date: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_workday_date, row["detail_url"], headers): row["detail_url"] for row in candidates}
        for future in as_completed(futures):
            try:
                url_to_date[futures[future]] = future.result(timeout=FUTURE_RESULT_TIMEOUT)
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    for row in candidates:
        row["date_posted"] = url_to_date.get(row.pop("detail_url"), "")
        rows.append(row)

    return rows


# ── iCIMS ──────────────────────────────────────────────────────────────────

_ICIMS_SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def _fetch_icims_metadata(job_url: str) -> dict[str, str]:
    """Fetch an iCIMS job page and extract title, location, datePosted from JSON-LD."""
    result: dict[str, str] = {"title": "", "location": "", "date_posted": ""}
    try:
        hdrs = _make_headers({"Accept": "application/json"})
        resp = requests.get(job_url, headers=hdrs, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return result
        html = resp.text
        for match in re.finditer(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.DOTALL):
            try:
                ld = json.loads(match.group(1))
                if isinstance(ld, list):
                    ld = next((x for x in ld if x.get("@type") == "JobPosting"), None)
                if not ld or ld.get("@type") != "JobPosting":
                    continue
                result["title"] = str(ld.get("title") or "").strip()
                result["date_posted"] = str(ld.get("datePosted") or "").strip()
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
                    result["location"] = ", ".join(p for p in parts if p)
                return result
            except (json.JSONDecodeError, AttributeError, StopIteration):
                continue
    except Exception:
        pass
    return result


def _scrape_icims(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(ICIMS, slug):
        return []
    sitemap_url = f"https://careers-{slug}.icims.com/sitemap.xml"
    headers = _make_headers({"Accept": "application/xml"})
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(sitemap_url, headers=headers, timeout=REQUEST_TIMEOUT)
        except Exception:
            return []
        if resp.status_code == 200:
            break
        if resp.status_code in (404, 410):
            _mark_dead(ICIMS, slug)
            return []
        if resp.status_code in (429, 503, 502) and attempt < 2:
            time.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
            headers["User-Agent"] = random.choice(USER_AGENTS)
            continue
        return []
    if resp is None or resp.status_code != 200:
        return []
    try:
        root = ET.fromstring(resp.content)
    except Exception:
        return []

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
        if len(candidates) >= max_jobs:
            break

    if not candidates:
        return []

    url_to_meta: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_icims_metadata, row["job_url"]): row["job_url"] for row in candidates}
        for future in as_completed(futures):
            try:
                url_to_meta[futures[future]] = future.result(timeout=FUTURE_RESULT_TIMEOUT)
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    for row in candidates:
        meta = url_to_meta.get(row["job_url"], {})
        if meta.get("title"):
            row["title"] = meta["title"]
        if meta.get("date_posted"):
            row["date_posted"] = meta["date_posted"]
        row["location"] = meta.get("location", "")
        if not _matches_location(row["location"], location):
            continue
        rows.append(row)

    return rows


# ── BambooHR ──────────────────────────────────────────────────────────────

def _scrape_bamboohr(slug: str, keywords: str, location: str, max_jobs: int) -> list[dict[str, Any]]:
    if _is_dead(BAMBOOHR, slug):
        return []
    time.sleep(random.uniform(0.5, 2.0))
    url = f"https://{slug}.bamboohr.com/careers/list"
    headers = _make_headers()
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.SSLError:
            if attempt < 2:
                time.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
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
            backoff = (2 ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(backoff)
            headers["User-Agent"] = random.choice(USER_AGENTS)
            continue
        return []
    if resp is None or resp.status_code != 200:
        return []
    if "application/json" not in resp.headers.get("Content-Type", ""):
        _mark_dead(BAMBOOHR, slug)
        return []
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
        if not _matches_location(loc, location):
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
        if len(rows) >= max_jobs:
            break
    return rows


# ── Platform dispatch ───────────────────────────────────────────────────────

_SCRAPERS: dict[str, Any] = {
    GREENHOUSE: _scrape_greenhouse,
    LEVER: _scrape_lever,
    ASHBY: _scrape_ashby,
    WORKDAY: _scrape_workday,
    ICIMS: _scrape_icims,
    BAMBOOHR: _scrape_bamboohr,
}


def scrape_ats_platform(
    platform: str,
    keywords: str,
    location: str,
    results_wanted: int = 20,
    company_slugs: list[str] | None = None,
) -> list[dict[str, Any]]:
    scraper = _SCRAPERS.get(platform)
    if not scraper:
        return []

    if company_slugs is None:
        company_slugs = load_company_lists().get(platform, [])

    if not company_slugs:
        return []

    max_per_company = results_wanted if results_wanted > 0 else 10_000
    workers = min(PLATFORM_WORKERS.get(platform, 10), len(company_slugs))
    all_rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scraper, slug, keywords, location, max_per_company): slug
            for slug in company_slugs
        }
        for future in as_completed(futures):
            try:
                rows = future.result(timeout=FUTURE_RESULT_TIMEOUT)
                all_rows.extend(rows)
            except Exception:
                continue

    flush_dead_slugs()
    return all_rows
