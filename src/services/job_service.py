from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import hashlib
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

DEFAULT_JOBSPY_EXE = Path(sys.executable)
JOBBANK_CANADA_SITE = "jobbank_canada"
JOBBANK_BASE_URL = "https://www.jobbank.gc.ca"
JOBBANK_SEARCH_URL = f"{JOBBANK_BASE_URL}/jobsearch/jobsearch"
JOBBANK_USER_AGENT = "Mozilla/5.0 (compatible; RebuiltJobWatcher/1.0)"
JOBBANK_DEFAULT_NATIVE_QUERY = (
    "flg=E&fgeo=27234&fgeo=6363&fgeo=9219&fn21=20012&fn21=21103&fn21=21109&fn21=21211&"
    "fn21=21220&fn21=21222&fn21=21223&fn21=21230&fn21=21231&fn21=21232&fn21=21233&fn21=21234&"
    "fn21=21311&fn21=21322&fn21=21330&fn21=22101&fn21=22111&fn21=22212&fn21=22214&fn21=22220&"
    "fn21=22221&fn21=22222&fn21=22230&fn21=22312&fn21=22313&page=1&sort=M&fsrc=21"
)
FALLBACK_JOBSPY_SITES = [
    "bayt",
    "bdjobs",
    "google",
    "indeed",
    "linkedin",
    "naukri",
    "zip_recruiter",
]
CUSTOM_SCRAPER_SITES = {"jobbank_canada", "glassdoor", "zip_recruiter", "greenhouse", "lever", "ashby", "workday", "icims", "bamboohr"}
JOBSPY_SITE_LABELS = {
    "all": "All supported sites",
    JOBBANK_CANADA_SITE: "Job Bank Canada",
    "bayt": "Bayt",
    "bdjobs": "BDJobs",
    "glassdoor": "Glassdoor",
    "google": "Google Jobs",
    "indeed": "Indeed",
    "linkedin": "LinkedIn",
    "naukri": "Naukri",
    "zip_recruiter": "ZipRecruiter",
    "greenhouse": "Greenhouse",
    "lever": "Lever",
    "ashby": "Ashby",
    "workday": "Workday",
    "icims": "iCIMS",
}
DISTANCE_UNSUPPORTED_SITES = {"bayt"}
JOB_BOARD_HOST_PATTERNS = {
    JOBBANK_CANADA_SITE: ("jobbank.gc.ca",),
    "bayt": ("bayt.",),
    "bdjobs": ("bdjobs.",),
    "glassdoor": ("glassdoor.",),
    "indeed": ("indeed.",),
    "linkedin": ("linkedin.",),
    "naukri": ("naukri.",),
    "zip_recruiter": ("ziprecruiter.",),
    "greenhouse": ("greenhouse.io", "boards.greenhouse.io"),
    "lever": ("lever.co", "jobs.lever.co"),
    "ashby": ("ashbyhq.com", "jobs.ashbyhq.com"),
    "workday": ("myworkdayjobs.com",),
    "icims": ("icims.com",),
}
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "gh_src",
    "mc_cid",
    "mc_eid",
    "ref",
    "refid",
    "trk",
    "trkinfo",
}
SPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_TITLE_CHUNK_RE = re.compile(r"\s+[-–/]\s+")
JOBBANK_POSTING_LINK_PATTERN = re.compile(r"/jobsearch/jobposting/[^\"'\s<>]+", re.IGNORECASE)
JOBBANK_POSTING_ID_PATTERN = re.compile(r"/jobsearch/jobposting/(\d+)", re.IGNORECASE)
JOBBANK_EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
JOBBANK_PHONE_PATTERN = re.compile(
    r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)\s*|\d{3}[-.\s])\d{3}[-.\s]\d{4}",
    re.IGNORECASE,
)
JOBBANK_LOCATION_STOP_WORDS = {
    "work location",
    "salary",
    "to be negotiated",
    "job requirements",
    "overview",
}
CANADIAN_PROVINCE_NAMES = {
    "alberta": "AB",
    "british columbia": "BC",
    "manitoba": "MB",
    "new brunswick": "NB",
    "newfoundland and labrador": "NL",
    "nova scotia": "NS",
    "ontario": "ON",
    "prince edward island": "PE",
    "quebec": "QC",
    "saskatchewan": "SK",
    "northwest territories": "NT",
    "nunavut": "NU",
    "yukon": "YT",
}
CANADIAN_PROVINCE_CODES = {value: value for value in CANADIAN_PROVINCE_NAMES.values()}
GLASSDOOR_LOCATION_LOCALITY_PATTERN = re.compile(r'"addressLocality"\s*:\s*"([^"]+)"', re.IGNORECASE)
GLASSDOOR_LOCATION_REGION_PATTERN = re.compile(r'"addressRegion"\s*:\s*"([^"]+)"', re.IGNORECASE)
GLASSDOOR_CARD_AGE_PATTERN = re.compile(r'(?<!\d)(\d+)\s*([hd])\+?(?!\d)', re.IGNORECASE)
JOBBANK_MAX_SEARCH_PAGES: int = 5
JOBBANK_USER_AGENT_OVERRIDE: str | None = None  # set via configure_job_service

# ── Semantic plugin (overridden at startup from settings.toml via configure_job_service) ──
SEMANTIC_PLUGIN_ENABLED: bool = True
SEMANTIC_PLUGIN_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_PLUGIN_THRESHOLD: float = 0.30
SEMANTIC_MATCH_TARGET: str = "description"
SEMANTIC_DESCRIPTION_CHAR_LIMIT: int = 2200

# ── Dedup FIFO constants (overridden at startup) ──────────────────────────────
DEDUP_MAX_FIFO_FILES: int = 6
DEDUP_MAX_ENTRIES_PER_FILE: int = 500
DEDUP_MONTHS_THRESHOLD: int = 2
DEDUP_SEEN_LINKS_CAP: int = 80_000

# ── Network timeouts (overridden at startup) ──────────────────────────────────
HTTP_REQUEST_TIMEOUT: int = 20
SUBPROCESS_SCRAPE_TIMEOUT: int = 90
MAX_KEYWORD_VARIANTS: int = 8

# ── Per-site concurrency limiting (overridden at startup) ────────────────────
SITE_CONCURRENCY_LIMIT: int = 2
SITE_SEMAPHORE_TIMEOUT: int = 120

_site_semaphores: dict[str, threading.Semaphore] = {}
_site_semaphores_lock = threading.Lock()


def _get_site_semaphore(site: str) -> threading.Semaphore:
    sem = _site_semaphores.get(site)
    if sem is not None:
        return sem
    with _site_semaphores_lock:
        sem = _site_semaphores.get(site)
        if sem is None:
            sem = threading.Semaphore(SITE_CONCURRENCY_LIMIT)
            _site_semaphores[site] = sem
        return sem


@contextmanager
def _site_scrape_slot(site: str, timeout: int | None = None):
    effective_timeout = timeout if timeout is not None else SITE_SEMAPHORE_TIMEOUT
    sem = _get_site_semaphore(site)
    acquired = sem.acquire(timeout=effective_timeout)
    try:
        yield acquired
    finally:
        if acquired:
            sem.release()


def message_to_ascii_signature(message: str) -> str:
    """Convert message to space-separated ASCII sums by word.
    
    Extracts alphanumeric characters from each word and sums their ASCII values.
    Returns space-separated sums for each word.
    
    Examples:
    - "Hello World" 
      → "Hello": H(72) + e(101) + l(108) + l(108) + o(111) = 500
      → "World": W(87) + o(111) + r(114) + l(108) + d(100) = 520
      → Result: "500 520"
    
    - "Senior Python Developer"
      → "Senior": S(83) + e(101) + n(110) + i(105) + o(111) + r(114) = 624
      → "Python": P(80) + y(121) + t(116) + h(104) + o(111) + n(110) = 642
      → "Developer": D(68) + e(101) + v(118) + e(101) + l(108) + o(111) + p(112) + e(101) + r(114) = 914
      → Result: "624 642 914"
    """
    # Split by whitespace and punctuation, keep only words with alphanumeric chars
    words = []
    current_word = []
    
    for char in message:
        if char.isalnum():  # Include letters AND numbers
            current_word.append(char)
        elif current_word:
            words.append(''.join(current_word))
            current_word = []
    
    if current_word:
        words.append(''.join(current_word))
    
    # Calculate ASCII sum for each word
    word_sums = []
    for word in words:
        word_sum = sum(ord(char) for char in word)
        word_sums.append(str(word_sum))
    
    return " ".join(word_sums)


def compress_timestamp() -> int:
    """Get current timestamp as Unix epoch integer."""
    return int(datetime.now().timestamp())


def extract_title_from_message(message_content: str) -> str:
    """Extract job title (first line before URL) from a formatted Discord message."""
    lines = message_content.strip().split('\n')
    if not lines:
        return ""
    # First line is typically [Site] Title (Company, Location)
    # We want to use that as the title
    title_line = lines[0]
    # Remove the [Site] prefix if present
    if title_line.startswith('['):
        title_line = title_line[title_line.find(']')+1:].strip()
    return title_line


def dedup_namespace_from_listing_file(listing_file: Path) -> str:
    """Build a safe, collision-resistant dedup namespace key.

    Canonical watcher files preserve their historical namespace shape so existing
    channel-specific dedup folders continue to work:
    - .message_listing_<channel_id>_job.json -> message_listing_<channel_id>_job
    - .message_listing_<channel_id>_reddit.json -> message_listing_<channel_id>_reddit

    Non-canonical names get a deterministic hash suffix so different paths cannot
    collapse into the same dedup directory via sanitization.
    """
    stem = listing_file.stem
    canonical = re.search(r"message_listing_(\d+)_(job|reddit)$", stem)
    if canonical:
        return f"message_listing_{canonical.group(1)}_{canonical.group(2)}"

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    # Include parent path to prevent collisions across different directories.
    fingerprint = hashlib.sha1(str(listing_file.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    if safe:
        return f"{safe}_{fingerprint}"
    return f"listing_{fingerprint}"


def dedup_namespace_token_for_listing_file(listing_file: Path) -> str:
    """Return a stable numeric token for watcher-scoped dedup signatures.

    Uses the channel id in canonical listing names like:
    - .message_listing_<channel_id>_job.json
    - .message_listing_<channel_id>_reddit.json

    Falls back to a deterministic numeric checksum for non-canonical names.
    """
    stem = dedup_namespace_from_listing_file(listing_file)
    match = re.search(r"message_listing_(\d+)_(job|reddit)$", stem)
    if match:
        return match.group(1)

    checksum = sum(ord(ch) for ch in stem)
    return str(checksum if checksum > 0 else 1)


def dedup_directory_for_listing_file(listing_file: Path) -> Path:
    """Return the watcher-scoped dedup directory derived from listing_file."""
    return listing_file.parent / "dedup_listings" / dedup_namespace_from_listing_file(listing_file)


def ensure_dedup_directory_for_listing_file(listing_file: Path) -> Path:
    """Ensure watcher-scoped dedup directory exists and return its path."""
    listings_dir = dedup_directory_for_listing_file(listing_file)
    listings_dir.mkdir(parents=True, exist_ok=True)
    return listings_dir


def _iter_listing_files_by_index(listings_dir: Path, include_extra: bool = True) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for file_path in listings_dir.glob("listing_*.json"):
        try:
            stem = file_path.stem
            if not stem.startswith("listing_"):
                continue
            idx = int(stem.split("_", 1)[1])
        except Exception:
            continue
        if include_extra or idx < DEDUP_MAX_FIFO_FILES:
            pairs.append((idx, file_path))
    pairs.sort(key=lambda p: p[0])
    return pairs


def _read_fifo_rows(listings_dir: Path, include_extra: bool = True) -> list[str]:
    """Read FIFO rows in logical order (newest -> oldest)."""
    rows: list[str] = []
    for _, file_path in _iter_listing_files_by_index(listings_dir, include_extra=include_extra):
        try:
            file_rows = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            continue
        rows.extend(file_rows)
    return rows


def _write_fifo_rows(listings_dir: Path, rows: list[str]) -> bool:
    """Write logical FIFO rows back into split listing_i files.

    Rows are expected in newest -> oldest order.
    """
    changed = False

    # Remove out-of-range files first.
    for idx, extra_path in _iter_listing_files_by_index(listings_dir, include_extra=True):
        if idx >= DEDUP_MAX_FIFO_FILES:
            try:
                extra_path.unlink(missing_ok=True)
                changed = True
            except Exception:
                continue

    max_total_rows = DEDUP_MAX_FIFO_FILES * DEDUP_MAX_ENTRIES_PER_FILE
    trimmed_rows = rows[:max_total_rows]

    for file_idx in range(DEDUP_MAX_FIFO_FILES):
        start = file_idx * DEDUP_MAX_ENTRIES_PER_FILE
        end = start + DEDUP_MAX_ENTRIES_PER_FILE
        chunk = trimmed_rows[start:end]
        file_path = listings_dir / f"listing_{file_idx}.json"

        if chunk:
            new_text = "\n".join(chunk) + "\n"
            current_text = ""
            if file_path.exists():
                try:
                    current_text = file_path.read_text(encoding="utf-8")
                except Exception:
                    current_text = ""
            if new_text != current_text:
                try:
                    file_path.write_text(new_text, encoding="utf-8")
                    changed = True
                except Exception:
                    continue
        elif file_path.exists():
            try:
                file_path.unlink(missing_ok=True)
                changed = True
            except Exception:
                continue

    return changed


def _row_timestamp(row: str) -> int:
    parts = row.strip().split()
    try:
        return int(parts[-1])
    except (IndexError, ValueError):
        return 0


def enforce_dedup_fifo_structure_for_listing_file(listing_file: Path) -> bool:
    """Enforce FIFO structure limits for watcher-scoped dedup storage.

    Guarantees:
    - Only listing_0 .. listing_(DEDUP_MAX_FIFO_FILES-1) are kept.
    - Each listing_i file has at most DEDUP_MAX_ENTRIES_PER_FILE non-empty rows.
    - Rows are sorted newest-first (descending timestamp) so the cascade
      insert's rows.pop() always evicts the oldest entry.
    """
    try:
        listings_dir = ensure_dedup_directory_for_listing_file(listing_file)
    except Exception:
        return False

    rows = _read_fifo_rows(listings_dir, include_extra=True)
    rows.sort(key=_row_timestamp, reverse=True)
    return _write_fifo_rows(listings_dir, rows)


def dedup_signature_for_message(message_content: str, listing_file: Path) -> str | None:
    """Return watcher-scoped numeric signature (ascii sums + namespace token)."""
    title = extract_title_from_message(message_content)
    if not title:
        return None
    title_sig = message_to_ascii_signature(title)
    if not title_sig:
        return None
    namespace_token = dedup_namespace_token_for_listing_file(listing_file)
    return f"{title_sig} {namespace_token}"


def dedup_legacy_signature_from_scoped(scoped_signature: str, listing_file: Path) -> str:
    """Return legacy signature form without namespace token.

    Legacy rows are stored as: <ascii sums> <timestamp>
    Scoped rows are stored as: <ascii sums> <namespace_token> <timestamp>
    """
    token = dedup_namespace_token_for_listing_file(listing_file)
    parts = [p for p in scoped_signature.split() if p]
    if parts and parts[-1] == token:
        parts = parts[:-1]
    return " ".join(parts)


def normalize_dedup_storage_for_listing_file(listing_file: Path) -> bool:
    """Normalize dedup rows to plain 'numeric_signature timestamp' format.

    Also converts legacy namespace/version rows like:
    - v2:message_listing_123_job:500 520 1700000000
    into:
    - 500 520 <namespace_token> 1700000000
    """
    try:
        listings_dir = ensure_dedup_directory_for_listing_file(listing_file)
    except Exception:
        return False

    changed = False
    namespace_token = dedup_namespace_token_for_listing_file(listing_file)

    for file_idx in range(DEDUP_MAX_FIFO_FILES):
        file_path = listings_dir / f"listing_{file_idx}.json"
        if not file_path.exists():
            continue
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        normalized_lines: list[str] = []
        for raw_line in lines:
            line = raw_line.lstrip("\ufeff").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sig = " ".join(parts[:-1])
            ts_raw = parts[-1]
            try:
                int(ts_raw)
            except ValueError:
                continue
            clean_sig = sig
            if ":" in clean_sig and clean_sig.startswith("v"):
                # Convert namespaced/versioned signatures back to plain ascii signature.
                clean_sig = clean_sig.rsplit(":", 1)[-1].strip()
            if not clean_sig:
                continue
            sig_parts = [part for part in clean_sig.split() if part]
            if not sig_parts:
                continue
            if sig_parts[-1] != namespace_token:
                sig_parts.append(namespace_token)
            clean_sig = " ".join(sig_parts)
            normalized_lines.append(f"{clean_sig} {ts_raw}")

        if normalized_lines != [line.lstrip("\ufeff").strip() for line in lines if line.lstrip("\ufeff").strip()]:
            changed = True
            if normalized_lines:
                try:
                    file_path.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")
                except Exception:
                    continue
            else:
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    continue

    # After row normalization, enforce hard FIFO structure limits.
    if enforce_dedup_fifo_structure_for_listing_file(listing_file):
        changed = True

    return changed


def is_message_duplicate(message_content: str, listing_file: Path, months_threshold: int = 2) -> bool:
    """Check whether message title exists inside threshold in watcher-scoped dedup storage."""
    ascii_sig = dedup_signature_for_message(message_content, listing_file)
    if not ascii_sig:
        return False
    legacy_sig = dedup_legacy_signature_from_scoped(ascii_sig, listing_file)

    current_time = compress_timestamp()
    threshold_seconds = months_threshold * 30 * 24 * 3600

    try:
        listings_dir = ensure_dedup_directory_for_listing_file(listing_file)
    except Exception:
        return False

    expired_cutoff = current_time - threshold_seconds

    for file_idx in range(DEDUP_MAX_FIFO_FILES):
        file_path = listings_dir / f"listing_{file_idx}.json"
        if not file_path.exists():
            continue
        try:
            file_exhausted = False
            for line in file_path.read_text(encoding="utf-8").splitlines():
                line = line.lstrip("\ufeff").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) <= 1:
                    continue
                try:
                    timestamp = int(parts[-1])
                except ValueError:
                    continue
                if timestamp < expired_cutoff:
                    file_exhausted = True
                    break
                sig = " ".join(parts[:-1])
                if sig == ascii_sig or sig == legacy_sig:
                    return True
            if file_exhausted:
                break
        except Exception:
            continue

    return False


def record_message_for_dedup(message_content: str, listing_file: Path) -> bool:
    """Record message title signature in watcher-scoped split FIFO files.

    True cascading insert: new entry goes to front of listing_0. If that file
    exceeds DEDUP_MAX_ENTRIES_PER_FILE, its last row overflows to the front of
    listing_1, and so on through each file. Overflow past the final file is
    permanently deleted.
    """
    ascii_sig = dedup_signature_for_message(message_content, listing_file)
    if not ascii_sig:
        return False

    current_time = compress_timestamp()
    new_entry = f"{ascii_sig} {current_time}"

    try:
        listings_dir = ensure_dedup_directory_for_listing_file(listing_file)
        enforce_dedup_fifo_structure_for_listing_file(listing_file)
    except Exception:
        return False

    overflow: str | None = new_entry
    for file_idx in range(DEDUP_MAX_FIFO_FILES):
        if overflow is None:
            break

        file_path = listings_dir / f"listing_{file_idx}.json"
        rows: list[str] = []
        if file_path.exists():
            try:
                rows = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except Exception:
                rows = []

        rows.insert(0, overflow)

        if len(rows) > DEDUP_MAX_ENTRIES_PER_FILE:
            overflow = rows.pop()
        else:
            overflow = None

        try:
            file_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        except Exception:
            return False

    return True


def check_and_record_message(message_content: str, listing_file: Path, months_threshold: int = 2) -> bool:
    """Check if job title is a duplicate and record it in FIFO rotating files.
    
    Args:
        message_content: The Discord message to check
        listing_file: Base path used to derive watcher-specific dedup namespace
        months_threshold: Number of months to look back for duplicates
        
    Returns:
        True if title is a duplicate (should be filtered), False if new
        
    Uses split FIFO files in dedup_listings/<namespace>/ folder:
    - Max 6 JSON files
    - Max 500 entries per file
    - listing_0 stores newest segment of the queue
    - New rows are inserted at top of listing_0 and overflow cascades downward
    - Rows beyond the last segment are dropped (oldest removed)
    """
    if is_message_duplicate(message_content, listing_file, months_threshold=months_threshold):
        return True

    # Preserve legacy behavior: treat storage failures as "not duplicate".
    record_message_for_dedup(message_content, listing_file)
    return False


@lru_cache(maxsize=8)
def jobspy_runtime_metadata(configured_exe: str | None = None) -> dict[str, tuple[str, ...]]:
    python_executable = jobspy_python_executable(configured_exe)
    if python_executable is None:
        return {"sites": tuple(FALLBACK_JOBSPY_SITES), "params": tuple()}

    code = (
        "import inspect, json\n"
        "payload={'sites': [], 'params': []}\n"
        "try:\n"
        "    from jobspy import Site, scrape_jobs\n"
        "    members=getattr(Site, '__members__', {})\n"
        "    payload['sites']=sorted(str(name).lower() for name in members.keys())\n"
        "    payload['params']=sorted(str(name) for name in inspect.signature(scrape_jobs).parameters.keys())\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps(payload, ensure_ascii=True))\n"
    )
    try:
        proc = subprocess.run(
            [str(python_executable), "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        proc = None

    if proc and proc.returncode == 0:
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if lines:
            try:
                payload = json.loads(lines[-1])
                sites = tuple(str(site).strip().lower() for site in payload.get("sites", []) if str(site).strip())
                params = tuple(str(param).strip() for param in payload.get("params", []) if str(param).strip())
                if sites:
                    return {"sites": sites, "params": params}
            except Exception:
                pass

    try:
        import inspect

        from jobspy import Site as JobSpySite
        from jobspy import scrape_jobs

        return {
            "sites": tuple(sorted(str(name).lower() for name in JobSpySite.__members__.keys())),
            "params": tuple(sorted(str(name) for name in inspect.signature(scrape_jobs).parameters.keys())),
        }
    except Exception:
        return {"sites": tuple(FALLBACK_JOBSPY_SITES), "params": tuple()}


def supported_jobspy_sites(configured_exe: str | None = None) -> list[str]:
    sites = list(jobspy_runtime_metadata(configured_exe).get("sites", ()))
    if not sites:
        return list(FALLBACK_JOBSPY_SITES)
    return sites


def jobspy_site_options() -> list[tuple[str, str]]:
    site_values = ["all", JOBBANK_CANADA_SITE, *supported_jobspy_sites()]
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for site in site_values:
        if site in seen:
            continue
        seen.add(site)
        options.append((JOBSPY_SITE_LABELS.get(site, site.replace("_", " ").title()), site))
    return options


def normalize_indeed_country(country_indeed: str | None, location: str) -> str:
    requested = str(country_indeed or "").strip().upper()
    if requested and requested != "AUTO":
        return requested

    from services.ats_service import _infer_country
    cc = _infer_country(location)
    if cc == "CA":
        return "CANADA"
    return "USA"


def infer_job_region(row: dict[str, Any]) -> str:
    from services.ats_service import _infer_country
    location_text = str(row.get("location") or "").strip()
    cc = _infer_country(location_text)
    if cc == "CA":
        return "canada"
    if cc == "US":
        return "us"
    if cc is not None:
        return "other"

    raw_url = str(row.get("job_url") or row.get("url") or row.get("job_url_direct") or "").strip()
    if not raw_url:
        return "unknown"

    parsed = urlparse(raw_url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if host.endswith(".ca") or host.startswith("ca."):
        return "canada"
    if "/canada/" in path or "/jobs/ca/" in path:
        return "canada"
    if any(token in path for token in ("/united-states/", "/usa/", "/us/")):
        return "us"
    return "unknown"


def _posting_age_ok(row: dict[str, Any], hours_old: int) -> bool:
    """Return True if the job's date_posted is within hours_old of now.
    Rows with no date_posted (bamboohr, ashby) always pass.
    """
    posted = str(row.get("date_posted") or "").strip()
    if not posted:
        return True
    try:
        posted_dt = datetime.fromisoformat(posted.replace("Z", "+00:00"))
        if posted_dt.tzinfo is None:
            from datetime import timezone as _tz
            posted_dt = posted_dt.replace(tzinfo=_tz.utc)
        age_hours = (datetime.now(posted_dt.tzinfo) - posted_dt).total_seconds() / 3600
        return 0 <= age_hours <= hours_old
    except (ValueError, TypeError):
        return True


def filter_rows_by_region(rows: list[dict[str, Any]], allow_north_america: bool = False) -> list[dict[str, Any]]:
    if not rows:
        return []
    from services.ats_service import ATS_PLATFORMS as _ats_plats
    _ats_set = set(_ats_plats)
    allowed_regions = {"canada", "us"} if allow_north_america else {"canada"}
    result = []
    for row in rows:
        region = infer_job_region(row)
        if region in allowed_regions:
            result.append(row)
        elif region == "unknown":
            sites = set(row.get("_source_sites") or [row.get("_source_site", "")])
            if sites & _ats_set:
                result.append(row)
    return result


def _python_has_jobspy(executable: Path) -> bool:
    try:
        proc = subprocess.run(
            [str(executable), "-c", "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('jobspy') else 1)"],
            capture_output=True,
            timeout=15,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def jobspy_python_executable(configured_exe: str | None = None) -> Path | None:
    candidates: list[Path] = []

    if configured_exe:
        candidates.append(Path(configured_exe).expanduser())
    env_override = os.getenv("JOBSPY_PYTHON_EXE")
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates.append(DEFAULT_JOBSPY_EXE)
    candidates.append(Path(sys.executable))
    for minor in ("3.11", "3.12", "3.13", "3.14"):
        try:
            proc = subprocess.run(
                ["py", f"-{minor}", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                candidates.append(Path(proc.stdout.strip()))
        except Exception:
            pass

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and _python_has_jobspy(candidate):
            return candidate
    return None


def job_site_from_url(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    for site_name, patterns in JOB_BOARD_HOST_PATTERNS.items():
        if any(pattern in host for pattern in patterns):
            return site_name
    if "google." in host and "jobs" in url.lower():
        return "google"
    return None


def normalize_job_text(value: Any) -> str:
    compact = SPACE_PATTERN.sub(" ", str(value or "").strip().lower())
    return NON_ALNUM_PATTERN.sub(" ", compact).strip()


def canonicalize_job_link(url: str) -> str:
    raw_url = str(url or "").strip()
    if not raw_url:
        return ""

    parsed = urlparse(raw_url)
    if not parsed.scheme and not parsed.netloc:
        return raw_url

    filtered_query: list[tuple[str, str]] = []
    for key, values in parse_qs(parsed.query, keep_blank_values=False).items():
        lowered = key.lower()
        if lowered in TRACKING_QUERY_KEYS or any(lowered.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        for value in values:
            filtered_query.append((key, value))

    path = parsed.path.rstrip("/") or parsed.path
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            urlencode(sorted(filtered_query), doseq=True),
            "",
        )
    )


def fix_text_encoding(value: Any) -> str:
    """Attempt to repair common mojibake/encoding issues in scraped text.

    Tries a couple of common encode/decode fallbacks when replacement
    characters or typical mojibake markers are present. Returns the
    original string when no repair is possible.
    """
    import unicodedata

    s = str(value or "")
    if not s:
        return s
    # Quick heuristics: replacement char or common mojibake lead bytes
    # Normalize Unicode and replace common smart punctuation with ASCII
    try:
        s = unicodedata.normalize('NFKC', s)
    except Exception:
        pass
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2013', '-').replace('\u2014', '-')

    # Try ftfy if available (best-effort repair of mojibake and garbled text)
    if "�" in s or "Ã" in s or "Â" in s:
        try:
            import ftfy

            try:
                fixed = ftfy.fix_text(s)
                if fixed:
                    return fixed
            except Exception:
                pass
        except Exception:
            pass

        try:
            return s.encode("latin1").decode("utf-8")
        except Exception:
            try:
                return s.encode("utf-8").decode("latin1")
            except Exception:
                # Final fallback: replace replacement character with apostrophe
                return s.replace("�", "'")

    return s


def normalize_canadian_province(value: Any) -> str:
    text = fix_text_encoding(value).strip()
    if not text:
        return ""
    normalized = SPACE_PATTERN.sub(" ", text).strip().rstrip(".")
    lowered = normalized.lower()
    if lowered in CANADIAN_PROVINCE_NAMES:
        return CANADIAN_PROVINCE_NAMES[lowered]
    upper = lowered.upper()
    if upper in CANADIAN_PROVINCE_CODES:
        return upper
    return normalized


def glassdoor_location_from_detail_page(session: requests.Session, job_url: str, fallback_location: str) -> str:
    fallback = fix_text_encoding(fallback_location).strip()
    detail_url = str(job_url or "").strip()
    if not detail_url:
        return fallback

    try:
        resp = session.get(detail_url, timeout=15)
    except Exception:
        return fallback

    if resp.status_code != 200:
        return fallback

    text = resp.text or ""
    city = ""
    region = ""

    match = GLASSDOOR_LOCATION_LOCALITY_PATTERN.search(text)
    if match:
        city = fix_text_encoding(match.group(1)).strip()

    match = GLASSDOOR_LOCATION_REGION_PATTERN.search(text)
    if match:
        region = normalize_canadian_province(match.group(1))

    if city and region:
        return f"{city}, {region}"
    if city:
        return city
    return fallback


def glassdoor_base_url(country_indeed: str | None, location: str) -> str:
    country = normalize_indeed_country(country_indeed, location)
    if country == "CANADA":
        return "https://www.glassdoor.ca"
    return "https://www.glassdoor.com"


def glassdoor_card_age_hours(job_card: Any) -> int | None:
    text = fix_text_encoding(job_card.get_text(" ", strip=True)).lower()
    if not text:
        return None
    if any(token in text for token in ("just posted", "today", "new")):
        return 0
    match = GLASSDOOR_CARD_AGE_PATTERN.search(text)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    return amount if unit == "h" else amount * 24


def job_row_dedupe_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    link = canonicalize_job_link(str(row.get("job_url") or row.get("url") or row.get("job_url_direct") or ""))
    if link:
        keys.append(f"link:{link}")

    title = normalize_job_text(row.get("title"))
    company = normalize_job_text(row.get("company"))
    location = normalize_job_text(row.get("location"))
    if title and company:
        keys.append(f"role:{title}|{company}|{location}")
    elif title and location:
        keys.append(f"role:{title}|{location}")
    elif title:
        keys.append(f"role:{title}")
    return keys


def collect_job_sites(row: dict[str, Any]) -> list[str]:
    raw_sites = row.get("_source_sites") or row.get("sites") or row.get("site") or row.get("_source_site")
    if isinstance(raw_sites, (list, tuple, set)):
        values = [str(site).strip().lower() for site in raw_sites]
    else:
        values = [str(raw_sites).strip().lower()] if raw_sites else []

    sites: list[str] = []
    for site in values:
        if site and site not in sites:
            sites.append(site)
    return sites


def site_labels_for_sites(site_names: list[str]) -> str:
    labels = [JOBSPY_SITE_LABELS.get(site, site.replace("_", " ").title()) for site in site_names if site]
    return ", ".join(labels)


def dedupe_job_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_rows_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        keys = job_row_dedupe_keys(row)
        existing = next((seen_rows_by_key[key] for key in keys if key in seen_rows_by_key), None)
        if existing is not None:
            merged_sites = collect_job_sites(existing)
            for site in collect_job_sites(row):
                if site not in merged_sites:
                    merged_sites.append(site)
            if merged_sites:
                existing["_source_sites"] = merged_sites
            continue

        normalized = dict(row)
        source_sites = collect_job_sites(normalized)
        if source_sites:
            normalized["_source_sites"] = source_sites
        deduped.append(normalized)
        for key in keys:
            seen_rows_by_key[key] = normalized
    return deduped


def build_google_search_term(keywords: str, location: str, hours_old: int) -> str:
    base = str(keywords or "jobs").strip() or "jobs"
    if location:
        base = f"{base} jobs in {location}"
    else:
        base = f"{base} jobs"

    if hours_old <= 24:
        return f"{base} since yesterday"
    if hours_old <= 72:
        return f"{base} past 3 days"
    return base


def parse_proxy_pool(raw: str | None) -> list[str]:
    if not raw:
        return []
    normalized = raw.replace("\n", ",").replace(";", ",")
    proxies: list[str] = []
    for part in normalized.split(","):
        candidate = part.strip()
        if candidate and candidate not in proxies:
            proxies.append(candidate)
    return proxies


def pick_proxy(proxy_pool: list[str], cursor: int) -> str | None:
    if not proxy_pool:
        return None
    return proxy_pool[cursor % len(proxy_pool)]


def build_keyword_variants(keywords: str) -> list[str]:
    import re
    normalized = SPACE_PATTERN.sub(" ", str(keywords or "").strip())
    if not normalized:
        return ["jobs"]

    # Regex to match quoted phrases or single words
    token_pattern = re.compile(r'"([^"]+)"|(\S+)')
    tokens = []
    for match in token_pattern.finditer(normalized):
        if match.group(1):
            tokens.append(match.group(1))  # Quoted phrase
        elif match.group(2):
            tokens.append(match.group(2))  # Unquoted word

    if len(tokens) <= 1:
        return [normalized]

    last_term = tokens[-1]
    variants: list[str] = []

    def add_variant(value: str) -> None:
        cleaned = SPACE_PATTERN.sub(" ", str(value or "").strip())
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

    # First pass should always use the user's full keyword phrase.
    add_variant(normalized)

    # Then add bounded fallback variants that preserve the final role token.
    # This is intentionally capped so keyword expansion stays finite and predictable.
    for token in tokens[:-1]:
        if len(variants) >= MAX_KEYWORD_VARIANTS:
            break
        add_variant(f"{token} {last_term}")

    return variants


def first_query_value(query: dict[str, list[str]], *keys: str, default: str = "") -> str:
    for key in keys:
        values = query.get(key)
        if not values:
            continue
        value = str(values[0]).strip()
        if value:
            return value
    return default


def normalize_jobbank_search_params(
    keywords: str,
    location: str,
    search_query: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "searchstring": str(keywords or "jobs").strip() or "jobs",
        "locationstring": str(location or "Canada").strip() or "Canada",
    }
    if not search_query:
        return params

    for key, values in search_query.items():
        query_key = str(key or "").strip()
        if not query_key:
            continue
        lowered = query_key.lower()
        if lowered in {"searchstring", "locationstring", "q", "keywords", "l", "location"}:
            continue

        cleaned_values = [str(value).strip() for value in values if str(value).strip()]
        if not cleaned_values:
            continue

        # Preserve native Job Bank filters. Keep all provided values for repeated keys.
        params[query_key] = cleaned_values if len(cleaned_values) > 1 else cleaned_values[0]

    return params


def parse_jobbank_native_query(raw_query: str) -> dict[str, list[str]]:
    raw = str(raw_query or "").strip()
    if not raw:
        return {}
    normalized = raw[1:] if raw.startswith("?") else raw
    parsed = parse_qs(normalized, keep_blank_values=False)
    return {
        str(key): [str(value).strip() for value in values if str(value).strip()]
        for key, values in parsed.items()
        if str(key).strip()
    }


def effective_jobbank_native_query(custom_query: str) -> dict[str, list[str]]:
    defaults = parse_jobbank_native_query(JOBBANK_DEFAULT_NATIVE_QUERY)
    custom = parse_jobbank_native_query(custom_query)
    merged: dict[str, list[str]] = {key: list(values) for key, values in defaults.items()}

    for key, values in custom.items():
        existing = merged.setdefault(key, [])
        for value in values:
            if value not in existing:
                existing.append(value)

    return merged


def paginate_jobbank_posting_links(
    session: requests.Session,
    base_params: dict[str, Any],
    target_links: int,
) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()

    start_page_raw = str(base_params.get("page") or "").strip()
    try:
        start_page = max(1, int(start_page_raw)) if start_page_raw else 1
    except ValueError:
        start_page = 1

    for page_offset in range(JOBBANK_MAX_SEARCH_PAGES):
        page_params = dict(base_params)
        page_params["page"] = str(start_page + page_offset)
        try:
            search_response = session.get(JOBBANK_SEARCH_URL, params=page_params, timeout=20)
            search_response.raise_for_status()
        except Exception:
            if page_offset == 0:
                return []
            break

        remaining = max(1, target_links - len(links))
        page_links = extract_jobbank_posting_links(search_response.text, max_links=remaining)
        if not page_links:
            break

        added_this_page = 0
        for detail_url in page_links:
            if detail_url in seen:
                continue
            seen.add(detail_url)
            links.append(detail_url)
            added_this_page += 1
            if len(links) >= target_links:
                return links

        if added_this_page == 0:
            break

    return links


def normalize_requested_sites(site_names: list[str], configured_python_exe: str | None = None) -> list[str]:
    normalized_sites: list[str] = []
    for site in site_names:
        candidate = str(site).strip().lower()
        if candidate and candidate not in normalized_sites:
            normalized_sites.append(candidate)

    if "all" in normalized_sites:
        normalized_sites = all_supported_job_sites(configured_python_exe)

    custom_sites = set(CUSTOM_SCRAPER_SITES)
    runtime_sites = set(jobspy_runtime_metadata(configured_python_exe).get("sites", ()))
    if runtime_sites:
        normalized_sites = [site for site in normalized_sites if site in runtime_sites or site in custom_sites]
    return normalized_sites


def all_supported_job_sites(configured_python_exe: str | None = None) -> list[str]:
    """Return every scraper site this runtime can query, ensuring Indeed is present."""
    ordered: list[str] = [JOBBANK_CANADA_SITE, "glassdoor", "zip_recruiter"]
    for site in supported_jobspy_sites(configured_python_exe):
        candidate = str(site).strip().lower()
        if candidate and candidate not in ordered:
            ordered.append(candidate)

    if "indeed" not in ordered:
        ordered.append("indeed")

    from services.ats_service import ATS_PLATFORMS
    for ats_site in ATS_PLATFORMS:
        if ats_site not in ordered:
            ordered.append(ats_site)

    return ordered


def normalize_jobbank_posting_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return ""
    match = JOBBANK_POSTING_ID_PATTERN.search(url)
    if not match:
        return ""
    return f"{JOBBANK_BASE_URL}/jobsearch/jobposting/{match.group(1)}"


def extract_jobbank_posting_links(search_html: str, max_links: int) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for match in JOBBANK_POSTING_LINK_PATTERN.findall(search_html or ""):
        clean_url = normalize_jobbank_posting_url(match)
        if not clean_url or clean_url in seen:
            continue
        seen.add(clean_url)
        links.append(clean_url)
        if len(links) >= max(1, max_links):
            break
    return links


def lines_from_html(html: str) -> list[str]:
    text = BeautifulSoup(html or "", "html.parser").get_text("\n", strip=True)
    return [line.strip() for line in text.splitlines() if line.strip()]


def value_after_label(lines: list[str], label: str, lookahead: int = 5) -> str:
    target = str(label or "").strip().lower()
    if not target:
        return ""
    for index, line in enumerate(lines):
        if line.strip().lower() != target:
            continue
        for candidate in lines[index + 1 : index + 1 + max(1, lookahead)]:
            cleaned = candidate.strip()
            if cleaned and cleaned.lower() != target and cleaned != ",":
                return cleaned
    return ""


def jobbank_location_from_lines(lines: list[str]) -> str:
    for index, line in enumerate(lines):
        if line.strip().lower() != "location":
            continue

        parts: list[str] = []
        for token in lines[index + 1 : index + 8]:
            cleaned = token.strip()
            lowered = cleaned.lower()
            if not cleaned:
                continue
            if lowered in JOBBANK_LOCATION_STOP_WORDS:
                break
            parts.append(cleaned)
            if len(parts) >= 3:
                break

        if parts:
            return " ".join(parts).replace(" , ", ", ")
    return ""


def extract_jobbank_contacts(html: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(html or "", "html.parser")
    lines = lines_from_html(html)
    text = "\n".join(lines)

    emails: set[str] = set()
    phones: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        lowered = href.lower()
        if lowered.startswith("mailto:"):
            email = unquote(href[7:].split("?", 1)[0]).strip()
            if email and "jobbank" not in email.lower() and "example" not in email.lower():
                emails.add(email)
        elif lowered.startswith("tel:"):
            phone = unquote(href[4:].split("?", 1)[0]).strip()
            if phone:
                phones.add(phone)

    emails.update(
        {
            value
            for value in JOBBANK_EMAIL_PATTERN.findall(text)
            if "jobbank" not in value.lower() and "example" not in value.lower()
        }
    )
    phones.update({value.strip() for value in JOBBANK_PHONE_PATTERN.findall(text)})

    return sorted({value.strip() for value in emails if value.strip()}), sorted(
        {value.strip() for value in phones if value.strip()}
    )


def extract_jobbank_apply_link(*html_docs: str) -> str:
    selectors = (
        "a#externalJobLink[href]",
        "#externalJobLink[href]",
        "a[data-exturl]",
        "a[href*='apply']",
    )
    for html in html_docs:
        soup = BeautifulSoup(html or "", "html.parser")
        for selector in selectors:
            for anchor in soup.select(selector):
                href = str(anchor.get("href") or "").strip()
                if href.startswith("http"):
                    return href
                data_ext = str(anchor.get("data-exturl") or "").strip()
                if data_ext.startswith("http"):
                    return data_ext

        for anchor in soup.select("a[href]"):
            href = str(anchor.get("href") or "").strip()
            if href.startswith("http") and "jobbank.gc.ca" not in href.lower():
                return href
    return ""


def normalize_description_text(raw_value: Any, max_chars: int = SEMANTIC_DESCRIPTION_CHAR_LIMIT) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""

    # Convert html-ish text into plain text so embeddings compare semantics, not markup noise.
    if "<" in text and ">" in text:
        try:
            text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        except Exception:
            pass

    text = SPACE_PATTERN.sub(" ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def compact_job_description(
    posting_url: str,
    title: str,
    company: str | None = None,
    location: str | None = None,
    *,
    site: str | None = None,
    row: dict[str, Any] | None = None,
    detail_text: str | None = None,
    max_chars: int = 7000,
) -> str:
    """Build a compact plain-text job description payload for downstream consumers.

    Works for both direct page scraping (`detail_text`) and structured row results
    from board scrapers (`row`) so callers can use one consistent pathway.
    """
    safe_limit = max(512, int(max_chars))
    row = row or {}

    resolved_site = str(site or row.get("site") or row.get("_source_site") or "").strip()
    resolved_title = str(title or row.get("title") or "Job Posting").strip() or "Job Posting"
    resolved_company = str(company or row.get("company") or "").strip()
    resolved_location = str(location or row.get("location") or "").strip()

    apply_link = str(row.get("apply_link") or "").strip()
    contact_emails = [str(value).strip() for value in row.get("contact_emails", []) if str(value).strip()]
    contact_phones = [str(value).strip() for value in row.get("contact_phones", []) if str(value).strip()]

    detail_candidates = [
        row.get("description"),
        row.get("job_description"),
        row.get("summary"),
        row.get("snippet"),
        detail_text,
    ]
    description_body = ""
    for candidate in detail_candidates:
        normalized = normalize_description_text(candidate, max_chars=safe_limit)
        if normalized:
            description_body = normalized
            break

    lines = [f"Posting URL: {posting_url}", f"Title: {resolved_title}"]
    if resolved_site:
        lines.insert(0, f"Source site: {resolved_site}")
    if resolved_company:
        lines.append(f"Company: {resolved_company}")
    if resolved_location:
        lines.append(f"Location: {resolved_location}")
    if apply_link:
        lines.append(f"Apply link: {apply_link}")
    if contact_emails:
        lines.append(f"Contact emails: {', '.join(contact_emails[:3])}")
    if contact_phones:
        lines.append(f"Contact phones: {', '.join(contact_phones[:3])}")
    if description_body:
        lines.append("Description:")
        lines.append(description_body)

    compact = "\n".join(line for line in lines if str(line).strip())
    if len(compact) <= safe_limit:
        return compact
    return compact[:safe_limit].rstrip()


def extract_jobbank_description(detail_html: str) -> str:
    soup = BeautifulSoup(detail_html or "", "html.parser")
    selectors = (
        "section#jobdetails",
        "div#job-details",
        "div.jobposting-details",
        "main",
    )
    for selector in selectors:
        block = soup.select_one(selector)
        if block is None:
            continue
        cleaned = normalize_description_text(block.get_text(" ", strip=True))
        if len(cleaned) >= 80:
            return cleaned

    return normalize_description_text("\n".join(lines_from_html(detail_html)))


def build_jobbank_posting_row(session: requests.Session, detail_url: str, detail_html: str) -> dict[str, Any]:
    detail_lines = lines_from_html(detail_html)
    title = BeautifulSoup(detail_html, "html.parser").select_one("h1")
    company = value_after_label(detail_lines, "Employer details")
    location_text = jobbank_location_from_lines(detail_lines)

    apply_html = fetch_jobbank_apply_html(session, detail_url, detail_html)
    contact_emails, contact_phones = extract_jobbank_contacts(apply_html)
    apply_link = extract_jobbank_apply_link(detail_html, apply_html)
    description = extract_jobbank_description(detail_html)

    row: dict[str, Any] = {
        "title": title.get_text(" ", strip=True) if title else "(job)",
        "company": company,
        "location": location_text,
        "job_url": normalize_jobbank_posting_url(detail_url) or detail_url,
        "_source_site": JOBBANK_CANADA_SITE,
        "_source_sites": [JOBBANK_CANADA_SITE],
    }
    if contact_emails:
        row["contact_emails"] = contact_emails
    if contact_phones:
        row["contact_phones"] = contact_phones
    if apply_link:
        row["apply_link"] = apply_link
    if description:
        row["description"] = description
    return row


def fetch_jobbank_apply_html(session: requests.Session, detail_url: str, detail_html: str) -> str:
    soup = BeautifulSoup(detail_html or "", "html.parser")
    form = soup.select_one("form#externallinkactivity")
    if form is None:
        return detail_html

    action = str(form.get("action") or "/jobsearch/pers/jobposting.xhtml").strip()
    job_link = soup.select_one("#externalJobLink")
    js_job_id = str(job_link.get("data-jsjobid") or "").strip() if job_link else ""

    payload = {
        str(input_tag.get("name")): str(input_tag.get("value") or "")
        for input_tag in form.select("input[name]")
        if input_tag.get("name")
    }
    payload.update({"jsJobId": js_job_id, "action": "applynowbutton", "jobid": js_job_id})

    post_url = requests.compat.urljoin(detail_url, action)
    try:
        response = session.post(post_url, data=payload, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception:
        return detail_html


def scrape_jobbank_canada_postings(
    keywords: str,
    location: str,
    results_wanted: int,
    search_query: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": JOBBANK_USER_AGENT_OVERRIDE or JOBBANK_USER_AGENT})

    params = normalize_jobbank_search_params(keywords, location, search_query)
    posting_links = paginate_jobbank_posting_links(session, params, target_links=max(5, results_wanted * 3))
    if not posting_links:
        return []

    rows: list[dict[str, Any]] = []
    for detail_url in posting_links:
        if len(rows) >= max(1, results_wanted):
            break
        try:
            detail_response = session.get(detail_url, timeout=20)
            detail_response.raise_for_status()
        except Exception:
            continue

        rows.append(build_jobbank_posting_row(session, detail_response.url, detail_response.text))

    return rows


def scrape_jobbank_canada_posting(detail_url: str) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": JOBBANK_USER_AGENT_OVERRIDE or JOBBANK_USER_AGENT})
    try:
        detail_response = session.get(detail_url, timeout=HTTP_REQUEST_TIMEOUT)
        detail_response.raise_for_status()
    except Exception:
        return []

    return [build_jobbank_posting_row(session, detail_response.url, detail_response.text)]


def build_site_scrape_code(
    site: str,
    keywords: str,
    location: str,
    results_wanted: int,
    hours_old: int,
    radius_miles: int,
    resolved_country_indeed: str,
    google_search_term: str,
    selected_proxy: str | None,
) -> str:
    return (
        "import json\n"
        "import inspect\n"
        "from jobspy import scrape_jobs\n"
        f"site_name={site!r}\n"
        "supported_sites=set()\n"
        "try:\n"
        "    from jobspy import Site as _JobSpySite\n"
        "    members=getattr(_JobSpySite, '__members__', {})\n"
        "    supported_sites={str(name).lower() for name in members.keys()}\n"
        "except Exception:\n"
        "    supported_sites=set()\n"
        "\nif supported_sites and site_name not in supported_sites:\n"
        "    print('[]')\n"
        "    raise SystemExit(0)\n"
        "params=set(inspect.signature(scrape_jobs).parameters)\n"
        f"kwargs=dict(site_name=[site_name], search_term={keywords!r}, location={location!r}, results_wanted={max(1, results_wanted)})\n"
        f"if 'hours_old' in params: kwargs['hours_old']={max(1, hours_old)}\n"
        f"if 'distance' in params and site_name not in {DISTANCE_UNSUPPORTED_SITES!r}: kwargs['distance']={max(1, radius_miles)}\n"
        f"if 'country_indeed' in params and site_name in {{'indeed', 'glassdoor'}}: kwargs['country_indeed']={resolved_country_indeed!r}\n"
        f"if 'google_search_term' in params and site_name == 'google': kwargs['google_search_term']={google_search_term!r}\n"
        f"selected_proxy={selected_proxy!r}\n"
        "if selected_proxy:\n"
        "    if 'proxies' in params: kwargs['proxies']=[selected_proxy]\n"
        "    elif 'proxy' in params: kwargs['proxy']=selected_proxy\n"
        "\nif site_name == 'bdjobs':\n"
        "    # Proactive patch: some JobSpy builds pass user_agent to BDJobs.__init__, which it doesn't accept.\n"
        "    try:\n"
        "        from jobspy.bdjobs import BDJobs\n"
        "        original_init = BDJobs.__init__\n"
        "        init_params = set(inspect.signature(original_init).parameters)\n"
        "\n        def patched_init(self, *args, **inner_kwargs):\n"
        "            filtered = {k: v for k, v in inner_kwargs.items() if k in init_params}\n"
        "            return original_init(self, *args, **filtered)\n"
        "\n        BDJobs.__init__ = patched_init\n"
        "    except Exception:\n"
        "        pass\n"
        "\ndef run_with_bdjobs_patch(run_kwargs):\n"
        "    try:\n"
        "        return scrape_jobs(**run_kwargs)\n"
        "    except TypeError as exc:\n"
        "        # Workaround for python-jobspy BDJobs constructor mismatch in some versions.\n"
        "        if site_name == 'bdjobs' and 'unexpected keyword argument' in str(exc):\n"
        "            from jobspy.bdjobs import BDJobs\n"
        "            original_init = BDJobs.__init__\n"
        "            init_params = set(inspect.signature(original_init).parameters)\n"
        "\n            def patched_init(self, *args, **inner_kwargs):\n"
        "                filtered = {k: v for k, v in inner_kwargs.items() if k in init_params}\n"
        "                return original_init(self, *args, **filtered)\n"
        "\n            BDJobs.__init__ = patched_init\n"
        "            return scrape_jobs(**run_kwargs)\n"
        "        raise\n"
        "\ndef execute_with_site_workarounds(base_kwargs):\n"
        "    if site_name != 'glassdoor':\n"
        "        return run_with_bdjobs_patch(base_kwargs)\n"
        "\n    # Glassdoor often fails location parsing and certain filters; retry with a simpler query shape.\n"
        "    # Print-and-exit as soon as the first attempt yields results so the subprocess output is\n"
        "    # flushed before the 90-second kill window; later attempts only run if the earlier ones\n"
        "    # came back empty.\n"
        "    location_value = str(base_kwargs.get('location') or '').strip()\n"
        "    city_only = location_value.split(',')[0].strip() if location_value else location_value\n"
        "    country_value = str(base_kwargs.get('country_indeed') or '').strip()\n"
        "    attempts = [dict(base_kwargs)]\n"
        "\n    normalized = dict(base_kwargs)\n"
        "    if city_only:\n"
        "        normalized['location'] = city_only\n"
        "    if country_value:\n"
        "        normalized['country_indeed'] = country_value.title()\n"
        "    attempts.append(normalized)\n"
        "\n    relaxed = dict(normalized)\n"
        "    relaxed.pop('hours_old', None)\n"
        "    relaxed.pop('distance', None)\n"
        "    attempts.append(relaxed)\n"
        "\n    last_jobs = None\n"
        "    for attempt_kwargs in attempts:\n"
        "        jobs = run_with_bdjobs_patch(attempt_kwargs)\n"
        "        if jobs is not None and len(jobs) > 0:\n"
        "            rows = jobs.to_dict('records')\n"
        "            print(json.dumps(rows, ensure_ascii=True, default=str))\n"
        "            raise SystemExit(0)\n"
        "        last_jobs = jobs\n"
        "    return last_jobs\n"
        "\njobs=execute_with_site_workarounds(kwargs)\n"
        "rows=[] if jobs is None else jobs.to_dict('records')\n"
        "print(json.dumps(rows, ensure_ascii=True, default=str))\n"
    )


def run_site_scrape_subprocess(python_executable: Path, site: str, code: str) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [str(python_executable), "-c", code],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_SCRAPE_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"Job site '{site}' timed out after {SUBPROCESS_SCRAPE_TIMEOUT}s; skipping variant")
        return []
    if proc.returncode != 0:
        stderr_lines = [line.strip() for line in (proc.stderr or "").splitlines() if line.strip()]
        if stderr_lines:
            print(f"Job site '{site}' failed: {stderr_lines[-1]}")
        return []

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return []

    try:
        rows = json.loads(lines[-1])
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def shape_job_item(row: dict[str, Any], source: str) -> dict[str, Any]:
    title = fix_text_encoding(row.get("title") or "(job)")
    company = fix_text_encoding(row.get("company") or "").strip()
    location_text = fix_text_encoding(row.get("location") or "").strip()
    source_sites = collect_job_sites(row)
    source_label = site_labels_for_sites(source_sites)
    from services.ats_service import ATS_PLATFORMS as _ats_platforms
    _ats_set = set(_ats_platforms)
    _company_lower = company.lower().replace("-", " ").replace("_", " ").strip()
    _is_job_board_name = _company_lower in JOBSPY_SITE_LABELS or _company_lower in {s.replace("_", " ") for s in JOBSPY_SITE_LABELS}
    if company and source_sites and source_sites[0] in _ats_set and not _is_job_board_name:
        source_label = f"{source_label}/{company.replace('-', ' ').title()}"
    suffix = ", ".join(part for part in [company, location_text] if part)
    if suffix:
        title = f"{title} ({suffix})"

    # Final normalization: replace replacement characters and collapse whitespace
    title = title.replace('\uFFFD', 'e')
    title = SPACE_PATTERN.sub(' ', title).strip()

    link = canonicalize_job_link(str(row.get("job_url") or row.get("url") or row.get("job_url_direct") or source)) or source
    item = {
        "title": title[:300],
        "link": link,
        "source_url": source,
        "type": "job",
        "site": source_sites[0] if source_sites else None,
        "sites": source_sites,
        "site_label": source_label,
    }
    description = normalize_description_text(
        row.get("description") or row.get("snippet") or row.get("summary") or row.get("job_description")
    )
    if description:
        item["description"] = description
    contact_emails = [str(value).strip() for value in row.get("contact_emails", []) if str(value).strip()]
    contact_phones = [str(value).strip() for value in row.get("contact_phones", []) if str(value).strip()]
    if contact_emails:
        item["contact_emails"] = contact_emails
    if contact_phones:
        item["contact_phones"] = contact_phones
    apply_link = str(row.get("apply_link") or "").strip()
    if apply_link:
        item["apply_link"] = apply_link
    return item


def scrape_glassdoor_postings(
    keywords: str,
    location: str,
    hours_old: int | None = None,
    radius_miles: int | None = None,
    country_indeed: str | None = None,
    results_wanted: int = 20,
) -> list[dict[str, Any]]:
    """
    Glassdoor job scraper that bypasses 403 Forbidden by:
    1. First fetching homepage to get session cookies
    2. Then requesting search results with valid session
    3. Parsing job listings with tested CSS selectors
    """
    results = []
    try:
        hours_old = 168 if hours_old is None else int(hours_old)
        radius_miles = 25 if radius_miles is None else int(radius_miles)
        country_indeed = country_indeed or "AUTO"

        # Create session with realistic browser headers
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })
        
        # Step 1: Get cookies from homepage (required to bypass 403)
        try:
            session.get("https://www.glassdoor.com/", timeout=10)
        except Exception:
            pass  # Continue anyway

        # Resolve location to Glassdoor's internal locId/locT via autocomplete API
        from services.jba.geo_db import resolve_glassdoor_location
        loc_id, loc_type = resolve_glassdoor_location(location, session)

        # Step 2: Request job search page with valid session
        search_url = f"{glassdoor_base_url(country_indeed, location)}/Job/jobs.htm"
        params = {
            "sc.keyword": keywords,
            "locT": loc_type,
            "locId": loc_id,
        }
        if radius_miles:
            params["radius"] = str(max(1, int(radius_miles)))
        url_str = f"{search_url}?{urlencode(params)}"
        
        resp = session.get(url_str, timeout=15)
        
        # Only process if we got a successful response
        if resp.status_code != 200:
            return results
        
        # Ensure requests uses the best available encoding before parsing
        try:
            resp.encoding = resp.apparent_encoding or "utf-8"
        except Exception:
            resp.encoding = "utf-8"
        # Prefer parsing from raw bytes so BeautifulSoup can detect encoding
        soup = BeautifulSoup(resp.content or resp.text, "html.parser")
        
        # Step 3: Extract job listings using tested CSS selectors
        # .jobCard is the container, tested and working
        for job_card in soup.select('.jobCard'):
            try:
                card_age_hours = glassdoor_card_age_hours(job_card)
                if card_age_hours is not None and card_age_hours > max(1, int(hours_old)):
                    continue

                # Extract title - works with tested selector
                title_elem = job_card.select_one('a[class*="jobtitle"], a[class*="jobTitle"]')
                if not title_elem:
                    continue
                title_text = title_elem.get_text(strip=True)
                title_text = fix_text_encoding(title_text)
                if not title_text:
                    continue
                
                # Extract company name - tested selector
                company_elem = job_card.select_one('span[class*="EmployerProfile_compactEmployerName"]')
                company_text = company_elem.get_text(strip=True) if company_elem else ""
                company_text = fix_text_encoding(company_text)
                
                # Extract location - try multiple selectors
                location_elem = job_card.select_one('[class*="jobLocation"]') or \
                                job_card.select_one('[data-test="job-location"]') or \
                                job_card.select_one('[class*="location"]')
                location_text = location_elem.get_text(strip=True) if location_elem else location
                location_text = fix_text_encoding(location_text)
                
                # Extract job URL - tested selector that works
                link_elem = job_card.select_one('a[href*="/job-listing"]')
                if not link_elem:
                    continue
                job_url = link_elem.get("href", "")
                if not job_url:
                    continue
                
                # Make absolute URL if relative
                if job_url and not job_url.startswith("http"):
                    job_url = f"https://www.glassdoor.com{job_url}"

                enriched_location = glassdoor_location_from_detail_page(session, job_url, location_text)
                
                results.append({
                    "title": title_text,
                    "company": company_text,
                    "location": enriched_location,
                    "job_url": job_url,
                    "url": job_url,
                    "job_url_direct": job_url,
                    "_source_site": "glassdoor",
                    "_source_sites": ["glassdoor"],
                })
                if len(results) >= results_wanted:
                    break
            except Exception:
                # Skip individual job extraction errors
                continue
                
    except Exception as e:
        # Silently handle errors - Glassdoor may block or change structure
        pass
    
    return results


def scrape_ziprecruiter_postings(
    keywords: str,
    location: str,
    results_wanted: int = 20,
) -> list[dict[str, Any]]:
    """Scrape ZipRecruiter using Playwright to bypass JS/Cloudflare protections.

    Falls back to returning [] if Playwright is not available.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    try:
        search_url = (
            "https://www.ziprecruiter.com/candidate/search?" +
            f"search={urlencode({'': keywords})[1:]}&location={urlencode({'': location})[1:]}"
        )
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
            # Navigate without waiting for networkidle (Cloudflare may block that); wait for DOM
            # Visit homepage first to establish cookies and bypass challenge
            try:
                page.goto('https://www.ziprecruiter.com/', wait_until='domcontentloaded', timeout=30000)
                page.wait_for_timeout(1500)
            except Exception:
                pass
            page.goto(search_url, wait_until='domcontentloaded', timeout=45000)

            # Try several selectors; job link anchors are preferred
            selectors = ['a[data-qa="job-link"]', 'a.job_link', 'article.job_result', 'div.justified-job-card']
            elements = []
            for sel in selectors:
                try:
                    page.wait_for_selector(sel, timeout=10000)
                    elements = page.query_selector_all(sel)
                except Exception:
                    elements = []
                if elements:
                    break

            # If nothing found yet, try scrolling to trigger lazy load
            if not elements:
                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(2000)
                except Exception:
                    pass
                for sel in selectors:
                    try:
                        page.wait_for_selector(sel, timeout=5000)
                        elements = page.query_selector_all(sel)
                    except Exception:
                        elements = []
                    if elements:
                        break

            for el in (elements[:results_wanted] if elements else []):
                try:
                    href = el.get_attribute('href') or ''
                    title = el.inner_text().strip() or ''
                    title = fix_text_encoding(title)
                    # If anchor is nested, try to extract title from child
                    if not title:
                        child = el.query_selector('h3') or el.query_selector('h2')
                        title = child.inner_text().strip() if child else ''

                    # If we got an article/div, try to find nested link
                    if href and not href.startswith('http'):
                        href = 'https://www.ziprecruiter.com' + href

                    company = ''
                    try:
                        comp = el.query_selector('[data-qa="company-name"]') or el.query_selector('.company')
                        if comp:
                            company = comp.inner_text().strip()
                            company = fix_text_encoding(company)
                    except Exception:
                        company = ''

                    location_text = ''
                    try:
                        loc = el.query_selector('[data-qa="location"]') or el.query_selector('.location')
                        if loc:
                            location_text = loc.inner_text().strip()
                            location_text = fix_text_encoding(location_text)
                    except Exception:
                        location_text = location

                    if title and href:
                        results.append({
                            'title': title,
                            'company': company,
                            'location': location_text or location,
                            'job_url': href,
                            '_source_site': 'zip_recruiter',
                            '_source_sites': ['zip_recruiter'],
                        })
                except Exception:
                    continue
            try:
                browser.close()
            except Exception:
                pass
    except Exception:
        return []
    return results


def scrape_job_postings(
    site_names: list[str],
    keywords: str,
    location: str,
    configured_python_exe: str | None = None,
    hours_old: int = 168,
    results_wanted: int = 20,
    radius_miles: int = 25,
    country_indeed: str = "USA",
    source_url: str | None = None,
    allow_north_america: bool = False,
    jobbank_search_query: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    if site_names:
        normalized_sites = normalize_requested_sites(site_names, configured_python_exe)
    else:
        normalized_sites = all_supported_job_sites(configured_python_exe)

    if not normalized_sites:
        return []

    raw: list[dict[str, Any]] = []
    target_count = max(1, results_wanted)

    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    scrape_tasks: list[Any] = []

    if JOBBANK_CANADA_SITE in normalized_sites:
        def _jobbank_task():
            with _site_scrape_slot(JOBBANK_CANADA_SITE) as acquired:
                if not acquired:
                    print(f"Skipping {JOBBANK_CANADA_SITE}: concurrency limit reached")
                    return []
                return scrape_jobbank_canada_postings(
                    keywords=keywords, location=location,
                    results_wanted=target_count, search_query=jobbank_search_query,
                )
        scrape_tasks.append(_jobbank_task)

    if "glassdoor" in normalized_sites:
        def _glassdoor_task():
            with _site_scrape_slot("glassdoor") as acquired:
                if not acquired:
                    print("Skipping glassdoor: concurrency limit reached")
                    return []
                return scrape_glassdoor_postings(
                    keywords, location, hours_old=hours_old,
                    radius_miles=radius_miles, country_indeed=country_indeed,
                    results_wanted=target_count,
                )
        scrape_tasks.append(_glassdoor_task)

    if "zip_recruiter" in normalized_sites:
        def _ziprec_task():
            with _site_scrape_slot("zip_recruiter") as acquired:
                if not acquired:
                    print("Skipping zip_recruiter: concurrency limit reached")
                    return []
                return scrape_ziprecruiter_postings(keywords, location, target_count)
        scrape_tasks.append(_ziprec_task)

    from services.ats_service import ATS_PLATFORMS as _ATS_PLATFORMS, _matches_keywords, _matches_location
    requested_ats = {s for s in normalized_sites if s in _ATS_PLATFORMS}
    if requested_ats:
        try:
            from services.jba.merge_data import load_daily_log, _today_str
            from datetime import date, timedelta
            today = date.today()
            seen_keys: set[str] = set()
            cached: list[dict[str, Any]] = []
            for delta in (0, 1):
                d = (today - timedelta(days=delta)).strftime("%Y-%m-%d")
                for row in load_daily_log(d):
                    key = row.get("job_url") or row.get("_dedup_key") or ""
                    if key and key not in seen_keys:
                        seen_keys.add(key)
                        cached.append(row)
            if cached:
                cached_rows = [
                    row for row in cached
                    if row.get("_source_site") in requested_ats
                    and _matches_keywords(str(row.get("title") or ""), keywords)
                    and _matches_location(str(row.get("location") or ""), location)
                    and _posting_age_ok(row, hours_old)
                ]
                if cached_rows:
                    raw.extend(cached_rows)
                    print(f"[jba-log] {len(cached_rows)} ATS jobs from 2-day window ({len(cached)} total)")
        except Exception as exc:
            print(f"[jba-log] Failed to read daily log: {exc}")

    from services.ats_service import ATS_PLATFORMS as _ATS_PLATS_SET
    _ats_names = set(_ATS_PLATS_SET)

    jobspy_sites = [site for site in normalized_sites if site not in CUSTOM_SCRAPER_SITES]
    python_executable = jobspy_python_executable(configured_python_exe)
    if jobspy_sites and python_executable is not None:
        resolved_country_indeed = normalize_indeed_country(country_indeed, location)
        proxy_pool = parse_proxy_pool(os.getenv("JOBSPY_PROXIES"))
        keyword_variants = build_keyword_variants(keywords)
        for site_idx, site in enumerate(jobspy_sites):
            for variant_index, keyword_variant in enumerate(keyword_variants):
                def _make_jobspy_task(si=site_idx, s=site, vi=variant_index, kv=keyword_variant):
                    def _task():
                        with _site_scrape_slot(s) as acquired:
                            if not acquired:
                                print(f"Skipping {s}: concurrency limit reached")
                                return []
                            proxy = pick_proxy(proxy_pool, si)
                            code = build_site_scrape_code(
                                site=s, keywords=kv, location=location,
                                results_wanted=results_wanted, hours_old=hours_old,
                                radius_miles=radius_miles,
                                resolved_country_indeed=resolved_country_indeed,
                                google_search_term=build_google_search_term(kv, location, hours_old),
                                selected_proxy=proxy,
                            )
                            rows = run_site_scrape_subprocess(python_executable, s, code)
                            for row in rows:
                                row.setdefault("_source_site", s)
                                row.setdefault("_source_sites", [s])
                                row.setdefault("_search_variant", kv)
                                row.setdefault("_search_variant_index", vi)
                            return rows
                    return _task
                scrape_tasks.append(_make_jobspy_task())

    if scrape_tasks:
        with ThreadPoolExecutor(max_workers=len(scrape_tasks)) as executor:
            futures = [executor.submit(t) for t in scrape_tasks]
            for future in _as_completed(futures):
                try:
                    raw.extend(future.result())
                except Exception as exc:
                    print(f"Scrape task error: {exc}")

    if not raw:
        return []

    deduped_rows = dedupe_job_rows(raw)
    filtered_rows = filter_rows_by_region(deduped_rows, allow_north_america=allow_north_america)
    source = source_url or f"jobspy:{','.join(normalized_sites)}"
    ats_rows = [r for r in filtered_rows if r.get("_source_site") in _ats_names]
    non_ats_rows = [r for r in filtered_rows if r.get("_source_site") not in _ats_names]
    return [shape_job_item(row, source) for row in non_ats_rows + ats_rows]


def scrape_job_descriptions_from_all_sites(
    keywords: str,
    location: str,
    site_names: list[str] | None = None,
    configured_python_exe: str | None = None,
    hours_old: int = 168,
    results_per_site: int = 5,
    max_descriptions: int = 60,
    radius_miles: int = 25,
    country_indeed: str = "AUTO",
    allow_north_america: bool = False,
    jobbank_search_query: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Scrape job descriptions across all supported sites, including Indeed.

    Returns one normalized record per posting with source metadata and a compact
    description payload suitable for downstream summarization/resume tailoring.
    """
    if site_names:
        sites = normalize_requested_sites(site_names, configured_python_exe)
    else:
        sites = all_supported_job_sites(configured_python_exe)
    target_per_site = max(1, int(results_per_site))
    max_total = max(1, int(max_descriptions))

    descriptions: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for site in sites:
        if len(descriptions) >= max_total:
            break

        site_items = scrape_job_postings(
            site_names=[site],
            keywords=keywords,
            location=location,
            configured_python_exe=configured_python_exe,
            hours_old=hours_old,
            results_wanted=target_per_site,
            radius_miles=radius_miles,
            country_indeed=country_indeed,
            allow_north_america=allow_north_america,
            jobbank_search_query=jobbank_search_query,
        )

        for item in site_items:
            posting_url = canonicalize_job_link(str(item.get("link") or item.get("source_url") or ""))
            title = fix_text_encoding(item.get("title") or "Job Posting")
            dedupe_key = posting_url or f"{site}|{normalize_job_text(title)}"
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            description_raw = compact_job_description(
                posting_url=posting_url or str(item.get("link") or ""),
                title=title,
                site=site,
                row=item,
                detail_text=item.get("description"),
            )

            entry: dict[str, Any] = {
                "site": site,
                "site_label": JOBSPY_SITE_LABELS.get(site, site.replace("_", " ").title()),
                "title": title,
                "link": posting_url or str(item.get("link") or ""),
                "description": description_raw,
            }

            source_sites = collect_job_sites(item)
            if source_sites:
                entry["sites"] = source_sites
            apply_link = str(item.get("apply_link") or "").strip()
            if apply_link:
                entry["apply_link"] = apply_link

            descriptions.append(entry)
            if len(descriptions) >= max_total:
                break

    return descriptions


def scrape_jobs_from_board_url(url: str, configured_python_exe: str | None = None, max_items: int = 20) -> list[dict[str, Any]]:
    site = job_site_from_url(url)
    if not site:
        return []

    if site == JOBBANK_CANADA_SITE:
        parsed = urlparse(url)
        if "/jobsearch/jobposting/" in parsed.path.lower():
            return scrape_jobbank_canada_posting(url)

        query = parse_qs(parsed.query)
        term = first_query_value(query, "searchstring", "q", "keywords", default="jobs")
        location = first_query_value(query, "locationstring", "l", "location", default="Canada")
        return scrape_jobbank_canada_postings(term, location, max_items, search_query=query)

    query = parse_qs(urlparse(url).query)
    term = first_query_value(query, "q", "keywords", default="python developer")
    location = first_query_value(query, "l", "location", default="United States")
    radius_raw = (query.get("radius") or query.get("distance") or query.get("rd") or ["25"])[0]
    try:
        radius_miles = max(1, int(str(radius_raw).strip()))
    except ValueError:
        radius_miles = 25
    return scrape_job_postings(
        site_names=[site],
        keywords=term,
        location=location,
        configured_python_exe=configured_python_exe,
        results_wanted=max_items,
        radius_miles=radius_miles,
        source_url=url,
    )


def matches_role_filters(title: str, role_filters: list[str]) -> bool:
    if not role_filters:
        return True
    import re as _re
    normalized = title.lower()
    keyword_map = {
        "internship": ["intern", "internship", "co-op", "coop"],
        "entry": ["entry", "entry-level", "new grad", "graduate", "fresher"],
        "junior": ["junior", "jr"],
        "mid": ["mid", "intermediate"],
        "senior": ["senior", "sr", "lead", "staff", "principal"],
    }
    for role in role_filters:
        for keyword in keyword_map.get(role, []):
            if _re.search(r'\b' + _re.escape(keyword) + r'\b', normalized):
                return True
    return False


def matches_exclusion_terms(job_data: dict[str, Any], exclusion_terms: list[str]) -> bool:
    """Check if job matches any exclusion terms (regex-based filter).
    
    Returns True if job should be EXCLUDED (matches exclusion terms).
    Searches in title and company name.
    Falls back to literal substring match if regex is invalid.
    """
    if not exclusion_terms:
        return False
    
    title = str(job_data.get("title") or "").lower()
    company = str(job_data.get("company") or "").lower()
    searchable = f"{title} {company}"
    
    for term in exclusion_terms:
        term = str(term).strip()
        if not term:
            continue
        
        try:
            # Try regex first
            pattern = re.compile(term, re.IGNORECASE)
            if pattern.search(searchable):
                return True
        except re.error:
            # Fall back to literal substring if regex is invalid
            if term.lower() in searchable:
                return True
    
    return False


def configure_job_service(config: Any) -> None:
    """Push settings.toml values (via AppConfig) into this module's globals.
    Call once from app.py after load_config().
    """
    global SEMANTIC_PLUGIN_ENABLED, SEMANTIC_PLUGIN_MODEL_NAME, SEMANTIC_PLUGIN_THRESHOLD, SEMANTIC_MATCH_TARGET
    global SEMANTIC_DESCRIPTION_CHAR_LIMIT, JOBBANK_MAX_SEARCH_PAGES, JOBBANK_USER_AGENT_OVERRIDE
    global DEDUP_MAX_FIFO_FILES, DEDUP_MAX_ENTRIES_PER_FILE, DEDUP_MONTHS_THRESHOLD, DEDUP_SEEN_LINKS_CAP
    global HTTP_REQUEST_TIMEOUT, SUBPROCESS_SCRAPE_TIMEOUT
    global SITE_CONCURRENCY_LIMIT, SITE_SEMAPHORE_TIMEOUT

    SEMANTIC_PLUGIN_ENABLED = bool(getattr(config, "semantic_enabled", SEMANTIC_PLUGIN_ENABLED))
    SEMANTIC_PLUGIN_MODEL_NAME = str(getattr(config, "semantic_model", SEMANTIC_PLUGIN_MODEL_NAME))
    SEMANTIC_PLUGIN_THRESHOLD = float(getattr(config, "semantic_threshold", SEMANTIC_PLUGIN_THRESHOLD))
    SEMANTIC_MATCH_TARGET = str(getattr(config, "semantic_match_target", SEMANTIC_MATCH_TARGET)).strip().lower()
    SEMANTIC_DESCRIPTION_CHAR_LIMIT = int(getattr(config, "semantic_description_char_limit", SEMANTIC_DESCRIPTION_CHAR_LIMIT))
    JOBBANK_MAX_SEARCH_PAGES = int(getattr(config, "jobbank_max_search_pages", JOBBANK_MAX_SEARCH_PAGES))
    JOBBANK_USER_AGENT_OVERRIDE = str(getattr(config, "jobbank_user_agent", JOBBANK_USER_AGENT))
    DEDUP_MAX_FIFO_FILES = int(getattr(config, "dedup_max_fifo_files", DEDUP_MAX_FIFO_FILES))
    DEDUP_MAX_ENTRIES_PER_FILE = int(getattr(config, "dedup_max_entries_per_file", DEDUP_MAX_ENTRIES_PER_FILE))
    DEDUP_MONTHS_THRESHOLD = int(getattr(config, "dedup_months_threshold", DEDUP_MONTHS_THRESHOLD))
    DEDUP_SEEN_LINKS_CAP = int(getattr(config, "dedup_seen_links_cap", DEDUP_SEEN_LINKS_CAP))
    HTTP_REQUEST_TIMEOUT = int(getattr(config, "http_timeout_seconds", HTTP_REQUEST_TIMEOUT))
    SUBPROCESS_SCRAPE_TIMEOUT = int(getattr(config, "subprocess_scrape_timeout_seconds", SUBPROCESS_SCRAPE_TIMEOUT))
    SITE_CONCURRENCY_LIMIT = int(getattr(config, "site_concurrency_limit", SITE_CONCURRENCY_LIMIT))
    SITE_SEMAPHORE_TIMEOUT = int(getattr(config, "site_semaphore_timeout_seconds", SITE_SEMAPHORE_TIMEOUT))
    with _site_semaphores_lock:
        _site_semaphores.clear()
    # Clear cached model so it re-loads with updated settings if needed
    load_semantic_plugin_model.cache_clear()
    semantic_plugin_available.cache_clear()


@lru_cache(maxsize=1)
def semantic_plugin_available() -> bool:
    if not SEMANTIC_PLUGIN_ENABLED:
        return False
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def load_semantic_plugin_model() -> Any | None:
    if not semantic_plugin_available():
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(SEMANTIC_PLUGIN_MODEL_NAME)
    except Exception as exc:
        print(f"Semantic plugin disabled (model load failed): {exc}")
        return None


def listing_text_for_semantic_match(job_data: dict[str, Any]) -> str:
    title = str(job_data.get("title") or "").strip()
    company = str(job_data.get("company") or "").strip()
    location_text = str(job_data.get("location") or "").strip()
    site_label = str(job_data.get("site_label") or job_data.get("site") or "").strip()
    description = normalize_description_text(
        job_data.get("description") or job_data.get("snippet") or job_data.get("summary") or job_data.get("job_description")
    )
    return " | ".join(part for part in [title, company, location_text, site_label, description] if part)


def title_text_for_semantic_match(job_data: dict[str, Any]) -> str:
    return " | ".join(
        part for part in [str(job_data.get("title") or "").strip(), str(job_data.get("company") or "").strip(), str(job_data.get("location") or "").strip()] if part
    )


def search_text_for_semantic_match(keywords: str, location: str, role_filters: list[str]) -> str:
    role_text = " ".join(str(value).strip() for value in role_filters if str(value).strip())
    return " | ".join(part for part in [str(keywords).strip(), str(location).strip(), role_text] if part)


def _title_chunk_texts(title_text: str) -> list[str]:
    """Return title_text plus variants with the title portion split on common delimiters.

    e.g. 'Fall Co-op - Mechanical Engineering Technician | Lockheed Martin | Ottawa' yields:
        ['Fall Co-op - Mechanical Engineering Technician | Lockheed Martin | Ottawa',
         'Fall Co-op | Lockheed Martin | Ottawa',
         'Mechanical Engineering Technician | Lockheed Martin | Ottawa']
    Company/location context is preserved in each chunk so the embedder has full signal.
    """
    sep_idx = title_text.find(" | ")
    title_part = title_text[:sep_idx] if sep_idx != -1 else title_text
    context = title_text[sep_idx:] if sep_idx != -1 else ""
    result = [title_text]
    for sub in _TITLE_CHUNK_RE.split(title_part):
        sub = sub.strip()
        if sub and sub != title_part and len(sub) >= 4:
            result.append(sub + context)
    return result


def semantic_similarity_score(text_a: str, text_b: str) -> float:
    model = load_semantic_plugin_model()
    if model is None:
        return 1.0

    try:
        from sentence_transformers import util

        embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
        score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        return score
    except Exception as exc:
        print(f"Semantic plugin scoring failed: {exc}")
        return 1.0


def semantic_similarity_score_max(texts: list[str], search_text: str) -> float:
    """Batch-encode all texts + search_text in one pass, return the max cosine similarity."""
    if not texts:
        return 0.0
    if len(texts) == 1:
        return semantic_similarity_score(texts[0], search_text)
    model = load_semantic_plugin_model()
    if model is None:
        return 1.0
    try:
        from sentence_transformers import util
        all_texts = [search_text] + texts
        embeddings = model.encode(all_texts, normalize_embeddings=True)
        search_emb = embeddings[0]
        scores = [float(util.cos_sim(search_emb, embeddings[i + 1]).item()) for i in range(len(texts))]
        return max(scores)
    except Exception as exc:
        print(f"Semantic max-chunk scoring failed: {exc}")
        return 1.0


def matches_search_parameters_semantic(
    job_data: dict[str, Any],
    keywords: str,
    location: str,
    role_filters: list[str],
    threshold: float | None = None,
) -> bool:
    listing_text = listing_text_for_semantic_match(job_data)
    title_text = title_text_for_semantic_match(job_data)
    description_text = normalize_description_text(
        job_data.get("description") or job_data.get("snippet") or job_data.get("summary") or job_data.get("job_description")
    )
    search_text = search_text_for_semantic_match(keywords, location, role_filters)
    if not listing_text or not search_text:
        return True

    effective_threshold = SEMANTIC_PLUGIN_THRESHOLD if threshold is None else float(threshold)
    if SEMANTIC_MATCH_TARGET == "title" and title_text:
        return semantic_similarity_score_max(_title_chunk_texts(title_text), search_text) >= effective_threshold
    if description_text:
        # Prefer description-body semantics when available to avoid title-only false positives.
        description_score = semantic_similarity_score(description_text, search_text)
        return description_score >= effective_threshold

    # Fallback: no description — chunk the title for better coverage of long/delimited titles.
    return semantic_similarity_score_max(_title_chunk_texts(title_text or listing_text), search_text) >= effective_threshold
