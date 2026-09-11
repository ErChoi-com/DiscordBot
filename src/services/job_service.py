from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import subprocess
import sys
import hashlib
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from services import capacity, platform_support
from services.net_util import parse_proxy_pool, pick_proxy, retry_backoff_delay

DEFAULT_JOBSPY_EXE = Path(sys.executable)
FALLBACK_JOBSPY_SITES = [
    "bayt",
    "bdjobs",
    "google",
    "indeed",
    "linkedin",
    "naukri",
    "zip_recruiter",
]
CUSTOM_SCRAPER_SITES = {"glassdoor", "zip_recruiter", "greenhouse", "lever", "ashby", "workday", "icims", "bamboohr"}


def custom_scraper_sites() -> set[str]:
    """Every site this module scrapes itself instead of handing to JobSpy.

    CUSTOM_SCRAPER_SITES is a hand-written literal and it fell behind: it named
    six ATS platforms long after ats_service had grown to eighteen. The twelve
    it omitted were then dropped by normalize_requested_sites, so a channel that
    asked for "all" received FEWER sources than one that named none at all --
    Oracle, Paylocity, SmartRecruiters, Teamtailor, Recruitee, Workable and the
    rest were unreachable however a channel was configured. Worse, had they
    survived that filter they would have been routed to JobSpy, which has no
    scraper for them.

    Derived from ATS_PLATFORMS rather than restated, so adding a platform
    cannot silently make it unreachable again. Imported locally to match the
    rest of this module, which never imports ats_service at module scope.
    """
    from services.ats_service import ATS_PLATFORMS

    return set(CUSTOM_SCRAPER_SITES) | set(ATS_PLATFORMS)

# Boards that list a single country or region. Asking one of them for a
# location it does not serve costs a subprocess and its 90s timeout per keyword
# variant and has never returned a row; measured in the archive, naukri, bdjobs
# and bayt account for 0 of 4,046 rows and 10 of 29 timeouts in one run.
# Sites absent from this map are treated as global. Only consulted when a
# channel asked for "all" -- a site someone named explicitly is always asked.
JOBSPY_SITE_COUNTRIES: dict[str, frozenset[str]] = {
    "naukri": frozenset({"IN"}),
    "bdjobs": frozenset({"BD"}),
    "bayt": frozenset({"AE", "SA", "QA", "KW", "BH", "OM", "JO", "LB", "EG", "IQ", "MA", "TN", "DZ", "PK"}),
}


def sites_serving(sites: list[str], country: str | None) -> list[str]:
    """`sites` minus the single-region boards that do not list `country`.

    Order kept. A None/empty country (undecidable location) keeps every site
    -- no evidence is not evidence of absence.
    """
    if not country:
        return list(sites)
    return [
        site for site in sites
        if site not in JOBSPY_SITE_COUNTRIES or country in JOBSPY_SITE_COUNTRIES[site]
    ]


JOBSPY_SITE_LABELS = {
    "all": "All supported sites",
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
PROVINCE_ABBR_TO_FULL = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "MB": "Manitoba",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "YT": "Yukon",
}
COUNTRY_CODE_TO_NAME = {
    "CA": "Canada",
    "US": "United States",
    "GB": "United Kingdom",
    "UK": "United Kingdom",
}
GLASSDOOR_LOCATION_LOCALITY_PATTERN = re.compile(r'"addressLocality"\s*:\s*"([^"]+)"', re.IGNORECASE)
GLASSDOOR_LOCATION_REGION_PATTERN = re.compile(r'"addressRegion"\s*:\s*"([^"]+)"', re.IGNORECASE)
GLASSDOOR_LOCATION_COUNTRY_PATTERN = re.compile(r'"addressCountry"\s*:\s*"([^"]+)"', re.IGNORECASE)
GLASSDOOR_CARD_AGE_PATTERN = re.compile(r'(?<!\d)(\d+)\s*([hd])\+?(?!\d)', re.IGNORECASE)
# ── Semantic plugin (overridden at startup from settings.toml via configure_job_service) ──
SEMANTIC_PLUGIN_ENABLED: bool = True
SEMANTIC_PLUGIN_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_PLUGIN_THRESHOLD: float = 0.30
SEMANTIC_MATCH_TARGET: str = "description"
SEMANTIC_DESCRIPTION_CHAR_LIMIT: int = 2200
# Texts per model call. Thirty candidates is a typical channel; the cap only
# matters for the odd channel that asks for hundreds.
SEMANTIC_ENCODE_BATCH: int = 32
# How many CPU threads inference may use. torch defaults to every core and
# the HF tokenizers to a pool of the same size, so one channel's filter
# claimed all twelve and two channels' filters fought each other for them
# alongside five hundred scrape threads: measured at 2,800s per channel for
# about thirty short texts. MiniLM over thirty texts is well under a second
# on four uncontended threads -- and with this bound, they are uncontended.
SEMANTIC_INFERENCE_THREADS_MAX: int = 4
# One inference at a time. The model already parallelises inside a call;
# two channels calling it at once only thrash the same cores.
_SEMANTIC_INFERENCE_LOCK = threading.Lock()


def semantic_inference_threads() -> int:
    """Threads for model inference: a third of the host, capped, never zero."""
    return max(1, min(SEMANTIC_INFERENCE_THREADS_MAX, int(capacity.cpu_limit()) // 3))


def semantic_encode_batch() -> int:
    """Texts per encode call: memory-proportional, not just CPU-proportional.

    Thirty-odd short strings through MiniLM is nothing on the 16 GB box this
    was tuned on, but on a 1 GB VPS -- where the scrape pools are already
    competing for that memory -- the same batch is a real allocation. This is
    local torch work, not a politeness ceiling, so it is one of the pools
    allowed to scale up on bigger hardware too.
    """
    return capacity.workers(SEMANTIC_ENCODE_BATCH, minimum=4, maximum=128)

# ── Dedup FIFO constants (overridden at startup) ──────────────────────────────
DEDUP_MAX_FIFO_FILES: int = 6
DEDUP_MAX_ENTRIES_PER_FILE: int = 500
DEDUP_MONTHS_THRESHOLD: int = 1
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


def is_message_duplicate(message_content: str, listing_file: Path, months_threshold: int = 1) -> bool:
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


# orphan-ok: deliberately unwired, and DO NOT wire it into the send path.
# Recording is a separate step there on purpose -- manager._send_watcher_message
# checks first, sends, and records only once the send succeeded. Folding the two
# together means a job whose send raises is still marked seen, so it is silently
# never posted. The two halves it composes (is_message_duplicate,
# record_message_for_dedup) are what callers use.
def check_and_record_message(message_content: str, listing_file: Path, months_threshold: int = 1) -> bool:
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
    site_values = ["all", *supported_jobspy_sites()]
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
    # `py` is the Windows Python launcher. On Linux this spawned four doomed
    # processes per call, and worse: an unrelated `py` binary (the
    # python-launcher package) would be handed a -3.x flag it does not
    # understand and could return the wrong interpreter.
    if platform_support.is_windows():
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
    else:
        for name in ("python3.14", "python3.13", "python3.12", "python3.11", "python3"):
            resolved = shutil.which(name)
            if resolved:
                candidates.append(Path(resolved))

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
        return PROVINCE_ABBR_TO_FULL[CANADIAN_PROVINCE_NAMES[lowered]]
    upper = lowered.upper()
    if upper in PROVINCE_ABBR_TO_FULL:
        return PROVINCE_ABBR_TO_FULL[upper]
    return normalized


def normalize_country_name(value: Any) -> str:
    text = fix_text_encoding(value).strip()
    if not text:
        return ""
    normalized = SPACE_PATTERN.sub(" ", text).strip().rstrip(".")
    upper = normalized.upper()
    if upper in COUNTRY_CODE_TO_NAME:
        return COUNTRY_CODE_TO_NAME[upper]
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

    # Glassdoor sometimes embeds this JSON-LD as an escaped string inside another
    # JSON blob (\"addressLocality\":\"...\"), so unescape before matching.
    text = (resp.text or "").replace('\\"', '"')
    city = ""
    region = ""
    country = ""

    match = GLASSDOOR_LOCATION_LOCALITY_PATTERN.search(text)
    if match:
        city = fix_text_encoding(match.group(1)).strip()

    match = GLASSDOOR_LOCATION_REGION_PATTERN.search(text)
    if match:
        region = normalize_canadian_province(match.group(1))

    match = GLASSDOOR_LOCATION_COUNTRY_PATTERN.search(text)
    if match:
        country = normalize_country_name(match.group(1))

    parts = [part for part in (city, region, country) if part]
    if parts:
        return ", ".join(parts)
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


def normalize_requested_sites(site_names: list[str], configured_python_exe: str | None = None) -> list[str]:
    normalized_sites: list[str] = []
    for site in site_names:
        candidate = str(site).strip().lower()
        if candidate and candidate not in normalized_sites:
            normalized_sites.append(candidate)

    if "all" in normalized_sites:
        normalized_sites = all_supported_job_sites(configured_python_exe)

    custom_sites = custom_scraper_sites()
    runtime_sites = set(jobspy_runtime_metadata(configured_python_exe).get("sites", ()))
    if runtime_sites:
        normalized_sites = [site for site in normalized_sites if site in runtime_sites or site in custom_sites]
    return normalized_sites


def all_supported_job_sites(configured_python_exe: str | None = None) -> list[str]:
    """Return every scraper site this runtime can query, ensuring Indeed is present."""
    ordered: list[str] = ["glassdoor", "zip_recruiter"]
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
    # Below normal priority: up to twelve of these run at once, each a fresh
    # interpreter importing pandas, and they were contending on equal terms
    # with the event loop and with interactive commands.
    cmd, extra = platform_support.low_priority_popen_args([str(python_executable), "-c", code])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_SCRAPE_TIMEOUT,
            check=False,
            **extra,
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


def _readable_company(company: str) -> str:
    """Company name for the "Site/Company" label.

    ATS boards give slugs ("trench-group"), which need the hyphens opened up and
    titlecasing to read as a name. A name that already reads as one must be left
    exactly alone: blanket .replace("-", " ").title() turned
    "CaptiveAire - Region 114 Western PA" into "Captiveaire   Region 114 Western
    Pa" -- three spaces where the dash was, and the casing of both "CaptiveAire"
    and "PA" destroyed. A slug has no whitespace and no capitals of its own, so
    that is the only shape that gets rewritten.
    """
    if company.strip() and not any(ch.isspace() for ch in company) and company.islower():
        return company.replace("-", " ").replace("_", " ").title()
    return company


def shape_job_item(row: dict[str, Any], source: str) -> dict[str, Any]:
    title = fix_text_encoding(row.get("title") or "(job)")
    company = fix_text_encoding(row.get("company") or "").strip()
    location_text = fix_text_encoding(row.get("location") or "").strip()
    source_sites = collect_job_sites(row)
    source_label = site_labels_for_sites(source_sites)
    _company_lower = company.lower().replace("-", " ").replace("_", " ").strip()
    # A placeholder company is one naming the board this row actually came from
    # ("LinkedIn" on a LinkedIn posting). Testing against every known board
    # instead treats real employers as placeholders -- Google, Lever, Greenhouse,
    # Ashby and Workday are all job boards AND companies that post their own
    # jobs, and a Google internship listed on LinkedIn lost its name entirely.
    _own_sites = {s.lower().replace("_", " ") for s in source_sites}
    _own_sites |= {JOBSPY_SITE_LABELS.get(s, "").lower() for s in source_sites}
    _is_job_board_name = bool(_company_lower) and _company_lower in _own_sites
    # Every source -- ATS boards and JobSpy sites alike -- labels as "Site/Company"
    # so listings read the same regardless of where they came from.
    _company_in_label = bool(company and source_sites and not _is_job_board_name)
    if _company_in_label:
        source_label = f"{source_label}/{_readable_company(company)}"
    # The parenthetical is the location; the company belongs in the label. It
    # stays here only when the label could not carry it, and a "company" that is
    # just the board's own name is dropped rather than repeated next to the city.
    _keep_company = bool(company) and not _company_in_label and not _is_job_board_name
    _suffix_parts = [company, location_text] if _keep_company else [location_text]
    suffix = ", ".join(part for part in _suffix_parts if part)
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
    # Real fields, not only folded into the title: the archive stores this dict
    # verbatim, and job_match._display_fields had to recover them by splitting
    # the title at the first comma only because they were missing.
    # Not _is_job_board_name: "LinkedIn" as the company would reach the archive
    # and let job_match._job_identity collapse unrelated postings that share it.
    if company and not _is_job_board_name:
        item["company"] = company
    if location_text:
        item["location"] = location_text
    # Carry the posting date the scraper extracted. Without this the watcher
    # path stamps today's date on archive (manager.py: `if not
    # _item.get("date_posted")`), so a listing Lever says was created in March
    # is archived as posted today. The background ATS loop archives raw rows and
    # keeps the real date, so the archive ends up holding two populations with
    # different date semantics -- and merge_data._collides keys repost detection
    # on exactly this field, so a genuine repost and a first sighting become
    # indistinguishable.
    posted = str(row.get("date_posted") or "").strip()
    if posted:
        item["date_posted"] = posted
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

                row = {
                    "title": title_text,
                    "company": company_text,
                    "location": enriched_location,
                    "job_url": job_url,
                    "url": job_url,
                    "job_url_direct": job_url,
                    "_source_site": "glassdoor",
                    "_source_sites": ["glassdoor"],
                }
                # The card's age was already parsed to filter on it, and was
                # then thrown away -- so every Glassdoor row reached the
                # pipeline dateless, and the watcher stamped it with the day it
                # happened to be scraped. That is a fabricated posting date: a
                # listing the card called "7d" old was archived as posted
                # today, and repost detection keys on exactly that field.
                # Glassdoor was the only source doing this; ZipRecruiter
                # recovers real timestamps from the page.
                #
                # Day granularity because that is what the card carries ("3d",
                # "24h"). Omitted rather than guessed when the card says
                # nothing, since no date is honestly "unknown" and is already
                # handled downstream, whereas a wrong one is not detectable.
                if card_age_hours is not None:
                    row["date_posted"] = (
                        datetime.now(timezone.utc) - timedelta(hours=card_age_hours)
                    ).strftime("%Y-%m-%d")
                results.append(row)
                if len(results) >= results_wanted:
                    break
            except Exception:
                # Skip individual job extraction errors
                continue
                
    except Exception as e:
        # Silently handle errors - Glassdoor may block or change structure
        pass
    
    return results


# ZipRecruiter sits behind a Cloudflare WAF that fingerprints the TLS/JA3
# handshake, not just headers or JS behaviour. Playwright-driven Chromium was
# refused with a 403 "Just a moment..." interstitial on EVERY configuration
# tried -- headless and headful, with and without webdriver/UA stealth patches
# -- so the browser implementation that used to live here could not return a
# row under any circumstance. curl_cffi replays a real browser's TLS
# fingerprint and clears the WAF without launching a browser at all, which
# also takes this scraper back out of browser_service's dispatch gate.
ZIPRECRUITER_HOME_URL = "https://www.ziprecruiter.com/"
ZIPRECRUITER_SEARCH_URL = "https://www.ziprecruiter.com/jobs-search"
# Verified against the live WAF, 3 consecutive requests each. Note that the
# bare "chrome" alias tracks curl_cffi's NEWEST fingerprint and that is exactly
# the one Cloudflare refuses (0/3 success, vs 3/3 for each target below).
# Do not "modernize" these to the bare alias. Rotating across three engines
# also absorbs the occasional single-fingerprint block.
ZIPRECRUITER_IMPERSONATIONS = ("chrome124", "firefox133", "safari")

# Which of the above last cleared the WAF, so the next fetch starts there.
#
# The order above is a fixed preference, and the WAF's is not: measured
# 2026-09-06, chrome124 now draws a 403 on every attempt while safari clears
# 3/3 -- the reverse of what the comment above recorded when it was written.
# Nothing was broken by that, because the rotation still reached a working
# fingerprint, but every page of every search paid a doomed request plus a
# backoff sleep first, and each page repeats it.
#
# Remembering the winner keeps the rotation exactly as it is -- still three
# engines, still in the same order once the remembered one is tried -- while
# making the common case one request instead of two. It also means the next
# time the WAF's preference moves, this follows it without an edit, which a
# reordered tuple would not.
_ziprecruiter_last_good: str | None = None
_ziprecruiter_last_good_lock = threading.Lock()


def _ziprecruiter_impersonation_order() -> tuple[str, ...]:
    """Configured order, with the last fingerprint that worked moved first."""
    with _ziprecruiter_last_good_lock:
        good = _ziprecruiter_last_good
    if not good or good not in ZIPRECRUITER_IMPERSONATIONS:
        return ZIPRECRUITER_IMPERSONATIONS
    return (good, *(i for i in ZIPRECRUITER_IMPERSONATIONS if i != good))


def _note_ziprecruiter_impersonation(impersonate: str) -> None:
    global _ziprecruiter_last_good
    with _ziprecruiter_last_good_lock:
        _ziprecruiter_last_good = impersonate
ZIPRECRUITER_RESULTS_PER_PAGE = 20
ZIPRECRUITER_MAX_PAGES = 5
_ZIPRECRUITER_LD_JSON_RE = re.compile(
    r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', re.S
)
_ZIPRECRUITER_JID_RE = re.compile(r"jid=([0-9a-f]+)")
_ZIPRECRUITER_SEAT_RE = re.compile(r'"openSeatId":"([0-9a-f]+)"')
_ZIPRECRUITER_POSTED_RE = re.compile(r'"rollingPostedAtUtc":"([^"]+)"')


def _ziprecruiter_ld_json_items(html: str) -> list[dict[str, str]]:
    """Ordered title/URL pairs from the search page's ld+json ItemList.

    The two-pane layout no longer puts the posting URL on any card anchor --
    those now point at company profiles (/co/...) and refine-search links --
    so the ItemList block is the only place the real posting links survive.
    """
    for match in _ZIPRECRUITER_LD_JSON_RE.finditer(html):
        try:
            data = json.loads(match.group(1).strip())
        except Exception:
            continue
        if not isinstance(data, dict) or data.get("@type") != "ItemList":
            continue
        items: list[dict[str, str]] = []
        for entry in data.get("itemListElement") or []:
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("name") or "").strip()
            job_url = str(entry.get("url") or "").strip()
            if title and job_url:
                items.append({"title": title, "job_url": job_url})
        if items:
            return items
    return []


def _ziprecruiter_card_details(html: str) -> list[dict[str, str]]:
    """Per-card company/location in the same order as the ItemList entries."""
    soup = BeautifulSoup(html, "html.parser")
    details: list[dict[str, str]] = []
    for card in soup.select("div.job_result_two_pane_v2"):
        # Each card is rendered twice (mobile + desktop panes) with identical
        # values, so the first node of each kind is the one to read.
        company = card.select_one('[data-testid="job-card-company"]')
        card_location = card.select_one('[data-testid="job-card-location"]')
        details.append({
            "company": company.get_text(" ", strip=True) if company else "",
            "location": card_location.get_text(" ", strip=True) if card_location else "",
        })
    return details


def _ziprecruiter_posted_dates(html: str) -> dict[str, str]:
    """Map a posting's `jid` to its exact UTC post timestamp.

    ZipRecruiter's own date filter is day-granular (`days=N`), but the page
    embeds per-job metadata carrying `rollingPostedAtUtc` to the second, which
    lets callers apply an exact hours_old window client-side.

    The metadata is JSON escaped inside a JS string, and its objects contain
    nested `listingKey` fields, so splitting on those to group each job's
    values does NOT work. What is stable is the ordering: every job emits its
    timestamp before the `openSeatId` that matches the ld+json `jid`, so each
    seat id pairs with the nearest preceding timestamp.
    """
    unescaped = html.replace('\\"', '"')
    stamps = [(m.start(), m.group(1)) for m in _ZIPRECRUITER_POSTED_RE.finditer(unescaped)]
    if not stamps:
        return {}
    offsets = [offset for offset, _ in stamps]
    dates: dict[str, str] = {}
    for match in _ZIPRECRUITER_SEAT_RE.finditer(unescaped):
        index = bisect.bisect_left(offsets, match.start()) - 1
        if index >= 0:
            dates.setdefault(match.group(1), stamps[index][1])
    return dates


def parse_ziprecruiter_search_html(html: str, fallback_location: str = "") -> list[dict[str, Any]]:
    """Turn one ZipRecruiter search page into job rows."""
    items = _ziprecruiter_ld_json_items(html)
    if not items:
        return []
    details = _ziprecruiter_card_details(html)
    posted_dates = _ziprecruiter_posted_dates(html)
    # Enrichment is positional, so it is only trustworthy when every ItemList
    # entry has exactly one card. A mismatch (sponsored slots, partial render)
    # would silently attach the wrong company to a job, which is worse than
    # shipping the row without one.
    aligned = details if len(details) == len(items) else []

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        detail = aligned[index] if index < len(aligned) else {}
        row = {
            "title": fix_text_encoding(item["title"]),
            "company": fix_text_encoding(detail.get("company") or ""),
            "location": fix_text_encoding(detail.get("location") or "") or fallback_location,
            "job_url": item["job_url"],
            "_source_site": "zip_recruiter",
            "_source_sites": ["zip_recruiter"],
        }
        jid = _ZIPRECRUITER_JID_RE.search(item["job_url"])
        if jid and jid.group(1) in posted_dates:
            row["date_posted"] = posted_dates[jid.group(1)]
        rows.append(row)
    return rows


def _ziprecruiter_fetch_page(
    params: dict[str, Any],
    proxy_pool: list[str],
    cursor: int,
) -> str | None:
    """Fetch one search page, rotating TLS fingerprints until one clears."""
    from curl_cffi import requests as curl_requests

    order = _ziprecruiter_impersonation_order()
    last_attempt = len(order) - 1
    for attempt, impersonate in enumerate(order):
        proxy = pick_proxy(proxy_pool, cursor + attempt)
        proxies = {"http": proxy, "https": proxy} if proxy else None
        try:
            session = curl_requests.Session(impersonate=impersonate)
            # The homepage hands out the __cf_bm clearance cookie that the
            # search route expects; without it the first search request is
            # noticeably more likely to draw a challenge.
            try:
                session.get(ZIPRECRUITER_HOME_URL, timeout=20, proxies=proxies)
            except Exception:
                pass
            response = session.get(
                ZIPRECRUITER_SEARCH_URL, params=params, timeout=30, proxies=proxies
            )
            if response.status_code == 200:
                _note_ziprecruiter_impersonation(impersonate)
                return response.content.decode("utf-8", "replace")
            print(f"[ziprecruiter] {impersonate} -> HTTP {response.status_code}")
        except Exception as exc:
            print(f"[ziprecruiter] {impersonate} request failed: {exc}")
        if attempt < last_attempt:
            time.sleep(retry_backoff_delay(attempt))
    return None


def scrape_ziprecruiter_postings(
    keywords: str,
    location: str,
    results_wanted: int = 20,
    hours_old: int = 168,
    radius_miles: int = 25,
) -> list[dict[str, Any]]:
    """Scrape ZipRecruiter search results over its Cloudflare-fronted HTML.

    Returns [] when curl_cffi is unavailable or every fingerprint is refused.
    """
    try:
        import curl_cffi  # noqa: F401
    except Exception:
        print("[ziprecruiter] curl_cffi is not installed; skipping")
        return []

    proxy_pool = parse_proxy_pool(os.getenv("JOBSPY_PROXIES"))
    target = max(1, int(results_wanted or 1))
    max_pages = min(
        ZIPRECRUITER_MAX_PAGES,
        -(-target // ZIPRECRUITER_RESULTS_PER_PAGE),
    )

    base_params: dict[str, Any] = {"search": keywords, "location": location}
    if radius_miles:
        base_params["radius"] = max(1, int(radius_miles))
    if hours_old:
        # ZipRecruiter's filter is day-granular, so hours are converted UP to
        # whole days: a 30h window must ask for 2 days or the 24-30h postings
        # never arrive. The requested window is then narrowed back to the exact
        # hour below, against each posting's real timestamp.
        base_params["days"] = max(1, min(30, -(-int(hours_old) // 24)))

    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for page in range(1, max_pages + 1):
        params = dict(base_params)
        if page > 1:
            params["page"] = page
        html = _ziprecruiter_fetch_page(params, proxy_pool, page)
        if not html:
            break
        page_rows = parse_ziprecruiter_search_html(html, location)
        if not page_rows:
            break
        for row in page_rows:
            if row["job_url"] in seen_urls:
                continue
            seen_urls.add(row["job_url"])
            # Exact-hour trim of the day-granular window. Rows whose timestamp
            # could not be recovered carry no date_posted and are kept, matching
            # _posting_age_ok's treatment of dateless rows elsewhere.
            if hours_old and not _posting_age_ok(row, int(hours_old)):
                continue
            rows.append(row)
            if len(rows) >= target:
                return rows
    return rows


# Cross-channel scrape dedup: two channels watching the SAME criteria used to
# run the identical external scrape independently every cycle (per-channel
# watcher tasks have no coordination). Results are cached by the full scrape
# criteria — deliberately EXCLUDING source_url, which is per-channel labeling
# applied after the cache — for just under the 60s minimum refresh interval.
# A per-key in-flight lock makes a simultaneous second caller wait for the
# first scrape's result instead of duplicating it.
# Stretched with the scrape budget above: if a scrape takes longer than this
# TTL the cache can never hit, and every channel re-runs the identical scrape --
# exactly the duplication this cache exists to prevent. On slow hardware a
# fixed 55s is the cliff that triggers it.
JOB_SCRAPE_CACHE_TTL_SECONDS = capacity.timeout(55.0)
_scrape_cache: dict[tuple, tuple[float, list[dict[str, Any]]]] = {}
_scrape_cache_lock = threading.Lock()
_scrape_key_locks: dict[tuple, threading.Lock] = {}


# orphan-ok: test-isolation reset for module-global state, like
# ats_service.clear_dead_slugs. Production has a TTL for this
# (JOB_SCRAPE_CACHE_TTL_SECONDS) and never needs to drop the cache wholesale.
def clear_job_scrape_cache() -> None:
    with _scrape_cache_lock:
        _scrape_cache.clear()
        _scrape_key_locks.clear()


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
) -> list[dict[str, Any]]:
    if site_names:
        normalized_sites = normalize_requested_sites(site_names, configured_python_exe)
    else:
        normalized_sites = all_supported_job_sites(configured_python_exe)

    if not normalized_sites:
        return []

    asked_for_all = not site_names or any(str(s).strip().lower() == "all" for s in site_names)
    if asked_for_all:
        from services.jba import geo_priority

        country = geo_priority.country_of(location)
        kept = sites_serving(normalized_sites, country)
        if len(kept) != len(normalized_sites):
            dropped = [site for site in normalized_sites if site not in kept]
            print(
                f"[jobs] {location!r} is in {country}; not asking "
                f"{', '.join(dropped)} (single-region boards)"
            )
            normalized_sites = kept

    cache_key = (
        tuple(normalized_sites),
        keywords,
        location,
        hours_old,
        results_wanted,
        radius_miles,
        country_indeed,
        allow_north_america,
    )
    with _scrape_cache_lock:
        key_lock = _scrape_key_locks.setdefault(cache_key, threading.Lock())
    with key_lock:
        now = time.monotonic()
        with _scrape_cache_lock:
            hit = _scrape_cache.get(cache_key)
        if hit is not None and now - hit[0] <= JOB_SCRAPE_CACHE_TTL_SECONDS:
            filtered_rows = [dict(row) for row in hit[1]]
        else:
            filtered_rows = _scrape_filtered_rows_uncached(
                normalized_sites,
                keywords,
                location,
                configured_python_exe,
                hours_old,
                results_wanted,
                radius_miles,
                country_indeed,
                allow_north_america,
            )
            with _scrape_cache_lock:
                _scrape_cache[cache_key] = (time.monotonic(), [dict(row) for row in filtered_rows])
                # Opportunistic purge keeps the dict bounded to live criteria.
                expired = [
                    k for k, (ts, _) in _scrape_cache.items()
                    if time.monotonic() - ts > JOB_SCRAPE_CACHE_TTL_SECONDS
                ]
                for k in expired:
                    _scrape_cache.pop(k, None)
                    _scrape_key_locks.pop(k, None)

    if not filtered_rows:
        return []
    from services.ats_service import ATS_PLATFORMS as _ats_platforms

    _ats_names = set(_ats_platforms)
    source = source_url or f"jobspy:{','.join(normalized_sites)}"
    ats_rows = [r for r in filtered_rows if r.get("_source_site") in _ats_names]
    non_ats_rows = [r for r in filtered_rows if r.get("_source_site") not in _ats_names]
    return [shape_job_item(row, source) for row in non_ats_rows + ats_rows]


def _scrape_filtered_rows_uncached(
    normalized_sites: list[str],
    keywords: str,
    location: str,
    configured_python_exe: str | None,
    hours_old: int,
    results_wanted: int,
    radius_miles: int,
    country_indeed: str,
    allow_north_america: bool,
) -> list[dict[str, Any]]:

    raw: list[dict[str, Any]] = []
    target_count = max(1, results_wanted)

    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    scrape_tasks: list[Any] = []

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
                return scrape_ziprecruiter_postings(
                    keywords, location, target_count,
                    hours_old=hours_old, radius_miles=radius_miles,
                )
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
            # Read enough archive days to cover the channel's own freshness
            # window. A hardcoded (0, 1) capped every channel at 48h no matter
            # what it asked for -- hours_old=72 silently became 48, and the
            # rows were dropped before _posting_age_ok ever saw them, so the
            # setting looked honoured while two thirds of its range was
            # unreachable. The extra day covers the lag between a job being
            # posted and the cycle that scrapes it writing that day's log.
            span = max(2, -(-int(hours_old or 0) // 24) + 1)
            for delta in range(span):
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
                    print(f"[jba-log] {len(cached_rows)} ATS jobs from {span}-day window ({len(cached)} total)")
        except Exception as exc:
            print(f"[jba-log] Failed to read daily log: {exc}")

    from services.ats_service import ATS_PLATFORMS as _ATS_PLATS_SET
    _ats_names = set(_ATS_PLATS_SET)

    jobspy_sites = [site for site in normalized_sites if site not in custom_scraper_sites()]
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
        # Capped, not len(scrape_tasks): sites x keyword-variants used to size
        # this pool unbounded (dozens of threads, each possibly a JobSpy
        # subprocess). This nested pool is invisible to PriorityWorkScheduler's
        # worker accounting (the caller occupies ONE scheduler slot), so the
        # cap is what keeps the hidden concurrency honest.
        # Each task is a DIFFERENT site, so widening this adds parallel sites
        # rather than more requests per site -- a local resource cost, hence an
        # explicit `maximum` that lets it grow on bigger hardware.
        pool_size = min(capacity.workers(6, minimum=2, maximum=12), len(scrape_tasks))
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = [executor.submit(t) for t in scrape_tasks]
            for future in _as_completed(futures):
                try:
                    raw.extend(future.result())
                except Exception as exc:
                    print(f"Scrape task error: {exc}")

    if not raw:
        return []

    deduped_rows = dedupe_job_rows(raw)
    return filter_rows_by_region(deduped_rows, allow_north_america=allow_north_america)


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


def matches_role_filters(
    title: str, role_filters: list[str], description: str = "", employment_type: str = ""
) -> bool:
    """Does this posting satisfy the channel's role filter?

    Two passes, unioned. The keyword pass below is the original: a substring
    search over the title. The level pass asks job_level.classify, which reads
    the term ("Winter 2027") and the platform's own employment_type -- the only
    signals on a posting whose title carries no level word at all. "Software
    Developer (Winter 2027)" is a student posting that the keyword pass can
    never see, and it is the shape this whole classifier was built for.

    Unioned rather than replaced on purpose: every title that matched before
    still matches, so this can only add results. The keyword pass also stays
    the authority on filter names the level mapping does not know.
    """
    if not role_filters:
        return True

    try:
        from services import job_level

        if job_level.matches_role_filter(title, role_filters, description, employment_type):
            return True
    except Exception:
        # The keyword pass below is a complete implementation on its own, so a
        # failure here costs recall, never correctness.
        pass

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
    global SEMANTIC_DESCRIPTION_CHAR_LIMIT
    global DEDUP_MAX_FIFO_FILES, DEDUP_MAX_ENTRIES_PER_FILE, DEDUP_MONTHS_THRESHOLD, DEDUP_SEEN_LINKS_CAP
    global HTTP_REQUEST_TIMEOUT, SUBPROCESS_SCRAPE_TIMEOUT
    global SITE_CONCURRENCY_LIMIT, SITE_SEMAPHORE_TIMEOUT

    SEMANTIC_PLUGIN_ENABLED = bool(getattr(config, "semantic_enabled", SEMANTIC_PLUGIN_ENABLED))
    SEMANTIC_PLUGIN_MODEL_NAME = str(getattr(config, "semantic_model", SEMANTIC_PLUGIN_MODEL_NAME))
    SEMANTIC_PLUGIN_THRESHOLD = float(getattr(config, "semantic_threshold", SEMANTIC_PLUGIN_THRESHOLD))
    SEMANTIC_MATCH_TARGET = str(getattr(config, "semantic_match_target", SEMANTIC_MATCH_TARGET)).strip().lower()
    SEMANTIC_DESCRIPTION_CHAR_LIMIT = int(getattr(config, "semantic_description_char_limit", SEMANTIC_DESCRIPTION_CHAR_LIMIT))
    DEDUP_MAX_FIFO_FILES = int(getattr(config, "dedup_max_fifo_files", DEDUP_MAX_FIFO_FILES))
    DEDUP_MAX_ENTRIES_PER_FILE = int(getattr(config, "dedup_max_entries_per_file", DEDUP_MAX_ENTRIES_PER_FILE))
    DEDUP_MONTHS_THRESHOLD = int(getattr(config, "dedup_months_threshold", DEDUP_MONTHS_THRESHOLD))
    # Scaled down on small hosts: this set is held per watched channel, and the
    # send path builds a second canonicalized copy each cycle, so the configured
    # value is really "memory per channel". Down-only -- a bigger box gains
    # nothing user-visible from remembering more links.
    DEDUP_SEEN_LINKS_CAP = capacity.scaled_cap(
        int(getattr(config, "dedup_seen_links_cap", DEDUP_SEEN_LINKS_CAP)), minimum=5_000
    )
    HTTP_REQUEST_TIMEOUT = int(getattr(config, "http_timeout_seconds", HTTP_REQUEST_TIMEOUT))
    # Stretched on slow hardware: this kills a JobSpy subprocess mid-flight, so
    # a budget that is generous on the reference box loses ALL results from a
    # site on a Pi rather than merely some.
    SUBPROCESS_SCRAPE_TIMEOUT = int(capacity.timeout(
        int(getattr(config, "subprocess_scrape_timeout_seconds", SUBPROCESS_SCRAPE_TIMEOUT))
    ))
    SITE_CONCURRENCY_LIMIT = int(getattr(config, "site_concurrency_limit", SITE_CONCURRENCY_LIMIT))
    SITE_SEMAPHORE_TIMEOUT = int(getattr(config, "site_semaphore_timeout_seconds", SITE_SEMAPHORE_TIMEOUT))
    with _site_semaphores_lock:
        _site_semaphores.clear()
    # Clear cached model so it re-loads with updated settings if needed
    load_semantic_plugin_model.cache_clear()
    semantic_plugin_available.cache_clear()


# The MiniLM weights are ~90 MB, but the torch runtime around them pushes real
# RSS to several hundred MB. On a 1 GB VPS that is the difference between the
# bot running and the OOM killer taking it, so below this the semantic path
# stays off and matching falls back to keywords.
SEMANTIC_MIN_MEMORY_GB: float = 2.0


@lru_cache(maxsize=1)
def semantic_plugin_available() -> bool:
    if not SEMANTIC_PLUGIN_ENABLED:
        return False
    if not capacity.can_afford(SEMANTIC_MIN_MEMORY_GB):
        print(
            f"Semantic matching disabled: needs ~{SEMANTIC_MIN_MEMORY_GB:g}GB, "
            f"host has {capacity.describe()}. Falling back to keyword matching."
        )
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
    # Before the import: the tokenizers' thread pool is sized at import time.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SEMANTIC_PLUGIN_MODEL_NAME)
    except Exception as exc:
        print(f"Semantic plugin disabled (model load failed): {exc}")
        return None
    _bound_inference_threads()
    return model


def _bound_inference_threads() -> None:
    """Cap torch's intra-op pool; a missing or odd torch must not cost the model."""
    try:
        import torch

        torch.set_num_threads(semantic_inference_threads())
    except Exception:
        pass


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


def _semantic_candidate_texts(job_data: dict[str, Any]) -> list[str] | None:
    """The texts an item is judged on; None when it carries nothing to judge.

    Description first, because a title alone produces false positives; the
    title split into its chunks otherwise, so a long delimited title is
    matched on its best part rather than on the whole.
    """
    listing_text = listing_text_for_semantic_match(job_data)
    if not listing_text:
        return None
    title_text = title_text_for_semantic_match(job_data)
    if SEMANTIC_MATCH_TARGET == "title" and title_text:
        return _title_chunk_texts(title_text)
    description_text = normalize_description_text(
        job_data.get("description") or job_data.get("snippet") or job_data.get("summary") or job_data.get("job_description")
    )
    if description_text:
        return [description_text]
    return _title_chunk_texts(title_text or listing_text)


def matches_search_parameters_semantic(
    job_data: dict[str, Any],
    keywords: str,
    location: str,
    role_filters: list[str],
    threshold: float | None = None,
) -> bool:
    """One item's verdict. The watcher judges a whole batch at once through
    semantic_filter_items; this remains for callers with a single item."""
    search_text = search_text_for_semantic_match(keywords, location, role_filters)
    texts = _semantic_candidate_texts(job_data)
    if not texts or not search_text:
        return True
    effective_threshold = SEMANTIC_PLUGIN_THRESHOLD if threshold is None else float(threshold)
    return semantic_similarity_score_max(texts, search_text) >= effective_threshold


def semantic_filter_items(
    items: list[dict[str, Any]],
    keywords: str,
    location: str,
    role_filters: list[str],
    threshold: float | Callable[[dict[str, Any]], float] | None = None,
) -> list[dict[str, Any]]:
    """The items whose best candidate text scores at least their threshold
    against the search text -- judged in one model call for the whole batch.

    Per item, the old path encoded the search text again and the item's
    texts with it: for thirty candidates, thirty model calls, thirty
    re-encodings of the same query, and thirty chances to queue behind the
    other channel doing the same. Now the query is encoded once, every
    candidate's texts go in the same batch, and cosine is a dot product over
    normalised embeddings. Items with nothing to judge are kept, as before:
    no evidence is not evidence against.

    `threshold` may be a number or a callable of the item, so a caller can
    hold different sources to different bars without a second pass. Each
    kept item gets its score under "semantic_score" for anyone ranking later.
    """
    items = list(items)
    search_text = search_text_for_semantic_match(keywords, location, role_filters)
    if not items or not search_text:
        return items

    texts: list[str] = [search_text]
    spans: list[tuple[int, int] | None] = []
    for item in items:
        candidates = _semantic_candidate_texts(item)
        if not candidates:
            spans.append(None)
            continue
        start = len(texts)
        texts.extend(candidates)
        spans.append((start, len(texts)))
    if len(texts) == 1:
        return items

    model = load_semantic_plugin_model()
    if model is None:
        return items
    try:
        # Sized before the lock is taken: it reads the host's memory, which is
        # no business of the one call at a time this lock exists to enforce.
        batch = semantic_encode_batch()
        with _SEMANTIC_INFERENCE_LOCK:
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=batch)
        # Normalised, so cosine is the dot product. Spelled out rather than
        # through numpy: it is thirty rows of 384 floats, and numpy is a
        # dependency of the optional model rather than of this module.
        query = list(embeddings[0])
        scores = [float(sum(x * y for x, y in zip(row, query))) for row in embeddings[1:]]
    except Exception as exc:
        print(f"Semantic plugin scoring failed: {exc}")
        return items

    kept: list[dict[str, Any]] = []
    for item, span in zip(items, spans):
        if span is None:
            kept.append(item)
            continue
        best = max(scores[span[0] - 1: span[1] - 1])
        if callable(threshold):
            bar = float(threshold(item))
        else:
            bar = SEMANTIC_PLUGIN_THRESHOLD if threshold is None else float(threshold)
        item["semantic_score"] = round(best, 4)
        if best >= bar:
            kept.append(item)
    return kept
