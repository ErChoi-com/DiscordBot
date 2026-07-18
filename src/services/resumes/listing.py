from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from config import load_env

from .. import (
    compact_job_description,
    job_site_from_url,
    scrape_job_descriptions_from_all_sites,
    scrape_jobs_from_board_url,
)
from .. import browser_service
from .cache import normalize_gemini_model_name
from .configkey import GeminiSettings
from .resume import LLM_PROVIDER_SWITCH_ORDER, PROVIDER_CAPABILITIES, TEMPLATE_PATH, extract_latex_document, read_resume_template, resolve_provider_capabilities
from .structured import (
    StructuredSelection,
    _effective_render_config,
    _extract_jd_tools,
    apply_grounding_verdicts,
    build_grounding_audit_items,
    build_structured_prompt,
    excerpt_job_description,
    extract_json_object,
    extract_listing_keywords,
    grounding_audit_prompt,
    load_structured_profile,
    parse_structured_response,
    render_structured_resume,
)

# rebuilt_app/.resume_cache — same root the compile logs and failed-latex
# artifacts use (via config.resume_cache_dir; this module has no config).
STRUCTURED_CACHE_ROOT = Path(__file__).resolve().parents[3] / ".resume_cache"
STRUCTURED_SELECTION_CACHE_DIR = STRUCTURED_CACHE_ROOT / "structured_selections"
STRUCTURED_TELEMETRY_PATH = STRUCTURED_CACHE_ROOT / "structured_builds.jsonl"
# One JSON line per scrape attempt recording which fallback rung rescued it
# (or that all rungs failed) — watch this for silent endpoint rot the way
# structured_builds.jsonl is watched for provider degradation.
SCRAPE_TELEMETRY_PATH = STRUCTURED_CACHE_ROOT / "scrape_events.jsonl"
SELECTION_CACHE_TTL_SECONDS = 86400.0

# A provider that just rate-limited us will rate-limit the next request too;
# skip it for a cooldown window instead of paying its timeout every build.
PROVIDER_COOLDOWN_SECONDS = 120.0
_PROVIDER_COOLDOWN_UNTIL: dict[str, float] = {}

SPACE_PATTERN = re.compile(r"\s+")
TITLE_PREFIX_PATTERN = re.compile(r"^\[[^\]]+\]\s*")
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
DESCRIPTION_HINT_PATTERN = re.compile(
    r"responsib|qualif|requirement|what you|about (the )?role|skills|experience",
    re.IGNORECASE,
)
DEFAULT_BASEINFO_PATH = Path(__file__).resolve().parent / "resumes_cache" / "baseinfo.txt"
DEFAULT_INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "resumes_cache" / "instructions.txt"
INDEED_GRAPHQL_ENDPOINTS = (
    "https://apis.indeed.com/graphql",
    "https://www.indeed.com/graphql",
)
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
@dataclass(slots=True)
class ResumeProviderCandidate:
    name: str
    api_key: str | None
    model: str


@dataclass(slots=True)
class JobContext:
    title: str
    posting_url: str
    apply_url: str | None
    source_message: str


@dataclass(slots=True)
class ScrapedJobPosting:
    title: str
    company: str | None
    location: str | None
    description: str
    highlights: list[str]
    source_url: str


@dataclass(slots=True)
class ResumeRewriteResult:
    status: str
    message: str
    rewritten_resume: str | None = None
    prompt_preview: str | None = None
    scraped_job: ScrapedJobPosting | None = None
    used_cache_name: str | None = None
    latex_document: str | None = None
    used_provider: str | None = None


def normalize_space(text: str) -> str:
    return SPACE_PATTERN.sub(" ", text).strip()


def clean_title(raw_title: str) -> str:
    title = TITLE_PREFIX_PATTERN.sub("", raw_title).strip()
    return title or raw_title.strip() or "Target Role"


def extract_first_url(text: str) -> str | None:
    match = URL_PATTERN.search(text)
    if match is None:
        return None
    return match.group(0).rstrip(")],.>")


def extract_message_lines(message_content: str) -> list[str]:
    return [line.strip() for line in message_content.splitlines() if line.strip()]


# Errors that typically mean "transient network/TLS hiccup" (including the
# JA3/TLS-fingerprint connection resets some job boards use for bot
# detection) rather than a hard, permanent block — worth a couple of quick
# retries before falling through to the next scraping strategy.
_TRANSIENT_REQUEST_EXCEPTIONS = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def _get_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int,
    max_attempts: int = 3,
    backoff_seconds: float = 0.6,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return requests.get(url, headers=headers, timeout=timeout)
        except _TRANSIENT_REQUEST_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(backoff_seconds * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _post_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_payload: Any = None,
    timeout: int,
    max_attempts: int = 3,
    backoff_seconds: float = 0.6,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return requests.post(url, headers=headers, json=json_payload, timeout=timeout)
        except _TRANSIENT_REQUEST_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(backoff_seconds * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _parse_company_from_soup(soup: BeautifulSoup) -> str | None:
    selectors = [
        "meta[property='og:site_name']",
        "meta[name='application-name']",
        "meta[name='twitter:site']",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        value = str(node.get("content") or "").strip()
        if value:
            return value.lstrip("@")
    return None


def _parse_location_from_text(description: str) -> str | None:
    for line in description.split("\n"):
        lowered = line.lower()
        if "location" not in lowered:
            continue
        _, _, suffix = line.partition(":")
        candidate = normalize_space(suffix or line)
        if candidate:
            return candidate[:120]
    return None


def _visible_text_from_soup(soup: BeautifulSoup) -> str:
    for node in soup(["script", "style", "noscript", "svg", "footer", "nav"]):
        node.extract()
    scope = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    lines = [normalize_space(line) for line in scope.get_text("\n", strip=True).splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_highlights(raw_text: str, limit: int = 10) -> list[str]:
    seen: set[str] = set()
    highlights: list[str] = []
    for line in raw_text.split("\n"):
        candidate = normalize_space(line)
        if len(candidate) < 35 or len(candidate) > 260:
            continue
        if not DESCRIPTION_HINT_PATTERN.search(candidate):
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        highlights.append(candidate)
        if len(highlights) >= limit:
            break
    return highlights


def _extract_indeed_job_key(posting_url: str) -> str | None:
    parsed = urlparse(posting_url)
    query = parse_qs(parsed.query or "")
    for key in ("jk", "vjk", "jobKey"):
        values = query.get(key)
        if not values:
            continue
        candidate = normalize_space(values[0])
        if candidate:
            return candidate
    return None


def _find_first_text(payload: Any, keys: set[str]) -> str | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and isinstance(value, str):
                candidate = normalize_space(value)
                if candidate:
                    return candidate
            nested = _find_first_text(value, keys)
            if nested:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _find_first_text(item, keys)
            if nested:
                return nested
    return None


def _read_dotenv_value(*keys: str) -> str:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    env_values = load_env(env_path)
    for key in keys:
        value = normalize_space(str(env_values.get(key) or ""))
        if value:
            return value
    return ""


def _scrape_indeed_job_posting_with_graphql(
    posting_url: str,
    timeout_seconds: int,
    user_agent: str,
) -> ScrapedJobPosting | None:
    job_key = _extract_indeed_job_key(posting_url)
    if not job_key:
        return None

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": "https://www.indeed.com",
        "Referer": posting_url,
        "User-Agent": user_agent,
    }
    indeed_api_key = normalize_space(
        os.getenv("INDEED_GRAPHQL_API_KEY")
        or os.getenv("INDEED_API_KEY")
        or _read_dotenv_value("INDEED_GRAPHQL_API_KEY", "INDEED_API_KEY")
    )
    if indeed_api_key:
        headers["x-api-key"] = indeed_api_key
        headers["Indeed-Api-Key"] = indeed_api_key
    indeed_bearer_token = normalize_space(
        os.getenv("INDEED_GRAPHQL_BEARER") or _read_dotenv_value("INDEED_GRAPHQL_BEARER")
    )
    if indeed_bearer_token:
        headers["Authorization"] = f"Bearer {indeed_bearer_token}"

    # Best-effort GraphQL probes against common Indeed endpoints.
    payloads: list[dict[str, Any]] = [
        {
            "operationName": "JobDataQuery",
            "query": (
                "query JobDataQuery($jobKeys:[ID!]){"
                "jobData(input:{jobKeys:$jobKeys}){"
                "results{job{title sourceEmployerName url location{city postalCode} description{text html}}}"
                "}"
                "}"
            ),
            "variables": {"jobKeys": [job_key]},
        },
    ]

    for endpoint in INDEED_GRAPHQL_ENDPOINTS:
        for payload in payloads:
            try:
                response = _post_with_retry(
                    endpoint,
                    headers=headers,
                    json_payload=payload,
                    timeout=max(5, int(timeout_seconds)),
                )
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError):
                continue

            graphql_data = data.get("data") if isinstance(data, dict) else None
            if not isinstance(graphql_data, dict):
                continue

            title: str | None = None
            company: str | None = None
            location: str | None = None
            detail_text: str | None = None

            results = None
            job_data = graphql_data.get("jobData") if isinstance(graphql_data, dict) else None
            if isinstance(job_data, dict):
                results = job_data.get("results")
            if isinstance(results, list) and results:
                first_result = results[0]
                job = first_result.get("job") if isinstance(first_result, dict) else None
                if isinstance(job, dict):
                    title = normalize_space(str(job.get("title") or "")) or None
                    company = normalize_space(str(job.get("sourceEmployerName") or "")) or None
                    job_location = job.get("location") if isinstance(job.get("location"), dict) else None
                    if isinstance(job_location, dict):
                        city = normalize_space(str(job_location.get("city") or ""))
                        postal = normalize_space(str(job_location.get("postalCode") or ""))
                        location = normalize_space(", ".join(part for part in (city, postal) if part)) or None
                    job_description = job.get("description") if isinstance(job.get("description"), dict) else None
                    if isinstance(job_description, dict):
                        text_value = normalize_space(str(job_description.get("text") or ""))
                        html_value = normalize_space(str(job_description.get("html") or ""))
                        detail_text = text_value or html_value or None

            # Keep generic extraction as a safety net for schema drift.
            title = title or _find_first_text(graphql_data, {"title", "jobTitle"})
            company = company or _find_first_text(graphql_data, {"sourceEmployerName", "company", "employerName"})
            location = location or _find_first_text(graphql_data, {"formattedLocation", "label", "location"})
            detail_text = detail_text or _find_first_text(graphql_data, {"description", "text", "html", "sanitizedHtml"})
            if not detail_text:
                continue

            normalized_title = title or "Job Posting"
            description_raw = compact_job_description(
                posting_url=posting_url,
                title=normalized_title,
                company=company,
                location=location,
                site="indeed",
                detail_text=detail_text,
            )
            highlights = _extract_highlights(detail_text or description_raw)
            if not highlights:
                highlights = [line for line in description_raw.splitlines() if ":" in line][:5]

            return ScrapedJobPosting(
                title=normalized_title,
                company=company,
                location=location,
                description=description_raw,
                highlights=highlights,
                source_url=posting_url,
            )

    return None


LINKEDIN_JOB_ID_PATTERN = re.compile(r"\d{8,12}")


def _extract_linkedin_job_id(posting_url: str) -> str | None:
    parsed = urlparse(posting_url)
    query = parse_qs(parsed.query or "")
    for key in ("currentJobId", "jobId"):
        values = query.get(key)
        if not values:
            continue
        candidate = normalize_space(values[0])
        if candidate.isdigit():
            return candidate

    matches = LINKEDIN_JOB_ID_PATTERN.findall(parsed.path)
    if matches:
        return matches[-1]
    return None


def _scrape_linkedin_job_posting_with_guest_api(
    posting_url: str,
    timeout_seconds: int,
    user_agent: str,
) -> ScrapedJobPosting | None:
    """Fetch a LinkedIn job posting via its public "guest" widget API.

    LinkedIn's normal job pages block plain `requests` clients via TLS/JA3
    fingerprinting (surfaces as SSLError/EOF), but this anonymous endpoint —
    used by LinkedIn itself to render embeddable job cards — serves the same
    title/company/location/description over plain HTTP(S) without a login
    wall or fingerprint check.
    """
    job_id = _extract_linkedin_job_id(posting_url)
    if not job_id:
        return None

    api_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": posting_url,
    }
    try:
        response = _get_with_retry(api_url, headers=headers, timeout=max(5, int(timeout_seconds)))
        response.raise_for_status()
        html_text = response.text
    except requests.RequestException:
        return None

    if not html_text or not html_text.strip():
        return None

    soup = BeautifulSoup(html_text, "html.parser")

    title_node = soup.select_one("h2.top-card-layout__title") or soup.select_one("h2.topcard__title")
    title = normalize_space(title_node.get_text(" ", strip=True)) if title_node else ""

    company_node = soup.select_one("a.topcard__org-name-link")
    company = normalize_space(company_node.get_text(" ", strip=True)) if company_node else None

    location_node = soup.select_one("span.topcard__flavor--bullet")
    location = normalize_space(location_node.get_text(" ", strip=True)) if location_node else None

    description_node = soup.select_one("div.show-more-less-html__markup")
    detail_text = ""
    if description_node is not None:
        for br in description_node.find_all("br"):
            br.replace_with("\n")
        for li in description_node.find_all("li"):
            li.insert(0, "\n")
        raw_lines = description_node.get_text("\n", strip=False).splitlines()
        detail_text = "\n".join(normalize_space(line) for line in raw_lines if normalize_space(line))

    criteria_lines: list[str] = []
    for item in soup.select("li.description__job-criteria-item"):
        header = item.select_one("h3.description__job-criteria-subheader")
        value = item.select_one("span.description__job-criteria-text")
        if header is None or value is None:
            continue
        header_text = normalize_space(header.get_text(" ", strip=True))
        value_text = normalize_space(value.get_text(" ", strip=True))
        if header_text and value_text:
            criteria_lines.append(f"{header_text} {value_text}")

    if not detail_text and not criteria_lines:
        return None

    full_detail_text = "\n".join(([detail_text] if detail_text else []) + criteria_lines)

    normalized_title = title or "Job Posting"
    description_raw = compact_job_description(
        posting_url=posting_url,
        title=normalized_title,
        company=company,
        location=location,
        site="linkedin",
        detail_text=full_detail_text,
    )
    highlights = _extract_highlights(full_detail_text or description_raw)
    if not highlights:
        highlights = [line for line in description_raw.splitlines() if ":" in line][:5]

    return ScrapedJobPosting(
        title=normalized_title,
        company=company,
        location=location,
        description=description_raw,
        highlights=highlights,
        source_url=posting_url,
    )


GREENHOUSE_JOB_PATH_PATTERN = re.compile(r"^/([^/]+)/jobs/(\d+)")


def _extract_greenhouse_job_ref(posting_url: str) -> tuple[str, str] | None:
    parsed = urlparse(posting_url)
    match = GREENHOUSE_JOB_PATH_PATTERN.match(parsed.path)
    if not match:
        return None
    return match.group(1), match.group(2)


def _scrape_greenhouse_job_posting_with_api(
    posting_url: str,
    timeout_seconds: int,
    user_agent: str,
) -> ScrapedJobPosting | None:
    """Fetch a Greenhouse job posting via its public boards API.

    `boards.greenhouse.io` / `job-boards.greenhouse.io` pages are ordinary
    server-rendered HTML and normally work fine with `requests`, but the
    JSON API is the more precise (and marginally more scrape-resistant)
    source, and matches the same "prefer the vendor's public API over
    scraping a rendered page" pattern used for LinkedIn/Indeed.
    """
    job_ref = _extract_greenhouse_job_ref(posting_url)
    if job_ref is None:
        return None
    board_token, job_id = job_ref

    api_url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs/{job_id}?content=true"
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    try:
        response = _get_with_retry(api_url, headers=headers, timeout=max(5, int(timeout_seconds)))
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    title = normalize_space(str(data.get("title") or "")) or "Job Posting"
    company = normalize_space(str(data.get("company_name") or "")) or None
    location_data = data.get("location") if isinstance(data.get("location"), dict) else None
    location = normalize_space(str(location_data.get("name") or "")) if location_data else None
    location = location or None

    content_html = str(data.get("content") or "")
    detail_text = normalize_space(BeautifulSoup(content_html, "html.parser").get_text("\n", strip=True)) if content_html else ""
    if not detail_text:
        return None

    description_raw = compact_job_description(
        posting_url=posting_url,
        title=title,
        company=company,
        location=location,
        site="greenhouse",
        detail_text=detail_text,
    )
    highlights = _extract_highlights(detail_text or description_raw)
    if not highlights:
        highlights = [line for line in description_raw.splitlines() if ":" in line][:5]

    return ScrapedJobPosting(
        title=title,
        company=company,
        location=location,
        description=description_raw,
        highlights=highlights,
        source_url=posting_url,
    )


LEVER_POSTING_PATH_PATTERN = re.compile(r"^/([^/]+)/([0-9a-fA-F-]{16,})")


def _extract_lever_posting_ref(posting_url: str) -> tuple[str, str] | None:
    parsed = urlparse(posting_url)
    match = LEVER_POSTING_PATH_PATTERN.match(parsed.path)
    if not match:
        return None
    return match.group(1), match.group(2)


def _scrape_lever_job_posting_with_api(
    posting_url: str,
    timeout_seconds: int,
    user_agent: str,
) -> ScrapedJobPosting | None:
    """Fetch a Lever job posting via its public postings API (same pattern
    as the Greenhouse/LinkedIn/Indeed direct-API fallbacks)."""
    posting_ref = _extract_lever_posting_ref(posting_url)
    if posting_ref is None:
        return None
    company, posting_id = posting_ref

    api_url = f"https://api.lever.co/v0/postings/{company}/{posting_id}?mode=json"
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    try:
        response = _get_with_retry(api_url, headers=headers, timeout=max(5, int(timeout_seconds)))
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    title = normalize_space(str(data.get("text") or "")) or "Job Posting"
    # Lever's postings API has no company-name field; the URL slug is the
    # company identifier, so humanize that instead (same convention used for
    # other ATS slugs in ats_service.py).
    company_name = normalize_space(company.replace("-", " ").replace("_", " ")).title() or None
    categories = data.get("categories") if isinstance(data.get("categories"), dict) else {}
    location = normalize_space(str(categories.get("location") or "")) or None
    team = normalize_space(str(categories.get("team") or categories.get("department") or ""))

    detail_html = str(data.get("descriptionPlain") or data.get("description") or "")
    detail_text = normalize_space(BeautifulSoup(detail_html, "html.parser").get_text("\n", strip=True)) if "<" in detail_html else normalize_space(detail_html)
    lists = data.get("lists") if isinstance(data.get("lists"), list) else []
    list_lines: list[str] = []
    if team:
        list_lines.append(f"Team: {team}")
    for section in lists:
        if not isinstance(section, dict):
            continue
        section_title = normalize_space(str(section.get("text") or ""))
        section_content = str(section.get("content") or "")
        section_text = normalize_space(BeautifulSoup(section_content, "html.parser").get_text("\n", strip=True)) if section_content else ""
        if section_title:
            list_lines.append(section_title)
        if section_text:
            list_lines.extend(line for line in section_text.split("\n") if line)

    full_detail_text = "\n".join(([detail_text] if detail_text else []) + list_lines)
    if not full_detail_text:
        return None

    description_raw = compact_job_description(
        posting_url=posting_url,
        title=title,
        company=company_name,
        location=location,
        site="lever",
        detail_text=full_detail_text,
    )
    highlights = _extract_highlights(full_detail_text or description_raw)
    if not highlights:
        highlights = [line for line in description_raw.splitlines() if ":" in line][:5]

    return ScrapedJobPosting(
        title=title,
        company=company_name,
        location=location,
        description=description_raw,
        highlights=highlights,
        source_url=posting_url,
    )


ASHBY_JOB_PATH_PATTERN = re.compile(r"^/([^/]+)/([0-9a-fA-F-]{16,})")


def _extract_ashby_job_ref(posting_url: str) -> tuple[str, str] | None:
    parsed = urlparse(posting_url)
    match = ASHBY_JOB_PATH_PATTERN.match(parsed.path)
    if not match:
        return None
    return match.group(1), match.group(2)


_ASHBY_JOB_POSTING_QUERY = """
query ApiJobPosting($organizationHostedJobsPageName: String!, $jobPostingId: String!) {
  jobPosting(organizationHostedJobsPageName: $organizationHostedJobsPageName, jobPostingId: $jobPostingId) {
    id
    title
    departmentName
    teamNames
    locationName
    employmentType
    descriptionHtml
  }
}
"""


def _scrape_ashby_job_posting_with_api(
    posting_url: str,
    timeout_seconds: int,
    user_agent: str,
) -> ScrapedJobPosting | None:
    """Fetch an Ashby job posting via its public non-user GraphQL API (same
    pattern as the LinkedIn/Greenhouse/Lever direct-API fallbacks)."""
    job_ref = _extract_ashby_job_ref(posting_url)
    if job_ref is None:
        return None
    slug, job_id = job_ref

    payload = {
        "operationName": "ApiJobPosting",
        "variables": {"organizationHostedJobsPageName": slug, "jobPostingId": job_id},
        "query": _ASHBY_JOB_POSTING_QUERY,
    }
    headers = {"User-Agent": user_agent, "Content-Type": "application/json"}
    try:
        response = _post_with_retry(
            "https://jobs.ashbyhq.com/api/non-user-graphql?op=ApiJobPosting",
            headers=headers,
            json_payload=payload,
            timeout=max(5, int(timeout_seconds)),
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    posting = (data or {}).get("data", {}).get("jobPosting") if isinstance(data, dict) else None
    if not isinstance(posting, dict):
        return None

    title = normalize_space(str(posting.get("title") or "")) or "Job Posting"
    company = normalize_space(slug.replace("-", " ").replace("_", " ")).title() or None
    location = normalize_space(str(posting.get("locationName") or "")) or None
    department = normalize_space(str(posting.get("departmentName") or ""))

    content_html = str(posting.get("descriptionHtml") or "")
    detail_text = normalize_space(BeautifulSoup(content_html, "html.parser").get_text("\n", strip=True)) if content_html else ""
    if department:
        detail_text = f"Department: {department}\n{detail_text}" if detail_text else f"Department: {department}"
    if not detail_text:
        return None

    description_raw = compact_job_description(
        posting_url=posting_url,
        title=title,
        company=company,
        location=location,
        site="ashby",
        detail_text=detail_text,
    )
    highlights = _extract_highlights(detail_text or description_raw)
    if not highlights:
        highlights = [line for line in description_raw.splitlines() if ":" in line][:5]

    return ScrapedJobPosting(
        title=title,
        company=company,
        location=location,
        description=description_raw,
        highlights=highlights,
        source_url=posting_url,
    )


def _extract_workday_job_ref(posting_url: str) -> tuple[str, str, str, str] | None:
    """Parse `{company}.{wd_num}.myworkdayjobs.com/{site_id}/job/{external_path}`
    into (company, wd_num, site_id, external_path)."""
    parsed = urlparse(posting_url)
    host_parts = parsed.hostname.split(".") if parsed.hostname else []
    if len(host_parts) < 4 or not host_parts[1].lower().startswith("wd"):
        return None
    company, wd_num = host_parts[0], host_parts[1]

    path = parsed.path.strip("/")
    if not path:
        return None
    segments = path.split("/")
    if "job" not in segments:
        return None
    job_index = segments.index("job")
    site_id = "/".join(segments[:job_index])
    external_path = "/" + "/".join(segments[job_index:])
    if not site_id or external_path == "/":
        return None
    return company, wd_num, site_id, external_path


def _scrape_workday_job_posting_with_api(
    posting_url: str,
    timeout_seconds: int,
    user_agent: str,
) -> ScrapedJobPosting | None:
    """Fetch a Workday job posting via its `wday/cxs` JSON API — the same
    endpoint Workday's own careers page calls client-side, and (unlike the
    rendered page) always plain JSON with no TLS fingerprinting involved."""
    job_ref = _extract_workday_job_ref(posting_url)
    if job_ref is None:
        return None
    company, wd_num, site_id, external_path = job_ref

    base_url = f"https://{company}.{wd_num}.myworkdayjobs.com"
    api_url = f"{base_url}/wday/cxs/{company}/{site_id}{external_path}"
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
        "Origin": base_url,
        "Referer": posting_url,
    }
    try:
        response = _get_with_retry(api_url, headers=headers, timeout=max(5, int(timeout_seconds)))
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(data, dict):
        return None
    info = data.get("jobPostingInfo") if isinstance(data.get("jobPostingInfo"), dict) else None
    if not info:
        return None

    title = normalize_space(str(info.get("title") or "")) or "Job Posting"
    hiring_org = data.get("hiringOrganization") if isinstance(data.get("hiringOrganization"), dict) else {}
    company_name = normalize_space(str(hiring_org.get("name") or "")) or None
    location = normalize_space(str(info.get("location") or "")) or None

    content_html = str(info.get("jobDescription") or "")
    detail_text = normalize_space(BeautifulSoup(content_html, "html.parser").get_text("\n", strip=True)) if content_html else ""
    if not detail_text:
        return None

    description_raw = compact_job_description(
        posting_url=posting_url,
        title=title,
        company=company_name,
        location=location,
        site="workday",
        detail_text=detail_text,
    )
    highlights = _extract_highlights(detail_text or description_raw)
    if not highlights:
        highlights = [line for line in description_raw.splitlines() if ":" in line][:5]

    return ScrapedJobPosting(
        title=title,
        company=company_name,
        location=location,
        description=description_raw,
        highlights=highlights,
        source_url=posting_url,
    )


_LD_JSON_PATTERN = re.compile(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.DOTALL)


def _scrape_icims_job_posting_with_ld_json(
    posting_url: str,
    timeout_seconds: int,
    user_agent: str,
) -> ScrapedJobPosting | None:
    """Fetch an iCIMS job posting's structured (JSON-LD) data.

    Many iCIMS career portals serve a client-side-rendered shell for the
    plain job URL (no JSON-LD present) but return the fully rendered page,
    JSON-LD included, for the same URL with `in_iframe=1` appended — the
    parameter iCIMS's own portal uses to load the job content into an
    iframe. Neither variant is fingerprint-gated; this just targets the one
    that actually contains the data.
    """
    parsed = urlparse(posting_url)
    query = parse_qs(parsed.query or "")
    query["in_iframe"] = ["1"]
    iframe_url = parsed._replace(query="&".join(f"{k}={v[0]}" for k, v in query.items())).geturl()

    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"}
    try:
        response = _get_with_retry(iframe_url, headers=headers, timeout=max(5, int(timeout_seconds)))
        response.raise_for_status()
        html_text = response.text
    except requests.RequestException:
        return None

    if not html_text:
        return None

    posting: dict[str, Any] | None = None
    for match in _LD_JSON_PATTERN.finditer(html_text):
        try:
            candidate = json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(candidate, list):
            candidate = next((item for item in candidate if isinstance(item, dict) and item.get("@type") == "JobPosting"), None)
        if isinstance(candidate, dict) and candidate.get("@type") == "JobPosting":
            posting = candidate
            break

    if posting is None:
        return None

    title = normalize_space(str(posting.get("title") or "")) or "Job Posting"
    hiring_org = posting.get("hiringOrganization") if isinstance(posting.get("hiringOrganization"), dict) else {}
    company = normalize_space(str(hiring_org.get("name") or "")) or None

    job_location = posting.get("jobLocation")
    if isinstance(job_location, list):
        job_location = job_location[0] if job_location else None
    location = None
    if isinstance(job_location, dict):
        address = job_location.get("address") or {}
        if isinstance(address, dict):
            parts = [
                normalize_space(str(address.get("addressLocality") or "")),
                normalize_space(str(address.get("addressRegion") or "")),
                normalize_space(str(address.get("addressCountry") or "")),
            ]
            location = ", ".join(part for part in parts if part) or None

    content_html = str(posting.get("description") or "")
    detail_text = normalize_space(BeautifulSoup(content_html, "html.parser").get_text("\n", strip=True)) if content_html else ""
    if not detail_text:
        return None

    description_raw = compact_job_description(
        posting_url=posting_url,
        title=title,
        company=company,
        location=location,
        site="icims",
        detail_text=detail_text,
    )
    highlights = _extract_highlights(detail_text or description_raw)
    if not highlights:
        highlights = [line for line in description_raw.splitlines() if ":" in line][:5]

    return ScrapedJobPosting(
        title=title,
        company=company,
        location=location,
        description=description_raw,
        highlights=highlights,
        source_url=posting_url,
    )


# Sites with a stable, public, non-fingerprint-gated JSON/HTML API for a
# *specific* posting. Tried before the generic page fetch, since it's the
# only one of these fallbacks precise enough to hit the exact posting (the
# job-board search fallback further below only does generic keyword search).
_DIRECT_API_SCRAPERS: dict[str, Callable[[str, int, str], "ScrapedJobPosting | None"]] = {
    "linkedin": _scrape_linkedin_job_posting_with_guest_api,
    "indeed": _scrape_indeed_job_posting_with_graphql,
    "greenhouse": _scrape_greenhouse_job_posting_with_api,
    "lever": _scrape_lever_job_posting_with_api,
    "ashby": _scrape_ashby_job_posting_with_api,
    "workday": _scrape_workday_job_posting_with_api,
    "icims": _scrape_icims_job_posting_with_ld_json,
}


def _fallback_scrape_job_posting_with_job_service(posting_url: str) -> ScrapedJobPosting | None:
    site = job_site_from_url(posting_url)
    if site is None:
        return None

    rows = scrape_jobs_from_board_url(posting_url, max_items=5)
    if not rows:
        # Secondary fallback: query the new description-export path, narrowed to the same board.
        default_location = "Canada" if ".ca/" in posting_url.lower() or ".ca" in posting_url.lower() else "United States"
        exported_rows = scrape_job_descriptions_from_all_sites(
            keywords="jobs",
            location=default_location,
            site_names=[site],
            results_per_site=5,
            max_descriptions=5,
        )
        if not exported_rows:
            return None

        exported = exported_rows[0]
        title = normalize_space(str(exported.get("title") or "Job Posting")) or "Job Posting"
        description_raw = normalize_space(str(exported.get("description") or ""))
        highlights = _extract_highlights(description_raw)
        if not highlights:
            highlights = [line for line in description_raw.splitlines() if ":" in line][:5]

        return ScrapedJobPosting(
            title=title,
            company=None,
            location=None,
            description=description_raw,
            highlights=highlights,
            source_url=posting_url,
        )

    first = rows[0]
    title = normalize_space(str(first.get("title") or "Job Posting")) or "Job Posting"
    company = normalize_space(str(first.get("company") or "")) or None
    location = normalize_space(str(first.get("location") or "")) or None
    description_raw = compact_job_description(
        posting_url=posting_url,
        title=title,
        company=company,
        location=location,
        site=site,
        row=first,
    )
    highlights = _extract_highlights(description_raw)
    if not highlights:
        highlights = [line for line in description_raw.splitlines() if ":" in line][:5]

    return ScrapedJobPosting(
        title=title,
        company=company,
        location=location,
        description=description_raw,
        highlights=highlights,
        source_url=posting_url,
    )


def _scrape_job_posting_with_browser(
    posting_url: str,
    timeout_seconds: int,
) -> str | None:
    """Fetch a posting page through the persistent Chrome context.

    Sites like LinkedIn block plain `requests` calls via TLS/JA3 fingerprinting
    or bot detection (surfaces as SSLError/EOF), but allow a real browser
    through. Returns rendered HTML, or None if the browser layer is
    unavailable or the fetch fails.
    """
    if not browser_service.ensure_ready():
        return None
    # priority=True: an interactive .resumebuild/.resumecoverbuild scrape must
    # not be starved out of the single Playwright slot by reddit watcher polls.
    return browser_service.fetch_html(
        posting_url,
        timeout_ms=max(5000, int(timeout_seconds) * 1000),
        priority=True,
    )


def _append_scrape_telemetry(record: dict) -> None:
    try:
        SCRAPE_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SCRAPE_TELEMETRY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        pass


def _record_scrape_outcome(posting_url: str, rung: str, started: float, error: str = "") -> None:
    _append_scrape_telemetry(
        {
            "ts": round(time.time(), 1),
            "url": posting_url[:300],
            "site": job_site_from_url(posting_url),
            "rung": rung,
            "ok": not error,
            "elapsed": round(time.monotonic() - started, 2),
            **({"error": error[:300]} if error else {}),
        }
    )


def scrape_job_posting(
    posting_url: str,
    timeout_seconds: int = 20,
    user_agent: str = "Mozilla/5.0 (compatible; RebuiltResumeBot/1.0)",
) -> ScrapedJobPosting:
    started = time.monotonic()
    # Some job sites (LinkedIn especially) block plain `requests` clients via
    # TLS/JA3 fingerprinting (surfaces as SSLError/EOF). For sites with a
    # public, non-fingerprint-gated API for a specific posting, try that
    # first — it's both more reliable and more precise than the generic
    # page-fetch path below (which, on fallback, can only search by keyword,
    # not fetch the exact posting).
    direct_api_scraper = _DIRECT_API_SCRAPERS.get(job_site_from_url(posting_url) or "")
    if direct_api_scraper is not None:
        try:
            direct_result = direct_api_scraper(posting_url, timeout_seconds, user_agent)
        except Exception:
            # A direct-API scraper bug (schema drift, parse edge) must degrade
            # to the next rung, never kill the whole chain.
            direct_result = None
        if direct_result is not None:
            _record_scrape_outcome(posting_url, "direct_api", started)
            return direct_result

    html_rung = "page"
    html_text: str | None = None
    try:
        response = _get_with_retry(
            posting_url,
            headers={"User-Agent": user_agent},
            timeout=max(5, int(timeout_seconds)),
        )
        response.raise_for_status()
        html_text = response.text
    except requests.RequestException as page_exc:
        html_rung = "browser"
        html_text = _scrape_job_posting_with_browser(posting_url, timeout_seconds)
        if html_text is None:
            try:
                fallback = _fallback_scrape_job_posting_with_job_service(posting_url)
            except Exception:
                fallback = None
            if fallback is not None:
                _record_scrape_outcome(posting_url, "job_service", started)
                return fallback
            _record_scrape_outcome(
                posting_url, "exhausted", started,
                error=f"{type(page_exc).__name__}: {page_exc}",
            )
            raise

    soup = BeautifulSoup(html_text, "html.parser")
    h1 = soup.select_one("h1")
    page_title = normalize_space(str((soup.title.string if soup.title else "") or (h1.get_text(" ", strip=True) if h1 else "")))
    visible_text = _visible_text_from_soup(soup)
    location = _parse_location_from_text(visible_text)
    company = _parse_company_from_soup(soup)
    site = job_site_from_url(posting_url)
    description_raw = compact_job_description(
        posting_url=posting_url,
        title=page_title or "Job Posting",
        company=company,
        location=location,
        site=site,
        detail_text=visible_text,
    )
    highlights = _extract_highlights(visible_text or description_raw)

    _record_scrape_outcome(posting_url, html_rung, started)
    return ScrapedJobPosting(
        title=page_title or "Job Posting",
        company=company,
        location=location,
        description=description_raw,
        highlights=highlights,
        source_url=posting_url,
    )


def load_baseinfo_text(extra_paths: list[Path] | None = None) -> str:
    candidates = list(extra_paths or []) + [DEFAULT_BASEINFO_PATH]
    sections: list[str] = []
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            if path.name == "baseinfo.txt":
                sections.append(f"<context>\n{text}\n</context>")
            else:
                sections.append(text)
    return "\n\n".join(sections)


def load_supporting_prompt_context(
    extra_paths: list[Path] | None = None,
    template_path: Path = TEMPLATE_PATH,
) -> str:
    candidates = list(extra_paths or []) + [DEFAULT_INSTRUCTIONS_PATH]
    sections: list[str] = []
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            if path.name == "instructions.txt":
                sections.append(f"<instructions>\n{text}\n</instructions>")
            else:
                sections.append(text)

    template_text = read_resume_template(template_path).strip() if template_path.exists() else ""
    if template_text:
        sections.append(f"<template>\n{template_text}\n</template>")

    return "\n\n".join(sections)


def build_resume_rewrite_prompt(job: JobContext, scraped_job: ScrapedJobPosting, baseinfo: str, supporting_context: str) -> str:
    highlights = "\n".join(f"- {line}" for line in scraped_job.highlights[:10]) or "- No specific highlights parsed"
    company_text = scraped_job.company or "Unknown company"
    location_text = scraped_job.location or "Unknown location"

    return (
        "<task>\n"
        "Tailor the resume for the job in <input>. Only use facts present in <context> — do not invent achievements, employers, dates, certifications, or technologies. Follow the structure and style of <template> exactly.\n"
        "Where sections or bullets are irrelevant to the listing, comment them out using % for single lines or \\begin{comment}...\\end{comment} for multi-line blocks. Where commented-out sections are relevant, uncomment them. Only include content that meaningfully matches the role.\n"
        "</task>\n\n"
        "<input>\n"
        "Job metadata:\n"
        f"- Requested title: {job.title}\n"
        f"- Posting URL: {job.posting_url}\n"
        f"- Page title: {scraped_job.title}\n"
        f"- Company: {company_text}\n"
        f"- Location: {location_text}\n\n"
        "Job highlights:\n"
        f"{highlights}\n\n"
        "Job description excerpt:\n"
        f"{excerpt_job_description(scraped_job.description)}\n"
        "</input>\n\n"
        f"{baseinfo or '<context>No baseinfo content available.</context>'}\n\n"
        f"{supporting_context or '<instructions>No additional instructions provided.</instructions>'}\n\n"
        "<output_format>\n"
        "1. rewritten_tex: a complete compilable standalone LaTeX document that follows template.tex.\n"
        "2. Append exactly one <latex>...</latex> block containing the same rewritten_tex document.\n"
        "3. Write concisely — do not pad bullets, summaries, or descriptions. Every line must earn its place or the PDF will overflow.\n"
        "4. PROSE QUALITY — every bullet must read as one coherent sentence a human editor would sign off on:\n"
        "   - Lead with a strong past-tense verb and keep one idea per bullet; tie each named tool to the specific thing it did rather than chaining names ('pipeline using Kafka using Python' and 'to serve as X, serving as Y' are forbidden shapes).\n"
        "   - Use 'using' at most once per bullet, and vary sentence structure across bullets so consecutive bullets never share the same shape or opening verb.\n"
        "   - Keep each metric attached to work that could plausibly have produced it, and never end a bullet with a vague gerund tail (', ensuring reliability and performance') — state a concrete outcome or stop at the fact.\n"
        "   - Calibrate credit to <context>: solo-ownership verbs (Owned, Led, Spearheaded, Directed) only where the background states individual ownership or leadership; otherwise open with an active collaborative verb (Built, Developed, Engineered, Co-led). Where the background states a scope fact (team size, user base, live operation), state it early in the bullet — scope before tool names.\n"
        "   - ADOPT THE LISTING'S LANGUAGE wherever it truthfully describes the work: mirror the posting's exact terminology and functional vocabulary, aiming for every bullet to carry at least one listing term. Spread DIFFERENT terms across bullets rather than repeating one, and never claim a named tool, system, or certification absent from <context>.\n"
        "   - Read each rewritten bullet back to itself: if any clause would make a domain expert wince or a recruiter stumble, restructure the sentence around its strongest fact before returning.\n"
        "5. LaTeX comment environments use \\begin{comment}...\\end{comment} to close — NEVER </comment> (that is an HTML closing tag and will cause a fatal pdflatex error).\n"
        "6. STRICTLY FORBIDDEN IN OUTPUT (examples):\n"
        "   - Adding non-template sections such as 'Summary', 'Objective', or standalone 'Contact Information'.\n"
        "   - Placeholder identity/contact text such as '[Insert Address]', '[Insert Phone Number]', 'example.com', or dummy numbers.\n"
        "   - Replacing real candidate details with fake identities (for example 'John Doe').\n"
        "   - HTML tags in LaTeX output (for example </comment>, </ul>, </li>).\n"
        "   - Unsupported domain claims/terms not grounded in baseinfo (for example power systems hardware, SCADA, HVAC, substation work).\n"
        "7. If any forbidden pattern appears, regenerate before returning.\n"
        "</output_format>"
    )


def _extract_generation_text(response: Any) -> str:
    direct = str(getattr(response, "text", "") or "").strip()
    if direct:
        return direct

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = str(getattr(part, "text", "") or "").strip()
            if text:
                return text

    return ""


def _build_cached_content_config(cache_name: str) -> Any:
    try:
        from google.genai import types

        return types.GenerateContentConfig(cached_content=cache_name)
    except Exception:
        # Keep fallback simple for test doubles and degraded environments.
        return {"cached_content": cache_name}


def _truncate_prompt_for_provider(prompt: str, max_chars: int) -> str:
    """Trim a prompt to stay within a provider's character ceiling.

    Tries to break at a paragraph boundary (double newline) to avoid
    cutting mid-sentence.  Appends a short notice so the model knows
    the input was trimmed.
    """
    if len(prompt) <= max_chars:
        return prompt
    boundary = prompt.rfind("\n\n", 0, max_chars)
    cut = boundary if boundary > max_chars // 2 else max_chars
    return prompt[:cut].rstrip() + "\n\n[...content truncated to fit provider limit]"


def _openrouter_api_key() -> str | None:
    value = normalize_space(
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("openRouter")
        or _read_dotenv_value("OPENROUTER_API_KEY", "openRouter")
    )
    return value or None


def _groq_api_key() -> str | None:
    value = normalize_space(
        os.getenv("GROQ_API_KEY") or os.getenv("groqAPI") or _read_dotenv_value("GROQ_API_KEY", "groqAPI")
    )
    return value or None


def _provider_switch_candidates(
    settings: GeminiSettings,
) -> list[ResumeProviderCandidate]:
    gemini_api_key = normalize_space(str(settings.api_key or "")) or None
    providers: dict[str, ResumeProviderCandidate] = {
        "gemini": ResumeProviderCandidate(
            name="gemini",
            api_key=gemini_api_key,
            model=normalize_gemini_model_name(settings.model),
        ),
        "gemini-flash": ResumeProviderCandidate(
            name="gemini-flash",
            api_key=gemini_api_key,
            model=normalize_gemini_model_name("gemini-3.1-flash-lite"),
        ),
        "openrouter": ResumeProviderCandidate(
            name="openrouter",
            api_key=settings.openrouter_api_key or _openrouter_api_key(),
            model=settings.openrouter_model,
        ),
        "groq": ResumeProviderCandidate(
            name="groq",
            api_key=settings.groq_api_key or _groq_api_key(),
            model=settings.groq_model,
        ),
    }
    return [providers[name] for name in LLM_PROVIDER_SWITCH_ORDER if name in providers]


def _build_json_response_config() -> Any:
    try:
        from google.genai import types

        return types.GenerateContentConfig(response_mime_type="application/json")
    except Exception:
        # Keep fallback simple for test doubles and degraded environments.
        return {"response_mime_type": "application/json"}


def _generate_with_gemini(
    settings: GeminiSettings,
    prompt: str,
    cache_name: str | None,
    client_factory: Callable[[str], Any] | None,
    model_override: str | None = None,
    json_response: bool = False,
) -> str:
    factory = client_factory
    if factory is None:
        from google import genai

        factory = lambda api_key: genai.Client(api_key=api_key)
    client = factory(settings.api_key)

    model = model_override or normalize_gemini_model_name(settings.model)
    kwargs: dict[str, Any] = {"model": model, "contents": prompt}
    if cache_name:
        kwargs["config"] = _build_cached_content_config(cache_name)
    elif json_response:
        kwargs["config"] = _build_json_response_config()

    response = client.models.generate_content(**kwargs)
    return _extract_generation_text(response)


def _extract_openai_compatible_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message") if isinstance(first.get("message"), dict) else None
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            value = item.get("text")
            if isinstance(value, str):
                text_parts.append(value)
        return "\n".join(part.strip() for part in text_parts if part.strip())
    return ""


def _generate_with_openai_compatible_provider(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    include_openrouter_headers: bool = False,
    timeout_seconds: int = 45,
    json_response: bool = False,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if include_openrouter_headers:
        headers["HTTP-Referer"] = "https://discord.com"
        headers["X-Title"] = "Rebuilt Resume Bot"

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    if json_response:
        body["response_format"] = {"type": "json_object"}

    response = requests.post(
        endpoint,
        headers=headers,
        json=body,
        timeout=max(10, int(timeout_seconds)),
    )
    if json_response and response.status_code == 400:
        # Some models behind OpenRouter/Groq reject response_format; retry
        # without it rather than losing the provider entirely.
        body.pop("response_format", None)
        response = requests.post(
            endpoint,
            headers=headers,
            json=body,
            timeout=max(10, int(timeout_seconds)),
        )
    response.raise_for_status()
    payload = response.json()
    return _extract_openai_compatible_content(payload)


def _iter_provider_responses(
    settings: GeminiSettings,
    prompt: str,
    client_factory: Callable[[str], Any] | None = None,
    *,
    json_response: bool = False,
    cache_name: str | None = None,
    candidates: list[ResumeProviderCandidate] | None = None,
    errors: list[str] | None = None,
):
    """Yield (provider_name, response_text) for each provider in the fallback
    chain that produces a non-empty response.

    The single home for per-provider plumbing that every LLM call site shares:
    API-key and rate-limit-cooldown gating, capability-based prompt truncation
    and timeouts, Gemini context-cache suppression for providers without
    cachedContents, transport dispatch, and 429/RESOURCE_EXHAUSTED cooldown
    marking. Callers keep their own response validation and stop iterating
    (break/return) once a response is accepted; iterating past a yield means
    "rejected, try the next provider".

    `candidates` overrides the default fallback order (e.g. the grounding
    audit puts gemini-flash first). `errors` collects per-provider skip and
    failure reasons when the caller wants to report them.
    """
    sink = errors if errors is not None else []
    for provider in (candidates if candidates is not None else _provider_switch_candidates(settings)):
        if not provider.api_key:
            sink.append(f"{provider.name}: missing API key")
            continue
        if time.monotonic() < _PROVIDER_COOLDOWN_UNTIL.get(provider.name, 0.0):
            sink.append(f"{provider.name}: cooling down after rate limit")
            continue

        capabilities = resolve_provider_capabilities(provider.name, provider.model)
        effective_prompt = (
            _truncate_prompt_for_provider(prompt, capabilities.max_prompt_chars)
            if capabilities and len(prompt) > capabilities.max_prompt_chars
            else prompt
        )
        effective_cache_name = (
            cache_name if capabilities and capabilities.supports_context_cache else None
        )
        timeout = capabilities.request_timeout_seconds if capabilities else 45

        try:
            if provider.name in ("gemini", "gemini-flash"):
                text = _generate_with_gemini(
                    settings,
                    effective_prompt,
                    effective_cache_name,
                    client_factory,
                    model_override=provider.model if provider.name == "gemini-flash" else None,
                    json_response=json_response,
                )
            elif provider.name == "openrouter":
                text = _generate_with_openai_compatible_provider(
                    endpoint=OPENROUTER_CHAT_COMPLETIONS_URL,
                    api_key=provider.api_key,
                    model=provider.model,
                    prompt=effective_prompt,
                    include_openrouter_headers=True,
                    timeout_seconds=timeout,
                    json_response=json_response,
                )
            elif provider.name == "groq":
                text = _generate_with_openai_compatible_provider(
                    endpoint=GROQ_CHAT_COMPLETIONS_URL,
                    api_key=provider.api_key,
                    model=provider.model,
                    prompt=effective_prompt,
                    timeout_seconds=timeout,
                    json_response=json_response,
                )
            else:
                sink.append(f"{provider.name}: unsupported provider")
                continue
        except Exception as exc:
            message = str(exc)
            if "429" in message or "RESOURCE_EXHAUSTED" in message or "rate limit" in message.lower():
                _PROVIDER_COOLDOWN_UNTIL[provider.name] = time.monotonic() + PROVIDER_COOLDOWN_SECONDS
            sink.append(f"{provider.name}: {exc}")
            continue

        if not text.strip():
            sink.append(f"{provider.name}: empty response")
            continue

        yield provider.name, text


def _selection_cache_path(
    job_url: str, template_path_text: str, job_content: str = "", mode: str = "",
) -> Path:
    # job_content ties the key to what was actually scraped; mode isolates
    # aggressive/strong-aggressive selections from each other.
    key = hashlib.sha256(
        f"{job_url}|{template_path_text}|{job_content}|{mode}".encode("utf-8")
    ).hexdigest()[:24]
    return STRUCTURED_SELECTION_CACHE_DIR / f"{key}.json"


def _store_cached_selection(path: Path, raw_response: str, provider: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"ts": time.time(), "provider": provider, "raw": raw_response}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _load_cached_selection(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if time.time() - float(payload.get("ts") or 0) > SELECTION_CACHE_TTL_SECONDS:
        return None
    raw = payload.get("raw")
    return raw if isinstance(raw, str) and raw.strip() else None


def _append_structured_telemetry(record: dict) -> None:
    try:
        STRUCTURED_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STRUCTURED_TELEMETRY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        pass


def _audit_selection_grounding(
    settings: GeminiSettings,
    catalog: Any,
    selection: StructuredSelection,
    client_factory: Callable[[str], Any] | None,
) -> tuple[str, list[str]]:
    """One batched judge call over every tailored bullet; strips bullets that
    claim a domain/industry/equipment the entry's verified background never
    mentions (they render canonical instead).

    Fail-open by design: if every provider fails or the response is
    malformed, the selection is left untouched and the status string records
    why — a judge outage must never block a build. Returns
    (status, flagged_reasons).
    """
    items, id_map = build_grounding_audit_items(catalog, selection)
    if not items:
        return "skipped", []

    prompt = grounding_audit_prompt(items, tuple(catalog.skill_anchors))
    errors: list[str] = []
    # Cheapest capable model first: the audit is a small classification task.
    candidates = sorted(
        _provider_switch_candidates(settings),
        key=lambda c: 0 if c.name == "gemini-flash" else 1,
    )
    for provider_name, text in _iter_provider_responses(
        settings,
        prompt,
        client_factory,
        json_response=True,
        candidates=candidates,
        errors=errors,
    ):
        payload = extract_json_object(text)
        verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
        if not isinstance(verdicts, list):
            errors.append(f"{provider_name}: no verdicts in response")
            continue
        flagged = apply_grounding_verdicts(selection, id_map, verdicts)
        return f"ok:{provider_name}", flagged

    return "failed: " + ("; ".join(errors) if errors else "no providers available"), []


def _generate_structured_rewrite(
    settings: GeminiSettings,
    job: JobContext,
    scraped_job: ScrapedJobPosting,
    catalog: Any,
    client_factory: Callable[[str], Any] | None,
    aggressive: bool = False,
    strong_aggressive: bool = False,
) -> ResumeRewriteResult:
    """Structured pipeline: LLM returns JSON decisions, Python renders LaTeX.

    Degrades in stages when providers fail: a fresh cached selection for the
    same listing is reused first; otherwise a deterministic render with
    keyword-extracted listing terms still gets skills ordering and tool
    emphasis right, so a useful resume is always produced.

    `aggressive` sets catalog.render_config.aggressive before the prompt is
    built and before rendering, so both stay in sync (see
    structured._effective_render_config): it never touches number/employer/
    title/date/certification grounding, only the style/conservatism guards.
    `strong_aggressive` is a superset that fabricates all content from the JD.
    """
    if strong_aggressive:
        catalog.render_config.strong_aggressive = True
        catalog.render_config.aggressive = True
    elif aggressive:
        catalog.render_config.aggressive = True
    job_text = f"{job.title}\n{scraped_job.title}\n{scraped_job.description}"
    prompt = build_structured_prompt(
        job_title=job.title,
        job_description=scraped_job.description,
        job_highlights=scraped_job.highlights,
        catalog=catalog,
        extra_guidance=catalog.guidance,
    )

    provider_errors: list[str] = []
    selection: StructuredSelection | None = None
    used_provider: str | None = None
    raw_response = ""

    # No context cache here: the cached profile files instruct the legacy
    # full-LaTeX output format, which conflicts with the JSON-only contract
    # of the structured prompt.
    for provider_name, text in _iter_provider_responses(
        settings,
        prompt,
        client_factory,
        json_response=True,
        errors=provider_errors,
    ):
        candidate = parse_structured_response(text, catalog)
        if candidate is None:
            provider_errors.append(f"{provider_name}: response contained no valid JSON decisions")
            continue

        selection = candidate
        used_provider = provider_name
        raw_response = text
        break

    # Keyed by listing URL + the profile's name/contact block (profile-unique)
    # + the scraped description, so a reposted/edited listing never reuses a
    # selection made against the old text.
    _cache_mode = "sa" if strong_aggressive else ("ag" if aggressive else "")
    cache_path = _selection_cache_path(job.posting_url, catalog.header_block, scraped_job.description, mode=_cache_mode)
    if selection is not None and raw_response:
        _store_cached_selection(cache_path, raw_response, used_provider or "")

    # Stage 2: every provider failed — reuse a fresh cached selection from a
    # previous successful build of the same listing before going deterministic.
    if selection is None:
        cached_raw = _load_cached_selection(cache_path)
        if cached_raw:
            cached_selection = parse_structured_response(cached_raw, catalog)
            if cached_selection is not None:
                selection = cached_selection
                used_provider = "cache"

    # Stage 3: fully deterministic — keyword-extracted listing terms still
    # drive skills reordering and anchor-tool emphasis without any LLM.
    deterministic_fallback = False
    if selection is None:
        selection = StructuredSelection(
            keywords=extract_listing_keywords(job_text, tuple(catalog.skill_anchors)),
        )
        deterministic_fallback = True

    # Grounding audit: catch domain fabrications (industries/equipment/processes
    # the candidate never touched) that per-bullet regex validation cannot see.
    # Cached selections are audited too — they are LLM output. Deterministic
    # renders carry no rewrites, so there is nothing to audit.
    # Gate on the EFFECTIVE config, not the raw profile config: aggressive and
    # strong-aggressive modes disable the audit by design (domain reframing /
    # fabrication is their entire point), and reading the raw flag here was
    # letting the audit blank nearly every strong-aggressive bullet, which the
    # keywordless filter then scrapped — 5-bullet, 0-tailored pages.
    grounding_audit_status = "disabled"
    grounding_flagged: list[str] = []
    if _effective_render_config(catalog.render_config).grounding_audit and selection.bullets:
        grounding_audit_status, grounding_flagged = _audit_selection_grounding(
            settings, catalog, selection, client_factory
        )

    jd_tools = (
        tuple(_extract_jd_tools(job_text, catalog.skill_anchors))
        if catalog.render_config.aggressive
        else ()
    )
    latex_document, report = render_structured_resume(
        catalog, selection, jd_tools, jd_text=scraped_job.description or ""
    )

    summary = {
        "mode": "structured",
        "rewrite_scope": report.rewrite_scope,
        "visible_entries": report.visible_entries,
        "hidden_entries": report.hidden_entries,
        "visible_bullet_count": report.visible_bullet_count,
        "tailored_bullets_used": report.tailored_bullets_used,
        "canonical_fallbacks": report.canonical_fallbacks + grounding_flagged,
        "grounding_audit": grounding_audit_status,
        "grounding_flagged": grounding_flagged,
        "excluded_entries": report.excluded_entries,
        "ignored_exclusions": report.ignored_exclusions,
        "header_tech_applied": report.header_tech_applied,
        "header_tech_rejected": report.header_tech_rejected,
        "fidelity_findings": report.fidelity_findings,
        "keywords": selection.keywords,
        "core_work": selection.core_work,
        "deterministic_fallback": deterministic_fallback,
        "provider_errors": provider_errors,
        "aggressive": aggressive,
        "strong_aggressive": strong_aggressive,
    }

    aggressive_suffix = (
        " [strong-aggressive mode]" if strong_aggressive
        else " [aggressive mode]" if aggressive
        else ""
    )
    if deterministic_fallback:
        message = (
            "Structured resume rendered deterministically with extracted listing keywords "
            "(all LLM providers failed: "
            + ("; ".join(provider_errors) or "none attempted")
            + ")."
        )
    elif used_provider == "cache":
        message = (
            f"Structured resume rebuilt from cached selection (providers unavailable): "
            f"{report.visible_bullet_count} bullets, "
            f"{report.tailored_bullets_used} tailored.{aggressive_suffix}"
        )
    else:
        message = (
            f"Structured resume tailored via {used_provider}: "
            f"{report.visible_bullet_count} bullets, {report.tailored_bullets_used} tailored.{aggressive_suffix}"
        )

    _append_structured_telemetry(
        {
            "ts": round(time.time(), 1),
            "title": job.title[:120],
            "url": job.posting_url,
            "provider": used_provider or ("deterministic" if deterministic_fallback else None),
            "bullets": report.visible_bullet_count,
            "tailored": report.tailored_bullets_used,
            "fallback_count": len(report.canonical_fallbacks),
            "fallback_reasons": report.canonical_fallbacks,
            "fidelity_count": len(report.fidelity_findings),
            "header_tech_applied": len(report.header_tech_applied),
            "grounding_audit": grounding_audit_status,
            "grounding_flagged": grounding_flagged,
            "deterministic": deterministic_fallback,
            "provider_error_count": len(provider_errors),
        }
    )

    return ResumeRewriteResult(
        status="ok",
        message=message,
        rewritten_resume=json.dumps(summary, indent=2, ensure_ascii=True),
        prompt_preview=prompt[:500],
        scraped_job=scraped_job,
        latex_document=latex_document,
        used_provider=used_provider,
    )


def generate_resume_rewrite(
    settings: GeminiSettings,
    job: JobContext,
    cache_name: str | None,
    baseinfo_paths: list[Path] | None = None,
    support_paths: list[Path] | None = None,
    template_path: Path = TEMPLATE_PATH,
    scraper: Callable[[str], ScrapedJobPosting] | None = None,
    client_factory: Callable[[str], Any] | None = None,
    aggressive: bool = False,
    strong_aggressive: bool = False,
) -> ResumeRewriteResult:
    scrape = scraper or scrape_job_posting
    try:
        scraped_job = scrape(job.posting_url)
    except Exception as exc:
        return ResumeRewriteResult(status="error", message=f"Failed to scrape posting URL: {exc}")

    # Profiles whose template uses the % [category] convention get the
    # structured pipeline: the LLM only returns JSON decisions and the LaTeX is
    # rendered deterministically from template parts, so it always compiles.
    baseinfo_path = (baseinfo_paths or [DEFAULT_BASEINFO_PATH])[0]
    catalog = load_structured_profile(template_path, baseinfo_path)
    if catalog is not None:
        return _generate_structured_rewrite(
            settings,
            job,
            scraped_job,
            catalog,
            client_factory,
            aggressive=aggressive,
            strong_aggressive=strong_aggressive,
        )

    baseinfo = load_baseinfo_text(baseinfo_paths)
    supporting_context = load_supporting_prompt_context(support_paths, template_path=template_path)
    prompt = build_resume_rewrite_prompt(job, scraped_job, baseinfo, supporting_context)

    provider_errors: list[str] = []
    text = ""
    used_provider: str | None = None

    # Context caching is Gemini-only; the iterator suppresses cache_name for
    # providers that do not support cachedContents to avoid silent failures.
    for provider_name, text in _iter_provider_responses(
        settings,
        prompt,
        client_factory,
        cache_name=cache_name,
        errors=provider_errors,
    ):
        # Validate LaTeX extraction before committing to this provider.
        # Some models return non-empty text that contains no valid LaTeX document
        # (e.g. a refusal, a truncated response, or a plain-text resume). Fall
        # through to the next provider in that case rather than returning
        # latex_document=None to the caller.
        _candidate: str | None = None
        try:
            _parsed_check = json.loads(text)
            if isinstance(_parsed_check, dict):
                for _key in ("rewritten_tex", "latex", "latex_document"):
                    _val = _parsed_check.get(_key)
                    if isinstance(_val, str) and "\\documentclass" in _val and "\\end{document}" in _val:
                        _candidate = _val.strip()
                        break
        except json.JSONDecodeError:
            pass
        if _candidate is None:
            _candidate = extract_latex_document(text)
        if _candidate is not None:
            used_provider = provider_name
            break
        provider_errors.append(f"{provider_name}: response contained no valid LaTeX document")

    if not text:
        details = "; ".join(provider_errors) if provider_errors else "No providers attempted"
        return ResumeRewriteResult(
            status="error",
            message=f"All resume providers failed: {details}",
            prompt_preview=prompt[:500],
            scraped_job=scraped_job,
            used_cache_name=cache_name,
        )

    parsed_payload: dict[str, Any] | None = None

    # Keep response machine-readable when possible.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed_payload = parsed
        pretty_text = json.dumps(parsed, indent=2, ensure_ascii=True)
    except json.JSONDecodeError:
        pretty_text = text

    latex_document: str | None = None
    if parsed_payload is not None:
        for key in ("rewritten_tex", "latex", "latex_document"):
            value = parsed_payload.get(key)
            if not isinstance(value, str):
                continue
            candidate = value.strip()
            if "\\documentclass" in candidate and "\\end{document}" in candidate:
                latex_document = candidate
                break
    if latex_document is None:
        latex_document = extract_latex_document(text)

    return ResumeRewriteResult(
        status="ok",
        message=f"Generated tailored resume content from job posting context via {used_provider or 'unknown provider'}.",
        rewritten_resume=pretty_text,
        prompt_preview=prompt[:500],
        scraped_job=scraped_job,
        used_cache_name=cache_name,
        latex_document=latex_document,
        used_provider=used_provider or None,
    )


def build_latex_repair_prompt(broken_latex: str, compile_error_excerpt: str) -> str:
    return (
        "<task>\n"
        "The LaTeX document in <broken_latex> failed to compile with pdflatex/xelatex. "
        "The compiler's error output is in <compile_error>. Fix ONLY the syntax that is "
        "breaking compilation. Do not rewrite, reword, shorten, expand, or restructure "
        "any content — every word, bullet, section, and command that isn't the direct "
        "cause of the compile error must remain byte-for-byte identical.\n"
        "</task>\n\n"
        "<compile_error>\n"
        f"{compile_error_excerpt[-3000:]}\n"
        "</compile_error>\n\n"
        "<broken_latex>\n"
        f"{broken_latex}\n"
        "</broken_latex>\n\n"
        "<output_format>\n"
        "1. Return the complete corrected LaTeX document — not just the fixed line/section.\n"
        "2. Append exactly one <latex>...</latex> block containing that document.\n"
        "3. Do not add commentary, explanations, or markdown fences outside the tag.\n"
        "</output_format>"
    )


def generate_validated_with_providers(
    prompt: str,
    settings: GeminiSettings,
    validate: Callable[[str], Any],
    client_factory: Callable[[str], Any] | None = None,
) -> tuple[Any, str | None]:
    """Run a prompt through the provider fallback chain (Gemini -> OpenRouter
    -> Groq) and return the first response that `validate` accepts.

    `validate` maps raw response text to a usable value, or None to reject it
    (rejection advances to the next provider). Returns (value, provider_name),
    or (None, None) when no configured provider produced an accepted response.
    """
    for provider_name, text in _iter_provider_responses(settings, prompt, client_factory):
        value = validate(text)
        if value is not None:
            return value, provider_name

    return None, None


def _generate_fixed_latex_with_providers(
    prompt: str,
    original_latex: str,
    settings: GeminiSettings,
    client_factory: Callable[[str], Any] | None = None,
) -> tuple[str | None, str | None]:
    """Run a LaTeX-fixing prompt through the provider fallback chain and
    return the first response containing a complete LaTeX document different
    from the input. Returns (fixed_latex, provider_name) or (None, None)."""

    def _validate(text: str) -> str | None:
        fixed = extract_latex_document(text)
        if fixed and "\\documentclass" in fixed and "\\end{document}" in fixed and fixed != original_latex:
            return fixed
        return None

    return generate_validated_with_providers(prompt, settings, _validate, client_factory)


def repair_latex_with_llm(
    broken_latex: str,
    compile_error_excerpt: str,
    settings: GeminiSettings,
    client_factory: Callable[[str], Any] | None = None,
) -> tuple[str | None, str | None]:
    """Ask an available LLM provider to fix a LaTeX document that failed to
    compile, as a last resort after the deterministic auto-fix passes in
    resume.py's `compile_latex_to_pdf` have been exhausted.

    Returns (fixed_latex, provider_name), or (None, None) if no configured
    provider produced a usable fix.
    """
    prompt = build_latex_repair_prompt(broken_latex, compile_error_excerpt)
    return _generate_fixed_latex_with_providers(prompt, broken_latex, settings, client_factory)


def repair_latex_until_compiles(
    latex_document: str,
    initial_result: Any,
    settings: GeminiSettings,
    compiler: Callable[[str], Any],
    max_rounds: int = 2,
    client_factory: Callable[[str], Any] | None = None,
) -> tuple[str, Any, str | None, int]:
    """Iteratively LLM-repair a failing LaTeX document, feeding each new
    compile error back to the model, until it compiles or rounds run out.

    A single repair pass often trades one compile error for another (fixing
    an unclosed brace can expose a bad macro further down); iterating on the
    *fresh* error each round converges where the one-shot approach gave up.

    `compiler` maps a LaTeX document to a LatexCompileResult-like object
    (duck-typed: `.status` and `.log_excerpt` are used). Returns
    (final_document, final_result, provider_name, rounds_attempted); the
    original inputs come back unchanged when no repair succeeded, so callers
    can use the result unconditionally.
    """
    current_document = latex_document
    current_result = initial_result
    rounds_attempted = 0

    for _ in range(max(0, int(max_rounds))):
        if getattr(current_result, "status", None) != "error" or not getattr(current_result, "log_excerpt", None):
            break
        fixed_latex, provider = repair_latex_with_llm(
            current_document,
            current_result.log_excerpt,
            settings,
            client_factory,
        )
        if not fixed_latex:
            break
        rounds_attempted += 1
        repaired_result = compiler(fixed_latex)
        if getattr(repaired_result, "status", None) == "ok":
            return fixed_latex, repaired_result, provider, rounds_attempted
        current_document = fixed_latex
        current_result = repaired_result

    return latex_document, initial_result, None, rounds_attempted


def build_latex_condense_prompt(latex_document: str, page_count: int, max_pages: int) -> str:
    return (
        "<task>\n"
        f"The LaTeX resume in <overflowing_latex> compiles cleanly but produces {page_count} pages; "
        f"the expected format is at most {max_pages} page(s). Reduce it to fit by commenting out the "
        "LEAST relevant bullet points using a leading % (never delete lines, never use "
        "\\begin{comment}). Prefer commenting out whole bullets over rewording. You may tighten "
        "wording only where a bullet barely wraps onto a new line. Do not remove or comment out "
        "section headers, education, contact details, or entire experience/project blocks, and do "
        "not change the preamble, packages, or any formatting commands.\n"
        "</task>\n\n"
        "<overflowing_latex>\n"
        f"{latex_document}\n"
        "</overflowing_latex>\n\n"
        "<output_format>\n"
        "1. Return the complete condensed LaTeX document — every line of the original must still be "
        "present, either active or %-commented.\n"
        "2. Append exactly one <latex>...</latex> block containing that document.\n"
        "3. Do not add commentary, explanations, or markdown fences outside the tag.\n"
        "</output_format>"
    )


def condense_latex_if_overflowing(
    latex_document: str,
    compile_result: Any,
    max_pages: int,
    settings: GeminiSettings,
    compiler: Callable[[str], Any],
    client_factory: Callable[[str], Any] | None = None,
) -> tuple[str, Any, str | None]:
    """If a compiled resume exceeds the page budget, ask an LLM to comment
    out the least-relevant bullets and recompile. Strictly best-effort: the
    condensed version is only adopted when it compiles AND has fewer pages
    than the original, so the caller can never end up worse off.

    Returns (document, compile_result, provider_name); provider_name is None
    when the original was kept.
    """
    page_count = getattr(compile_result, "page_count", None)
    if (
        getattr(compile_result, "status", None) != "ok"
        or not page_count
        or page_count <= max(1, int(max_pages))
    ):
        return latex_document, compile_result, None

    prompt = build_latex_condense_prompt(latex_document, page_count, max_pages)
    condensed_latex, provider = _generate_fixed_latex_with_providers(prompt, latex_document, settings, client_factory)
    if not condensed_latex:
        return latex_document, compile_result, None

    condensed_result = compiler(condensed_latex)
    condensed_pages = getattr(condensed_result, "page_count", None)
    if (
        getattr(condensed_result, "status", None) == "ok"
        and condensed_pages is not None
        and condensed_pages < page_count
    ):
        return condensed_latex, condensed_result, provider

    return latex_document, compile_result, None


def extract_job_urls(lines: list[str], message_content: str) -> tuple[str | None, str | None]:
    posting_url: str | None = None
    apply_url: str | None = None

    for line in lines[1:]:
        lowered = line.lower()
        if lowered.startswith("apply:"):
            apply_url = apply_url or extract_first_url(line)
            continue
        posting_url = posting_url or extract_first_url(line)

    return posting_url or extract_first_url(message_content), apply_url


def extract_job_context_from_message(message_content: str) -> JobContext | None:
    lines = extract_message_lines(message_content)
    if not lines:
        return None

    posting_url, apply_url = extract_job_urls(lines, message_content)
    if posting_url is None:
        return None

    return JobContext(
        title=clean_title(lines[0]),
        posting_url=posting_url,
        apply_url=apply_url,
        source_message=message_content.strip(),
    )