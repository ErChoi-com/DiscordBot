from __future__ import annotations

import json
import os
import re
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
from .cache import normalize_gemini_model_name
from .configkey import GeminiSettings
from .resume import LLM_PROVIDER_SWITCH_ORDER, PROVIDER_CAPABILITIES, TEMPLATE_PATH, extract_latex_document, read_resume_template

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
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
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


def scrape_job_posting(
    posting_url: str,
    timeout_seconds: int = 20,
    user_agent: str = "Mozilla/5.0 (compatible; RebuiltResumeBot/1.0)",
) -> ScrapedJobPosting:
    try:
        response = requests.get(
            posting_url,
            headers={"User-Agent": user_agent},
            timeout=max(5, int(timeout_seconds)),
        )
        response.raise_for_status()
    except requests.RequestException:
        if job_site_from_url(posting_url) == "indeed":
            graphql_fallback = _scrape_indeed_job_posting_with_graphql(
                posting_url=posting_url,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
            )
            if graphql_fallback is not None:
                return graphql_fallback

        fallback = _fallback_scrape_job_posting_with_job_service(posting_url)
        if fallback is not None:
            return fallback
        raise

    soup = BeautifulSoup(response.text, "html.parser")
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
            sections.append(f"Source: {path.name}\n{text}")
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
            sections.append(f"Source: {path.name}\n{text}")

    template_text = read_resume_template(template_path).strip() if template_path.exists() else ""
    if template_text:
        sections.append(f"Source: template.tex\n{template_text}")

    return "\n\n".join(sections)


def build_resume_rewrite_prompt(job: JobContext, scraped_job: ScrapedJobPosting, baseinfo: str, supporting_context: str) -> str:
    highlights = "\n".join(f"- {line}" for line in scraped_job.highlights[:10]) or "- No specific highlights parsed"
    company_text = scraped_job.company or "Unknown company"
    location_text = scraped_job.location or "Unknown location"

    return (
        "Rewrite and tailor the resume content for this job. Keep claims factual and grounded in cached resume context.\n"
        "Do not invent achievements, employers, dates, certifications, or technologies that are not present in the cached candidate context or base info.\n"
        "The LaTeX output must be derived from the provided template.tex structure and style.\n"
        "Return JSON with keys exactly: summary, targeted_bullets, revised_profile, rewritten_tex.\n\n"
        "Job metadata:\n"
        f"- Requested title: {job.title}\n"
        f"- Posting URL: {job.posting_url}\n"
        f"- Page title: {scraped_job.title}\n"
        f"- Company: {company_text}\n"
        f"- Location: {location_text}\n\n"
        "Job highlights:\n"
        f"{highlights}\n\n"
        "Job description excerpt:\n"
        f"{scraped_job.description[:4500]}\n\n"
        "Base information:\n"
        f"{baseinfo or 'No baseinfo content available.'}\n\n"
        "LaTeX instructions and template context:\n"
        f"{supporting_context or 'No additional LaTeX instructions provided.'}\n\n"
        "Output contract:\n"
        "1. summary: 3 to 5 sentence rationale for changes.\n"
        "2. targeted_bullets: 6 to 10 resume bullet points tailored to this role.\n"
        "3. revised_profile: one concise professional summary paragraph tailored to this role.\n"
        "4. rewritten_tex: a complete compilable standalone LaTeX document that follows template.tex.\n"
        "5. Also append exactly one <latex>...</latex> block containing the same rewritten_tex document."
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


def _provider_switch_candidates(settings: GeminiSettings) -> list[ResumeProviderCandidate]:
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
            model=normalize_gemini_model_name("gemini-2.5-flash"),
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


def _generate_with_gemini(
    settings: GeminiSettings,
    prompt: str,
    cache_name: str | None,
    client_factory: Callable[[str], Any] | None,
    model_override: str | None = None,
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
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if include_openrouter_headers:
        headers["HTTP-Referer"] = "https://discord.com"
        headers["X-Title"] = "Rebuilt Resume Bot"

    response = requests.post(
        endpoint,
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=max(10, int(timeout_seconds)),
    )
    response.raise_for_status()
    payload = response.json()
    return _extract_openai_compatible_content(payload)


def generate_resume_rewrite(
    settings: GeminiSettings,
    job: JobContext,
    cache_name: str | None,
    baseinfo_paths: list[Path] | None = None,
    support_paths: list[Path] | None = None,
    template_path: Path = TEMPLATE_PATH,
    scraper: Callable[[str], ScrapedJobPosting] | None = None,
    client_factory: Callable[[str], Any] | None = None,
) -> ResumeRewriteResult:
    scrape = scraper or scrape_job_posting
    try:
        scraped_job = scrape(job.posting_url)
    except Exception as exc:
        return ResumeRewriteResult(status="error", message=f"Failed to scrape posting URL: {exc}")

    baseinfo = load_baseinfo_text(baseinfo_paths)
    supporting_context = load_supporting_prompt_context(support_paths, template_path=template_path)
    prompt = build_resume_rewrite_prompt(job, scraped_job, baseinfo, supporting_context)

    provider_errors: list[str] = []
    text = ""
    used_provider: str | None = None

    for provider in _provider_switch_candidates(settings):
        if not provider.api_key:
            provider_errors.append(f"{provider.name}: missing API key")
            continue

        capabilities = PROVIDER_CAPABILITIES.get(provider.name)
        effective_prompt = (
            _truncate_prompt_for_provider(prompt, capabilities.max_prompt_chars)
            if capabilities and len(prompt) > capabilities.max_prompt_chars
            else prompt
        )
        # Context caching is Gemini-only; suppress cache_name for providers
        # that do not support cachedContents to avoid silent failures.
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
                )
            elif provider.name == "openrouter":
                text = _generate_with_openai_compatible_provider(
                    endpoint=OPENROUTER_CHAT_COMPLETIONS_URL,
                    api_key=provider.api_key,
                    model=provider.model,
                    prompt=effective_prompt,
                    include_openrouter_headers=True,
                    timeout_seconds=timeout,
                )
            elif provider.name == "groq":
                text = _generate_with_openai_compatible_provider(
                    endpoint=GROQ_CHAT_COMPLETIONS_URL,
                    api_key=provider.api_key,
                    model=provider.model,
                    prompt=effective_prompt,
                    timeout_seconds=timeout,
                )
            else:
                provider_errors.append(f"{provider.name}: unsupported provider")
                continue
        except Exception as exc:
            provider_errors.append(f"{provider.name}: {exc}")
            continue

        if text.strip():
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
                used_provider = provider.name
                break
            provider_errors.append(f"{provider.name}: response contained no valid LaTeX document")
            continue

        provider_errors.append(f"{provider.name}: empty response")

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