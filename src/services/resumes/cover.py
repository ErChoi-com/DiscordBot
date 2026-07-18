"""Cover letter generation that complements a tailored resume.

The cover letter's PRIMARY material is everything the structured resume
pipeline could NOT fit for this listing: hidden entries, honoured exclusions,
inclusions that lost the budget race, and their baseinfo facts. The LLM only
returns JSON paragraphs; the LaTeX letter is rendered deterministically from a
fixed, dependency-free preamble so it always compiles.

Degradation mirrors the structured resume pipeline: providers -> cached resume
selection (so the letter pairs with the resume actually built for the same
listing) -> fully deterministic letter assembled from omitted content and
extracted listing keywords.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .configkey import GeminiSettings
from . import listing as _listing
from .listing import (
    JobContext,
    ScrapedJobPosting,
    STRUCTURED_CACHE_ROOT,
    _iter_provider_responses,
    _provider_switch_candidates,
    scrape_job_posting,
)
from .structured import (
    StructuredSelection,
    TemplateCatalog,
    extract_json_object,
    extract_listing_keywords,
    grounding_audit_prompt,
    load_structured_profile,
    parse_structured_response,
    render_structured_resume,
)

COVER_TELEMETRY_PATH = STRUCTURED_CACHE_ROOT / "cover_builds.jsonl"

MIN_BODY_PARAGRAPHS = 2
MAX_BODY_PARAGRAPHS = 4
MAX_PARAGRAPH_CHARS = 900

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_PATTERN = re.compile(r"(?:\+?1[\s.\-])?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}")
_LINK_PATTERN = re.compile(r"(?:linkedin\.com|github\.com)/[A-Za-z0-9_./\-]+")
_NAME_LINE_PATTERN = re.compile(r"^\s*-?\s*Name:\s*(.+)$", re.MULTILINE)
_HTML_TAG_PATTERN = re.compile(r"</?[a-zA-Z][^>]*>")
_LATEX_COMMAND_PATTERN = re.compile(r"\\[a-zA-Z@]+")
_NUMBER_TOKEN_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


@dataclass(slots=True)
class OmittedEntry:
    title: str
    bullets: list[str]
    extra_facts: str = ""


@dataclass(slots=True)
class CoverInventory:
    omitted: list[OmittedEntry]
    visible_titles: list[str]
    selection_source: str  # "cache" | "deterministic" | "legacy"
    profile_facts: str = ""
    keywords: list[str] = field(default_factory=list)
    skill_anchors: tuple[str, ...] = ()


@dataclass(slots=True)
class CoverLetterResult:
    status: str
    message: str
    latex_document: str | None = None
    used_provider: str | None = None
    scraped_job: ScrapedJobPosting | None = None
    summary: dict[str, Any] | None = None


def _append_cover_telemetry(record: dict) -> None:
    try:
        COVER_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with COVER_TELEMETRY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        pass


def latex_escape(text: str) -> str:
    """Escape user/LLM-derived plain text for safe LaTeX interpolation."""
    result = []
    for ch in str(text or ""):
        if ch == "\\":
            result.append(r"\textbackslash{}")
        elif ch in "&%$#_{}":
            result.append(f"\\{ch}")
        elif ch == "~":
            result.append(r"\textasciitilde{}")
        elif ch == "^":
            result.append(r"\textasciicircum{}")
        else:
            result.append(ch)
    return "".join(result)


def latex_to_plain(bullet: str) -> str:
    """Flatten a template bullet's LaTeX markup into plain prose."""
    text = str(bullet or "")
    text = re.sub(r"\\(textbf|textit|emph|texttt)\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\(textbf|textit|emph|texttt)\{([^{}]*)\}", r"\2", text)  # nested pass
    text = text.replace(r"\%", "%").replace(r"\&", "&").replace(r"\$", "$")
    text = text.replace(r"\_", "_").replace(r"\#", "#")
    text = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", "").replace("}", "").replace("\\\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_candidate_identity(baseinfo_text: str, header_block: str = "") -> dict[str, str]:
    """Best-effort name/contact extraction; every field may be empty."""
    combined = f"{baseinfo_text}\n{header_block}"
    name_match = _NAME_LINE_PATTERN.search(baseinfo_text)
    name = name_match.group(1).strip() if name_match else ""
    if not name and header_block:
        bold = re.search(r"\\textbf\{([^{}]+)\}", header_block)
        huge = re.search(r"\\(?:Huge|LARGE|Large)\s*(?:\\scshape\s*)?([A-Za-z .'\-]{3,60})", header_block)
        candidate = (bold.group(1) if bold else "") or (huge.group(1) if huge else "")
        name = latex_to_plain(candidate)
    email = _EMAIL_PATTERN.search(combined)
    phone = _PHONE_PATTERN.search(combined)
    links = list(dict.fromkeys(_LINK_PATTERN.findall(combined)))[:2]
    return {
        "name": name,
        "email": email.group(0) if email else "",
        "phone": phone.group(0).strip() if phone else "",
        "links": " | ".join(links),
    }


def _reproduce_resume_selection(
    catalog: TemplateCatalog,
    job: JobContext,
    scraped_job: ScrapedJobPosting,
) -> tuple[StructuredSelection, str]:
    """Recover the selection the resume build used for this exact listing.

    Reads the same 24h selection cache generate_resume_rewrite writes, so a
    cover built after `.resumebuild` complements that resume precisely. With
    no cached selection (standalone cover), falls back to the deterministic
    canonical selection with extracted listing keywords.
    """
    job_text = f"{job.title}\n{scraped_job.title}\n{scraped_job.description}"
    cache_path = _listing._selection_cache_path(
        job.posting_url, catalog.header_block, scraped_job.description
    )
    cached_raw = _listing._load_cached_selection(cache_path)
    if cached_raw:
        cached_selection = parse_structured_response(cached_raw, catalog)
        if cached_selection is not None:
            return cached_selection, "cache"

    return (
        StructuredSelection(
            keywords=extract_listing_keywords(job_text, tuple(catalog.skill_anchors)),
        ),
        "deterministic",
    )


def build_cover_inventory(
    catalog: TemplateCatalog,
    selection: StructuredSelection,
    selection_source: str,
) -> CoverInventory:
    """Render the resume in-memory and collect everything it left out."""
    _, report = render_structured_resume(catalog, selection)

    # hidden_entries is everything that did not make this render (excluded
    # entries included), so it is the complete omission list by construction.
    omitted_ids = list(report.hidden_entries)

    omitted: list[OmittedEntry] = []
    for entry_id in omitted_ids:
        entry = catalog.entry(entry_id)
        if entry is None:
            continue
        block_facts = "\n".join(
            catalog.baseinfo_blocks.get(category.lower(), "")
            for category in entry.categories
        ).strip()
        omitted.append(
            OmittedEntry(
                title=latex_to_plain(entry.title),
                bullets=[latex_to_plain(bullet) for bullet in entry.bullets],
                extra_facts=block_facts,
            )
        )

    visible_titles = [
        latex_to_plain(entry.title)
        for entry_id in report.visible_entries
        if (entry := catalog.entry(entry_id)) is not None
    ]

    return CoverInventory(
        omitted=omitted,
        visible_titles=visible_titles,
        selection_source=selection_source,
        profile_facts=catalog.baseinfo_profile.strip(),
        keywords=list(selection.keywords),
        skill_anchors=tuple(catalog.skill_anchors),
    )


def build_legacy_inventory(baseinfo_text: str) -> CoverInventory:
    """Legacy (untagged) profiles have no omission report; the whole baseinfo
    is the letter's material."""
    bullets = [
        line.lstrip("- ").strip()
        for line in baseinfo_text.splitlines()
        if line.strip().startswith("- ")
    ]
    return CoverInventory(
        omitted=[OmittedEntry(title="Candidate background", bullets=bullets)],
        visible_titles=[],
        selection_source="legacy",
        profile_facts=baseinfo_text.split("==", 1)[0].strip(),
    )


def extract_listing_points(scraped_job: ScrapedJobPosting, keywords: list[str]) -> list[str]:
    """Everything useful the description asks for: the scraper's
    requirement/responsibility highlight lines first, then extracted listing
    keywords not already covered by a highlight."""
    points: list[str] = []
    seen: set[str] = set()
    for line in scraped_job.highlights:
        normalized = re.sub(r"\s+", " ", str(line or "")).strip()
        key = normalized.lower()
        if len(normalized) < 15 or key in seen:
            continue
        seen.add(key)
        points.append(normalized)
        if len(points) >= 10:
            break
    highlight_blob = " ".join(points).lower()
    for keyword in keywords:
        cleaned = str(keyword or "").strip()
        if cleaned and cleaned.lower() not in highlight_blob and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            points.append(cleaned)
        if len(points) >= 14:
            break
    return points


def build_cover_letter_prompt(
    job: JobContext,
    scraped_job: ScrapedJobPosting,
    inventory: CoverInventory,
    identity: dict[str, str],
    listing_points: list[str],
) -> str:
    omitted_lines: list[str] = []
    for entry in inventory.omitted:
        omitted_lines.append(f"* {entry.title}")
        omitted_lines.extend(f"  - {bullet}" for bullet in entry.bullets)
        if entry.extra_facts:
            omitted_lines.append(f"  Additional true facts: {entry.extra_facts}")
    omitted_text = "\n".join(omitted_lines) or "(nothing was omitted; draw on the resume material)"
    on_resume_text = "\n".join(f"* {title}" for title in inventory.visible_titles) or "(unknown)"
    points_text = "\n".join(f"{i}. {point}" for i, point in enumerate(listing_points, 1)) or "(none parsed)"

    return (
        "<task>\n"
        "Write the body of a one-page cover letter for the job in <job>. Work through <listing_requirements> — everything useful the description asks for — and address every requirement the candidate can truthfully support. Prefer evidence from <omitted> (the resume could not include it, so the letter is its only home); requirements already proven by <on_resume> items get at most a brief reference, never a restatement. Skip requirements the provided material cannot support — never fabricate coverage.\n"
        "</task>\n\n"
        "<job>\n"
        f"Role: {job.title}\n"
        f"Company: {scraped_job.company or 'Unknown'}\n"
        f"Location: {scraped_job.location or 'Unknown'}\n"
        "Description excerpt:\n"
        f"{scraped_job.description[:3000]}\n"
        "</job>\n\n"
        "<listing_requirements>\n"
        f"{points_text}\n"
        "</listing_requirements>\n\n"
        "<omitted>\n"
        f"{omitted_text}\n"
        "</omitted>\n\n"
        "<on_resume>\n"
        f"{on_resume_text}\n"
        "</on_resume>\n\n"
        "<candidate>\n"
        f"Name: {identity.get('name') or 'The candidate'}\n"
        f"{inventory.profile_facts}\n"
        "</candidate>\n\n"
        "<rules>\n"
        "1. Use ONLY facts present in <omitted>, <on_resume>, <candidate>, or <job>. Never invent employers, tools, metrics, certifications, or dates.\n"
        "2. Every number you write must appear verbatim in the provided material.\n"
        "3. Plain text only: no LaTeX, HTML, markdown, bullet lists, or headings.\n"
        f"4. {MIN_BODY_PARAGRAPHS}-{MAX_BODY_PARAGRAPHS} body paragraphs, 50-110 words each. Do NOT write a greeting or sign-off; those are added separately.\n"
        "5. Name the company and the role title within the first paragraph.\n"
        "6. Organize by the listing's own priorities: cover the highest-numbered-relevance requirements first, using the listing's vocabulary as plain text. Work the listing's own terminology into every paragraph where it truthfully applies to the candidate's material.\n"
        "7. Specific and factual; no clichés (passionate, team player, fast learner), no filler.\n"
        "8. Where the material states ownership or scope (sole developer, team size, user base, live operation), open the relevant paragraph with that framing — scope and ownership before tool names. Never claim leadership or solo credit the material does not state.\n"
        "</rules>\n\n"
        "<output_format>\n"
        'Return ONLY a JSON object: {"body_paragraphs": ["...", "..."], "closing_sentence": "..."}\n'
        "</output_format>"
    )


def _grounding_text(inventory: CoverInventory, scraped_job: ScrapedJobPosting, identity: dict[str, str]) -> str:
    parts: list[str] = [scraped_job.description, scraped_job.title, inventory.profile_facts]
    parts.extend(identity.values())
    parts.extend(inventory.visible_titles)
    for entry in inventory.omitted:
        parts.append(entry.title)
        parts.extend(entry.bullets)
        parts.append(entry.extra_facts)
    return "\n".join(part for part in parts if part)


def _candidate_grounding_text(inventory: CoverInventory, identity: dict[str, str]) -> str:
    """Candidate-only facts, deliberately WITHOUT the job description.

    Used for the grounding audit (unlike `_grounding_text`'s number-check
    grounding): the job posting's own vocabulary must not count as "true
    background" or a listing that asks for SCADA would make claiming SCADA
    look grounded.
    """
    parts: list[str] = [inventory.profile_facts]
    parts.extend(identity.values())
    parts.extend(inventory.visible_titles)
    for entry in inventory.omitted:
        parts.append(entry.title)
        parts.extend(entry.bullets)
        parts.append(entry.extra_facts)
    return "\n".join(part for part in parts if part)


def _audit_cover_paragraphs(
    settings: GeminiSettings,
    paragraphs: list[str],
    grounding: str,
    skill_anchors: tuple[str, ...],
    client_factory: Callable[[str], Any] | None,
) -> tuple[set[int], list[str]]:
    """LLM judge pass over cover-letter paragraphs, mirroring the resume
    pipeline's grounding audit — the replacement for the old static
    forbidden-terms list. Flags any paragraph making a concrete claim
    (domain, named platform/technology, equipment, function substitution)
    the candidate's real background does not support.

    Fail-open: a judge outage returns no flags, so paragraphs are never
    dropped just because every provider was unavailable. Returns
    (flagged_indices, reasons).
    """
    if not paragraphs:
        return set(), []
    items = [
        {"id": i, "section": "Cover Letter", "grounding": grounding, "bullet": text}
        for i, text in enumerate(paragraphs)
    ]
    prompt = grounding_audit_prompt(items, skill_anchors)
    candidates = sorted(
        _provider_switch_candidates(settings),
        key=lambda c: 0 if c.name == "gemini-flash" else 1,
    )
    for _provider_name, text in _iter_provider_responses(
        settings,
        prompt,
        client_factory,
        json_response=True,
        candidates=candidates,
    ):
        payload = extract_json_object(text)
        verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
        if not isinstance(verdicts, list):
            continue

        flagged: set[int] = set()
        reasons: list[str] = []
        for verdict in verdicts:
            if not isinstance(verdict, dict) or not verdict.get("fabricated"):
                continue
            try:
                vid = int(verdict.get("id"))
            except (TypeError, ValueError):
                continue
            if 0 <= vid < len(paragraphs):
                flagged.add(vid)
                phrase = " ".join(str(verdict.get("phrase") or "").split())[:60]
                detail = f" {phrase!r}" if phrase else ""
                reasons.append(f"paragraph {vid}: ungrounded claim{detail}")
        return flagged, reasons

    return set(), []


def validate_cover_paragraph(
    text: str,
    grounding: str,
) -> tuple[str | None, str]:
    """Returns (cleaned paragraph, "") or (None, reason).

    Domain/tool fabrication is NOT checked here — there is no per-candidate
    blocklist to check against. That job belongs to the LLM grounding audit
    (`_audit_cover_paragraphs`), the same plausibility arbiter the resume
    pipeline uses, run once over all surviving paragraphs together.
    """
    candidate = re.sub(r"\s+", " ", str(text or "")).strip()
    if not candidate or len(candidate) < 40:
        return None, "too short"
    if len(candidate) > MAX_PARAGRAPH_CHARS:
        candidate = candidate[:MAX_PARAGRAPH_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "."
    if _HTML_TAG_PATTERN.search(candidate):
        return None, "contains HTML"
    if _LATEX_COMMAND_PATTERN.search(candidate):
        return None, "contains LaTeX markup"
    if "**" in candidate or candidate.lstrip().startswith(("#", "-", "*")):
        return None, "contains markdown"

    grounding_digits = set(_NUMBER_TOKEN_PATTERN.findall(grounding.replace(",", "")))
    for token in _NUMBER_TOKEN_PATTERN.findall(candidate.replace(",", "")):
        if token not in grounding_digits:
            return None, f"ungrounded number {token!r}"

    return candidate, ""


def _point_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9+#.]{3,}", text.lower())}


def _best_omitted_match(point: str, inventory: CoverInventory) -> tuple[OmittedEntry, str] | None:
    """The omitted bullet with the strongest token overlap against a listing
    requirement; None when nothing overlaps at all."""
    point_tokens = _point_tokens(point)
    best: tuple[int, OmittedEntry, str] | None = None
    for entry in inventory.omitted:
        for bullet in entry.bullets:
            overlap = len(point_tokens & _point_tokens(f"{entry.title} {bullet}"))
            if overlap and (best is None or overlap > best[0]):
                best = (overlap, entry, bullet)
    if best is None:
        return None
    return best[1], best[2]


def deterministic_cover_paragraphs(
    job: JobContext,
    scraped_job: ScrapedJobPosting,
    inventory: CoverInventory,
    listing_points: list[str],
) -> list[str]:
    """Stage-3 fallback: a grounded, requirement-driven letter with zero LLM
    involvement — each covered point pairs a description ask with the omitted
    fact that answers it."""
    company = scraped_job.company or "your team"
    keywords = inventory.keywords[:4]
    focus = ", ".join(keywords) if keywords else "the role's core requirements"

    paragraphs = [
        (
            f"I am writing to apply for the {job.title} position at {company}. "
            f"Alongside the attached resume, I want to address the posting's requirements directly, "
            f"including experience the one-page format could not accommodate, particularly {focus}."
        )
    ]

    used_bullets: set[str] = set()
    for point in listing_points:
        match = _best_omitted_match(point, inventory)
        if match is None:
            continue
        entry, bullet = match
        if bullet in used_bullets:
            continue
        used_bullets.add(bullet)
        anchor = point if len(point) <= 90 else point[:90].rsplit(" ", 1)[0]
        paragraphs.append(f"Where the posting asks for {anchor}: {entry.title}. {bullet}")
        if len(paragraphs) >= MAX_BODY_PARAGRAPHS - 1:
            break

    # Nothing matched a requirement — fall back to surfacing omitted material.
    if len(paragraphs) == 1:
        for entry in inventory.omitted[:2]:
            if entry.bullets:
                paragraphs.append(f"Beyond my resume: {entry.title}. {entry.bullets[0]}")

    paragraphs.append(
        "I would welcome the opportunity to discuss how this additional background "
        "supports the goals described in the posting. Thank you for your consideration."
    )
    return paragraphs[:MAX_BODY_PARAGRAPHS]


def render_cover_letter_latex(
    identity: dict[str, str],
    job: JobContext,
    scraped_job: ScrapedJobPosting,
    paragraphs: list[str],
    closing_sentence: str = "",
) -> str:
    """Deterministic letter document with a fixed, dependency-free preamble."""
    name = identity.get("name") or "Candidate"
    contact_bits = [bit for bit in (identity.get("email"), identity.get("phone"), identity.get("links")) if bit]
    contact_line = latex_escape("  |  ".join(contact_bits))
    company = scraped_job.company or ""
    location = scraped_job.location or ""
    recipient_lines = [latex_escape(part) for part in (company, location) if part]

    body_parts: list[str] = []
    for paragraph in paragraphs:
        body_parts.append(latex_escape(paragraph))
    if closing_sentence:
        body_parts.append(latex_escape(closing_sentence))
    body = "\n\n".join(body_parts)

    recipient_block = ("\\\n".join(recipient_lines) + "\n\n") if recipient_lines else ""
    greeting_target = latex_escape(f"{company} Hiring Team" if company else "Hiring Team")

    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{parskip}\n"
        "\\pagenumbering{gobble}\n"
        "\\begin{document}\n\n"
        f"{{\\LARGE \\textbf{{{latex_escape(name)}}}}}\n\n"
        + (f"{contact_line}\n\n" if contact_line else "")
        + "\\bigskip\n\n"
        "\\today\n\n"
        "\\bigskip\n\n"
        + recipient_block
        + f"Re: {latex_escape(job.title)}\n\n"
        "\\bigskip\n\n"
        f"Dear {greeting_target},\n\n"
        f"{body}\n\n"
        "\\bigskip\n\n"
        "Sincerely,\n\n"
        f"{latex_escape(name)}\n\n"
        "\\end{document}\n"
    )


def _parse_cover_response(text: str) -> tuple[list[str], str] | None:
    payload = extract_json_object(text)
    if not isinstance(payload, dict):
        return None
    raw_paragraphs = payload.get("body_paragraphs")
    if isinstance(raw_paragraphs, str):
        raw_paragraphs = [chunk for chunk in re.split(r"\n\s*\n", raw_paragraphs) if chunk.strip()]
    if not isinstance(raw_paragraphs, list):
        return None
    paragraphs = [str(item).strip() for item in raw_paragraphs if str(item).strip()]
    if not paragraphs:
        return None
    closing = str(payload.get("closing_sentence") or "").strip()
    return paragraphs[:MAX_BODY_PARAGRAPHS], closing


def generate_cover_letter(
    settings: GeminiSettings,
    job: JobContext,
    template_path: Path,
    baseinfo_path: Path,
    scraper: Callable[[str], ScrapedJobPosting] | None = None,
    client_factory: Callable[[str], Any] | None = None,
) -> CoverLetterResult:
    scrape = scraper or scrape_job_posting
    try:
        scraped_job = scrape(job.posting_url)
    except Exception as exc:
        return CoverLetterResult(status="error", message=f"Failed to scrape posting URL: {exc}")

    try:
        baseinfo_text = baseinfo_path.read_text(encoding="utf-8")
    except OSError:
        baseinfo_text = ""

    catalog = load_structured_profile(template_path, baseinfo_path)
    if catalog is not None:
        selection, selection_source = _reproduce_resume_selection(catalog, job, scraped_job)
        inventory = build_cover_inventory(catalog, selection, selection_source)
        identity = parse_candidate_identity(baseinfo_text, catalog.header_block)
    else:
        inventory = build_legacy_inventory(baseinfo_text)
        identity = parse_candidate_identity(baseinfo_text)

    if not inventory.keywords:
        job_text = f"{job.title}\n{scraped_job.title}\n{scraped_job.description}"
        inventory.keywords = extract_listing_keywords(job_text, ())

    listing_points = extract_listing_points(scraped_job, inventory.keywords)
    prompt = build_cover_letter_prompt(job, scraped_job, inventory, identity, listing_points)
    grounding = _grounding_text(inventory, scraped_job, identity)
    candidate_grounding = _candidate_grounding_text(inventory, identity)

    provider_errors: list[str] = []
    used_provider: str | None = None
    paragraphs: list[str] = []
    closing_sentence = ""
    dropped: list[str] = []

    for provider_name, text in _iter_provider_responses(
        settings,
        prompt,
        client_factory,
        json_response=True,
        errors=provider_errors,
    ):
        parsed = _parse_cover_response(text)
        if parsed is None:
            provider_errors.append(f"{provider_name}: response contained no valid JSON paragraphs")
            continue

        candidate_paragraphs, closing_sentence = parsed
        validated: list[str] = []
        for raw_paragraph in candidate_paragraphs:
            cleaned, reason = validate_cover_paragraph(raw_paragraph, grounding)
            if cleaned is None:
                dropped.append(reason)
                continue
            validated.append(cleaned)
        if closing_sentence:
            cleaned_closing, reason = validate_cover_paragraph(closing_sentence, grounding)
            closing_sentence = cleaned_closing or ""
            if cleaned_closing is None:
                dropped.append(f"closing: {reason}")

        # LLM grounding audit: the plausibility arbiter for domain/tool claims
        # (replaces the old static forbidden-terms list). Audited together in
        # one call; the closing sentence rides along as the last item.
        audit_input = list(validated)
        closing_index = len(audit_input) if closing_sentence else None
        if closing_sentence:
            audit_input.append(closing_sentence)
        if audit_input:
            flagged, audit_reasons = _audit_cover_paragraphs(
                settings, audit_input, candidate_grounding, inventory.skill_anchors, client_factory
            )
            dropped.extend(audit_reasons)
            if closing_index is not None and closing_index in flagged:
                closing_sentence = ""
            validated = [p for i, p in enumerate(validated) if i not in flagged]

        if len(validated) < MIN_BODY_PARAGRAPHS:
            provider_errors.append(
                f"{provider_name}: only {len(validated)} paragraph(s) survived validation"
            )
            continue

        paragraphs = validated
        used_provider = provider_name
        break

    deterministic = False
    if not paragraphs:
        paragraphs = deterministic_cover_paragraphs(job, scraped_job, inventory, listing_points)
        closing_sentence = ""
        deterministic = True

    # The company/role anchor sentence is non-negotiable for a cover letter;
    # if the model ignored rule 5, prepend the deterministic intro.
    company_name = (scraped_job.company or "").strip()
    first_paragraph = paragraphs[0].lower() if paragraphs else ""
    if company_name and company_name.lower() not in first_paragraph:
        intro = deterministic_cover_paragraphs(job, scraped_job, inventory, listing_points)[0]
        paragraphs = [intro] + paragraphs[: MAX_BODY_PARAGRAPHS - 1]

    latex_document = render_cover_letter_latex(identity, job, scraped_job, paragraphs, closing_sentence)

    omitted_count = len(inventory.omitted)
    if deterministic:
        message = (
            f"Cover letter rendered deterministically from {omitted_count} omitted resume item(s) "
            "(all LLM providers failed: " + ("; ".join(provider_errors) or "none attempted") + ")."
        )
    elif inventory.selection_source == "legacy":
        message = f"Cover letter written via {used_provider} from full baseinfo (legacy profile)."
    else:
        pairing = (
            "paired with the cached resume selection"
            if inventory.selection_source == "cache"
            else "against the canonical resume selection (run .resumebuild first for exact pairing)"
        )
        message = (
            f"Cover letter written via {used_provider} from {omitted_count} omitted resume item(s), "
            f"{pairing}."
        )

    summary = {
        "mode": "cover",
        "selection_source": inventory.selection_source,
        "omitted_entries": [entry.title for entry in inventory.omitted],
        "visible_entries": inventory.visible_titles,
        "listing_points": listing_points,
        "paragraphs": len(paragraphs),
        "dropped_paragraphs": dropped,
        "deterministic": deterministic,
        "provider_errors": provider_errors,
    }

    _append_cover_telemetry(
        {
            "ts": round(time.time(), 1),
            "title": job.title[:120],
            "url": job.posting_url,
            "provider": used_provider or "deterministic",
            "selection_source": inventory.selection_source,
            "omitted": omitted_count,
            "paragraphs": len(paragraphs),
            "dropped": len(dropped),
            "deterministic": deterministic,
            "provider_error_count": len(provider_errors),
        }
    )

    return CoverLetterResult(
        status="ok",
        message=message,
        latex_document=latex_document,
        used_provider=used_provider,
        scraped_job=scraped_job,
        summary=summary,
    )
