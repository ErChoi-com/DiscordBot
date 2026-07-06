"""Structured resume tailoring: the LLM returns JSON decisions, Python renders LaTeX.

The legacy pipeline asked the LLM to rewrite the entire .tex document, which
produced a long tail of structural failures (broken preambles, HTML tags,
unclosed comment environments, invented entries, dropped metrics). This module
removes LaTeX authoring from the LLM entirely:

  1. `parse_template_catalog` reads the profile's template.tex into a catalog of
     entries keyed by the `% [category]` tag convention. Preamble, headers,
     skills, and education are kept verbatim and never touch the LLM.
  2. `parse_role_guide` reads the `== ROLE TYPE SELECTION GUIDE ==` block from
     baseinfo.txt into MUST SHOW / SHOW / HIDE rules per role family.
  3. The LLM is asked for a small JSON object: role family, relevance ranking,
     and tailored bullet texts. Nothing else.
  4. `render_structured_resume` enforces visibility rules, bullet counts, metric
     preservation, and bullet-text sanitisation, then renders the document from
     template parts. Any bullet that fails validation falls back to its
     canonical template text, so the output always compiles.

Profiles whose template has no `% [category]` tags keep the legacy free-form
path (see listing.generate_resume_rewrite).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Bullet-count window required by the profile instructions ("fill the page").
# The ceiling is 16 rather than the instructions' historical 18: sample compiles
# showed 17 bullets across 6 entries overflow to a second page. Profiles can
# override these via structured_config.json (see RenderConfig).
MIN_VISIBLE_BULLETS = 14
MAX_VISIBLE_BULLETS = 16
MIN_MUST_SHOW_BULLETS = 2
MAX_BULLET_CHARS = 400
# Cap on HIDE entries the model may promote per render. Keeps a runaway
# "include everything" response from overriding the role guide wholesale.
MAX_INCLUSIONS = 2
# Appended "…, demonstrating <posting phrase>" clauses are keyword padding in a
# trench coat. If the canonical bullet did not use the construction, reject it.
# The second group covers the empty purpose-clause tails HR review flagged
# ("…, ensuring clean, performant rendering", "…, driving successful customer
# relations"): gerunds that assert an outcome without adding a fact.
FILLER_CLAUSE_PATTERN = re.compile(
    r",\s*(demonstrating|showcasing|highlighting|underscoring|evidencing|exemplifying"
    r"|ensuring|enabling|supporting|driving|prioritizing|prioritising|facilitating|streamlining|empowering)\b",
    re.IGNORECASE,
)

STRUCTURED_CONFIG_FILENAME = "structured_config.json"
STRUCTURED_GUIDANCE_FILENAME = "structured_guidance.txt"

# LaTeX commands allowed inside an LLM-tailored bullet. Anything else means the
# model tried to write structure, which is exactly what this pipeline forbids.
ALLOWED_BULLET_COMMANDS = frozenset(
    {"textbf", "textit", "texttt", "emph", "times", "%", "&", "_", "#", "$"}
)

# Forbidden terms are per-candidate data: one candidate's "never claim Rust"
# is another candidate's core skill. They therefore live in each profile's
# structured_config.json ("forbidden_terms"), never in this module. A tailored
# bullet that introduces one of a profile's terms (when the canonical bullet
# does not contain it) is discarded in favour of the canonical bullet.

CATEGORY_TAG_PATTERN = re.compile(r"^%\s*\[([a-z][a-z0-9_, ]*)\]\s*$", re.MULTILINE)
# Both \section{...} and \section*{...} — templates differ.
SECTION_PATTERN = re.compile(r"\\section\*?\{([^}]*)\}")

# Known bullet-list conventions, tried in order. Each is (list_begin_pattern,
# list_end_pattern, item_style). "env" = \begin{itemize}...\item...\end{itemize};
# "braced" = Jake-style \resumeItemListStart...\resumeItem{...}...\resumeItemListEnd.
LIST_CONVENTIONS: tuple[tuple[str, str, str, str], ...] = (
    (r"\\begin\{itemize\}", r"\\end\{itemize\}", r"\\item\b", "env"),
    (r"\\resumeItemListStart", r"\\resumeItemListEnd", r"\\resumeItem", "braced"),
)
LATEX_COMMAND_PATTERN = re.compile(r"\\([a-zA-Z]+|.)")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
UNESCAPED_SPECIAL_PATTERN = re.compile(r"(?<!\\)([&%#_])")


@dataclass(slots=True)
class RenderConfig:
    """Per-profile rendering knobs, optionally loaded from structured_config.json."""

    min_visible_bullets: int = MIN_VISIBLE_BULLETS
    max_visible_bullets: int = MAX_VISIBLE_BULLETS
    min_must_show_bullets: int = MIN_MUST_SHOW_BULLETS
    max_bullet_chars: int = MAX_BULLET_CHARS
    # Total characters across all rendered bullets; the one-page ceiling.
    # Canonical bullets average ~190 chars; 14-15 x 190 ≈ 2800.
    max_total_bullet_chars: int = 2800
    # Profile-specific terms a tailored bullet may never introduce. Empty by
    # default: what counts as an off-limits claim differs per candidate.
    forbidden_terms: tuple[str, ...] = ()
    # Appended on top of forbidden_terms (legacy alias, kept for config compat).
    extra_forbidden_terms: tuple[str, ...] = ()
    # How much text authority the LLM gets. Selection decisions (role family,
    # ranking, ordering, exclusions) always come from the model; this dial
    # controls only bullet REWRITES, the sole source of prose defects:
    #   "full"      — every visible bullet may be rewritten (legacy behavior)
    #   "limited"   — only the first `limited_rewrite_bullets` bullets in
    #                 final document order may be rewritten; the rest render
    #                 canonical. Bounds slop to the bullets recruiters read.
    #   "selection" — no rewrites at all; every bullet renders canonical.
    rewrite_scope: str = "full"
    limited_rewrite_bullets: int = 5
    # Bolding beyond this many \textbf{} per bullet reads as keyword farming;
    # extras are demoted to plain text (first-come priority).
    max_bold_per_bullet: int = 3
    # Per-role-family ordering for Skills-section items, keyed by family KEY
    # ("EMBEDDED", "ML", ...). Applied after listing-keyword matches, so an
    # embedded resume leads Languages with C/C++ even when the listing text
    # never names a language. Falls back to the family's detection keywords.
    family_skill_priority: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "RenderConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(payload, dict):
            return cls()
        config = cls()
        for field_name in (
            "min_visible_bullets",
            "max_visible_bullets",
            "min_must_show_bullets",
            "max_bullet_chars",
            "max_total_bullet_chars",
            "limited_rewrite_bullets",
            "max_bold_per_bullet",
        ):
            value = payload.get(field_name)
            if isinstance(value, int) and value > 0:
                setattr(config, field_name, value)
        scope = payload.get("rewrite_scope")
        if isinstance(scope, str) and scope.strip().lower() in ("full", "limited", "selection"):
            config.rewrite_scope = scope.strip().lower()
        priority_map = payload.get("family_skill_priority")
        if isinstance(priority_map, dict):
            config.family_skill_priority = {
                str(key).upper(): tuple(str(item) for item in values if str(item).strip())
                for key, values in priority_map.items()
                if isinstance(values, list)
            }
        replacement = payload.get("forbidden_terms")
        if isinstance(replacement, list):
            config.forbidden_terms = tuple(
                str(term).lower() for term in replacement if str(term).strip()
            )
        terms = payload.get("extra_forbidden_terms")
        if isinstance(terms, list):
            config.extra_forbidden_terms = tuple(
                str(term).lower() for term in terms if str(term).strip()
            )
        if config.min_visible_bullets > config.max_visible_bullets:
            config.min_visible_bullets, config.max_visible_bullets = (
                config.max_visible_bullets,
                config.min_visible_bullets,
            )
        return config


@dataclass(slots=True)
class TemplateEntry:
    entry_id: str
    section: str
    categories: tuple[str, ...]
    header: str  # verbatim header line(s), including trailing \\ and % comments
    bullets: tuple[str, ...]  # verbatim item bodies
    title: str
    smallskip: bool = False  # entry was preceded by \smallskip in the template
    # The entry's own list convention, captured from the template so the
    # renderer emits exactly the structure the template used.
    list_begin: str = "\\begin{itemize}"
    list_end: str = "\\end{itemize}"
    item_command: str = "\\item"
    item_style: str = "env"  # "env" -> "\item text"; "braced" -> "\cmd{text}"


@dataclass(slots=True)
class TemplateCatalog:
    preamble: str  # through \begin{document}
    header_block: str  # NAME + CONTACT block, verbatim
    entry_sections: tuple[str, ...]  # section names containing tagged entries
    entries: list[TemplateEntry]
    tail: str  # Skills/Education/etc., verbatim, up to \end{document}
    render_config: RenderConfig = field(default_factory=RenderConfig)
    guidance: str = ""  # optional per-profile prompt guidance
    # Tools/technologies the candidate genuinely has (parsed from baseinfo's
    # SKILL ANCHORS section). Empty tuple disables the invented-tool check.
    skill_anchors: tuple[str, ...] = ()
    # Verbatim \section command per entry section (templates vary: \section vs
    # \section*), and any section-level wrapper lines that sit between the
    # section command and the first entry / after the last entry (e.g.
    # Jake-style \resumeSubHeadingListStart / \resumeSubHeadingListEnd).
    section_headers: dict[str, str] = field(default_factory=dict)
    section_chrome: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Extra verified facts from baseinfo.txt, keyed by category tag — baseinfo
    # entries carry the same [tag] labels as the template. Shown to the model
    # per-entry and used as additional validation grounding.
    baseinfo_blocks: dict[str, str] = field(default_factory=dict)
    # Candidate-level facts (the baseinfo header before the first == section).
    baseinfo_profile: str = ""

    def entry(self, entry_id: str) -> TemplateEntry | None:
        for item in self.entries:
            if item.entry_id == entry_id:
                return item
        return None


@dataclass(slots=True)
class RoleFamily:
    key: str  # short stable key, e.g. "EMBEDDED"
    name: str  # full heading, e.g. "EMBEDDED / FIRMWARE / HARDWARE roles"
    keywords: tuple[str, ...]
    must_show: frozenset[str]
    show: frozenset[str]
    hide: frozenset[str]


@dataclass(slots=True)
class StructuredSelection:
    """Validated LLM decisions (or deterministic defaults when the LLM fails)."""

    role_family_key: str
    ranking: list[str] = field(default_factory=list)
    bullets: dict[str, list[str]] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    # SHOW entries the model judges a poor fit for this specific listing.
    # MUST SHOW entries can never be excluded; exclusions are rolled back if
    # the page cannot be filled without them.
    exclusions: list[str] = field(default_factory=list)
    # HIDE entries the model wants promoted because the listing explicitly
    # calls for them. Bounded: budget-gated and capped (MAX_INCLUSIONS).
    inclusions: list[str] = field(default_factory=list)
    # Per-entry bullet indices the model considers weakest (weakest first);
    # used to steer trimming when the render is over budget.
    weak_bullets: dict[str, list[int]] = field(default_factory=dict)
    # Per-entry bullet display order, most relevant first ("front-load the
    # page"). Canonical indices; invalid/partial orders are merged tolerantly.
    bullet_order: dict[str, list[int]] = field(default_factory=dict)


@dataclass(slots=True)
class RenderReport:
    role_family_key: str
    visible_entries: list[str]
    hidden_entries: list[str]
    visible_bullet_count: int
    tailored_bullets_used: int
    canonical_fallbacks: list[str]  # "entry_id[index]: reason"
    excluded_entries: list[str] = field(default_factory=list)  # exclusions honoured
    ignored_exclusions: list[str] = field(default_factory=list)  # must-show or budget rollback
    included_extras: list[str] = field(default_factory=list)  # HIDE entries promoted
    ignored_inclusions: list[str] = field(default_factory=list)  # no budget / over cap
    fidelity_findings: list[str] = field(default_factory=list)  # template-structure drift
    rewrite_scope: str = "full"  # which rewrite dial rendered this document


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "entry"


def _extract_entry_title(header: str) -> str:
    match = re.search(r"\\textbf\{([^}]*)\}", header)
    if match:
        title = match.group(1).strip().rstrip(",")
        brace = re.search(r"\\textbf\{[^}]*\}\s*\{([^}]*)\}", header)
        if brace and brace.group(1).strip():
            title = f"{title} {brace.group(1).strip()}"
        return title
    # Macro-style headers (\resumeSubheading{Title}{...}{...}{...}): the first
    # braced argument is the title.
    macro = re.search(r"\\[a-zA-Z]+\s*\{([^{}]*)\}", header)
    if macro and macro.group(1).strip():
        return macro.group(1).strip()
    return header.strip()


def parse_template_catalog(template_text: str) -> TemplateCatalog | None:
    """Parse a template that follows the `% [category]` entry convention.

    Returns None when the template does not use the convention, signalling the
    caller to keep the legacy free-form pipeline for this profile.
    """
    begin_doc = template_text.find("\\begin{document}")
    end_doc = template_text.rfind("\\end{document}")
    if begin_doc == -1 or end_doc == -1:
        return None
    if not CATEGORY_TAG_PATTERN.search(template_text):
        return None

    preamble = template_text[: begin_doc + len("\\begin{document}")]
    body = template_text[begin_doc + len("\\begin{document}") : end_doc]

    sections = list(SECTION_PATTERN.finditer(body))
    if not sections:
        return None

    header_block = body[: sections[0].start()].strip("\n")

    entries: list[TemplateEntry] = []
    entry_sections: list[str] = []
    section_headers: dict[str, str] = {}
    section_chrome: dict[str, tuple[str, str]] = {}
    tail_start: int | None = None

    for index, section_match in enumerate(sections):
        section_name = section_match.group(1).strip()
        section_end = sections[index + 1].start() if index + 1 < len(sections) else len(body)
        section_text = body[section_match.start() : section_end]

        section_entries, prefix, suffix = _parse_section_entries(section_name, section_text)
        if section_entries:
            entries.extend(section_entries)
            entry_sections.append(section_name)
            section_headers[section_name] = section_match.group(0)
            section_chrome[section_name] = (prefix, suffix)
        elif tail_start is None and entry_sections:
            tail_start = section_match.start()

    if not entries:
        return None

    tail = body[tail_start:].strip("\n") if tail_start is not None else ""

    seen: set[str] = set()
    for entry in entries:
        base = entry.entry_id
        suffix_n = 2
        while entry.entry_id in seen:
            entry.entry_id = f"{base}-{suffix_n}"
            suffix_n += 1
        seen.add(entry.entry_id)

    return TemplateCatalog(
        preamble=preamble,
        header_block=header_block,
        entry_sections=tuple(entry_sections),
        entries=entries,
        tail=tail,
        section_headers=section_headers,
        section_chrome=section_chrome,
    )


# Lines that are template plumbing rather than entry content. They must never
# survive into a parsed header: the renderer emits its own structure, so a
# leaked \begin{comment} or \smallskip would unbalance the output.
_HEADER_NOISE_PATTERN = re.compile(
    r"^\s*(\\begin\{comment\}|\\end\{comment\}|\\smallskip|\\medskip|\\bigskip)\s*$"
)


def _clean_header_lines(raw_header: str) -> tuple[str, bool]:
    """Strip structural noise from a captured header; returns (header, smallskip)."""
    kept: list[str] = []
    smallskip = False
    for line in raw_header.splitlines():
        if _HEADER_NOISE_PATTERN.match(line):
            if "skip" in line:
                smallskip = True
            continue
        if line.strip():
            kept.append(line.rstrip())
    return "\n".join(kept).strip(), smallskip


def _preceded_by_smallskip(section_text: str, match_start: int) -> bool:
    """True when the lines just before an entry's tag contain \\smallskip."""
    preceding = section_text[:match_start].splitlines()
    for line in reversed(preceding[-4:]):
        stripped = line.strip()
        if not stripped or stripped in ("\\begin{comment}", "\\end{comment}"):
            continue
        return stripped in ("\\smallskip", "\\medskip", "\\bigskip")
    return False


def _extract_braced_items(body: str, item_command_pattern: str) -> list[str]:
    """Extract brace-balanced arguments of item macros like \\resumeItem{...}."""
    items: list[str] = []
    for match in re.finditer(item_command_pattern + r"\s*\{", body):
        depth = 1
        index = match.end()
        start = index
        while index < len(body) and depth:
            char = body[index]
            if char == "{" and body[index - 1] != "\\":
                depth += 1
            elif char == "}" and body[index - 1] != "\\":
                depth -= 1
            index += 1
        if depth == 0:
            item = body[start : index - 1].strip()
            if item:
                items.append(item)
    return items


def _clean_chrome_lines(raw: str) -> str:
    """Keep section-level wrapper lines, dropping comment/skip plumbing."""
    kept = [
        line.rstrip()
        for line in raw.splitlines()
        if line.strip() and not _HEADER_NOISE_PATTERN.match(line)
    ]
    return "\n".join(kept)


def _parse_section_entries(
    section_name: str, section_text: str
) -> tuple[list[TemplateEntry], str, str]:
    """Parse a section's tagged entries; returns (entries, prefix, suffix).

    prefix/suffix are section-level wrapper lines around the entries (e.g.
    Jake-style \\resumeSubHeadingListStart / \\resumeSubHeadingListEnd) that
    must be re-emitted for the render to match the template's structure.
    """
    for list_begin_pat, list_end_pat, item_pat, item_style in LIST_CONVENTIONS:
        pattern = re.compile(
            r"^%\s*\[([a-z][a-z0-9_, ]*)\]\s*\n(.*?)"
            + list_begin_pat
            + r"(.*?)"
            + list_end_pat,
            re.MULTILINE | re.DOTALL,
        )
        matches = list(pattern.finditer(section_text))
        if not matches:
            continue

        entries: list[TemplateEntry] = []
        for match in matches:
            categories = tuple(
                part.strip() for part in match.group(1).split(",") if part.strip()
            )
            header, header_smallskip = _clean_header_lines(match.group(2))
            if not header:
                continue
            if item_style == "env":
                bullets = tuple(
                    bullet.strip()
                    for bullet in re.split(item_pat, match.group(3))
                    if bullet.strip()
                )
            else:
                bullets = tuple(_extract_braced_items(match.group(3), item_pat))
            if not bullets:
                continue
            title = _extract_entry_title(header)
            entries.append(
                TemplateEntry(
                    entry_id=_slugify(title),
                    section=section_name,
                    categories=categories,
                    header=header,
                    bullets=bullets,
                    title=title,
                    smallskip=header_smallskip
                    or _preceded_by_smallskip(section_text, match.start()),
                    list_begin=list_begin_pat.replace("\\\\", "\\").replace("\\{", "{").replace("\\}", "}"),
                    list_end=list_end_pat.replace("\\\\", "\\").replace("\\{", "{").replace("\\}", "}"),
                    item_command=item_pat.replace("\\\\", "\\").replace(r"\b", ""),
                    item_style=item_style,
                )
            )
        if not entries:
            continue

        section_command_end = SECTION_PATTERN.search(section_text)
        content_start = section_command_end.end() if section_command_end else 0
        prefix = _clean_chrome_lines(section_text[content_start : matches[0].start()])
        suffix = _clean_chrome_lines(section_text[matches[-1].end() :])
        return entries, prefix, suffix

    return [], "", ""


def parse_baseinfo_blocks(baseinfo_text: str) -> tuple[dict[str, str], str]:
    """Extract per-entry background facts and candidate-level profile facts.

    Baseinfo entries start with the same "[tag] Title..." labels the template
    uses, so each block maps to catalog entries by category tag. Returns
    ({tag: block_text}, profile_header_text). The ROLE TYPE SELECTION GUIDE
    and later machine-oriented sections are excluded.
    """
    guide_start = re.search(r"==\s*ROLE TYPE SELECTION GUIDE\s*==", baseinfo_text)
    scope = baseinfo_text[: guide_start.start()] if guide_start else baseinfo_text

    first_section = re.search(r"^==\s", scope, re.MULTILINE)
    profile = scope[: first_section.start()].strip() if first_section else scope.strip()

    blocks: dict[str, str] = {}
    block_pattern = re.compile(
        r"^\[([a-z][a-z0-9_, ]*)\]\s*(.+?)(?=^\[[a-z]|^==\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for match in block_pattern.finditer(scope):
        text = match.group(2).strip()
        if not text:
            continue
        for tag in match.group(1).split(","):
            tag = tag.strip()
            if tag:
                blocks[tag] = f"{blocks[tag]}\n{text}" if tag in blocks else text
    return blocks, profile


def parse_skill_anchors(baseinfo_text: str) -> tuple[str, ...]:
    """Parse the candidate's real toolset from baseinfo's SKILL ANCHORS section.

    Lines look like "Languages: JavaScript/Node.js, Java, Python, ..." — every
    comma-separated token after a label is an anchor. Returns lowercase tokens;
    empty when the profile has no such section (check disabled).
    """
    match = re.search(
        r"==\s*SKILL ANCHORS\s*==(.*?)(?:\n==\s|\Z)",
        baseinfo_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ()
    anchors: list[str] = []
    for line in match.group(1).splitlines():
        _, _, values = line.partition(":")
        for token in (values if values else line).split(","):
            cleaned = token.strip().lower()
            if cleaned and len(cleaned) < 60:
                anchors.append(cleaned)
                # "JavaScript/Node.js" should also anchor its parts.
                for part in cleaned.split("/"):
                    part = part.strip()
                    if part and part not in anchors:
                        anchors.append(part)
    return tuple(dict.fromkeys(anchors))


def parse_role_guide(baseinfo_text: str) -> list[RoleFamily]:
    """Parse the `== ROLE TYPE SELECTION GUIDE ==` block from baseinfo.txt."""
    match = re.search(
        r"==\s*ROLE TYPE SELECTION GUIDE\s*==(.*?)(?:\n==\s|\Z)",
        baseinfo_text,
        re.DOTALL,
    )
    if not match:
        return []
    guide_text = match.group(1)

    families: list[RoleFamily] = []
    family_pattern = re.compile(
        r"^([A-Z][A-Z /&-]+?)\s+roles\s*\n(.*?)(?=^[A-Z][A-Z /&-]+?\s+roles\s*$|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for family_match in family_pattern.finditer(guide_text):
        name = f"{family_match.group(1).strip()} roles"
        block = family_match.group(2)

        keywords = _parse_guide_list(block, "Keywords")
        must_show = _parse_guide_list(block, "MUST SHOW")
        show = _parse_guide_list(block, "SHOW")
        hide = _parse_guide_list(block, "HIDE")
        if not (must_show or show or hide):
            continue

        key = family_match.group(1).split("/")[0].strip().split()[0].upper()
        families.append(
            RoleFamily(
                key=key,
                name=name,
                keywords=tuple(k.lower() for k in keywords),
                must_show=frozenset(c.lower() for c in must_show),
                show=frozenset(c.lower() for c in show),
                hide=frozenset(c.lower() for c in hide),
            )
        )
    return families


def _parse_guide_list(block: str, label: str) -> list[str]:
    match = re.search(
        rf"^\s*{re.escape(label)}:\s*(.*?)(?=^\s*(?:Keywords|MUST SHOW|SHOW|HIDE|Extra rule)\b|\Z)",
        block,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        return []
    raw = " ".join(line.strip() for line in match.group(1).splitlines())
    return [part.strip() for part in raw.split(",") if part.strip()]


_GUIDE_NAME_STOPWORDS = frozenset({"roles", "role", "and", "or"})


def _family_name_tokens(family: RoleFamily) -> list[str]:
    tokens = re.split(r"[^a-z0-9]+", family.name.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _GUIDE_NAME_STOPWORDS]


def detect_role_family(job_text: str, families: list[RoleFamily]) -> RoleFamily | None:
    """Deterministic keyword-scored family detection (fallback when LLM fails).

    Guide keywords score by (capped) substring count. Family-name tokens score
    as whole words so a listing that only says "software" or "frontend" still
    lands on the right family. If nothing matches at all, the most inclusive
    family (largest MUST SHOW + SHOW coverage) wins — a generic listing is
    better served by showing more than by an arbitrary first-in-file pick.
    """
    if not families:
        return None
    lowered = job_text.lower()
    best: RoleFamily | None = None
    best_score = 0
    for family in families:
        score = 0
        for keyword in family.keywords:
            # Token-boundary match, not substring: "register" must not score
            # inside "Registered Nurse", nor "signal" inside "signalling
            # bonus". Lookarounds (rather than \b) keep keywords ending in
            # non-word chars ("c++") matchable.
            pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
            occurrences = len(re.findall(pattern, lowered))
            score += min(occurrences, 3)
        for token in _family_name_tokens(family):
            if re.search(rf"\b{re.escape(token)}\b", lowered):
                score += 1
        if score > best_score:
            best = family
            best_score = score
    if best is not None:
        return best
    return max(families, key=lambda f: len(f.must_show | f.show))


def find_role_family(key_or_name: str, families: list[RoleFamily]) -> RoleFamily | None:
    wanted = key_or_name.strip().upper()
    if not wanted:
        return None
    for family in families:
        if family.key == wanted or family.name.upper() == wanted:
            return family
    for family in families:
        if wanted in family.name.upper():
            return family
    return None


# ---------------------------------------------------------------------------
# Prompt + response parsing
# ---------------------------------------------------------------------------

def build_structured_prompt(
    job_title: str,
    job_description: str,
    job_highlights: list[str],
    catalog: TemplateCatalog,
    families: list[RoleFamily],
    extra_guidance: str = "",
) -> str:
    family_lines = "\n".join(f"- {family.key}: {family.name}" for family in families)
    highlight_lines = "\n".join(f"- {line}" for line in job_highlights[:10]) or "- (none parsed)"

    scope = catalog.render_config.rewrite_scope
    if scope == "selection":
        rewrite_scope_note = (
            "14. THIS PROFILE RENDERS ALL BULLET TEXT CANONICALLY: your 'bullets' values\n"
            "    are ignored, so return an empty object for 'bullets' and spend your\n"
            "    effort on role_family, ranking, bullet_order, weak_bullets,\n"
            "    exclude/include, and keywords instead.\n"
        )
    elif scope == "limited":
        rewrite_scope_note = (
            f"14. REWRITE BUDGET: only the first {catalog.render_config.limited_rewrite_bullets}\n"
            "    bullets of the final document use your rewrites — everything after\n"
            "    renders canonically. Concentrate your best rewrites in the entries you\n"
            "    rank most relevant; skip rewriting bullets that will fall outside the\n"
            "    budget.\n"
        )
    else:
        rewrite_scope_note = ""

    entry_blocks: list[str] = []
    for entry in catalog.entries:
        bullet_lines = "\n".join(
            f'    "{index}": {json.dumps(bullet)}' for index, bullet in enumerate(entry.bullets)
        )
        block = (
            f'entry_id: {entry.entry_id}\n'
            f'  section: {entry.section}\n'
            f'  title: {entry.title}\n'
            f'  canonical bullets:\n{bullet_lines}'
        )
        background_parts = [
            catalog.baseinfo_blocks[tag]
            for tag in entry.categories
            if tag in catalog.baseinfo_blocks
        ]
        if background_parts:
            background = "\n".join(dict.fromkeys(background_parts))
            indented = "\n".join(f"    {line}" for line in background.splitlines())
            block += f"\n  verified background (draw on ANY of these facts):\n{indented}"
        entry_blocks.append(block)
    entries_text = "\n\n".join(entry_blocks)

    profile_section = (
        "\n<candidate_facts>\n"
        f"{catalog.baseinfo_profile}\n"
        "</candidate_facts>\n"
        if catalog.baseinfo_profile
        else ""
    )

    guidance_section = f"\n<profile_guidance>\n{extra_guidance.strip()}\n</profile_guidance>\n" if extra_guidance.strip() else ""
    anchors_section = (
        "\n<skill_anchors>\n"
        "Tools, systems, methods, and certifications the candidate GENUINELY has\n"
        "(their full real skill set, whatever the field). You may weave any of\n"
        "these into bullets where they are relevant to the listing:\n"
        f"{', '.join(catalog.skill_anchors)}\n"
        "</skill_anchors>\n"
        if catalog.skill_anchors
        else ""
    )

    return (
        "<task>\n"
        "PRIMARY GOAL: rewrite this resume AROUND the job description. Every visible\n"
        "bullet should be re-aimed at this specific listing — its language, its\n"
        "requirements, its tools. A recruiter should feel the resume was written for\n"
        "this job. The candidate is the UNDERDOG for this role: find every genuine\n"
        "angle in their experience that answers the listing, and if hard technical\n"
        "overlap is thin, fit through the listing's softer vocabulary — how the work\n"
        "was done and what it achieved. You do NOT write LaTeX documents; you return\n"
        "ONLY a JSON object with your decisions, and the caller renders the resume.\n"
        "</task>\n\n"
        "<job>\n"
        f"Title: {job_title}\n"
        f"Highlights:\n{highlight_lines}\n"
        f"Description excerpt:\n{job_description[:4500]}\n"
        "</job>\n\n"
        "<role_families>\n"
        f"{family_lines}\n"
        "</role_families>\n\n"
        "<resume_entries>\n"
        f"{entries_text}\n"
        "</resume_entries>\n"
        f"{profile_section}"
        f"{anchors_section}"
        f"{guidance_section}\n"
        "<output_format>\n"
        "Return ONLY a JSON object (no prose, no markdown fences) with exactly these keys:\n"
        '{\n'
        '  "role_family": "<one KEY from <role_families>, e.g. EMBEDDED>",\n'
        '  "keywords": ["top 8-10 hard skills/tools/competencies quoted verbatim from the listing"],\n'
        '  "ranking": ["every entry_id, ordered most relevant to the listing first"],\n'
        '  "exclude": ["entry_ids that genuinely do NOT fit this listing (optional)"],\n'
        '  "include": ["entry_ids normally hidden for this role family that the LISTING\n'
        '              EXPLICITLY requires (optional, rare — max 2 honoured)"],\n'
        '  "weak_bullets": {"<entry_id>": [indices of its weakest bullets, weakest first]},\n'
        '  "bullet_order": {"<entry_id>": [bullet indices, MOST relevant to this listing\n'
        '                   first — the reader sees the first bullet before any other]},\n'
        '  "bullets": {"<entry_id>": ["rewritten bullet 0", "rewritten bullet 1", ...], ...}\n'
        '}\n'
        "Notes on exclude/include/weak_bullets: use exclude only for entries whose\n"
        "content would look out of place to a recruiter for this specific listing — not\n"
        "merely the least relevant ones (ranking already handles that). Core entries for\n"
        "the role family cannot be excluded and such requests are ignored. Use include\n"
        "only when the listing directly asks for an entry's domain (e.g. a backend role\n"
        "that explicitly wants graphics experience) — never to pad the page. weak_bullets\n"
        "guides which bullet is dropped first if space runs out; still provide rewrites\n"
        "for every bullet.\n"
        "Rules for rewritten bullets:\n"
        "1. Provide the SAME NUMBER of bullets per entry as the canonical list, same\n"
        "   order. Each rewrite must be grounded in the FACTS of the canonical bullet at\n"
        "   that index, but you have full freedom over the sentence itself: restructure\n"
        "   it, change its emphasis, reframe what was built in terms of the listing's\n"
        "   problems. You are re-aiming each bullet at this job, not lightly editing it.\n"
        "   Do NOT move facts or numbers between indices — bullets[i] rewrites canonical\n"
        "   bullet i. To change which bullet appears first, use bullet_order instead.\n"
        "   Each entry's 'verified background' lines are additional TRUE facts about\n"
        "   that same work — mine them for details, metrics, and phrasing the canonical\n"
        "   bullet leaves out, whenever they strengthen the fit with the listing.\n"
        "2. REWRITE AGGRESSIVELY around the listing. Extract its keywords first, then\n"
        "   rework every bullet to speak the listing's language, mirroring its exact\n"
        "   terminology (if it says 'RESTful APIs', write 'RESTful APIs', not 'REST\n"
        "   APIs'). Metrics are real outcomes: keep them (digit-for-digit) where they\n"
        "   strengthen the bullet — which is almost always — but you may drop one when\n"
        "   the rewrite genuinely reads better without it. NEVER alter digits or invent\n"
        "   new measurements. Light extrapolation from what an entry clearly implies is\n"
        "   fine (deployment implies configuration; testing implies debugging) — named\n"
        "   tools, employers, dates, and certifications are not extrapolation.\n"
        + (
            "3. INCLUDE AS MANY RELEVANT TOOLS AS POSSIBLE. Lead with the canonical bullet's\n"
            "   own named tools/systems/methods, and freely weave in additional ones from\n"
            "   <skill_anchors> wherever they are relevant to the listing AND plausibly\n"
            "   connect to that bullet's work (e.g. \\textbf{Git} in a deployment bullet,\n"
            "   \\textbf{Epic} in a charting bullet). Named skills NOT in <skill_anchors>\n"
            "   must never appear — a listing asking for Kubernetes (or a certification\n"
            "   the candidate lacks) does not make it true.\n"
            if catalog.skill_anchors
            else "3. Only mention named tools, systems, or methods the canonical bullet\n"
            "   itself contains — a listing asking for a skill the candidate lacks does\n"
            "   not make it true.\n"
        )
        +
        "4. The rewritten sentence must read as if a human editor wrote it, and every\n"
        "   claim must stay COHERENT for the candidate's field and interview-defensible.\n"
        "   Never import jargon from an adjacent domain that contradicts the entry's\n"
        "   actual equipment, scale, or role (chip-design vocabulary on a\n"
        "   microcontroller project, ICU terminology on a clinic placement,\n"
        "   enterprise-scale language on a course project) — a domain expert reading the\n"
        "   bullet must not wince. Keep each metric attached to the activity the\n"
        "   canonical entry says produced it. When a plain truthful phrasing and an\n"
        "   impressive-sounding one conflict, the plain one wins. NEVER bolt a keyword\n"
        "   onto a sentence where it does not fit grammatically — restructure the\n"
        "   sentence around the keyword instead.\n"
        "   Example — canonical: 'Built a REST service in Python handling 10k requests/day.'\n"
        "   Listing keyword: 'low-latency APIs'.\n"
        "   BAD (bolted):  'Built a REST service in Python handling 10k requests/day and\n"
        "   low-latency APIs performance.'\n"
        "   GOOD (integrated): 'Built a low-latency REST API in Python handling 10k\n"
        "   requests/day.'\n"
        "5. Plain text plus optional \\textbf{...}/\\textit{...} emphasis only. No other\n"
        "   LaTeX commands, no HTML, no line breaks inside a bullet. Escape % as \\%.\n"
        "   \\textbf may wrap ONLY named tools, systems, software, equipment, methods,\n"
        "   or certifications — never soft phrases from the posting ('analysis\n"
        "   experience', 'client relationships'), business vocabulary, verbs, or\n"
        "   metrics.\n"
        "6. Keep each bullet a single concise sentence (under 260 characters). Lead with\n"
        "   a strong PAST-TENSE active verb (Developed, Built — never the listing's\n"
        "   imperative mood) calibrated to the listing's seniority. Do not reuse the\n"
        "   same keyword in more than two bullets — spread coverage across the listing's\n"
        "   requirements instead of stuffing one phrase. Never append the same\n"
        "   listing-derived phrase to bullet after bullet ('for engine data analysis'\n"
        "   tacked onto five bullets reads machine-generated); each bullet must connect\n"
        "   to the listing in its own specific way, or not at all.\n"
        "7. Canonical fallback is a last resort, not a default: only return a bullet\n"
        "   unchanged when the listing offers genuinely nothing to tailor it toward.\n"
        "8. DIFFERENT-DOMAIN or NON-TECHNICAL listings (aerospace, banking, policy,\n"
        "   admin): still rewrite every bullet — work the candidate's experience INTO\n"
        "   the listing's frame. Adopt the listing's soft and functional vocabulary as\n"
        "   plain text wherever it truthfully describes the work: monitoring,\n"
        "   diagnostics, reliability, safety, compliance, reporting, data analysis,\n"
        "   documentation, process improvement, stakeholder communication. A testing\n"
        "   bullet becomes a quality-assurance bullet for a QA-flavoured listing; a\n"
        "   data-pipeline bullet becomes a monitoring-and-diagnostics bullet for a\n"
        "   health-management listing. What stays off-limits: NAMED domain systems,\n"
        "   standards, certifications, and physical equipment the candidate never\n"
        "   touched (specific aircraft systems, trading licences, lab instruments) —\n"
        "   and never bold or quote posting phrases, and never append them as\n"
        "   ', demonstrating <phrase>' clauses; integrate them as the sentence's own\n"
        "   subject, verb, or object.\n"
        "9. COVER THE REQUIREMENTS, front-load the important ones. Requirements listed\n"
        "   FIRST in the posting matter most — weight them heaviest when extracting\n"
        "   keywords, put the bullets that answer them first via bullet_order, and aim\n"
        "   to truthfully cover as many distinct requirements as possible across all\n"
        "   bullets (modern ATS scores requirement coverage and penalises stuffing one\n"
        "   phrase everywhere).\n"
        "10. When the listing uses an acronym (CI/CD, API, ML), mirror the listing's\n"
        "    form; if the listing uses both the acronym and the expansion, use both once.\n"
        "    Place each metric where it lands hardest — usually right after the verb\n"
        "    phrase it proves, not buried at the end of the sentence.\n"
        "11. NO EMPTY PURPOSE CLAUSES. Never end a bullet with a comma + gerund tail\n"
        "    that asserts an outcome without adding a fact (', ensuring clean and\n"
        "    reliable rendering', ', driving successful customer engagement',\n"
        "    ', prioritizing safety and correctness'). If the outcome is real, make it\n"
        "    the sentence's object with its metric; if it is not measurable, cut it.\n"
        "12. NEVER REDUCE SPECIFICITY. Do not replace a concrete tool, library, or\n"
        "    product name from the canonical bullet with a vaguer phrase ('GLFW and\n"
        "    GLUT' must never become 'specialized libraries'). A rewrite may drop a\n"
        "    tool entirely, but whatever it keeps must stay at least as specific.\n"
        "13. VARY SENTENCE SHAPE. Use 'using' at most once per bullet and in fewer than\n"
        "    half the bullets overall; rotate structures (verb + object + outcome;\n"
        "    verb + tool + what it enabled; outcome-first with the metric up front).\n"
        "    Fifteen bullets shaped 'Verbed X using A and B to improve Y' read\n"
        "    machine-generated to a recruiter.\n"
        + rewrite_scope_note +
        "</output_format>"
    )


_VALID_JSON_STRING_ESCAPES = set('"\\/bfnrtu')

# LaTeX commands whose first letter collides with a valid JSON escape: a model
# emitting ``\textbf`` unescaped parses as TAB + ``extbf`` and the PDF renders
# literal "extbf". When the text after the backslash spells one of these, the
# backslash is LaTeX, not a JSON escape.
_LATEX_JSON_ESCAPE_COLLISIONS = re.compile(
    r"(?:textbf|textit|texttt|textsc|times|frac|bullet|newline|underline)(?![a-zA-Z])"
)


def extract_json_object(text: str) -> dict | None:
    """Tolerantly pull the first balanced JSON object out of an LLM response.

    Models routinely emit LaTeX bullet text verbatim inside JSON string
    values (e.g. ``reduced errors by 41\\%``). ``\\%`` is not a valid JSON
    escape, so a strict ``json.loads`` rejects the entire payload. While
    scanning for the balanced ``{...}`` span, double any backslash not
    followed by a recognized JSON escape character so those sequences
    survive parsing instead of silently killing the whole response. A
    backslash followed by a valid escape letter is still doubled when the
    letters spell a LaTeX command (``\\textbf`` is not a tab).
    """
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    repaired: list[str] = []
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
                repaired.append(char)
                continue
            if char == "\\":
                escape = True
                nxt = cleaned[index + 1] if index + 1 < len(cleaned) else ""
                if nxt not in _VALID_JSON_STRING_ESCAPES or _LATEX_JSON_ESCAPE_COLLISIONS.match(
                    cleaned, index + 1
                ):
                    repaired.append("\\")
                repaired.append(char)
                continue
            if char == '"':
                in_string = False
            repaired.append(char)
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        repaired.append(char)
        if char == "}" and depth == 0:
            candidate = "".join(repaired)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def parse_structured_response(
    text: str,
    catalog: TemplateCatalog,
    families: list[RoleFamily],
    job_text: str,
) -> StructuredSelection | None:
    """Turn raw LLM output into a validated selection, or None if unusable."""
    payload = extract_json_object(text)
    if payload is None:
        return None

    family = find_role_family(str(payload.get("role_family", "")), families)
    if family is None:
        family = detect_role_family(job_text, families)
    if family is None:
        return None

    known_ids = {entry.entry_id for entry in catalog.entries}
    # Models occasionally key by entry title or a re-slugged variant instead of
    # the exact entry_id; resolve those instead of discarding the work.
    alias_map: dict[str, str] = {}
    for entry in catalog.entries:
        alias_map[entry.entry_id] = entry.entry_id
        alias_map[_slugify(entry.title)] = entry.entry_id
        alias_map[entry.title.lower()] = entry.entry_id

    def _resolve_entry_id(raw_key: object) -> str | None:
        key = str(raw_key).strip()
        if key in known_ids:
            return key
        return alias_map.get(key.lower()) or alias_map.get(_slugify(key))

    ranking: list[str] = []
    ranking_raw = payload.get("ranking")
    if isinstance(ranking_raw, list):
        for item in ranking_raw:
            resolved = _resolve_entry_id(item)
            if resolved is not None and resolved not in ranking:
                ranking.append(resolved)

    def _coerce_bullet_text(item: object) -> str:
        # Tolerate {"text": "..."} / {"bullet": "..."} wrappers.
        if isinstance(item, dict):
            for key in ("text", "bullet", "content", "value"):
                value = item.get(key)
                if isinstance(value, str):
                    return value
            return ""
        return str(item) if isinstance(item, (str, int, float)) else ""

    bullets_raw = payload.get("bullets")
    bullets: dict[str, list[str]] = {}
    if isinstance(bullets_raw, dict):
        for entry_key, entry_bullets in bullets_raw.items():
            resolved = _resolve_entry_id(entry_key)
            if resolved is None or not isinstance(entry_bullets, list):
                continue
            bullets[resolved] = [_coerce_bullet_text(item) for item in entry_bullets]

    keywords_raw = payload.get("keywords")
    keywords = (
        [str(item) for item in keywords_raw if isinstance(item, (str, int, float))]
        if isinstance(keywords_raw, list)
        else []
    )

    exclusions: list[str] = []
    for key in ("exclude", "exclusions", "omit"):
        raw = payload.get(key)
        if isinstance(raw, list):
            for item in raw:
                resolved = _resolve_entry_id(item)
                if resolved is not None and resolved not in exclusions:
                    exclusions.append(resolved)
            break

    inclusions: list[str] = []
    for key in ("include", "inclusions", "force_include"):
        raw = payload.get(key)
        if isinstance(raw, list):
            for item in raw:
                resolved = _resolve_entry_id(item)
                if resolved is not None and resolved not in inclusions:
                    inclusions.append(resolved)
            break

    def _parse_index_map(key: str) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        raw = payload.get(key)
        if isinstance(raw, dict):
            for entry_key, indices in raw.items():
                resolved = _resolve_entry_id(entry_key)
                if resolved is None or not isinstance(indices, list):
                    continue
                cleaned = [
                    int(i) for i in indices if isinstance(i, (int, float)) and int(i) >= 0
                ]
                if cleaned:
                    result[resolved] = cleaned
        return result

    weak_bullets = _parse_index_map("weak_bullets")
    bullet_order = _parse_index_map("bullet_order")

    return StructuredSelection(
        role_family_key=family.key,
        ranking=ranking,
        bullets=bullets,
        keywords=keywords,
        exclusions=exclusions,
        inclusions=inclusions,
        weak_bullets=weak_bullets,
        bullet_order=bullet_order,
    )


# ---------------------------------------------------------------------------
# Bullet validation
# ---------------------------------------------------------------------------

def _matches_skill_anchor(normalized: str, skill_anchors: tuple[str, ...]) -> bool:
    """True when a bolded phrase corresponds to a tool the candidate has.

    Anchors match as whole words ("git" matches "git-based workflows" but the
    single-letter anchor "c" does not match every phrase containing a c), and
    a phrase may also be a fragment of a longer anchor ("pytorch" within
    "pytorch lightning").
    """
    for anchor in skill_anchors:
        if normalized == anchor:
            return True
        if len(normalized) >= 3 and normalized in anchor:
            return True
        if re.search(rf"(?<![a-z0-9]){re.escape(anchor)}(?![a-z0-9])", normalized):
            return True
    return False


# LLM typography noise that must be normalised (not rejected): smart quotes
# map to their ASCII equivalents so downstream checks and LaTeX see one form.
_SMART_QUOTE_MAP = {
    "“": '"',  # “
    "”": '"',  # ”
    "„": '"',  # „
    "‘": "'",  # ‘
    "’": "'",  # ’
    "«": '"',  # «
    "»": '"',  # »
}
_MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*")
_MARKDOWN_ITALIC_PATTERN = re.compile(r"(?<!\*)\*([^*\s][^*]*?)\*(?!\*)")
# A text command whose backslash was eaten (JSON escaping, model error):
# bare "textbf{" renders literally in the PDF. Preceded-by-letter/backslash
# guard keeps real words and correctly-escaped commands untouched.
_BARE_TEXT_COMMAND_PATTERN = re.compile(r"(?<![\\a-zA-Z])(textbf|textit|texttt|emph)\{")


def _normalize_llm_bullet_typography(text: str) -> str:
    """Normalise LLM output quirks that break rendering but have an obvious
    intended form: smart quotes, whole-bullet quote wrapping, markdown
    emphasis, and text commands that lost their backslash."""
    for smart, plain in _SMART_QUOTE_MAP.items():
        text = text.replace(smart, plain)
    text = text.strip()
    # A bullet wrapped entirely in quotes is the model quoting its own answer,
    # not content — unwrap it (interior quotes are handled separately).
    while len(text) >= 2 and text[0] == '"' and text[-1] == '"' and '"' not in text[1:-1]:
        text = text[1:-1].strip()
    text = _MARKDOWN_BOLD_PATTERN.sub(r"\\textbf{\1}", text)
    text = _MARKDOWN_ITALIC_PATTERN.sub(r"\\textit{\1}", text)
    text = _BARE_TEXT_COMMAND_PATTERN.sub(r"\\\1{", text)
    return text


def validate_tailored_bullet(
    tailored: str,
    canonical: str,
    config: RenderConfig | None = None,
    skill_anchors: tuple[str, ...] = (),
    entry_context: str = "",
) -> tuple[str | None, str | None]:
    """Return (sanitised_bullet, None) or (None, rejection_reason).

    `canonical` is the same-index bullet (numbers are validated against it,
    strictly). `entry_context` is all of the entry's canonical bullets joined —
    tool/emphasis grounding uses it so content legitimately moved between an
    entry's own bullets is not flagged as invented. Defaults to `canonical`.
    """
    limits = config or RenderConfig()
    # Backstop for JSON-escape collisions that slipped past extract_json_object:
    # an unescaped \textbf/\textit/\times parses as TAB + command remainder.
    text = (
        tailored.replace("\t" + "extbf{", "\\textbf{")
        .replace("\t" + "extit{", "\\textit{")
        .replace("\t" + "imes", "\\times")
    )
    text = _normalize_llm_bullet_typography(text)
    text = " ".join(text.split()).strip()
    if not text:
        return None, "empty"
    if len(text) > limits.max_bullet_chars:
        return None, "too long"
    if "<" in text or ">" in text:
        return None, "html/angle brackets"
    # ^ is a math-mode-only character; outside $...$ it is a fatal LaTeX error.
    if "^" in re.sub(r"(?<!\\)\$[^$]*(?<!\\)\$", "", text):
        return None, "caret outside math mode"

    for command in LATEX_COMMAND_PATTERN.findall(text):
        if command not in ALLOWED_BULLET_COMMANDS:
            return None, f"disallowed latex command \\{command}"

    lowered = text.lower()
    canonical_lowered = canonical.lower()
    for term in limits.forbidden_terms + limits.extra_forbidden_terms:
        # Whole-word match: "rust" must not fire inside "robust", nor "hmi"
        # inside "algorithmic".
        term_pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        if re.search(term_pattern, lowered) and not re.search(term_pattern, canonical_lowered):
            return None, f"forbidden term {term!r}"

    # Long bolded phrases that the canonical bullet never contained are
    # job-posting language pasted in for keyword effect, not technology names.
    # Short new bolded names must at least be tools the candidate really has
    # (skill anchors) — a listing mentioning Kubernetes never makes it true.
    grounding = entry_context or canonical
    grounding_normalized = grounding.lower().replace("\\", "")
    grounding_words = set(re.findall(r"[a-z0-9.+#-]+", grounding_normalized))
    for bold_phrase in re.findall(r"\\textbf\{([^}]*)\}", text):
        phrase = bold_phrase.strip()
        normalized = phrase.replace("\\", "").lower()
        if normalized in grounding_normalized:
            continue
        # Reworded canonical concept ("distributed, low-latency data pipeline"
        # -> "distributed data pipeline"): every word already in the entry.
        phrase_words = re.findall(r"[a-z0-9.+#-]+", normalized)
        if phrase_words and all(word in grounding_words for word in phrase_words):
            continue
        if skill_anchors and _matches_skill_anchor(normalized, skill_anchors):
            continue
        # Multi-word phrases and plain lowercase words are posting language,
        # not tool claims: demote to plain text (keep the rewrite, drop the
        # emphasis). A single product-shaped token (capitalised, or with
        # digits/./#/+) that isn't an anchor is a likely hallucinated tool —
        # reject the bullet.
        looks_like_product = bool(re.search(r"[A-Z0-9.#+]", phrase))
        if len(phrase_words) > 1 or not looks_like_product:
            text = text.replace(f"\\textbf{{{bold_phrase}}}", bold_phrase)
            continue
        if skill_anchors:
            return None, f"invented tool {phrase[:40]!r}"

    # Bold density: 4+ bolded tokens per bullet reads as keyword farming, not
    # emphasis. Keep the first N (earliest mentions carry the emphasis value),
    # demote the rest to plain text.
    bold_spans = re.findall(r"\\textbf\{([^}]*)\}", text)
    if len(bold_spans) > limits.max_bold_per_bullet:
        for extra_phrase in bold_spans[limits.max_bold_per_bullet:]:
            text = text.replace(f"\\textbf{{{extra_phrase}}}", extra_phrase, 1)

    # Straight double quotes are wrong LaTeX typography, and quoting a posting
    # phrase is the same padding as bolting it.
    if '"' in text and '"' not in canonical:
        return None, "quoted posting phrase"

    filler = FILLER_CLAUSE_PATTERN.search(text)
    if filler and not FILLER_CLAUSE_PATTERN.search(canonical):
        return None, f"appended filler clause {filler.group(1).lower()!r}"

    # Metrics
    # (41%, 14%, 12%, 50ms, ...) are real outcomes. A rewrite may drop or
    # relocate one when the sentence reads better without it, but may never
    # mint or alter a number — every digit must exist somewhere in the entry.
    grounding_numbers = set(NUMBER_PATTERN.findall(grounding))
    for number in NUMBER_PATTERN.findall(text):
        if number not in grounding_numbers:
            return None, f"invented number {number}"

    text = UNESCAPED_SPECIAL_PATTERN.sub(r"\\\1", text)

    if text.count("{") != text.count("}"):
        return None, "unbalanced braces"
    dollars = len(re.findall(r"(?<!\\)\$", text))
    if dollars % 2 != 0:
        return None, "unbalanced math delimiters"

    return text, None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _entry_visibility(entry: TemplateEntry, family: RoleFamily) -> str:
    categories = {category.lower() for category in entry.categories}
    if categories & family.must_show:
        return "must_show"
    if categories & family.show:
        return "show"
    return "hide"


def _rank_index(entry_id: str, ranking: list[str], template_order: list[str]) -> tuple[int, int]:
    try:
        return (0, ranking.index(entry_id))
    except ValueError:
        return (1, template_order.index(entry_id))


def lint_render_fidelity(
    catalog: TemplateCatalog,
    document: str,
    included: list[TemplateEntry],
) -> list[str]:
    """Verify the rendered document follows the template's structure exactly.

    The renderer builds from verbatim template parts, so any finding here is a
    bug, not a style issue. Checked: preamble and name/contact block byte-for-
    byte, every included entry's header verbatim, balanced list environments,
    and exactly one document environment.
    """
    findings: list[str] = []
    if catalog.preamble not in document:
        findings.append("preamble does not match template verbatim")
    if catalog.header_block and catalog.header_block not in document:
        findings.append("name/contact block does not match template verbatim")
    for entry in included:
        if entry.header not in document:
            findings.append(f"header altered for entry {entry.entry_id}")
    if document.count("\\begin{document}") != 1 or document.count("\\end{document}") != 1:
        findings.append("document environment not exactly one begin/end pair")
    for begin_marker, end_marker in {
        (e.list_begin, e.list_end) for e in included
    }:
        if document.count(begin_marker) != document.count(end_marker):
            findings.append(f"unbalanced list markers {begin_marker!r}")
    if "\\begin{comment}" in document or "\\end{comment}" in document:
        findings.append("comment environment leaked into output")
    return findings


def reorder_skills_for_keywords(
    tail: str,
    keywords: list[str],
    family_priority: tuple[str, ...] = (),
) -> str:
    """Reorder items within Skills-section lines so listing-relevant tools lead.

    Format-preserving and content-preserving: same labels, same items, same
    line breaks — only the order of comma-separated items within each
    `\\textbf{Label:} a, b, c \\\\` line changes. Items matching a listing
    keyword move to the front, then items matching the role family's skill
    priority (in priority order), then everything else in canonical order.
    Nothing is ever added or removed, so there is no invention risk.
    """
    if not keywords and not family_priority:
        return tail

    keyword_lowered = [k.lower().strip() for k in keywords if k.strip()]
    keyword_words = [set(re.findall(r"[a-z0-9.+#]+", k)) for k in keyword_lowered]
    # Family priority uses whole-token matching only: substring matching would
    # let a priority of "C" capture "CSS" and "scikit-learn".
    priority_words = [
        (position, set(re.findall(r"[a-z0-9.+#/]+", entry.lower())))
        for position, entry in enumerate(family_priority)
        if entry.strip()
    ]

    def item_rank(item: str) -> tuple[int, int]:
        item_lowered = item.lower()
        for keyword in keyword_lowered:
            if keyword == item_lowered or keyword in item_lowered or item_lowered in keyword:
                return (0, 0)
        item_words = set(re.findall(r"[a-z0-9.+#]+", item_lowered))
        for words in keyword_words:
            if item_words & words:
                return (1, 0)
        item_tokens = set(re.findall(r"[a-z0-9.+#/]+", item_lowered)) | item_words
        for position, tokens in priority_words:
            if item_tokens & tokens:
                return (2, position)
        return (3, 0)

    out_lines: list[str] = []
    in_skills = False
    for line in tail.splitlines():
        if line.strip().startswith("\\section*"):
            in_skills = "skills" in line.lower()
            out_lines.append(line)
            continue
        stripped = line.rstrip()
        has_break = stripped.endswith("\\\\")
        core = stripped[:-2].rstrip() if has_break else stripped
        label_match = re.match(r"^(\\textbf\{[^}]*\})\s*(.+)$", core) if in_skills else None
        if label_match and "," in label_match.group(2):
            items = [item.strip() for item in label_match.group(2).split(",") if item.strip()]
            items.sort(key=item_rank)  # stable: preserves canonical order within ranks
            rebuilt = f"{label_match.group(1)} {', '.join(items)}"
            out_lines.append(rebuilt + (" \\\\" if has_break else ""))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def render_structured_resume(
    catalog: TemplateCatalog,
    families: list[RoleFamily],
    selection: StructuredSelection,
) -> tuple[str, RenderReport]:
    """Deterministically render the resume from template parts + selection."""
    family = find_role_family(selection.role_family_key, families)
    if family is None:
        raise ValueError(f"Unknown role family: {selection.role_family_key}")

    config = catalog.render_config
    template_order = [entry.entry_id for entry in catalog.entries]

    must_show = [e for e in catalog.entries if _entry_visibility(e, family) == "must_show"]
    show = [e for e in catalog.entries if _entry_visibility(e, family) == "show"]
    hidden = [e for e in catalog.entries if _entry_visibility(e, family) == "hide"]

    show.sort(key=lambda e: _rank_index(e.entry_id, selection.ranking, template_order))

    # Exclusions: the model may drop SHOW entries that genuinely do not fit
    # this listing. MUST SHOW entries are guardrailed and cannot be excluded.
    must_show_ids = {e.entry_id for e in must_show}
    ignored_exclusions = [eid for eid in selection.exclusions if eid in must_show_ids]
    excluded_ids = {
        eid for eid in selection.exclusions
        if eid not in must_show_ids
    }

    # Bullet budget: all MUST SHOW entries at full count, then non-excluded
    # SHOW entries by relevance until the minimum is reached (or all are
    # placed while still under the maximum).
    kept_indices: dict[str, list[int]] = {
        e.entry_id: list(range(len(e.bullets))) for e in must_show
    }
    included: list[TemplateEntry] = list(must_show)
    total = sum(len(v) for v in kept_indices.values())

    def _try_add(entry: TemplateEntry) -> bool:
        nonlocal total
        if total >= config.min_visible_bullets and total + len(entry.bullets) > config.max_visible_bullets:
            return False
        included.append(entry)
        kept_indices[entry.entry_id] = list(range(len(entry.bullets)))
        total += len(entry.bullets)
        return True

    # Inclusions: the model may promote HIDE entries the listing explicitly
    # calls for. Placed before generic SHOW entries (an explicit request
    # outranks default relevance), hard-capped at MAX_INCLUSIONS, and only
    # when the whole entry fits under the ceiling.
    hidden_ids = {e.entry_id for e in hidden}
    included_extras: list[str] = []
    ignored_inclusions: list[str] = [
        eid for eid in selection.inclusions if eid not in hidden_ids
    ]
    inclusion_candidates = [e for e in hidden if e.entry_id in set(selection.inclusions)]
    inclusion_candidates.sort(
        key=lambda e: _rank_index(e.entry_id, selection.ranking, template_order)
    )
    for entry in inclusion_candidates:
        if (
            len(included_extras) < MAX_INCLUSIONS
            and total + len(entry.bullets) <= config.max_visible_bullets
        ):
            included.append(entry)
            kept_indices[entry.entry_id] = list(range(len(entry.bullets)))
            total += len(entry.bullets)
            included_extras.append(entry.entry_id)
        else:
            ignored_inclusions.append(entry.entry_id)

    for entry in show:
        if entry.entry_id in excluded_ids:
            continue
        _try_add(entry)

    # Budget rollback: if honouring the exclusions leaves the page under-full,
    # re-admit excluded entries in relevance order until the minimum is met.
    if total < config.min_visible_bullets:
        for entry in show:
            if entry.entry_id not in excluded_ids:
                continue
            if _try_add(entry):
                excluded_ids.discard(entry.entry_id)
                ignored_exclusions.append(entry.entry_id)
            if total >= config.min_visible_bullets:
                break

    excluded_entries = sorted(excluded_ids & {e.entry_id for e in show})

    # Front-load each entry: display bullets most-relevant-first when the
    # model provided an order ("place key qualifications where the reader
    # looks first"). Trimming below then cuts from the tail, i.e. the least
    # relevant bullets are sacrificed first.
    for entry in included:
        order = selection.bullet_order.get(entry.entry_id)
        if not order:
            continue
        kept = kept_indices[entry.entry_id]
        ordered = [index for index in order if index in kept]
        kept_indices[entry.entry_id] = ordered + [i for i in kept if i not in ordered]

    # Over budget: trim from the least relevant SHOW entries first, then from
    # MUST SHOW entries, never below their floors. Within an entry, trim the
    # model's flagged weakest bullet first, otherwise the last one. Budget is
    # both bullet COUNT and total bullet CHARACTERS: tailored bullets run
    # longer than canonical ones, and a 15-bullet render of long bullets
    # overflows to a second page even though the count fits.
    # The rewrite-scope dial: "selection" ignores model bullet text entirely;
    # "limited" applies it only to the first N bullets in document order (the
    # ones a skimming recruiter reads); "full" applies it everywhere.
    effective_bullets: dict[str, list[str]] = (
        {} if config.rewrite_scope == "selection" else selection.bullets
    )

    def _estimated_chars() -> int:
        chars = 0
        for entry in included:
            provided = effective_bullets.get(entry.entry_id, [])
            for index in kept_indices[entry.entry_id]:
                if index < len(provided) and provided[index].strip():
                    chars += min(len(provided[index]), config.max_bullet_chars)
                else:
                    chars += len(entry.bullets[index])
        return chars

    def _trim_order() -> list[TemplateEntry]:
        shows = [e for e in included if e not in must_show]
        return list(reversed(shows)) + list(reversed(must_show))

    def _trim_one(entry: TemplateEntry) -> None:
        nonlocal total
        kept = kept_indices[entry.entry_id]
        for weak_index in selection.weak_bullets.get(entry.entry_id, []):
            if weak_index in kept and len(kept) > 1:
                kept.remove(weak_index)
                total -= 1
                return
        kept.pop()
        total -= 1

    # Page fit beats the bullet minimum: character trimming may dip slightly
    # below min_visible_bullets, because a second page is the worse outcome.
    char_trim_floor = max(config.min_visible_bullets - 2, 1)
    while total > config.max_visible_bullets or (
        total > char_trim_floor
        and _estimated_chars() > config.max_total_bullet_chars
    ):
        trimmed = False
        for entry in _trim_order():
            floor = config.min_must_show_bullets if entry in must_show else 1
            if len(kept_indices[entry.entry_id]) > floor:
                _trim_one(entry)
                trimmed = True
                break
        if not trimmed:
            break

    # Order entries inside each section: MUST SHOW first, then by relevance.
    def _entry_sort_key(entry: TemplateEntry) -> tuple[int, tuple[int, int]]:
        priority = 0 if entry in must_show else 1
        return (priority, _rank_index(entry.entry_id, selection.ranking, template_order))

    fallbacks: list[str] = []
    tailored_used = 0
    rendered_bullet_count = 0

    def _render_entry(entry: TemplateEntry) -> str:
        nonlocal tailored_used, rendered_bullet_count
        provided = effective_bullets.get(entry.entry_id, [])
        lines: list[str] = []
        if entry.smallskip:
            lines.append("\\smallskip")
        lines.append(f"% [{', '.join(entry.categories)}]")
        lines.append(entry.header)
        lines.append(entry.list_begin)
        for index in kept_indices[entry.entry_id]:
            canonical = entry.bullets[index]
            text = canonical
            rewrite_allowed = (
                config.rewrite_scope != "limited"
                or rendered_bullet_count < config.limited_rewrite_bullets
            )
            rendered_bullet_count += 1
            if rewrite_allowed and index < len(provided):
                background = " ".join(
                    catalog.baseinfo_blocks.get(tag, "") for tag in entry.categories
                )
                sanitised, reason = validate_tailored_bullet(
                    provided[index],
                    canonical,
                    config,
                    catalog.skill_anchors,
                    entry_context=f"{' '.join(entry.bullets)} {background}".strip(),
                )
                if sanitised is not None:
                    text = sanitised
                    tailored_used += 1
                else:
                    fallbacks.append(f"{entry.entry_id}[{index}]: {reason}")
            if entry.item_style == "braced":
                lines.append(f"  {entry.item_command}{{{text}}}")
            else:
                lines.append(f"  {entry.item_command} {text}")
        lines.append(entry.list_end)
        return "\n".join(lines)

    section_chunks: list[str] = []
    for section_name in catalog.entry_sections:
        section_entries = sorted(
            (e for e in included if e.section == section_name),
            key=_entry_sort_key,
        )
        if not section_entries:
            continue
        rendered_entries = "\n\n".join(_render_entry(entry) for entry in section_entries)
        section_command = catalog.section_headers.get(
            section_name, f"\\section*{{{section_name}}}"
        )
        prefix, suffix = catalog.section_chrome.get(section_name, ("", ""))
        chunk_parts = [section_command]
        if prefix:
            chunk_parts.append(prefix)
        chunk_parts.append("")
        chunk_parts.append(rendered_entries)
        if suffix:
            chunk_parts.append(suffix)
        section_chunks.append("\n".join(chunk_parts))

    parts = [
        catalog.preamble,
        "",
        catalog.header_block,
        "",
        "\n\n".join(section_chunks),
    ]
    if catalog.tail:
        family_priority = config.family_skill_priority.get(family.key) or family.keywords
        parts += ["", reorder_skills_for_keywords(catalog.tail, selection.keywords, family_priority)]
    parts += ["", "\\end{document}", ""]
    document = "\n".join(parts)

    report = RenderReport(
        role_family_key=family.key,
        visible_entries=[entry.entry_id for entry in included],
        hidden_entries=[
            entry.entry_id for entry in hidden if entry.entry_id not in included_extras
        ],
        visible_bullet_count=total,
        tailored_bullets_used=tailored_used,
        canonical_fallbacks=fallbacks,
        excluded_entries=excluded_entries,
        ignored_exclusions=ignored_exclusions,
        included_extras=included_extras,
        ignored_inclusions=ignored_inclusions,
        fidelity_findings=lint_render_fidelity(catalog, document, included),
        rewrite_scope=config.rewrite_scope,
    )
    return document, report


def load_structured_profile(
    template_path: Path,
    baseinfo_path: Path,
) -> tuple[TemplateCatalog, list[RoleFamily]] | None:
    """Load catalog + role guide for a profile, or None if unsupported.

    Also picks up two optional per-profile files next to template.tex:
    - structured_config.json: rendering knobs (bullet budget, forbidden terms)
    - structured_guidance.txt: free-text prompt guidance (voice, seniority)
    """
    try:
        template_text = template_path.read_text(encoding="utf-8")
        baseinfo_text = baseinfo_path.read_text(encoding="utf-8")
    except OSError:
        return None
    catalog = parse_template_catalog(template_text)
    if catalog is None:
        return None
    families = parse_role_guide(baseinfo_text)
    if not families:
        return None
    catalog.skill_anchors = parse_skill_anchors(baseinfo_text)
    catalog.baseinfo_blocks, catalog.baseinfo_profile = parse_baseinfo_blocks(baseinfo_text)

    config_path = template_path.parent / STRUCTURED_CONFIG_FILENAME
    if config_path.is_file():
        catalog.render_config = RenderConfig.from_file(config_path)

    guidance_path = template_path.parent / STRUCTURED_GUIDANCE_FILENAME
    if guidance_path.is_file():
        try:
            catalog.guidance = guidance_path.read_text(encoding="utf-8").strip()
        except OSError:
            catalog.guidance = ""

    return catalog, families
