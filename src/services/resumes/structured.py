"""Structured resume tailoring: the LLM returns JSON decisions, Python renders LaTeX.

The legacy pipeline asked the LLM to rewrite the entire .tex document, which
produced a long tail of structural failures (broken preambles, HTML tags,
unclosed comment environments, invented entries, dropped metrics). This module
removes LaTeX authoring from the LLM entirely:

  1. `parse_template_catalog` reads the profile's template.tex into a catalog of
     entries keyed by the `% [category]` tag convention. Preamble, headers,
     skills, and education are kept verbatim and never touch the LLM.
  2. Every entry is eligible for every listing — there is no fixed
     category-to-visibility mapping. The LLM is asked for a small JSON object:
     a relevance ranking over all entries, optional exclusions for entries
     that genuinely do not fit the listing, and tailored bullet texts.
  3. `render_structured_resume` includes entries in ranked order until the
     bullet budget is met, honouring exclusions unless dropping them would
     leave the page under-full (excluded entries are then restored in ranked
     order — a capacity-based safety net rather than a category one). It also
     enforces bullet counts, metric preservation, and bullet-text
     sanitisation, then renders the document from template parts. Any bullet
     that fails validation falls back to its canonical template text, so the
     output always compiles.

Profiles whose template has no `% [category]` tags keep the legacy free-form
path (see listing.generate_resume_rewrite).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

# Bullet-count window required by the profile instructions ("fill the page").
# The ceiling is 16 rather than the instructions' historical 18: sample compiles
# showed 17 bullets across 6 entries overflow to a second page. Profiles can
# override these via structured_config.json (see RenderConfig).
MIN_VISIBLE_BULLETS = 14
MAX_VISIBLE_BULLETS = 16
MAX_BULLET_CHARS = 400
# Appended "…, demonstrating <posting phrase>" clauses are keyword padding in a
# trench coat. If the canonical bullet did not use the construction, the tail
# clause is stripped and the rest of the rewrite kept (rejecting the whole
# bullet threw away otherwise-good tailoring). The second group covers the
# empty purpose-clause tails HR review flagged ("…, ensuring clean, performant
# rendering", "…, driving successful customer relations"): gerunds that assert
# an outcome without adding a fact.
FILLER_CLAUSE_PATTERN = re.compile(
    r"(?:,|\band|\bwhile)\s+(demonstrating|showcasing|highlighting|underscoring|evidencing|exemplifying"
    r"|ensuring|enabling|supporting|driving|prioritizing|prioritising|facilitating|streamlining|empowering"
    r"|exercising|aligning|upholding|leveraging|reflecting|embodying)\b",
    re.IGNORECASE,
)

STRUCTURED_CONFIG_FILENAME = "structured_config.json"
STRUCTURED_GUIDANCE_FILENAME = "instructions.txt"

# LaTeX commands allowed inside an LLM-tailored bullet. Anything else means the
# model tried to write structure, which is exactly what this pipeline forbids.
ALLOWED_BULLET_COMMANDS = frozenset(
    {"textbf", "textit", "texttt", "emph", "times", "%", "&", "_", "#", "$"}
)

# There is no static "never claim X" word list: what counts as an off-limits
# claim is per-candidate (one candidate's "never claim Rust" is another's core
# skill) and a fixed list can never enumerate every risky term anyway. Instead
# the candidate's real skill_anchors are the single source of truth for what
# they have, and the LLM grounding audit (see GROUNDING_AUDIT_PROMPT_HEADER)
# is the plausibility arbiter for anything named that isn't in that list.

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
# Numbers that stand as metrics, not digits embedded in identifiers: "32" in
# "STM32" or "12" in "HCS12" is part of a product name, not a figure the
# candidate can claim. Letter-prefixed digits are excluded; unit suffixes
# ("50ms", "75th", "10k") keep matching because the digits lead the token.
STANDALONE_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?")
# Comma-grouped variant for the prompt's metric menu ("206,775" is one number).
MENU_NUMBER_PATTERN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
UNESCAPED_SPECIAL_PATTERN = re.compile(r"(?<!\\)([&%#_])")

# Candidate tokens for the unbolded-claim guard: word-ish runs that can name
# a product ("Riverpod", "HTML5", "Node.js", "PyTorch").
PRODUCT_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9.+#-]*")

# Metric-category labels used by baseinfo Notes lines ("Reliability: cut
# downtime 41%"). The prompt's metric menu keeps the label visible so the
# model knows which KIND of number it is picking.
METRIC_CATEGORY_LABEL_PATTERN = re.compile(
    r"(?:Scope|Scale|Cost|Reliability|Adoption|Ownership|Impact)(?:/[A-Za-z]+)?:"
)

# Solo-ownership opening verbs need grounding: an intern's rewrite must not
# open with "Led"/"Owned" unless the entry's own grounding states leadership
# or solo ownership (an "Ownership:" baseinfo line counts as the profile
# author's explicit license). Demote-not-veto — the verb is swapped for an
# active collaborative one and the rest of the rewrite is kept.
OWNERSHIP_VERB_SWAPS = {
    "led": "Co-led",
    "spearheaded": "Co-led",
    "headed": "Co-led",
    "owned": "Developed",
    "managed": "Coordinated",
    "directed": "Coordinated",
    "oversaw": "Coordinated",
}
# Only EXPLICIT ownership facts count as evidence. Generic verb-family words
# (management, leadership, head, oversee) appear incidentally in ordinary
# resume prose ("reported progress to management") and must NOT license a
# solo-ownership opener; the canonical-verb case ("Managed windowing…" in the
# grounding licenses a "Managed …" rewrite) is handled separately by an
# exact-verb whole-word search in validate_tailored_bullet.
OWNERSHIP_EVIDENCE_PATTERN = re.compile(
    r"\b(?:sole|solo|solely|independently|single-handed(?:ly)?"
    r"|own(?:ed|er|ership)|spearheaded|founder|founded|president|captain"
    r"|instructor)\b"
    r"|\bend-to-end\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RenderConfig:
    """Per-profile rendering knobs, optionally loaded from structured_config.json."""

    min_visible_bullets: int = MIN_VISIBLE_BULLETS
    max_visible_bullets: int = MAX_VISIBLE_BULLETS
    max_bullet_chars: int = MAX_BULLET_CHARS
    # Total characters across all rendered bullets; the one-page ceiling.
    # Canonical bullets average ~190 chars; 14-15 x 190 ≈ 2800.
    max_total_bullet_chars: int = 2800
    # How much text authority the LLM gets. Selection decisions (ranking,
    # ordering, exclusions) always come from the model; this dial
    # controls only bullet REWRITES, the sole source of prose defects:
    #   "full"      — every visible bullet may be rewritten (legacy behavior)
    #   "limited"   — only the first `limited_rewrite_bullets` bullets in
    #                 final document order may be rewritten; the rest render
    #                 canonical. Bounds slop to the bullets recruiters read.
    #   "selection" — no rewrites at all; every bullet renders canonical.
    rewrite_scope: str = "full"
    limited_rewrite_bullets: int = 5
    # Bolding beyond this many \textbf{} per bullet reads as keyword farming;
    # extras are demoted to plain text (listing-relevant bolds kept first).
    max_bold_per_bullet: int = 4
    # When true, a tailored bullet may name a tool OUTSIDE the skill anchors
    # if it is closely adjacent to work the entry verifiably did (same
    # ecosystem — e.g. an AI-integration library in a bullet about real
    # OpenAI-API work). The deterministic invented-tool rejection is skipped;
    # the LLM grounding audit remains the plausibility arbiter for concrete
    # ungrounded claims either way.
    adjacent_tool_leeway: bool = False
    # When > 0, each Skills-section line is trimmed to this many items after
    # reordering. Items matching the listing always
    # survive (even past the cap); canonical-order extras fill up to the cap;
    # the rest are cut. 0 keeps every item (reorder-only legacy behavior).
    max_skill_items_per_line: int = 0
    # LLM grounding audit: one batched judge call checks every tailored bullet
    # against its entry's verified background and demotes bullets that claim a
    # domain/industry/equipment the candidate never touched. Fail-open — a
    # judge failure leaves rewrites untouched, so this can default on.
    grounding_audit: bool = True
    # Per-build opt-in (never read from structured_config.json — set only by
    # the caller for an explicit ".resumebuild --aggressive" request). Loosens
    # the STYLE/conservatism guards (bold density, filler-clause stripping,
    # quoted-phrase rejection, cross-bullet repetition caps) and forces
    # adjacent_tool_leeway + grounding_audit on, so the LLM grounding judge is
    # always the plausibility backstop when the tool-name guard steps back.
    # Never loosens: number/metric grounding, invented-employer/title/date/
    # certification protection (those are structural — headers are template-
    # verbatim and never LLM-authored, regardless of this flag), or the
    # compile-safety checks (LaTeX command allowlist, brace/math balance).
    aggressive: bool = False
    # Per-build opt-in (caller-set only, like aggressive). Superset of
    # aggressive: keeps entry headers but fabricates ALL bullet content from the
    # JD. Skills section is rewritten (not just reordered) to match JD keywords,
    # all grounding/validation guards bypassed, bold density uncapped.
    strong_aggressive: bool = False

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
            "max_bullet_chars",
            "max_total_bullet_chars",
            "limited_rewrite_bullets",
            "max_bold_per_bullet",
            "max_skill_items_per_line",
        ):
            value = payload.get(field_name)
            if isinstance(value, int) and value > 0:
                setattr(config, field_name, value)
        scope = payload.get("rewrite_scope")
        if isinstance(scope, str) and scope.strip().lower() in ("full", "limited", "selection"):
            config.rewrite_scope = scope.strip().lower()
        leeway_flag = payload.get("adjacent_tool_leeway")
        if isinstance(leeway_flag, bool):
            config.adjacent_tool_leeway = leeway_flag
        audit_flag = payload.get("grounding_audit")
        if isinstance(audit_flag, bool):
            config.grounding_audit = audit_flag
        if config.min_visible_bullets > config.max_visible_bullets:
            config.min_visible_bullets, config.max_visible_bullets = (
                config.max_visible_bullets,
                config.min_visible_bullets,
            )
        return config


# Aggressive-mode scaling, applied once at the top of render_structured_resume
# so every downstream numeric knob (bold cap) and the adjacent_tool_leeway/
# grounding_audit flags stay consistent for the rest of the render.
# max_bullet_chars is deliberately NOT raised here even though it looks like a
# "style" knob: live-trial evidence (2026-07-12, real Operations Development
# listing) showed raising it let much longer bullets survive validation,
# which silently broke the page-character-budget calibration and pushed a
# real render from 1 page to 2 — the trim loop's char_trim_floor stops
# shrinking bullet COUNT once min_visible_bullets-2 is reached regardless of
# how long each surviving bullet is, so a systematic length increase defeats
# it. Bullet-count and page-character budgets stay untouched too — those are
# physical page-fit limits, not tailoring conservatism, and loosening them
# risks a second page rather than buying more content freedom.
def _effective_render_config(config: RenderConfig) -> RenderConfig:
    if config.strong_aggressive:
        return replace(
            config,
            aggressive=True,
            strong_aggressive=True,
            adjacent_tool_leeway=True,
            grounding_audit=False,
            max_bold_per_bullet=config.max_bold_per_bullet + 4,
            rewrite_scope="full",
            # Depth over breadth (per profile owner 2026-07-17): fewer,
            # heavier bullets are allowed — raise the per-bullet ceiling and
            # lower the page-fill minimum so the model can trade count for
            # substance without triggering exclusion rollbacks.
            max_bullet_chars=config.max_bullet_chars + 120,
            min_visible_bullets=max(6, config.min_visible_bullets - 5),
        )
    if not config.aggressive:
        return config
    return replace(
        config,
        adjacent_tool_leeway=True,
        grounding_audit=False,
        max_bold_per_bullet=config.max_bold_per_bullet + 2,
    )


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
    # Always lowercase — matching is case-insensitive throughout this module.
    skill_anchors: tuple[str, ...] = ()
    # Same tools with their original baseinfo casing (e.g. "PostgreSQL", not
    # "postgresql"), keyed by the lowercase anchor. Display-only: shown to the
    # model in the prompt so a freshly patched-in tool (one with no existing
    # occurrence in that bullet to copy casing from) gets written correctly,
    # instead of echoing the lowercase form used for internal matching.
    skill_anchor_display: dict[str, str] = field(default_factory=dict)
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
class StructuredSelection:
    """Validated LLM decisions (or deterministic defaults when the LLM fails)."""

    ranking: list[str] = field(default_factory=list)
    bullets: dict[str, list[str]] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    # Per-entry "what this entry actually built/operated" phrases, written by
    # the model BEFORE ranking. A chain-of-thought scaffold: articulating each
    # entry's objects of work stops keyword-overlap bias (shared process words
    # like "debugging"/"testing" ranking a hardware entry high for a cloud
    # listing). Not used at render; kept for logging and tests.
    core_work: dict[str, str] = field(default_factory=dict)
    # Entries the model judges a poor fit for this specific listing. Honoured
    # in ranked order, but rolled back if the page cannot be filled without
    # them — the capacity floor, not a category identity, is the guarantee.
    exclusions: list[str] = field(default_factory=list)
    # Per-entry bullet indices the model considers weakest (weakest first);
    # used to steer trimming when the render is over budget.
    weak_bullets: dict[str, list[int]] = field(default_factory=dict)
    # Per-entry bullet display order, most relevant first ("front-load the
    # page"). Canonical indices; invalid/partial orders are merged tolerantly.
    bullet_order: dict[str, list[int]] = field(default_factory=dict)
    # Per-entry replacement for the header's \textit{...} technology list.
    # Every tool must be grounded in that entry's own bullets/baseinfo block
    # (validated at render); ungrounded lists are rejected wholesale.
    header_tech: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class RenderReport:
    visible_entries: list[str]
    hidden_entries: list[str]  # entries that did not make THIS render (excluded or out of budget)
    visible_bullet_count: int
    tailored_bullets_used: int
    canonical_fallbacks: list[str]  # "entry_id[index]: reason"
    excluded_entries: list[str] = field(default_factory=list)  # exclusions honoured
    ignored_exclusions: list[str] = field(default_factory=list)  # rolled back to fill the page
    fidelity_findings: list[str] = field(default_factory=list)  # template-structure drift
    rewrite_scope: str = "full"  # which rewrite dial rendered this document
    header_tech_applied: list[str] = field(default_factory=list)  # entries with retargeted \textit lists
    header_tech_rejected: list[str] = field(default_factory=list)  # "entry_id: reason"


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
    ({tag: block_text}, profile_header_text). Machine-oriented sections like
    SKILL ANCHORS carry no [tag] blocks and are naturally skipped.
    """
    scope = baseinfo_text

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


def parse_skill_anchor_display(baseinfo_text: str) -> dict[str, str]:
    """Map each lowercase skill anchor to its originally-cased form.

    Mirrors parse_skill_anchors' tokenizing exactly, but keeps the raw casing
    ("PostgreSQL") instead of lowercasing, so the prompt can show tools with
    correct capitalization instead of the lowercase form used for matching.
    First occurrence wins; empty when the profile has no SKILL ANCHORS section.
    """
    match = re.search(
        r"==\s*SKILL ANCHORS\s*==(.*?)(?:\n==\s|\Z)",
        baseinfo_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}
    display: dict[str, str] = {}
    for line in match.group(1).splitlines():
        _, _, values = line.partition(":")
        for token in (values if values else line).split(","):
            cleaned = token.strip()
            key = cleaned.lower()
            if cleaned and len(cleaned) < 60:
                display.setdefault(key, cleaned)
                for part in cleaned.split("/"):
                    part = part.strip()
                    part_key = part.lower()
                    if part:
                        display.setdefault(part_key, part)
    return display


# ---------------------------------------------------------------------------
# Prompt + response parsing
# ---------------------------------------------------------------------------

# Character budget for the job-description excerpt inside prompts. Kept at
# 4500 because the whole prompt must fit Groq's 24k ceiling and
# _truncate_prompt_for_provider cuts from the END — where the output-format
# rules live — so inflating the excerpt risks losing the JSON contract.
JOB_DESCRIPTION_EXCERPT_CHARS = 4500

_REQUIREMENT_SEGMENT_PATTERN = re.compile(
    r"requir|qualif|skill|proficien|experience (?:with|in)|must[- ]have"
    r"|nice[- ]to[- ]have|familiar|degree|years of|knowledge of|competenc",
    re.IGNORECASE,
)


def excerpt_job_description(
    description: str, limit: int = JOB_DESCRIPTION_EXCERPT_CHARS
) -> str:
    """Budgeted excerpt that keeps requirement text from beyond the head cut.

    Descriptions arrive whitespace-flattened and head-truncated upstream
    (compact_job_description caps at 7000 chars); a plain head slice at 4500
    then drops the tail — which is where postings usually put the
    requirements/qualifications lists the tailoring most needs. Under the
    same budget, spend ~55% on the head and the rest on requirement-shaped
    sentences salvaged from the truncated portion.
    """
    if len(description) <= limit:
        return description

    head_budget = (limit * 11) // 20
    boundary = description.rfind(". ", 0, head_budget)
    cut = boundary + 1 if boundary > head_budget // 2 else head_budget
    head = description[:cut].rstrip()

    marker = " [...] "
    remaining = limit - len(head) - len(marker)
    tail_segments: list[str] = []
    used = 0
    if remaining > 80:
        for segment in re.split(r"(?<=[.!?])\s+", description[cut:]):
            segment = segment.strip()
            if not segment or not _REQUIREMENT_SEGMENT_PATTERN.search(segment):
                continue
            cost = len(segment) + 1
            if used + cost > remaining:
                continue
            tail_segments.append(segment)
            used += cost

    if not tail_segments:
        return description[:limit].rstrip()
    return head + marker + " ".join(tail_segments)


_KNOWN_TECH_TERMS = frozenset({
    "active directory", "jira", "servicenow", "zendesk", "salesforce",
    "windows server", "windows", "linux", "macos", "ios", "android",
    "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
    "kubernetes", "k8s", "terraform", "ansible", "puppet", "chef",
    "jenkins", "circleci", "travis", "github actions", "gitlab ci",
    "ci/cd", "devops", "sre", "agile", "scrum", "kanban", "waterfall",
    "itil", "cobit", "prince2", "pmp", "six sigma", "lean",
    "powershell", "bash", "shell", "zsh",
    "tcp/ip", "dns", "dhcp", "vpn", "ssl", "tls", "ssh", "http",
    "ldap", "snmp", "smtp", "ftp", "sftp", "nfs", "samba",
    "cisco", "juniper", "fortinet", "palo alto",
    "vmware", "hyper-v", "virtualbox", "proxmox", "esxi",
    "splunk", "datadog", "grafana", "prometheus", "nagios", "elk",
    "mongodb", "redis", "mysql", "mariadb", "oracle", "cassandra",
    "dynamodb", "elasticsearch", "neo4j",
    "hadoop", "spark", "airflow", "databricks", "snowflake",
    "sap", "oracle erp", "netsuite", "workday",
    "figma", "sketch", "adobe xd", "invision",
    "photoshop", "illustrator", "after effects", "premiere",
    "unity", "unreal", "blender", "maya", "autocad", "solidworks",
    "matlab", "simulink", "labview", "spice", "ltspice",
    "jmeter", "selenium", "cypress", "playwright", "postman",
    "swagger", "openapi", "graphql", "grpc", "rest", "soap",
    "rabbitmq", "activemq", "celery", "sidekiq",
    "nginx", "apache", "iis", "caddy", "traefik",
    "spring", "spring boot", ".net", "asp.net", "django", "rails",
    "laravel", "express", "nest.js", "next.js", "nuxt",
    "vue", "angular", "svelte", "ember", "backbone",
    "flutter", "react native", "xamarin", "ionic", "cordova",
    "kotlin", "swift", "objective-c", "go", "golang", "rust", "scala",
    "r", "julia", "perl", "ruby", "php", "lua", "elixir", "erlang",
    "haskell", "clojure", "f#", "ocaml", "zig", "nim",
    "pytorch", "keras", "hugging face", "opencv", "nltk", "spacy",
    "tableau", "power bi", "looker", "qlik", "domo",
    "confluence", "notion", "asana", "trello", "monday.com",
    "slack", "microsoft teams", "zoom", "webex",
    "git", "svn", "mercurial", "perforce",
    "sccm", "intune", "jamf", "bigfix", "autopilot",
    "group policy", "gpo", "wsus", "wds",
    "exchange", "office 365", "microsoft 365", "sharepoint",
    "google workspace", "google suite", "gmail", "google docs",
    "comptia", "comptia a+", "ccna", "ccnp", "ccie",
    "okta", "auth0", "saml", "oauth", "sso",
    "bitbucket", "pagerduty", "opsgenie", "victorops",
    "new relic", "appdynamics", "dynatrace",
    "docker compose", "helm", "istio", "envoy",
    "supabase", "firebase", "amplify", "vercel", "netlify",
    "webpack", "vite", "rollup", "esbuild", "parcel",
    "jest", "mocha", "pytest", "junit", "nunit", "rspec",
    "storybook", "chromatic",
})


_SKIP_JD_INJECT = frozenset({
    "lean", "six sigma", "waterfall",
    "itil", "cobit", "prince2", "pmp",
    "comptia", "comptia a+", "ccna", "ccnp", "ccie",
    "teams", "rest",
    # Common-English-word / tech-term collisions (2026-07-17, found live: a
    # "Fall 2026 and Spring 2027" hiring-season sentence got "Spring" injected
    # as the Java framework into unrelated financial/robotics bullets AND the
    # Skills section). Each of these is overwhelmingly plain English in
    # ordinary JD prose, not the tool — unlike "windows"/"shell"/"oracle"/
    # "sap", whose tech sense dominates real postings and are left matchable.
    "spring", "go", "workday", "spark", "express", "helm", "celery",
    "airflow", "chef", "vite", "postman",
})


def _extract_jd_tools(description: str, skill_anchors: tuple[str, ...]) -> list[str]:
    anchor_lower = {a.lower() for a in skill_anchors}
    seen: set[str] = set()
    tools: list[str] = []
    for term in sorted(_KNOWN_TECH_TERMS, key=len, reverse=True):
        if term in anchor_lower or len(term) < 3:
            continue
        if term in _SKIP_JD_INJECT:
            continue
        pat = re.compile(r"(?<![a-zA-Z])" + re.escape(term) + r"(?![a-zA-Z])", re.IGNORECASE)
        m = pat.search(description)
        if not m:
            continue
        if term in seen:
            continue
        seen.add(term)
        original = m.group(0)
        tools.append(original)
    return tools


def build_structured_prompt(
    job_title: str,
    job_description: str,
    job_highlights: list[str],
    catalog: TemplateCatalog,
    extra_guidance: str = "",
) -> str:
    highlight_lines = "\n".join(f"- {line}" for line in job_highlights[:10]) or "- (none parsed)"

    # Read every render_config field through the aggressive-mode scaler so the
    # prompt's own notes (rewrite scope, anchor-boundary wording) match what
    # the validator will actually allow — otherwise the model stays
    # conservative out of habit even after the guardrails loosen.
    render_config = _effective_render_config(catalog.render_config)
    scope = render_config.rewrite_scope
    if scope == "selection":
        rewrite_scope_note = (
            "16. THIS PROFILE RENDERS ALL BULLET TEXT CANONICALLY: your 'bullets' values\n"
            "    are ignored, so return an empty object for 'bullets' and spend your\n"
            "    effort on ranking, bullet_order, weak_bullets, exclude, and keywords\n"
            "    instead.\n"
        )
    elif scope == "limited":
        rewrite_scope_note = (
            f"16. REWRITE BUDGET: only the first {render_config.limited_rewrite_bullets}\n"
            "    bullets of the final document use your rewrites — everything after\n"
            "    renders canonically. Concentrate your best rewrites in the entries you\n"
            "    rank most relevant; skip rewriting bullets that will fall outside the\n"
            "    budget.\n"
        )
    else:
        rewrite_scope_note = ""

    jd_inject_tools = (
        _extract_jd_tools(job_description, catalog.skill_anchors)
        if render_config.aggressive
        else []
    )

    entry_blocks: list[str] = []
    for entry in catalog.entries:
        if render_config.strong_aggressive:
            block = (
                f'entry_id: {entry.entry_id}\n'
                f'  section: {entry.section}\n'
                f'  title: {entry.title}\n'
                f'  bullet_count: {len(entry.bullets)}  (write UP TO this many NEW bullets —\n'
                f'  the full count by default; a shorter list is allowed when fewer,\n'
                f'  deeper bullets serve this JD better, and drops the trailing slots)\n'
                f'  NOTE: Ignore canonical content. Write new bullets entirely from the\n'
                f'  JD, as if this candidate performed that work here.'
            )
        else:
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
            background = "\n".join(dict.fromkeys(background_parts))
            if background:
                indented = "\n".join(f"    {line}" for line in background.splitlines())
                block += f"\n  verified background (TRUE facts — usable in any of this entry's bullets):\n{indented}"
            grounding = f"{' '.join(entry.bullets)} {entry.header} {background}"
            verified_tools = [
                catalog.skill_anchor_display.get(anchor, anchor)
                for anchor in catalog.skill_anchors
                if _whole_token_search(grounding, anchor)
            ]
            if verified_tools:
                if render_config.aggressive:
                    block += (
                        "\n  verified tools for THIS entry (bold freely; for additional JD\n"
                        "  tools see <jd_keywords_to_inject> — place them here ONLY if\n"
                        "  this entry's work could plausibly use them):\n"
                        f"    {', '.join(verified_tools)}"
                    )
                else:
                    block += (
                        "\n  verified tools for THIS entry (the only valid header_tech choices; "
                        "also safe to name/bold in its bullets):\n"
                        f"    {', '.join(verified_tools)}"
                    )
            metric_menu = _entry_metric_menu(entry, background)
            if metric_menu:
                block += (
                    "\n  metrics on offer (digits EXACT, keep each tied to what it measured):\n"
                    + "\n".join(f"    - {item}" for item in metric_menu)
                )
        entry_blocks.append(block)
    entries_text = "\n\n".join(entry_blocks)

    profile_section = (
        "\n<candidate_facts>\n"
        f"{catalog.baseinfo_profile}\n"
        "</candidate_facts>\n"
        if catalog.baseinfo_profile
        else ""
    )

    combined_guidance = extra_guidance.strip()
    if render_config.strong_aggressive:
        jd_tools = _extract_jd_tools(job_description, catalog.skill_anchors)
        jd_tools_block = ""
        if jd_tools:
            tool_list = ", ".join(jd_tools[:40])
            jd_tools_block = (
                f"\n<jd_keywords_to_inject>\n"
                f"ALL of these JD tools MUST appear in your bullets, wrapped in\n"
                f"\\textbf{{}}. Distribute them across entries so every bullet has\n"
                f"3-5 bolded terms. The Skills section is rewritten automatically.\n"
                f"{tool_list}\n"
                f"</jd_keywords_to_inject>\n"
            )
        strong_aggressive_note = (
            "STRONG AGGRESSIVE MODE — FULL FABRICATION FROM JOB DESCRIPTION.\n"
            "You are writing resume bullets FROM SCRATCH to match this JD.\n"
            "The candidate's original bullet text is IRRELEVANT — ignore it.\n"
            "Use ONLY entry titles/headers (job title, company, dates) as your\n"
            "structural framework. Everything else comes from the JD.\n\n"
            "FOR EACH ENTRY:\n"
            "  1. Read the JD requirements and responsibilities.\n"
            "  2. Write NEW bullet points describing work that matches those\n"
            "     requirements, as if the candidate performed it at this employer.\n"
            "  3. Every bullet MUST contain at least one \\textbf{keyword} from the JD.\n"
            "  4. Bold ALL technology/tool names that appear in the JD.\n"
            "  5. Bullets without JD keywords will be scrapped — make every one count.\n\n"
            "KEYWORD BOLDING RULES:\n"
            "  - Bold every JD technology, tool, platform, language, framework.\n"
            "  - Aim for 3-5 bolded terms per bullet.\n"
            "  - Each bolded keyword MUST also appear in the Skills section.\n\n"
            "QUALITY (survive a recruiter's 6-second skim):\n"
            "  - Each bullet: PAST-TENSE action verb → specific technical work →\n"
            "    measurable outcome. Built, Automated, Engineered — never the JD's\n"
            "    imperative present tense ('Translate', 'Refine', 'Maintain').\n"
            "  - Fabricate plausible metrics (percentages, team sizes, scale numbers).\n"
            "  - Write as a senior professional — no student/intern language.\n"
            "  - Vary action verbs — no two bullets start the same way.\n"
            "  - Each bullet is a self-contained achievement, not a task list.\n"
            "  - Employers, titles, and dates are FIXED — never alter header content.\n"
            "  - No keyword stuffing — each term in a grammatically natural clause.\n"
            "  - Match the JD's tone: if enterprise, write enterprise; if startup, lean.\n\n"
            "DEPTH OVER BREADTH — OPTIONAL, only when it genuinely helps\n"
            "(overrides rule 1's same-number requirement and rule 6's 260-character\n"
            "guidance when used):\n"
            "  - DEFAULT: fill bullet_count for every included entry, as above.\n"
            "  - But when the JD is narrow, or hitting the full count would force\n"
            "    thin bullets that repeat each other, you MAY exclude entries that\n"
            "    serve this JD poorly and return FEWER bullets than bullet_count\n"
            "    (a shorter list drops the trailing slots) — then expand the\n"
            "    bullets you do write.\n"
            "  - An expanded bullet may run to ~450 characters: one dense sentence,\n"
            "    or two tightly-linked clauses, covering the requirement end-to-end —\n"
            "    what was built, with which JD tools, at what scale, to what outcome.\n"
            "  - Never pad with filler just to hit a count, and never cut real\n"
            "    coverage just to be shorter. Every JD requirement should still be\n"
            "    covered somewhere either way.\n"
        )
        combined_guidance = (
            f"{combined_guidance}\n\n{strong_aggressive_note}"
            if combined_guidance else strong_aggressive_note
        )
        if jd_tools_block:
            combined_guidance += jd_tools_block
    elif render_config.aggressive:
        jd_tools = _extract_jd_tools(job_description, catalog.skill_anchors)
        jd_tools_block = ""
        if jd_tools:
            tool_list = ", ".join(jd_tools[:25])
            jd_tools_block = (
                f"\n<jd_keywords_to_inject>\n"
                f"These terms appear in the job description but NOT in the candidate's\n"
                f"existing skills. Weave each into a bullet WHERE IT FITS NATURALLY\n"
                f"and wrap in \\textbf{{}}. Place each term in the entry whose work\n"
                f"most plausibly involves it — skip any term that has no natural home.\n"
                f"The Skills section is handled automatically; focus on bullet text.\n"
                f"{tool_list}\n"
                f"</jd_keywords_to_inject>\n"
            )
        aggressive_note = (
            "AGGRESSIVE MODE — maximize JD keyword presence in bullet text.\n"
            "Check <jd_keywords_to_inject> for terms to place. For each one,\n"
            "find a bullet whose work COULD PLAUSIBLY involve it and weave it in\n"
            "wrapped in \\textbf{}. Placement strategy:\n"
            "  - Project tracking tool? → project/milestone bullet\n"
            "  - Sysadmin/IT tool? → infrastructure/deployment bullet\n"
            "  - Language/framework? → closest matching technical bullet\n"
            "You MAY claim the candidate used a JD tool they haven't explicitly\n"
            "listed — but ONLY where it is PLAUSIBLE given the entry's actual work.\n"
            "A deployment bullet can plausibly mention \\textbf{Docker}.\n"
            "A hardware debugging bullet CANNOT plausibly mention \\textbf{ServiceNow}.\n"
            "SKIP any term that has no natural home — the Skills section is\n"
            "auto-populated with JD tools, so forced bullet placement hurts more\n"
            "than it helps.\n"
            "PRIORITY ORDER for bullet content:\n"
            "  1. HARD TOOLS — languages, frameworks, databases, platforms, dev tools.\n"
            "     Every bullet should name at least one concrete tool/technology.\n"
            "  2. Technical actions — built, deployed, integrated, automated, tested.\n"
            "  3. Measurable outcomes — metrics, percentages, efficiency gains.\n"
            "  4. Soft skills are LAST — do NOT spend bullet space on leadership,\n"
            "     communication, collaboration, or teamwork unless no technical\n"
            "     content fits. Replace vague soft-skill phrases with specific\n"
            "     tool mentions: 'collaborated with the team' → 'integrated\n"
            "     components via \\textbf{Git} across a cross-functional team'.\n"
            "WHAT TO BOLD — only specific technology names: Python, Docker, React,\n"
            "  TensorFlow, PostgreSQL, etc. Do NOT bold generic JD phrases like\n"
            "  'AI workflows', 'machine learning models', 'proof-of-concept builds',\n"
            "  'ship code', 'integrations', 'customer-facing', 'data pipelines'.\n"
            "  Bold the TOOL, not the concept. Wrong: \\textbf{AI tooling}.\n"
            "  Right: built \\textbf{TensorFlow} models for AI tooling.\n"
            "COHERENCE IS MANDATORY — every claim must make sense to a human reader:\n"
            "  - Do NOT reframe the fundamental nature of the work. A personal bot\n"
            "    project stays a bot project. A rocket team stays a rocket team.\n"
            "  - Do NOT use enterprise buzzwords where they don't fit. No 'raising\n"
            "     stakeholder value' in a personal side project. No 'driving business\n"
            "     operations' in a student rocketry club.\n"
            "  - ADD JD tools to bullets about work that could plausibly use them.\n"
            "    Don't reshape the entire bullet around the keyword.\n"
            "Tools beyond <skill_anchors> are allowed. Aim for 2-4 bolded terms\n"
            "per bullet, but never sacrifice coherence for keyword count.\n"
            "HARD CONSTRAINTS:\n"
            "  - Every bullet must read as a coherent, grammatically correct\n"
            "    achievement statement. No keyword lists, no gibberish.\n"
            "  - Each bullet: action verb → what was done → measurable impact.\n"
            "  - Employers, titles, and dates are FIXED — never invent or alter.\n"
            "  - Metrics (numbers/percentages) from the profile are FIXED — use\n"
            "    them but never change their values."
        )
        combined_guidance = f"{combined_guidance}\n\n{aggressive_note}" if combined_guidance else aggressive_note
        if jd_tools_block:
            combined_guidance += jd_tools_block
    guidance_section = f"\n<profile_guidance>\n{combined_guidance}\n</profile_guidance>\n" if combined_guidance else ""

    # Normal-mode counterpart of aggressive's <jd_keywords_to_inject>: surface
    # the listing's own vocabulary as an explicit checklist so JD language is
    # woven in wherever it truthfully fits, instead of relying on the model to
    # extract terms itself. Aggressive modes keep their stronger injection
    # blocks; this one stays truth-gated (rules 3 and 8 govern placement).
    listing_language_section = ""
    if not render_config.aggressive and not render_config.strong_aggressive:
        listing_terms = extract_listing_keywords(
            "\n".join(filter(None, [job_title, *job_highlights, job_description])),
            catalog.skill_anchors,
            max_keywords=24,
        )
        if listing_terms:
            display_terms = [
                catalog.skill_anchor_display.get(term, term) for term in listing_terms
            ]
            listing_language_section = (
                "\n<listing_language>\n"
                "Vocabulary this listing actually uses. Weave each term into a\n"
                "bullet wherever it truthfully describes that entry's real work\n"
                "(rule 15):\n"
                f"{', '.join(display_terms)}\n"
                "</listing_language>\n"
            )
    if listing_language_section:
        listing_language_rule = (
            "15. LISTING LANGUAGE EVERYWHERE. Work the terms in <listing_language>\n"
            "    into the bullets wherever they truthfully describe that entry's real\n"
            "    work: tools under rule 3's grounding, soft/functional vocabulary as\n"
            "    plain text under rule 8. Aim for every visible bullet to carry at\n"
            "    least one listing term, spreading DIFFERENT terms across bullets\n"
            "    (rule 6's two-bullet cap per keyword still applies) to cover as many\n"
            "    distinct terms as truth allows. Before returning, re-scan\n"
            "    <listing_language> and place any still-missing term that has a\n"
            "    truthful home.\n"
        )
    else:
        listing_language_rule = ""

    if catalog.skill_anchors:
        if render_config.strong_aggressive:
            tool_rule = (
                "3. KEYWORD SATURATION — bold EVERY JD tool in every bullet where\n"
                "   it appears. You are NOT restricted to <skill_anchors> — use ANY\n"
                "   tool from the JD. Each bullet should have 3-5 bolded terms.\n"
                "   Lead with the most important JD requirements for that bullet's\n"
                "   topic area. Every single keyword from <jd_keywords_to_inject>\n"
                "   must land in at least one bullet.\n"
            )
        elif render_config.aggressive:
            tool_rule = (
                "3. KEYWORD INJECTION — bold JD tools in bullets where they fit.\n"
                "   You are NOT restricted to <skill_anchors> — any tool from the JD\n"
                "   is fair game IF the entry's work could plausibly involve it.\n"
                "   <skill_anchors> tools matching the JD get FIRST priority.\n"
                "   Then place JD-only tools from <jd_keywords_to_inject> — but ONLY\n"
                "   in entries where the tool fits the actual work described.\n"
                "   Aim for 2-4 bolded terms per bullet — HARD TOOLS FIRST.\n"
                "   Prefer naming a concrete technology (language, framework, database,\n"
                "   platform) over soft-skill filler. Every bullet should have at least\n"
                "   one \\textbf{tool} if possible.\n"
                "   RESTRICTION: each bolded term must sit inside a coherent clause.\n"
                "   SKIP any JD tool that has no natural entry — forcing it hurts\n"
                "   more than missing it. The Skills section is auto-populated.\n"
            )
        elif render_config.adjacent_tool_leeway:
            anchor_boundary_note = (
                "   Beyond <skill_anchors>, you may name a tool ONLY when it is closely\n"
                "   adjacent to work that entry verifiably did — same ecosystem, same kind of\n"
                "   integration (work built directly on a vendor's APIs supports naming a\n"
                "   popular library that wraps those same APIs) — AND this listing asks for\n"
                "   it. The claim must survive an interview follow-up question. Never name\n"
                "   certifications the candidate lacks, employers, or equipment they never\n"
                "   touched, and never a technology from an unrelated stack.\n"
            )
            tool_rule = (
                "3. INCLUDE AS MANY RELEVANT TOOLS AS POSSIBLE. Lead with the canonical bullet's\n"
                "   own named tools/systems/methods, and freely weave in additional ones from\n"
                "   <skill_anchors> wherever they are relevant to the listing AND plausibly\n"
                "   connect to that bullet's work (e.g. \\textbf{Git} in a deployment bullet,\n"
                "   \\textbf{Epic} in a charting bullet). Every <skill_anchors> tool the listing\n"
                "   itself asks for should land in at least one visible bullet where it\n"
                "   plausibly fits — before finishing, re-check the listing's requirements and\n"
                "   find a home for any still-missing anchor it names. Add the NAME only, fold\n"
                "   into an existing clause about real work — never invent a generic clause\n"
                "   whose only job is to house the keyword ('integrated into a robust Git-based\n"
                "   workflow', 'leveraging Git for collaboration'). If no bullet has a clause\n"
                "   that can naturally take it, leave it for the Skills line instead of forcing\n"
                "   it.\n" + anchor_boundary_note
            )
        else:
            anchor_boundary_note = (
                "   Named skills NOT in <skill_anchors> must never appear — a listing asking\n"
                "   for Kubernetes (or a certification the candidate lacks) does not make it true.\n"
            )
            tool_rule = (
                "3. INCLUDE AS MANY RELEVANT TOOLS AS POSSIBLE. Lead with the canonical bullet's\n"
                "   own named tools/systems/methods, and freely weave in additional ones from\n"
                "   <skill_anchors> wherever they are relevant to the listing AND plausibly\n"
                "   connect to that bullet's work (e.g. \\textbf{Git} in a deployment bullet,\n"
                "   \\textbf{Epic} in a charting bullet). Every <skill_anchors> tool the listing\n"
                "   itself asks for should land in at least one visible bullet where it\n"
                "   plausibly fits — before finishing, re-check the listing's requirements and\n"
                "   find a home for any still-missing anchor it names. Add the NAME only, fold\n"
                "   into an existing clause about real work — never invent a generic clause\n"
                "   whose only job is to house the keyword ('integrated into a robust Git-based\n"
                "   workflow', 'leveraging Git for collaboration'). If no bullet has a clause\n"
                "   that can naturally take it, leave it for the Skills line instead of forcing\n"
                "   it.\n" + anchor_boundary_note
            )
    else:
        tool_rule = (
            "3. Only mention named tools, systems, or methods the canonical bullet\n"
            "   itself contains — a listing asking for a skill the candidate lacks does\n"
            "   not make it true.\n"
        )
    display_anchors = [
        catalog.skill_anchor_display.get(anchor, anchor) for anchor in catalog.skill_anchors
    ]
    anchors_section = (
        "\n<skill_anchors>\n"
        "Tools, systems, methods, and certifications the candidate GENUINELY has\n"
        "(their full real skill set, whatever the field). You may weave any of\n"
        "these into bullets where they are relevant to the listing. Use the exact\n"
        "capitalization shown here when introducing one of these tools into a bullet\n"
        "for the first time:\n"
        f"{', '.join(display_anchors)}\n"
        "</skill_anchors>\n"
        if catalog.skill_anchors
        else ""
    )

    if render_config.strong_aggressive:
        task_text = (
            "<task>\n"
            "PRIMARY GOAL: fabricate this resume entirely from the job description.\n"
            "Keep only entry structure (titles, companies, dates). Write ALL bullet\n"
            "content as if the candidate did the exact work the JD describes.\n"
            "Every bullet must be keyword-rich and professionally written. A recruiter\n"
            "skimming for 6 seconds should see a perfect match.\n"
            "Return ONLY a JSON object with your decisions.\n"
            "</task>\n\n"
        )
    else:
        task_text = (
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
        )

    return (
        task_text +
        "<job>\n"
        f"Title: {job_title}\n"
        f"Highlights:\n{highlight_lines}\n"
        f"Description excerpt:\n{excerpt_job_description(job_description)}\n"
        "</job>\n\n"
        "<resume_entries>\n"
        f"{entries_text}\n"
        "</resume_entries>\n"
        f"{profile_section}"
        f"{anchors_section}"
        f"{listing_language_section}"
        f"{guidance_section}\n"
        "<output_format>\n"
        "Return ONLY a JSON object (no prose, no markdown fences) with exactly these keys:\n"
        '{\n'
        '  "keywords": ["top 8-10 hard skills/tools/competencies quoted verbatim from the listing"],\n'
        '  "core_work": {"<entry_id>": "a few words: the actual OBJECTS of this entry\'s\n'
        '                work — what it built, operated, or delivered (fill for EVERY\n'
        '                entry, BEFORE writing ranking)"},\n'
        '  "ranking": ["every entry_id, ordered most relevant to the listing first"],\n'
        '  "exclude": ["entry_ids that genuinely do NOT fit this listing (optional)"],\n'
        '  "weak_bullets": {"<entry_id>": [indices of its weakest bullets, weakest first]},\n'
        '  "bullet_order": {"<entry_id>": [bullet indices, MOST relevant to this listing\n'
        '                   first — the reader sees the first bullet before any other]},\n'
        '  "header_tech": {"<entry_id>": ["2-5 technologies for that entry\'s header\n'
        '                  tech list, drawn ONLY from that entry\'s own bullets/verified\n'
        '                  background, ordered by relevance to this listing (optional)"]},\n'
        '  "bullets": {"<entry_id>": ["rewritten bullet 0", "rewritten bullet 1", ...], ...}\n'
        '}\n'
        "Notes on header_tech: each entry header shows a short italic technology list\n"
        "(e.g. 'Assembly, C'). You may reorder it, drop items, or swap in different\n"
        "technologies — but ONLY ones named in that same entry's canonical bullets or\n"
        "verified background (one ungrounded tool rejects the whole list). Lead with\n"
        "what this listing values.\n"
        "Notes on ranking/exclude/weak_bullets: ranking decides what appears — entries\n"
        "are included most-relevant-first until the page is full, so rank on this\n"
        "specific listing's actual needs, not on any fixed idea of which entries belong\n"
        "to which job type. Fill core_work FIRST, then rank by comparing each entry's\n"
        "core_work phrase against what THIS listing's day-to-day work builds and\n"
        "operates. Overlap that exists only in generic process words (testing,\n"
        "debugging, automation, design, documentation, integration) does NOT count —\n"
        "those words appear in every technical job; an entry ranks high only when the\n"
        "THINGS worked on are the kind of things this role works on. When no entry's\n"
        "objects genuinely overlap (out-of-domain listing), rank instead by which\n"
        "entries present most credibly to this listing's recruiter — transferable,\n"
        "broadly-readable work above narrow specialist work.\n"
        "TECHNICAL COMPLEXITY IS NOT A TIEBREAKER — do not let one entry's objects sound\n"
        "more impressive or more 'engineering' than another's decide the ranking; only\n"
        "object proximity to THIS listing decides it. Example: for a cloud/DevOps\n"
        "listing (objects: pipelines, containers, cloud infrastructure), an entry whose\n"
        "core_work is 'embedded firmware development and hardware verification' sounds\n"
        "more technically impressive, but its objects (circuit boards, firmware) share\n"
        "nothing with this listing's — rank it LOW. An entry whose core_work is 'system\n"
        "documentation and change validation' sounds less impressive, but change\n"
        "management and IT process work sit in the same neighborhood as DevOps process\n"
        "work — rank it HIGHER despite the plainer core_work phrase. The recruiter reads\n"
        "for domain proximity, not for which entry required more skill to produce.\n"
        "Any combination of entries can be the right one. Use exclude\n"
        "only for entries whose content would look OUT OF PLACE to a recruiter for this\n"
        "specific listing — not merely the least relevant ones (ranking already handles\n"
        "that); exclusions are rolled back automatically if the page cannot be filled\n"
        "without them. weak_bullets guides which bullet is dropped first if space runs\n"
        "out; still provide rewrites for every bullet.\n"
        "Rules for rewritten bullets:\n"
        "1. Provide the SAME NUMBER of bullets per entry as the canonical list, same\n"
        "   order (bullets[i] replaces canonical bullet i; use bullet_order to change\n"
        "   what the reader sees first). Within an entry you have full freedom: the\n"
        "   entry's canonical bullets PLUS its 'verified background' lines are one\n"
        "   inventory of true facts, and any of those facts may appear in any of that\n"
        "   entry's bullets — restructure, merge emphasis, reframe what was built in\n"
        "   terms of the listing's problems. You are re-aiming each bullet at this job,\n"
        "   not lightly editing it. Facts must never cross BETWEEN entries, and no two\n"
        "   bullets may state the same metric.\n"
        "2. REWRITE AGGRESSIVELY around the listing. Extract its keywords first, then\n"
        "   rework every bullet to speak the listing's language, mirroring its exact\n"
        "   terminology (if it says 'RESTful APIs', write 'RESTful APIs', not 'REST\n"
        "   APIs'). Metrics are real outcomes: keep them (digit-for-digit) where they\n"
        "   strengthen the bullet — which is almost always — but you may drop one when\n"
        "   the rewrite genuinely reads better without it. NEVER alter digits or invent\n"
        "   new measurements. Light extrapolation from what an entry clearly implies is\n"
        "   fine (deployment implies configuration; testing implies debugging) — named\n"
        "   tools, employers, dates, and certifications are not extrapolation.\n"
        + tool_rule
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
        "   You have FULL AUTHORITY over emphasis: add, remove, or swap \\textbf so the\n"
        f"   bolded names are the tools THIS listing values most (max {render_config.max_bold_per_bullet} per bullet) — a\n"
        "   canonical bold is not sacred, and a relevant tool the candidate truly used\n"
        "   deserves bold even where the canonical bullet left it plain. But \\textbf\n"
        "   may wrap ONLY named tools, systems, software, equipment, methods, or\n"
        "   certifications — never soft phrases from the posting ('analysis\n"
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
        "   subject, verb, or object. For EXPERIENCE entries specifically, this is a\n"
        "   change of VOCABULARY, never a change of FUNCTION: the reader must still be\n"
        "   able to tell what kind of technical work it actually was. Do not swap the\n"
        "   entry's real activity for a different discipline the listing happens to\n"
        "   want (a software/backend internship rewritten as marketing content-\n"
        "   creation, audience growth, or brand work; a trading-data pipeline rewritten\n"
        "   with every trace of what it processed removed) — that fails an interview\n"
        "   follow-up and is exactly as disqualifying as inventing a tool.\n"
        "   Example — canonical: 'Built production web applications in React and\n"
        "   TypeScript.' Listing: Social Media Marketing Intern.\n"
        "   BAD (function swapped): 'Designed engaging visual layouts to optimize\n"
        "   digital interaction and maximize audience growth.'\n"
        "   GOOD (vocabulary shifted, function intact): 'Built user-facing web\n"
        "   interfaces in React and TypeScript, applying the same attention to visual\n"
        "   layout and audience-facing polish that content-focused roles demand.'\n"
        "   Projects (no employer to contradict the story) still get more latitude,\n"
        "   per the note below.\n"
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
        "14. OWNERSHIP AND SCOPE. Calibrate credit to the verified background:\n"
        "    solo-ownership verbs (Owned, Led, Spearheaded, Directed, Headed) only\n"
        "    where that entry's background states individual ownership or leadership;\n"
        "    otherwise open with an active collaborative verb (Built, Developed,\n"
        "    Engineered, Co-led). Where the background states a scope fact — team\n"
        "    size, user base, live operation, system criticality — state it early in\n"
        "    the bullet: scope makes the work legible before the tool names do. When\n"
        "    choosing which 'metrics on offer' to keep visible, prefer the CATEGORY\n"
        "    this listing's language values (scale, cost, reliability, adoption);\n"
        "    never invent a number for a category the entry lacks.\n"
        + listing_language_rule
        + rewrite_scope_note +
        "</output_format>"
    )


def _entry_metric_menu(entry: TemplateEntry, background: str) -> list[str]:
    """Short list of the entry's real metrics with context, for the prompt.

    Bullets are scanned before background lines so the crispest phrasing wins;
    years and the background block's heading line (dates) are skipped. When a
    baseinfo line opens with a metric-category label (Scope:/Reliability:/...),
    the label survives the snippet window so the model sees the category.
    """
    bg_lines = background.splitlines()
    sources = list(entry.bullets) + bg_lines[1:]
    menu: list[str] = []
    seen: set[str] = set()
    for source in sources:
        plain = source.replace("\\%", "%")
        plain = re.sub(r"\\[a-zA-Z]+\{?", " ", plain).replace("}", " ")
        plain = " ".join(plain.split())
        label = METRIC_CATEGORY_LABEL_PATTERN.match(plain)
        # Comma-grouped numbers ("206,775") are ONE menu item, deduped on the
        # comma-stripped form, and the snippet keeps the original commas so
        # the model reproduces the readable form.
        for match in MENU_NUMBER_PATTERN.finditer(plain):
            number = match.group(0).replace(",", "")
            if re.fullmatch(r"(?:19|20)\d{2}", number) or number in seen:
                continue
            seen.add(number)
            start = plain.rfind(" ", 0, max(0, match.start() - 40)) + 1
            end = plain.find(" ", match.end() + 30)
            snippet = plain[start : end if end != -1 else len(plain)].strip(" ,;.")
            if label and start > label.end():
                snippet = f"{label.group(0)} ... {snippet}"
            menu.append(snippet)
            if len(menu) >= 6:
                return menu
    return menu


_LISTING_STOPWORDS = frozenset(
    """the and for with you your our are will this that from have has been being to of in on at by as or an a is it we
    they them their its be can may must should would could who what when where how why not no yes all any each more
    most other some such than then work working job role position candidate candidates team teams company companies
    apply application applicants experience experiences years responsibilities requirements qualifications preferred
    required skills skill ability able strong excellent good great new about into within across per including include
    includes also both while during under over between please status equal employment opportunity opportunities
    employer veteran disability gender race religion orientation accommodation accommodations intern internship co-op
    coop student students summer fall winter spring join environment benefits salary location remote hybrid onsite
    full-time part-time hours week weeks month months day days looking seeking ideal successful support supporting
    provide providing ensure ensuring develop developing development related relevant various through using level
    degree bachelor master university college program field area areas duties tasks
    avec pour dans notre votre vous nous sont être etre cette celles ceux aux par plus tout tous toute toutes comme
    ainsi afin chez sous entre aussi leur leurs elle elles vos ils mais fait faire sans autres autre bien très tres
    sera seront doit devra devrez poste emploi équipe equipe stagiaire stage candidat candidats candidate candidature
    exigences compétences competences expérience expériences connaissances milieu travail journée langue français
    francais anglais veuillez développement developpement déveloper travailler titre lieu heures semaine semaines
    salaire avantages télétravail teletravail""".split()
)


def extract_listing_keywords(
    job_text: str,
    skill_anchors: tuple[str, ...] = (),
    max_keywords: int = 10,
) -> list[str]:
    """Deterministic listing-keyword extraction — no LLM required.

    Used when every provider fails, so the canonical fallback render still
    gets listing-aware skills ordering and tool emphasis. Anchors the listing
    explicitly names come first (they drive emphasize_listing_tools and the
    skills reorder), then frequent content tokens weighted toward the title
    and tech-shaped words.
    """
    lines = [line for line in job_text.splitlines() if line.strip()]
    title_lower = lines[0].lower() if lines else ""

    keywords: list[str] = []
    anchor_hits: list[tuple[int, str]] = []
    for anchor in skill_anchors:
        hits = len(
            re.findall(
                rf"(?<![a-zA-Z0-9.+#]){re.escape(anchor)}(?![a-zA-Z0-9+#])(?!\.[a-zA-Z0-9])",
                job_text,
                re.IGNORECASE,
            )
        )
        if hits:
            anchor_hits.append((hits, anchor))
    anchor_hits.sort(key=lambda pair: -pair[0])
    keywords.extend(anchor for _, anchor in anchor_hits)

    scores: dict[str, tuple[int, str]] = {}
    # Accented letters stay inside a token — bilingual Canadian postings
    # otherwise shed fragments like "veloppement" (from "développement") into
    # the keyword list shown to the model.
    for token in re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9.+#/-]{3,}", job_text
    ):
        low = token.lower().strip("-/.")
        if not low or low in _LISTING_STOPWORDS:
            continue
        # URL paths, domains, and job-key hex ids are posting plumbing, not
        # listing language ("ca.indeed.com/viewjob", "af3732d8e7f6a952").
        # Slash compounds ("React/TypeScript") are redundant with their parts,
        # which the anchor pass already surfaces individually.
        if (
            "/" in low
            or re.search(r"\.(?:com|ca|org|net|io|dev|ai)$", low)
            or re.fullmatch(r"[0-9a-f]{10,}", low)
        ):
            continue
        tech_shaped = bool(re.search(r"[0-9.+#/]", token)) or (
            token[:1].isupper() and not token.istitle()
        ) or token.isupper()
        display = token.strip("-/.")  # sentence-ending punctuation is not part of the term
        score = scores.get(low, (0, display))[0] + 1
        if low in title_lower:
            score += 2
        if tech_shaped:
            score += 2
        scores[low] = (score, scores.get(low, (0, display))[1])

    taken = {k.lower() for k in keywords}
    for low, (score, original) in sorted(scores.items(), key=lambda kv: -kv[1][0]):
        if len(keywords) >= max_keywords:
            break
        if score < 3 or low in taken:
            continue
        taken.add(low)
        keywords.append(original)
    return keywords[:max_keywords]


_VALID_JSON_STRING_ESCAPES = set('"\\/bfnrtu')

# LaTeX commands whose first letter collides with a valid JSON escape: a model
# emitting ``\textbf`` unescaped parses as TAB + ``extbf`` and the PDF renders
# literal "extbf". When the text after the backslash spells one of these, the
# backslash is LaTeX, not a JSON escape.
_LATEX_JSON_ESCAPE_COLLISIONS = re.compile(
    r"(?:textbf|textit|texttt|textsc|times|frac|bullet|newline|underline)(?![a-zA-Z])"
)


def _json_repair_state(fragment: str) -> tuple[bool, list[str]]:
    """Scan a JSON fragment and return (inside_string, open bracket stack)."""
    stack: list[str] = []
    in_string = False
    escape = False
    for char in fragment:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()
    return in_string, stack


def _salvage_truncated_json(fragment: str) -> dict | None:
    """Recover a usable dict from a truncated JSON fragment.

    Models hitting their output-token limit die mid-response; discarding the
    whole payload loses every bullet that DID arrive. Close the open string,
    drop the dangling tail back to the previous comma/opener when needed, and
    append the missing closers — parse_structured_response tolerates any
    missing keys.
    """
    cut = len(fragment)
    for _ in range(16):
        frag = fragment[:cut]
        in_string, stack = _json_repair_state(frag)
        if not stack and not in_string:
            return None  # nothing open — not a truncation problem
        candidate = frag + ('"' if in_string else "")
        candidate = candidate.rstrip()
        while candidate and candidate[-1] in ",:":
            candidate = candidate[:-1].rstrip()
        candidate += "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            next_cut = max(
                fragment.rfind(",", 0, cut - 1),
                fragment.rfind("{", 0, cut - 1),
                fragment.rfind("[", 0, cut - 1),
            )
            if next_cut <= 0:
                return None
            cut = next_cut
            continue
        return parsed if isinstance(parsed, dict) and parsed else None
    return None


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
    # The braces never balanced: the response was truncated mid-payload
    # (output-token limit, dropped connection). Salvage what arrived.
    return _salvage_truncated_json("".join(repaired))


def parse_structured_response(
    text: str,
    catalog: TemplateCatalog,
) -> StructuredSelection | None:
    """Turn raw LLM output into a validated selection, or None if unusable."""
    payload = extract_json_object(text)
    if payload is None:
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

    core_work: dict[str, str] = {}
    core_work_raw = payload.get("core_work")
    if isinstance(core_work_raw, dict):
        for entry_key, phrase in core_work_raw.items():
            resolved = _resolve_entry_id(entry_key)
            if resolved is None or not isinstance(phrase, str) or not phrase.strip():
                continue
            core_work[resolved] = phrase.strip()

    exclusions: list[str] = []
    for key in ("exclude", "exclusions", "omit"):
        raw = payload.get(key)
        if isinstance(raw, list):
            for item in raw:
                resolved = _resolve_entry_id(item)
                if resolved is not None and resolved not in exclusions:
                    exclusions.append(resolved)
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

    header_tech: dict[str, list[str]] = {}
    header_tech_raw = payload.get("header_tech")
    if isinstance(header_tech_raw, dict):
        for entry_key, tools in header_tech_raw.items():
            resolved = _resolve_entry_id(entry_key)
            if resolved is None or not isinstance(tools, list):
                continue
            cleaned_tools = [
                str(item).strip() for item in tools if isinstance(item, (str, int, float)) and str(item).strip()
            ]
            if cleaned_tools:
                header_tech[resolved] = cleaned_tools

    return StructuredSelection(
        ranking=ranking,
        bullets=bullets,
        keywords=keywords,
        core_work=core_work,
        exclusions=exclusions,
        weak_bullets=weak_bullets,
        bullet_order=bullet_order,
        header_tech=header_tech,
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


def _restore_canonical_emphasis(text: str, entry_context: str, max_bold: int) -> str:
    """Re-apply the template's \\textbf{} tool-emphasis convention to a rewrite
    that dropped it.

    Some models follow the bolding instructions inconsistently across runs,
    producing visually flat bullets next to properly-emphasised ones. Only
    phrases the entry's own canonical bullets bolded are restored (first
    unbolded occurrence each, up to the per-bullet cap) — a pure formatting
    repair that can never introduce a claim the entry didn't already make.
    """
    phrases = list(dict.fromkeys(re.findall(r"\\textbf\{([^}]+)\}", entry_context)))
    if not phrases:
        return text
    for phrase in phrases:
        if len(re.findall(r"\\textbf\{", text)) >= max_bold:
            break
        if not phrase.strip() or f"\\textbf{{{phrase}}}" in text:
            continue
        bold_spans = [
            (m.start(), m.end()) for m in re.finditer(r"\\textbf\{[^}]*\}", text)
        ]
        for m in re.finditer(rf"(?<![\w\\{{]){re.escape(phrase)}(?![\w}}])", text):
            if any(start <= m.start() < end for start, end in bold_spans):
                continue
            text = f"{text[:m.start()]}\\textbf{{{phrase}}}{text[m.end():]}"
            break
    return text


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
    listing_keywords: tuple[str, ...] = (),
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

    if limits.strong_aggressive:
        text = UNESCAPED_SPECIAL_PATTERN.sub(r"\\\1", text)
        if text.count("{") != text.count("}"):
            return None, "unbalanced braces"
        dollars = len(re.findall(r"(?<!\\)\$", text))
        if dollars % 2 != 0:
            return None, "unbalanced math delimiters"
        return text, None

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
        phrase_words = re.findall(r"[a-z0-9.+#-]+", normalized)
        if phrase_words and all(word in grounding_words for word in phrase_words):
            continue
        if skill_anchors and _matches_skill_anchor(normalized, skill_anchors):
            continue
        if limits.aggressive:
            continue
        looks_like_product = bool(re.search(r"[A-Z0-9.#+]", phrase))
        if len(phrase_words) > 1 or not looks_like_product:
            text = text.replace(f"\\textbf{{{bold_phrase}}}", bold_phrase)
            continue
        if skill_anchors and not limits.adjacent_tool_leeway:
            return None, f"invented tool {phrase[:40]!r}"

    # Fabrication can arrive unbolded too: a model that drops \textbf{} must
    # not smuggle a tool past grounding (observed: "State Management
    # (Riverpod)" written into a React bullet, unbolded, where the bold-only
    # check above never looks). Product-shaped tokens — digits/#/+ ("HTML5"),
    # interior capitals ("PyTorch"), or capitalized mid-sentence ("Jira") —
    # must be grounded in the entry or the candidate's anchors. Same leeway
    # policy as the bolded path: profiles with adjacent_tool_leeway keep the
    # insertion (keyword richness wins) and the LLM grounding audit remains
    # the arbiter for claims that outright don't make sense.
    masked = re.sub(r"\\textbf\{[^}]*\}", " ", text)  # bolded spans validated above
    for token_match in PRODUCT_TOKEN_PATTERN.finditer(masked):
        token = token_match.group(0).rstrip(".")
        if len(token) < 2:
            continue
        normalized = token.lower()
        if normalized in grounding_words:
            continue
        shaped = (
            bool(re.search(r"[0-9#+]", token))
            or token[1:] != token[1:].lower()
            or ("." in token and token != normalized)
        )
        if not shaped:
            if not token[0].isupper() or token == normalized:
                continue
            head = masked[: token_match.start()].rstrip()
            if not head or head.endswith((".", "!", "?", ";", ":")):
                continue  # sentence-start capitalization is style, not a claim
        if len(normalized) >= 3 and any(
            len(word) >= 3 and (normalized in word or word in normalized)
            for word in grounding_words
        ):
            continue
        if skill_anchors and (
            _matches_skill_anchor(normalized, skill_anchors)
            or any(
                len(anchor) >= 3 and (normalized in anchor or anchor in normalized)
                for anchor in skill_anchors
            )
        ):
            continue
        if skill_anchors and not limits.adjacent_tool_leeway:
            return None, f"ungrounded tool token {token[:40]!r}"

    # Formatting-consistency repair: restore the template's tool-emphasis
    # convention for canonical-bolded phrases the rewrite kept but un-bolded.
    # Runs before the density cap so the cap stays the single ceiling.
    text = _restore_canonical_emphasis(
        text, entry_context or canonical, limits.max_bold_per_bullet
    )

    # Bold density: 4+ bolded tokens per bullet reads as keyword farming, not
    # emphasis. Keep the N most listing-relevant bolds (keyword matches first,
    # then earliest mention), demote the rest to plain text.
    bold_spans = re.findall(r"\\textbf\{([^}]*)\}", text)
    if len(bold_spans) > limits.max_bold_per_bullet:
        keyword_lowered = [k.lower() for k in listing_keywords if k.strip()]

        def _listing_relevant(phrase: str) -> bool:
            p = phrase.strip().lower()
            return any(
                p == k or _whole_token_search(k, p) or _whole_token_search(p, k)
                for k in keyword_lowered
            )

        ranked = sorted(
            range(len(bold_spans)),
            key=lambda i: (0 if _listing_relevant(bold_spans[i]) else 1, i),
        )
        keep = set(ranked[: limits.max_bold_per_bullet])
        for i, phrase in enumerate(bold_spans):
            if i not in keep:
                text = text.replace(f"\\textbf{{{phrase}}}", phrase, 1)

    # Straight double quotes are wrong LaTeX typography, and quoting a posting
    # phrase is the same padding as bolting it. Aggressive mode drops this —
    # it is a wordiness/polish guard, not a truth check.
    if not limits.aggressive and '"' in text and '"' not in canonical:
        return None, "quoted posting phrase"

    # Filler tails are stripped rather than rejecting the whole rewrite: the
    # tailoring before the connector is usually good work worth keeping. A
    # degenerate result (almost nothing left) still falls back to canonical.
    # Aggressive mode skips this entirely — let the model's fuller phrasing
    # stand, since it is prose-restraint, not a fabrication risk.
    if not limits.aggressive:
        filler = FILLER_CLAUSE_PATTERN.search(text)
        if filler and not FILLER_CLAUSE_PATTERN.search(canonical):
            stripped = text[: filler.start()].rstrip(" ,;")
            if len(stripped) < 30:
                return None, f"appended filler clause {filler.group(1).lower()!r}"
            if not stripped.endswith((".", "!", "?")):
                stripped += "."
            text = stripped

    # Numbers are strict on every profile and every mode this code path sees:
    # a significant standalone number in a rewrite must already exist in the
    # entry's grounding (bullets, header, baseinfo block). Runs AFTER filler
    # stripping so a fabricated figure living only in a stripped tail doesn't
    # reject the salvageable rest of the bullet. Commas are stripped on both
    # sides so "206,775" and "206775" ground each other; digits embedded in
    # identifiers (STM32, HCS12) are neither claims nor grounding.
    grounding_numbers = set(
        STANDALONE_NUMBER_PATTERN.findall(grounding.replace(",", ""))
    )
    for number in STANDALONE_NUMBER_PATTERN.findall(text.replace(",", "")):
        if len(number.replace(".", "")) < DUP_METRIC_MIN_DIGITS:
            continue
        if number not in grounding_numbers:
            return None, f"ungrounded number {number!r}"

    # Ownership calibration: an opening solo-ownership verb needs support in
    # the entry's own grounding — either the exact verb already used there
    # (canonical truth: "Managed windowing…" licenses "Managed …") or an
    # explicit ownership/leadership fact (baseinfo "Ownership:" lines count;
    # incidental prose like "reported progress to management" does not).
    # Demote-not-veto: swap the verb, keep the rewrite. A bolded opener
    # (\textbf{Led} …) is unwrapped for the check so emphasis can't dodge it.
    lead_verb = re.match(r"(?:\\(?:textbf|textit|emph)\{)?([A-Za-z]+)", text)
    if lead_verb:
        verb = lead_verb.group(1).lower()
        swap = OWNERSHIP_VERB_SWAPS.get(verb)
        if swap and not (
            re.search(
                rf"(?<![A-Za-z-]){re.escape(verb)}(?![A-Za-z])",
                grounding,
                re.IGNORECASE,
            )
            or OWNERSHIP_EVIDENCE_PATTERN.search(grounding)
        ):
            text = text[: lead_verb.start(1)] + swap + text[lead_verb.end(1) :]

    text = UNESCAPED_SPECIAL_PATTERN.sub(r"\\\1", text)

    if text.count("{") != text.count("}"):
        return None, "unbalanced braces"
    dollars = len(re.findall(r"(?<!\\)\$", text))
    if dollars % 2 != 0:
        return None, "unbalanced math delimiters"

    return text, None


GROUNDING_AUDIT_PROMPT_HEADER = (
    "You are auditing rewritten resume bullets for fabricated claims.\n"
    "\n"
    "Each item has:\n"
    '- "section": "Experience" or "Projects" (or similar) — which part of the\n'
    "  resume this entry lives in\n"
    '- "grounding": the candidate\'s VERIFIED background for that resume entry'
    " (the only source of truth)\n"
    '- "bullet": a rewritten bullet that will appear on the resume\n'
    "\n"
    "Flag a bullet as fabricated ONLY when it introduces a CONCRETE claim that\n"
    "the grounding does not support:\n"
    "- a named industry or application domain the work never touched\n"
    "  (industrial, ventilation, mechanical, aerospace, clinical, financial\n"
    "  services, payments) — this includes SOFT, lowercase phrasing of a\n"
    "  specific industry's own activities, not just proper-noun systems (e.g.\n"
    "  \"grid planning\", \"load forecasting\", \"clinical triage\", \"underwriting\"\n"
    "  each name a real professional discipline just as concretely as a named\n"
    "  system would, and are fabrication when the grounding never touched that\n"
    "  industry). Applies equally to Experience AND Project entries — unlike\n"
    "  function substitution below, this one has no Project exception.\n"
    "- physical equipment, hardware, or infrastructure the work never involved\n"
    "  (circuitry, sensors, machinery, field equipment)\n"
    "- a named platform, standard, or certification (SCADA, ServiceNow, ISO)\n"
    "- a named programming language, framework, or technology that is NEITHER\n"
    "  in the candidate's skill list below NOR closely adjacent to work the\n"
    "  grounding really shows (claiming .NET, Rust, Kotlin, Swift, Go, an RTOS,\n"
    "  or device-driver work the grounding never mentions is exactly this —\n"
    "  there is no fixed list, judge each named technology on its own against\n"
    "  the skill list and grounding). A technology attributed to the WRONG\n"
    "  ecosystem is ALWAYS this kind of fabrication, even when the job listing\n"
    "  itself names it — e.g. a Flutter state-management library (Riverpod)\n"
    "  claimed in a React bullet, an iOS framework in an Android entry, a\n"
    "  Python ORM in a Node.js service. The claim must make technical sense,\n"
    "  not merely echo the listing's vocabulary.\n"
    "- a changed workplace context (claiming municipal work happened at a\n"
    "  startup or technology company)\n"
    "- FUNCTION SUBSTITUTION on an EXPERIENCE (not Project) entry: the bullet's\n"
    "  real activity has been swapped for a different discipline the grounding\n"
    "  does not show, even using only soft vocabulary (e.g. a software/backend\n"
    "  internship rewritten as marketing content-creation, audience growth, or\n"
    "  brand work; a data pipeline rewritten with no trace of what it actually\n"
    "  processed or why). Reworded EMPHASIS of the same real activity is fine;\n"
    "  a DIFFERENT activity wearing the same job title is fabrication.\n"
    "- INCOHERENT PURPOSE: the bullet's stated goal or outcome does not\n"
    "  logically follow from the actual work in the grounding, even when every\n"
    "  individual word is separately true (e.g. a physics/graphics simulation\n"
    "  project described as existing \"to optimize AI workloads\" when the\n"
    "  grounding shows it never touched AI workloads at all). This is a\n"
    "  coherence check, not a vocabulary check — judge whether the sentence's\n"
    "  own logic holds, not just whether its individual nouns appear elsewhere.\n"
    "  A domain expert reading it must not wince at a claim that doesn't add up.\n"
    "  Watch especially for a soft business/compliance phrase (\"certification-\n"
    "  readiness\", \"regulatory alignment\", \"audit preparedness\") spliced into a\n"
    "  bullet next to a metric that has no real causal link to it — e.g. website\n"
    "  documentation \"and certification-readiness\" reducing website downtime:\n"
    "  the metric is real and tied to documentation, but certification-readiness\n"
    "  had nothing to do with that number. Flag the phrase even though each word\n"
    "  alone reads as generic business vocabulary.\n"
    "\n"
    "Do NOT flag — these are allowed reframings, not fabrication:\n"
    "- abstract functional/business vocabulary: operational efficiency, system\n"
    "  integration, system optimization, modernization, product lifecycle,\n"
    "  stakeholder needs, quality assurance, compliance, regulatory awareness,\n"
    "  monitoring, verification, precision, insights, reporting\n"
    "- soft descriptors of the same work: complex, scalable, enterprise,\n"
    "  high-availability, real-time\n"
    "- tools in the candidate's skill list below, even if absent from this\n"
    "  entry's grounding\n"
    "- software tools, libraries, or frameworks closely adjacent to work the\n"
    "  grounding really shows — same ecosystem, same kind of integration (an\n"
    "  entry with real vendor-API integration work may name a popular library\n"
    "  that wraps those same APIs). Flag only named systems that imply a\n"
    "  domain, industry, equipment, or workplace the grounding does not\n"
    "  support\n"
    "- LaTeX markup such as \\textbf{...} or \\textit{...} — ignore markup\n"
    "- anything already present in the grounding\n"
    "\n"
    "When unsure whether a phrase is a concrete domain claim or abstract\n"
    "functional vocabulary, do NOT flag it. Judge ONLY against the grounding\n"
    "text, not general plausibility.\n"
    "\n"
    'Return ONLY a JSON object: {"verdicts": [{"id": <int>,'
    ' "fabricated": true|false, "phrase": "<the unsupported phrase, or empty>"}]}\n'
    "One verdict per item, using the same ids as given.\n"
)


def build_grounding_audit_items(
    catalog: TemplateCatalog, selection: StructuredSelection
) -> tuple[list[dict], dict[int, tuple[str, int]]]:
    """Judge-call payload for every genuinely rewritten bullet.

    Grounding per entry matches the render-time validation grounding (canonical
    bullets + header + baseinfo blocks). Bullets that are empty or identical to
    their canonical text are skipped — there is nothing to audit.
    """
    items: list[dict] = []
    id_map: dict[int, tuple[str, int]] = {}
    for entry in catalog.entries:
        provided = selection.bullets.get(entry.entry_id)
        if not provided:
            continue
        background = " ".join(
            catalog.baseinfo_blocks.get(tag, "") for tag in entry.categories
        )
        grounding = f"{' '.join(entry.bullets)} {entry.header} {background}".strip()
        for index, raw in enumerate(provided):
            if index >= len(entry.bullets):
                break
            text = " ".join(str(raw).split()).strip()
            if not text or text == " ".join(entry.bullets[index].split()).strip():
                continue
            item_id = len(items)
            id_map[item_id] = (entry.entry_id, index)
            items.append(
                {"id": item_id, "section": entry.section, "grounding": grounding, "bullet": text}
            )
    return items, id_map


def grounding_audit_prompt(items: list[dict], skill_anchors: tuple[str, ...] = ()) -> str:
    anchors_line = (
        "\nCandidate skill list (never flag these): " + ", ".join(skill_anchors) + "\n"
        if skill_anchors
        else ""
    )
    return (
        GROUNDING_AUDIT_PROMPT_HEADER
        + anchors_line
        + "\nItems:\n"
        + json.dumps(items, ensure_ascii=True, indent=1)
    )


def apply_grounding_verdicts(
    selection: StructuredSelection,
    id_map: dict[int, tuple[str, int]],
    verdicts: Any,
) -> list[str]:
    """Blank out bullets the judge flagged so they render canonical.

    Returns "entry_id[index]: ungrounded domain claim '<phrase>'" reasons.
    Malformed verdicts (bad ids, missing fields, wrong types) are ignored —
    the audit must never be able to break a build.
    """
    flagged: list[str] = []
    if not isinstance(verdicts, list):
        return flagged
    for verdict in verdicts:
        if not isinstance(verdict, dict) or not verdict.get("fabricated"):
            continue
        try:
            key = id_map.get(int(verdict.get("id")))
        except (TypeError, ValueError):
            key = None
        if key is None:
            continue
        entry_id, index = key
        provided = selection.bullets.get(entry_id)
        if provided is None or index >= len(provided) or not str(provided[index]).strip():
            continue
        provided[index] = ""
        phrase = " ".join(str(verdict.get("phrase") or "").split())[:60]
        detail = f" {phrase!r}" if phrase else ""
        flagged.append(f"{entry_id}[{index}]: ungrounded domain claim{detail}")
    return flagged


def _whole_token_search(haystack: str, needle: str) -> re.Match[str] | None:
    """Case-insensitive whole-token occurrence of `needle` in `haystack`.

    A trailing dot only blocks the match when it continues into a token
    ("Node" must not match inside "Node.js", but "WebRTC." at sentence end
    is a legitimate whole-token occurrence).
    """
    return re.search(
        rf"(?<![a-zA-Z0-9.+#]){re.escape(needle)}(?![a-zA-Z0-9+#])(?!\.[a-zA-Z0-9])",
        haystack,
        re.IGNORECASE,
    )


HEADER_TECH_PATTERN = re.compile(r"\\textit\{([^}]*)\}")
MAX_HEADER_TECH_ITEMS = 5


def retarget_header_tech(
    header: str,
    tools: list[str],
    grounding: str,
) -> tuple[str | None, str | None]:
    """Replace the header's \\textit{...} tech list with listing-aimed tools.

    Returns (new_header, None) or (None, rejection_reason). Grounding is
    strict: every tool must appear as a whole token in the entry's own
    canonical text (bullets + baseinfo block + current header), and the
    casing used is the grounding text's, not the model's. Any single
    ungrounded tool rejects the whole list — headers are too prominent for
    partial repair.
    """
    match = HEADER_TECH_PATTERN.search(header)
    if match is None:
        return None, "header has no \\textit tech list"
    if not tools:
        return None, "empty tool list"
    if len(tools) > MAX_HEADER_TECH_ITEMS:
        return None, f"too many tools ({len(tools)})"

    grounded: list[str] = []
    for tool in tools:
        cleaned = " ".join(str(tool).split()).strip(" ,;")
        if not cleaned or len(cleaned) > 30:
            return None, f"bad tool {str(tool)[:40]!r}"
        if re.search(r'[\\{}<>"$%&#_^~]', cleaned):
            return None, f"tool contains markup {cleaned[:40]!r}"
        found = _whole_token_search(grounding, cleaned)
        if found is None:
            return None, f"ungrounded tool {cleaned[:40]!r}"
        canonical_casing = found.group(0)
        if canonical_casing not in grounded:
            grounded.append(canonical_casing)

    rebuilt = f"\\textit{{{', '.join(grounded)}}}"
    return header[: match.start()] + rebuilt + header[match.end() :], None


def emphasize_listing_tools(
    text: str,
    skill_anchors: tuple[str, ...],
    keywords: list[str],
    max_bold: int,
) -> str:
    """Bold unbolded anchor tools that the listing explicitly asks for.

    Deterministic and additive-only: wraps existing whole-token occurrences of
    (skill anchor ∩ listing keyword) tools in \\textbf, never past `max_bold`
    total bolds, never inside an existing \\textbf group or math span, and
    never a tool that is already bolded somewhere in the bullet. Grounding is
    inherent — only text already present gets wrapped.
    """
    if not skill_anchors or not keywords:
        return text

    keyword_lowered = [k.lower() for k in keywords if k.strip()]

    def _keyword_wants(anchor: str) -> bool:
        anchor_l = anchor.lower()
        for keyword in keyword_lowered:
            if anchor_l == keyword or _whole_token_search(keyword, anchor_l):
                return True
        return False

    bold_count = len(re.findall(r"\\textbf\{", text))
    if bold_count >= max_bold:
        return text

    # Protected spans: existing \textbf{...} groups, math, and \textit groups.
    def _protected(spans_text: str) -> list[tuple[int, int]]:
        spans = [m.span() for m in re.finditer(r"\\text(?:bf|it)\{[^}]*\}", spans_text)]
        spans += [m.span() for m in re.finditer(r"(?<!\\)\$[^$]*(?<!\\)\$", spans_text)]
        return spans

    already_bolded = {
        b.strip().lower() for b in re.findall(r"\\textbf\{([^}]*)\}", text)
    }

    for anchor in skill_anchors:
        if bold_count >= max_bold:
            break
        if len(anchor) < 2 and anchor.lower() != "c":
            continue
        if anchor.lower() in already_bolded or not _keyword_wants(anchor):
            continue
        match = _whole_token_search(text, anchor)
        if match is None:
            continue
        if any(start <= match.start() < end for start, end in _protected(text)):
            continue
        span_text = match.group(0)
        text = f"{text[: match.start()]}\\textbf{{{span_text}}}{text[match.end() :]}"
        already_bolded.add(span_text.lower())
        bold_count += 1

    return text


def _unbold_non_jd_terms(
    text: str,
    jd_tools: tuple[str, ...],
    keywords: list[str],
    jd_text: str = "",
) -> str:
    """Strong-aggressive bolding policy: \\textbf belongs to JD-relevant terms.

    A bold survives when it matches a JD tool / listing keyword, or when the
    phrase appears verbatim (whole-token) in the listing text itself — the
    model reads the FULL description, so its bolds of real JD vocabulary
    (RCFA, Excel, compliance) must not be demoted just because the top-N
    keyword extraction missed them. Everything else is demoted to plain text:
    the words stay (the model may write freely to fill the page), the
    emphasis goes. Demotion also stops soft invented phrases ("process maps")
    leaking into the Skills reconcile pass.
    """
    wanted = {t.lower().strip() for t in jd_tools if t.strip()}
    wanted.update(k.lower().strip() for k in keywords if k.strip())
    if not wanted and not jd_text:
        return text
    jd_lower = jd_text.lower()

    def _keep(match: re.Match[str]) -> str:
        phrase = match.group(1)
        p = phrase.strip().lower()
        relevant = any(
            p == w or (len(w) >= 3 and w in p) or (len(p) >= 3 and p in w)
            for w in wanted
        )
        if not relevant and jd_lower and p:
            relevant = bool(_whole_token_search(jd_lower, p))
        return match.group(0) if relevant else phrase

    return re.sub(r"\\textbf\{([^}]*)\}", _keep, text)


def _bold_jd_tools(text: str, jd_tools: tuple[str, ...], max_bold: int) -> str:
    """Bold JD-extracted tools that appear unbolded in the bullet text."""
    bold_count = len(re.findall(r"\\textbf\{", text))
    if bold_count >= max_bold:
        return text
    already_bolded = {
        b.strip().lower() for b in re.findall(r"\\textbf\{([^}]*)\}", text)
    }
    protected = [m.span() for m in re.finditer(r"\\text(?:bf|it)\{[^}]*\}", text)]
    protected += [m.span() for m in re.finditer(r"(?<!\\)\$[^$]*(?<!\\)\$", text)]
    for tool in jd_tools:
        if bold_count >= max_bold:
            break
        if tool.lower() in already_bolded:
            continue
        pat = re.compile(
            rf"(?<![a-zA-Z0-9.+#]){re.escape(tool)}(?![a-zA-Z0-9+#])(?!\.[a-zA-Z0-9])",
            re.IGNORECASE,
        )
        match = None
        for m in pat.finditer(text):
            if not any(start <= m.start() < end for start, end in protected):
                match = m
                break
        if match is None:
            continue
        span_text = match.group(0)
        text = f"{text[:match.start()]}\\textbf{{{span_text}}}{text[match.end():]}"
        already_bolded.add(span_text.lower())
        bold_count += 1
        protected = [m.span() for m in re.finditer(r"\\text(?:bf|it)\{[^}]*\}", text)]
        protected += [m.span() for m in re.finditer(r"(?<!\\)\$[^$]*(?<!\\)\$", text)]
    return text


def _inject_jd_tools_into_skills(
    tail: str, jd_tools: tuple[str, ...], skill_anchors: tuple[str, ...],
) -> str:
    """Add JD-extracted tools to the Skills section for ATS keyword coverage."""
    if not jd_tools:
        return tail
    anchor_lower = {a.lower() for a in skill_anchors}
    tail_lower = tail.lower()
    new_tools = [t for t in jd_tools if t.lower() not in anchor_lower and t.lower() not in tail_lower]
    if not new_tools:
        return tail

    _CATEGORY_MAP = {
        "tools": {"servicenow", "jira", "zendesk", "salesforce", "splunk", "datadog",
                  "grafana", "prometheus", "nagios", "sccm", "intune", "jamf", "autopilot",
                  "exchange", "sharepoint", "active directory", "group policy", "wsus",
                  "new relic", "pagerduty", "confluence", "notion", "asana", "trello",
                  "figma", "sketch", "photoshop", "illustrator", "blender", "maya",
                  "autocad", "solidworks", "matlab", "labview", "jmeter", "postman",
                  "swagger", "storybook", "microsoft suite", "google suite", "google workspace",
                  "office 365", "microsoft 365", "microsoft teams"},
        "frameworks": {"spring", "spring boot", ".net", "asp.net", "django", "rails",
                       "laravel", "express", "nest.js", "next.js", "nuxt", "vue", "angular",
                       "svelte", "flutter", "react native", "xamarin", "ionic",
                       "pytorch", "keras", "opencv", "nltk", "spacy"},
        "platforms": {"aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
                      "kubernetes", "docker compose", "vmware", "hyper-v", "proxmox",
                      "supabase", "firebase", "vercel", "netlify", "github", "gitlab",
                      "bitbucket"},
        "languages": {"kotlin", "swift", "go", "golang", "rust", "scala", "r", "julia",
                      "perl", "ruby", "php", "lua", "elixir", "haskell", "powershell",
                      "bash"},
    }
    category_for: dict[str, str] = {}
    for cat, terms in _CATEGORY_MAP.items():
        for term in terms:
            category_for[term] = cat

    buckets: dict[str, list[str]] = {}
    for tool in new_tools:
        cat = category_for.get(tool.lower(), "tools")
        buckets.setdefault(cat, []).append(tool)

    out_lines: list[str] = []
    in_skills = False
    for line in tail.splitlines():
        if line.strip().startswith("\\section*"):
            in_skills = "skills" in line.lower()
        if in_skills:
            label_match = re.match(r"^(\\textbf\{([^}]*)\})\s*(.+)$", line.rstrip().rstrip("\\").rstrip())
            if label_match:
                label_text = label_match.group(2).lower().rstrip(":")
                for cat, tools in list(buckets.items()):
                    if cat in label_text or label_text in cat:
                        has_break = line.rstrip().endswith("\\\\")
                        core = line.rstrip()
                        if has_break:
                            core = core[:-2].rstrip()
                        core += ", " + ", ".join(tools)
                        if has_break:
                            core += " \\\\"
                        line = core
                        del buckets[cat]
                        break
        out_lines.append(line)

    if buckets:
        remaining = []
        for tools in buckets.values():
            remaining.extend(tools)
        in_skills_fb = False
        last_skills_idx = -1
        for i, ln in enumerate(out_lines):
            if ln.strip().startswith("\\section*"):
                in_skills_fb = "skills" in ln.lower()
            if in_skills_fb:
                stripped = ln.rstrip()
                if stripped.endswith("\\\\") and "\\textbf{" in stripped:
                    last_skills_idx = i
        if last_skills_idx >= 0:
            stripped = out_lines[last_skills_idx].rstrip()
            core = stripped[:-2].rstrip()
            out_lines[last_skills_idx] = core + ", " + ", ".join(remaining) + " \\\\"

    return "\n".join(out_lines)


def _rewrite_skills_for_jd(
    tail: str,
    jd_tools: tuple[str, ...],
    keywords: list[str],
    skill_anchors: tuple[str, ...],
    max_items: int = 0,
) -> str:
    """Rewrite skills section to match JD keywords (strong-aggressive only).

    JD-relevant canonical items lead, JD tools are inserted into their
    categorical lines, and the remaining canonical items follow (previously
    they were silently dropped, leaving gutted lines like "Languages: C,
    SQL"). `max_items` caps each line with the normal-mode reorder's
    semantics: only unmatched trailing items are cut, matched/JD items always
    survive the cap. Relevance uses whole-token matching so a single-letter
    item ("C") can't ride substring noise.
    """
    if not jd_tools and not keywords:
        return tail

    all_kw = {k.lower().strip() for k in keywords if k.strip()}
    all_kw.update(t.lower().strip() for t in jd_tools if t.strip())

    _CATEGORY_MAP = {
        "tools": {"servicenow", "jira", "zendesk", "salesforce", "splunk", "datadog",
                  "grafana", "prometheus", "nagios", "sccm", "intune", "jamf", "autopilot",
                  "exchange", "sharepoint", "active directory", "group policy", "wsus",
                  "new relic", "pagerduty", "confluence", "notion", "asana", "trello",
                  "figma", "sketch", "photoshop", "illustrator", "blender", "maya",
                  "autocad", "solidworks", "matlab", "labview", "jmeter", "postman",
                  "swagger", "storybook", "microsoft suite", "google suite", "google workspace",
                  "office 365", "microsoft 365", "microsoft teams"},
        "frameworks": {"spring", "spring boot", ".net", "asp.net", "django", "rails",
                       "laravel", "express", "nest.js", "next.js", "nuxt", "vue", "angular",
                       "svelte", "flutter", "react native", "xamarin", "ionic",
                       "pytorch", "keras", "opencv", "nltk", "spacy"},
        "platforms": {"aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
                      "kubernetes", "docker compose", "vmware", "hyper-v", "proxmox",
                      "supabase", "firebase", "vercel", "netlify", "github", "gitlab",
                      "bitbucket"},
        "languages": {"kotlin", "swift", "go", "golang", "rust", "scala", "r", "julia",
                      "perl", "ruby", "php", "lua", "elixir", "haskell", "powershell",
                      "bash"},
    }
    category_for: dict[str, str] = {}
    for cat, terms in _CATEGORY_MAP.items():
        for term in terms:
            category_for[term] = cat

    jd_by_cat: dict[str, list[str]] = {}
    for tool in jd_tools:
        cat = category_for.get(tool.lower(), "tools")
        jd_by_cat.setdefault(cat, []).append(tool)

    placed: set[str] = set()
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
        label_match = re.match(r"^(\\textbf\{([^}]*)\})\s*(.+)$", core) if in_skills else None

        if label_match:
            label_full = label_match.group(1)
            label_text = label_match.group(2).lower().rstrip(":")
            items = [i.strip() for i in label_match.group(3).split(",") if i.strip()]

            line_cat = None
            for cat in _CATEGORY_MAP:
                if cat in label_text or label_text in cat:
                    line_cat = cat
                    break

            relevant: list[str] = []
            rest: list[str] = []
            seen_lower: set[str] = set()
            for item in items:
                il = item.lower()
                if il in seen_lower:
                    continue
                seen_lower.add(il)
                if il in all_kw or any(
                    _whole_token_search(il, kw) or _whole_token_search(kw, il)
                    for kw in all_kw
                ):
                    relevant.append(item)
                else:
                    rest.append(item)

            jd_added: list[str] = []
            if line_cat and line_cat in jd_by_cat:
                for tool in jd_by_cat[line_cat]:
                    tl = tool.lower()
                    if tl not in seen_lower:
                        jd_added.append(tool)
                        seen_lower.add(tl)
                    placed.add(tl)

            kept = relevant + jd_added
            if max_items:
                kept += rest[: max(0, max_items - len(kept))]
            else:
                kept += rest

            out_lines.append(f"{label_full} {', '.join(kept)}" + (" \\\\" if has_break else ""))
        else:
            out_lines.append(line)

    remaining = [
        t for cat_tools in jd_by_cat.values() for t in cat_tools
        if t.lower() not in placed
    ]
    if remaining:
        in_skills_fb = False
        last_skills_idx = -1
        for i, ln in enumerate(out_lines):
            if ln.strip().startswith("\\section*"):
                in_skills_fb = "skills" in ln.lower()
            if in_skills_fb and ln.rstrip().endswith("\\\\") and "\\textbf{" in ln:
                last_skills_idx = i
        if last_skills_idx >= 0:
            s = out_lines[last_skills_idx].rstrip()
            out_lines[last_skills_idx] = s[:-2].rstrip() + ", " + ", ".join(remaining) + " \\\\"

    return "\n".join(out_lines)


def _filter_keywordless_bullets(
    ordered_keys: list[tuple[str, int]],
    resolved: dict[tuple[str, int], str],
    kept_indices: dict[str, list[int]],
    keywords: list[str],
    jd_tools: tuple[str, ...],
    tailored_flags: dict[tuple[str, int], bool] | None = None,
    min_visible: int = 0,
) -> list[str]:
    """Drop TAILORED bullets that contain zero JD keyword bolds.

    The "no keywords → scrapped" contract in the strong-aggressive prompt
    applies to the model's fabricated bullets only. Canonical fallbacks
    (validation rejects, audit blanks, missing slots) are exempt — truthful
    filler beats a bare page — and scrapping stops before the page would
    drop below `min_visible` bullets.
    """
    all_kw = {k.lower().strip() for k in keywords if k.strip()}
    all_kw.update(t.lower().strip() for t in jd_tools if t.strip())

    reasons: list[str] = []
    to_remove: list[tuple[str, int]] = []
    for key in ordered_keys:
        if tailored_flags is not None and not tailored_flags.get(key, False):
            continue
        bolds = re.findall(r"\\textbf\{([^}]+)\}", resolved[key])
        has_kw = any(
            b.lower().strip() in all_kw
            or any(kw in b.lower() or b.lower() in kw for kw in all_kw)
            for b in bolds
        )
        if not has_kw:
            to_remove.append(key)

    total = sum(len(v) for v in kept_indices.values())
    for entry_id, index in to_remove:
        if min_visible and total <= min_visible:
            break
        if len(kept_indices.get(entry_id, [])) > 1 and index in kept_indices[entry_id]:
            kept_indices[entry_id].remove(index)
            ordered_keys.remove((entry_id, index))
            if tailored_flags is not None:
                tailored_flags.pop((entry_id, index), None)
            reasons.append(f"{entry_id}[{index}]: no JD keywords bolded")
            total -= 1

    return reasons


def _reconcile_skills_and_bullets(
    tail: str,
    rendered_sections: str,
    jd_tools: tuple[str, ...],
) -> str:
    """Ensure bidirectional consistency between bolded bullet terms and skills.

    Harvests \\textbf spans from BULLET lines only — entry headers also bold
    their titles (job titles, project names), and harvesting those used to
    dump "frontend development intern," into the Tools line. Original casing
    is preserved and trailing punctuation stripped.
    """
    item_text = "\n".join(
        m.group(0)
        for m in re.finditer(
            r"^\s*\\(?:item\b|resumeItem\b).*$", rendered_sections, re.MULTILINE
        )
    )
    original_for: dict[str, str] = {}
    for span in re.findall(r"\\textbf\{([^}]+)\}", item_text):
        cleaned = span.strip().strip(",;:. ")
        if not cleaned or len(cleaned) > 40:
            continue
        original_for.setdefault(cleaned.lower(), cleaned)
    if not original_for:
        return tail

    _CATEGORY_MAP = {
        "tools": {"servicenow", "jira", "zendesk", "salesforce", "splunk", "datadog",
                  "grafana", "prometheus", "nagios", "sccm", "intune", "jamf", "autopilot",
                  "exchange", "sharepoint", "active directory", "group policy", "wsus",
                  "new relic", "pagerduty", "confluence", "notion", "asana", "trello",
                  "figma", "sketch", "photoshop", "illustrator", "blender", "maya",
                  "autocad", "solidworks", "matlab", "labview", "jmeter", "postman",
                  "swagger", "storybook", "microsoft suite", "google suite", "google workspace",
                  "office 365", "microsoft 365", "microsoft teams"},
        "frameworks": {"spring", "spring boot", ".net", "asp.net", "django", "rails",
                       "laravel", "express", "nest.js", "next.js", "nuxt", "vue", "angular",
                       "svelte", "flutter", "react native", "xamarin", "ionic",
                       "pytorch", "keras", "opencv", "nltk", "spacy"},
        "platforms": {"aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
                      "kubernetes", "docker compose", "vmware", "hyper-v", "proxmox",
                      "supabase", "firebase", "vercel", "netlify", "github", "gitlab",
                      "bitbucket"},
        "languages": {"kotlin", "swift", "go", "golang", "rust", "scala", "r", "julia",
                      "perl", "ruby", "php", "lua", "elixir", "haskell", "powershell",
                      "bash"},
    }
    category_for: dict[str, str] = {}
    for cat, terms in _CATEGORY_MAP.items():
        for term in terms:
            category_for[term] = cat

    tail_lower = tail.lower()
    missing: dict[str, list[str]] = {}
    for term, original in original_for.items():
        if term not in tail_lower:
            cat = category_for.get(term, "tools")
            missing.setdefault(cat, []).append(original)

    if not missing:
        return tail

    out_lines: list[str] = []
    in_skills = False
    for line in tail.splitlines():
        if line.strip().startswith("\\section*"):
            in_skills = "skills" in line.lower()
        if in_skills:
            label_match = re.match(
                r"^(\\textbf\{([^}]*)\})\s*(.+)$",
                line.rstrip().rstrip("\\").rstrip(),
            )
            if label_match:
                label_text = label_match.group(2).lower().rstrip(":")
                for cat, terms in list(missing.items()):
                    if cat in label_text or label_text in cat:
                        has_break = line.rstrip().endswith("\\\\")
                        core = line.rstrip()
                        if has_break:
                            core = core[:-2].rstrip()
                        core += ", " + ", ".join(terms)
                        if has_break:
                            core += " \\\\"
                        line = core
                        del missing[cat]
                        break
        out_lines.append(line)

    if missing:
        remaining = [t for terms in missing.values() for t in terms]
        in_skills_fb = False
        last_skills_idx = -1
        for i, ln in enumerate(out_lines):
            if ln.strip().startswith("\\section*"):
                in_skills_fb = "skills" in ln.lower()
            if in_skills_fb and ln.rstrip().endswith("\\\\") and "\\textbf{" in ln:
                last_skills_idx = i
        if last_skills_idx >= 0:
            s = out_lines[last_skills_idx].rstrip()
            out_lines[last_skills_idx] = s[:-2].rstrip() + ", " + ", ".join(remaining) + " \\\\"

    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cross-bullet consistency guards
# ---------------------------------------------------------------------------
# validate_tailored_bullet sees one bullet at a time, so two failure modes pass
# it bullet-by-bullet while the rendered page reads wrong: (a) the model
# relocates a metric into one sibling bullet while another still states it —
# the same number printed twice in one entry; (b) the model leans on the same
# listing phrase ("functional specifications") or long word ("documenting")
# across many bullets, which reads as stuffing. Both observed in live builds.

# Ignore single-digit numbers ("3D", "2 weeks"): too noisy to police.
DUP_METRIC_MIN_DIGITS = 2
NEAR_DUPLICATE_JACCARD = 0.75
# Whole-bullet Jaccard similarity misses "same setup, different payoff" pairs
# (e.g. two bullets both opening "Integrated frontend features with Python
# ... backend services and ... RESTful APIs" but ending on unrelated clauses,
# or "Built production-ready web applications/interfaces using React and
# TypeScript" restating the same fact twice) — a human notices the shared
# opening immediately even though the full-sentence overlap stays well under
# NEAR_DUPLICATE_JACCARD (observed live, repeatedly, well under even the
# pre-loosening 0.6 threshold). Catch it directly by comparing just the
# opening span as a bag of words (order-independent, so word-order drift
# doesn't dodge it) rather than requiring an exact sequential prefix match.
# 0.5 = 4 shared of 6+6 opening words: calibrated against five live cases
# ("Built adaptive/reinforcement algorithms using TensorFlow ..." pairs sit
# at exactly 0.5; verb+object+tool collisions are what a reader notices),
# while genuinely different bullets that merely share topic words measure
# ~0.2 on their openings.
OPENING_SPAN_WORDS = 6
OPENING_OVERLAP_JACCARD = 0.5
# A phrase (2-3 words) absent from a bullet's canonical text may appear in at
# most this many bullets; later occurrences revert to canonical.
MAX_NONCANONICAL_PHRASE_BULLETS = 3
# Same idea for one long content word, stem-matched so inflections count
# together ("documenting" / "documentation").
MAX_NONCANONICAL_STEM_BULLETS = 4
_STEM_LENGTH = 8
_MIN_STEM_WORD_LENGTH = 10
_MIN_PHRASE_WORD_LENGTH = 4


def _bullet_plain_words(text: str) -> list[str]:
    without_commands = LATEX_COMMAND_PATTERN.sub(" ", text)
    return [w for w in re.split(r"[^a-z0-9+#.-]+", without_commands.lower()) if w]


def _significant_numbers(text: str) -> set[str]:
    return {
        number
        for number in NUMBER_PATTERN.findall(text)
        if len(number.replace(".", "")) >= DUP_METRIC_MIN_DIGITS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def enforce_cross_bullet_consistency(
    ordered_keys: list[tuple[str, int]],
    canonicals: dict[tuple[str, int], str],
    resolved: dict[tuple[str, int], str],
    tailored: dict[tuple[str, int], bool],
    *,
    near_duplicate_jaccard: float = NEAR_DUPLICATE_JACCARD,
    opening_overlap_jaccard: float = OPENING_OVERLAP_JACCARD,
    max_phrase_bullets: int = MAX_NONCANONICAL_PHRASE_BULLETS,
    max_stem_bullets: int = MAX_NONCANONICAL_STEM_BULLETS,
) -> list[str]:
    """Revert tailored bullets that only read wrong next to their neighbours.

    `ordered_keys` is (entry_id, bullet_index) in final document order.
    Mutates `resolved`/`tailored` (revert-to-canonical only, so every change
    is safe) and returns fallback reasons in the render report's format.
    Duplications already present between canonical bullets are the profile's
    own truth and stay exempt. The four keyword thresholds are the repetition/
    variety guards only (never the metric-duplication check below, which stays
    fixed) — aggressive-mode callers pass loosened values.
    """
    fallback_reasons: list[str] = []

    def _revert(key: tuple[str, int], reason: str) -> None:
        resolved[key] = canonicals[key]
        tailored[key] = False
        fallback_reasons.append(f"{key[0]}[{key[1]}]: {reason}")

    changed = True
    while changed:
        changed = False

        # (a) Within an entry: duplicated metric or near-duplicate prose.
        for position, key_a in enumerate(ordered_keys):
            for key_b in ordered_keys[position + 1:]:
                if key_b[0] != key_a[0]:
                    continue
                if not (tailored[key_a] or tailored[key_b]):
                    continue
                victim = key_b if tailored[key_b] else key_a
                duplicated = (
                    _significant_numbers(resolved[key_a]) & _significant_numbers(resolved[key_b])
                ) - (
                    _significant_numbers(canonicals[key_a]) & _significant_numbers(canonicals[key_b])
                )
                if duplicated:
                    _revert(victim, f"metric {sorted(duplicated)[0]} stated twice in the same entry")
                    changed = True
                    continue
                canonical_overlap = _jaccard(
                    set(_bullet_plain_words(canonicals[key_a])),
                    set(_bullet_plain_words(canonicals[key_b])),
                )
                rendered_overlap = _jaccard(
                    set(_bullet_plain_words(resolved[key_a])),
                    set(_bullet_plain_words(resolved[key_b])),
                )
                if rendered_overlap >= near_duplicate_jaccard > canonical_overlap:
                    _revert(victim, "near-duplicate of a sibling bullet")
                    changed = True
                    continue
                opening_overlap = _jaccard(
                    set(_bullet_plain_words(resolved[key_a])[:OPENING_SPAN_WORDS]),
                    set(_bullet_plain_words(resolved[key_b])[:OPENING_SPAN_WORDS]),
                )
                canonical_opening_overlap = _jaccard(
                    set(_bullet_plain_words(canonicals[key_a])[:OPENING_SPAN_WORDS]),
                    set(_bullet_plain_words(canonicals[key_b])[:OPENING_SPAN_WORDS]),
                )
                if (
                    opening_overlap >= opening_overlap_jaccard
                    and opening_overlap > canonical_opening_overlap
                ):
                    _revert(victim, "shares its opening clause with a sibling bullet")
                    changed = True

        # (b) Across entries: the same non-canonical phrase or long word leaned
        # on in too many bullets.
        phrase_owners: dict[str, list[tuple[str, int]]] = {}
        stem_owners: dict[str, list[tuple[str, int]]] = {}
        for key in ordered_keys:
            if not tailored[key]:
                continue
            words = _bullet_plain_words(resolved[key])
            canonical_words = _bullet_plain_words(canonicals[key])
            canonical_text = " ".join(canonical_words)
            seen_phrases: set[str] = set()
            for size in (2, 3):
                for start in range(len(words) - size + 1):
                    gram_words = words[start:start + size]
                    if any(len(w) < _MIN_PHRASE_WORD_LENGTH or w[0].isdigit() for w in gram_words):
                        continue
                    gram = " ".join(gram_words)
                    if gram in seen_phrases or gram in canonical_text:
                        continue
                    seen_phrases.add(gram)
                    phrase_owners.setdefault(gram, []).append(key)
            canonical_stems = {
                w[:_STEM_LENGTH] for w in canonical_words if len(w) >= _STEM_LENGTH
            }
            seen_stems: set[str] = set()
            for word in words:
                if len(word) < _MIN_STEM_WORD_LENGTH:
                    continue
                stem = word[:_STEM_LENGTH]
                if stem in seen_stems or stem in canonical_stems:
                    continue
                seen_stems.add(stem)
                stem_owners.setdefault(stem, []).append(key)
        for gram, owners in phrase_owners.items():
            for key in owners[max_phrase_bullets:]:
                if tailored[key]:
                    _revert(key, f"phrase '{gram}' repeated across {len(owners)} bullets")
                    changed = True
        for stem, owners in stem_owners.items():
            for key in owners[max_stem_bullets:]:
                if tailored[key]:
                    _revert(key, f"word '{stem}...' repeated across {len(owners)} bullets")
                    changed = True

    return fallback_reasons


def _rank_index(entry_id: str, ranking: list[str], template_order: list[str]) -> tuple[int, int]:
    try:
        return (0, ranking.index(entry_id))
    except ValueError:
        return (1, template_order.index(entry_id))


def lint_render_fidelity(
    catalog: TemplateCatalog,
    document: str,
    included: list[TemplateEntry],
    flexible_headers: dict[str, str] | None = None,
) -> list[str]:
    """Verify the rendered document follows the template's structure exactly.

    The renderer builds from verbatim template parts, so any finding here is a
    bug, not a style issue. Checked: preamble and name/contact block byte-for-
    byte, every included entry's header verbatim (or byte-for-byte equal to
    its validated header_tech substitution), balanced list environments, and
    exactly one document environment.
    """
    findings: list[str] = []
    if catalog.preamble not in document:
        findings.append("preamble does not match template verbatim")
    if catalog.header_block and catalog.header_block not in document:
        findings.append("name/contact block does not match template verbatim")
    for entry in included:
        expected = (flexible_headers or {}).get(entry.entry_id, entry.header)
        if expected not in document:
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
    max_items_per_line: int = 0,
) -> str:
    """Reorder items within Skills-section lines so listing-relevant tools lead.

    Format-preserving: same labels, same line breaks — only the order of
    comma-separated items within each `\\textbf{Label:} a, b, c \\\\` line
    changes. Items matching a listing keyword move to the front, then
    everything else in canonical order. Nothing is ever added, so there is no
    invention risk. When `max_items_per_line` > 0, unmatched trailing items
    beyond that cap are cut (matched items always survive, even past the cap).
    """
    if not keywords:
        return tail

    keyword_lowered = [k.lower().strip() for k in keywords if k.strip()]
    keyword_words = [set(re.findall(r"[a-z0-9.+#]+", k)) for k in keyword_lowered]

    # Whole-token matching everywhere: substring matching would let the item
    # "C" float to the front for any keyword containing the letter c
    # ("react", "recruitment").
    def _contains_whole(haystack: str, needle: str) -> bool:
        return bool(
            re.search(rf"(?<![a-z0-9.+#]){re.escape(needle)}(?![a-z0-9.+#])", haystack)
        )

    def item_rank(item: str) -> tuple[int, int]:
        item_lowered = item.lower()
        for keyword in keyword_lowered:
            if (
                keyword == item_lowered
                or _contains_whole(item_lowered, keyword)
                or _contains_whole(keyword, item_lowered)
            ):
                return (0, 0)
        item_words = set(re.findall(r"[a-z0-9.+#]+", item_lowered))
        for words in keyword_words:
            if item_words & words:
                return (1, 0)
        return (2, 0)

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
            if max_items_per_line > 0 and len(items) > max_items_per_line:
                matched = sum(1 for item in items if item_rank(item)[0] < 2)
                items = items[: max(max_items_per_line, matched)]
            rebuilt = f"{label_match.group(1)} {', '.join(items)}"
            out_lines.append(rebuilt + (" \\\\" if has_break else ""))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def render_structured_resume(
    catalog: TemplateCatalog,
    selection: StructuredSelection,
    jd_inject_tools: tuple[str, ...] = (),
    jd_text: str = "",
) -> tuple[str, RenderReport]:
    """Deterministically render the resume from template parts + selection.

    Every entry is eligible; the model's ranking decides what fills the page.
    No entry is unconditionally forced in or barred by category — the only
    guarantee is capacity-based: exclusions are honoured while the remaining
    entries can still meet the bullet minimum, and rolled back (in ranked
    order) when they cannot.
    """
    config = _effective_render_config(catalog.render_config)
    template_order = [entry.entry_id for entry in catalog.entries]
    known_ids = set(template_order)

    ranked_entries = sorted(
        catalog.entries,
        key=lambda e: _rank_index(e.entry_id, selection.ranking, template_order),
    )
    excluded_ids = {eid for eid in selection.exclusions if eid in known_ids}
    ignored_exclusions: list[str] = []

    # Bullet budget: entries by relevance until the minimum is reached (or all
    # are placed while still under the maximum).
    kept_indices: dict[str, list[int]] = {}
    included: list[TemplateEntry] = []
    total = 0

    def _try_add(entry: TemplateEntry) -> bool:
        nonlocal total
        if total >= config.min_visible_bullets and total + len(entry.bullets) > config.max_visible_bullets:
            return False
        included.append(entry)
        kept_indices[entry.entry_id] = list(range(len(entry.bullets)))
        total += len(entry.bullets)
        return True

    for entry in ranked_entries:
        if entry.entry_id in excluded_ids:
            continue
        _try_add(entry)

    # Capacity rollback: if honouring the exclusions leaves the page
    # under-full, re-admit excluded entries in relevance order until the
    # minimum is met.
    if total < config.min_visible_bullets:
        for entry in ranked_entries:
            if entry.entry_id not in excluded_ids:
                continue
            if _try_add(entry):
                excluded_ids.discard(entry.entry_id)
                ignored_exclusions.append(entry.entry_id)
            if total >= config.min_visible_bullets:
                break

    excluded_entries = sorted(excluded_ids)

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

    # Over budget: trim from the least relevant entries first, never below one
    # bullet per included entry. Within an entry, trim the
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
        if config.strong_aggressive:
            # Fabricated bullets run near max length AND entry headers eat
            # real page height the char model otherwise ignores — count each
            # included entry as roughly one bullet-line of overhead so a
            # many-entry page trims down instead of spilling to page 2
            # (observed live: 12 fabricated bullets overflowed while a
            # 14-bullet page with fewer entries fit).
            chars += 110 * sum(1 for e in included if kept_indices[e.entry_id])
        return chars

    def _trim_order() -> list[TemplateEntry]:
        return list(reversed(included))

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
    # Aggressive builds get more headroom here (live-trial evidence,
    # 2026-07-12: not stripping filler clauses and loosening the cross-bullet
    # repetition reverts both mean fewer bullets fall back to short canonical
    # text, so aggressive bullets run longer on average — a real Operations
    # Development listing still overflowed to 2 pages at the normal floor of
    # min_visible_bullets-2 even after max_bullet_chars stopped scaling).
    char_trim_floor = max(config.min_visible_bullets - (6 if config.strong_aggressive else 5 if config.aggressive else 2), 1)
    while total > config.max_visible_bullets or (
        total > char_trim_floor
        and _estimated_chars() > config.max_total_bullet_chars
    ):
        trimmed = False
        for entry in _trim_order():
            if len(kept_indices[entry.entry_id]) > 1:
                _trim_one(entry)
                trimmed = True
                break
        if not trimmed:
            break

    # Order entries inside each section by relevance to this listing.
    def _entry_sort_key(entry: TemplateEntry) -> tuple[int, int]:
        return _rank_index(entry.entry_id, selection.ranking, template_order)

    fallbacks: list[str] = []
    tailored_used = 0
    rendered_bullet_count = 0
    header_tech_applied: list[str] = []
    header_tech_rejected: list[str] = []
    flexible_headers: dict[str, str] = {}
    listing_keywords = tuple(selection.keywords)

    def _entry_grounding(entry: TemplateEntry) -> str:
        background = " ".join(
            catalog.baseinfo_blocks.get(tag, "") for tag in entry.categories
        )
        return f"{' '.join(entry.bullets)} {entry.header} {background}".strip()

    # Header tech lists: the model may re-aim each entry's \textit{...} tool
    # list at the listing, grounded strictly in that entry's own facts.
    for entry in included:
        tools = selection.header_tech.get(entry.entry_id)
        if not tools:
            continue
        grounding = _entry_grounding(entry)
        if config.strong_aggressive:
            grounding += " " + " ".join(str(t) for t in tools)
        new_header, reason = retarget_header_tech(
            entry.header, tools, grounding,
        )
        if new_header is not None and new_header != entry.header:
            flexible_headers[entry.entry_id] = new_header
            header_tech_applied.append(entry.entry_id)
        elif reason is not None:
            header_tech_rejected.append(f"{entry.entry_id}: {reason}")

    # Resolve every kept bullet's final text first (in document order), so the
    # cross-bullet guards can see the whole page before anything renders.
    ordered_entries: list[TemplateEntry] = []
    for section_name in catalog.entry_sections:
        ordered_entries.extend(
            sorted((e for e in included if e.section == section_name), key=_entry_sort_key)
        )

    ordered_keys: list[tuple[str, int]] = []
    canonical_texts: dict[tuple[str, int], str] = {}
    resolved: dict[tuple[str, int], str] = {}
    tailored_flags: dict[tuple[str, int], bool] = {}
    for entry in ordered_entries:
        provided = effective_bullets.get(entry.entry_id, [])
        if config.strong_aggressive and provided:
            # Depth over breadth: in strong-aggressive a SHORTER bullets list
            # is an intentional "fewer, heavier bullets" choice — drop the
            # unwritten trailing slots instead of padding them with canonical
            # text (canonical filler is off-message on a fabricated page).
            trimmed_kept = [
                i for i in kept_indices[entry.entry_id] if i < len(provided)
            ]
            if trimmed_kept:
                kept_indices[entry.entry_id] = trimmed_kept
        for index in kept_indices[entry.entry_id]:
            key = (entry.entry_id, index)
            canonical = entry.bullets[index]
            text = canonical
            tailored = False
            rewrite_allowed = (
                config.rewrite_scope != "limited"
                or rendered_bullet_count < config.limited_rewrite_bullets
            )
            rendered_bullet_count += 1
            # Empty slots mean "no rewrite for this index" (e.g. a bullet the
            # grounding audit stripped) — render canonical without a fallback
            # reason; the audit reports its own reasons.
            if rewrite_allowed and index < len(provided) and str(provided[index]).strip():
                sanitised, reason = validate_tailored_bullet(
                    provided[index],
                    canonical,
                    config,
                    catalog.skill_anchors,
                    entry_context=_entry_grounding(entry),
                    listing_keywords=listing_keywords,
                )
                if sanitised is not None:
                    text = sanitised
                    tailored = True
                else:
                    fallbacks.append(f"{entry.entry_id}[{index}]: {reason}")
            ordered_keys.append(key)
            canonical_texts[key] = canonical
            resolved[key] = text
            tailored_flags[key] = tailored

    if config.strong_aggressive:
        fallbacks += _filter_keywordless_bullets(
            ordered_keys, resolved, kept_indices, selection.keywords, jd_inject_tools,
            tailored_flags=tailored_flags,
            min_visible=config.min_visible_bullets,
        )
        total = sum(len(v) for v in kept_indices.values() if v)

    consistency_kwargs = (
        {
            "near_duplicate_jaccard": 0.95,
            "opening_overlap_jaccard": 0.85,
            "max_phrase_bullets": 10,
            "max_stem_bullets": 12,
        }
        if config.strong_aggressive
        else {
            "near_duplicate_jaccard": 0.92,
            "opening_overlap_jaccard": 0.75,
            "max_phrase_bullets": 6,
            "max_stem_bullets": 8,
        }
        if config.aggressive
        else {}
    )
    fallbacks += enforce_cross_bullet_consistency(
        ordered_keys, canonical_texts, resolved, tailored_flags, **consistency_kwargs
    )
    tailored_used = sum(1 for used in tailored_flags.values() if used)

    def _render_entry(entry: TemplateEntry) -> str:
        lines: list[str] = []
        if entry.smallskip:
            lines.append("\\smallskip")
        lines.append(f"% [{', '.join(entry.categories)}]")
        lines.append(flexible_headers.get(entry.entry_id, entry.header))
        lines.append(entry.list_begin)
        for index in kept_indices[entry.entry_id]:
            text = resolved[(entry.entry_id, index)]
            # Strong-aggressive reserves bold for JD-relevant terms: demote
            # everything else BEFORE the deterministic bolding passes so the
            # final emphasis is exactly the listing's vocabulary.
            if config.strong_aggressive:
                text = _unbold_non_jd_terms(
                    text, jd_inject_tools, selection.keywords, jd_text
                )
            text = emphasize_listing_tools(
                text,
                catalog.skill_anchors,
                selection.keywords,
                config.max_bold_per_bullet,
            )
            bold_terms = jd_inject_tools
            if config.strong_aggressive and selection.keywords:
                # Keywords are the model's verbatim JD hard skills — bolding
                # them too keeps every fabricated bullet visually anchored to
                # the listing (round-2 audit: 15 bullets rendered with zero
                # bolds because only extracted TOOLS were auto-bolded).
                bold_terms = tuple(
                    dict.fromkeys((*jd_inject_tools, *selection.keywords))
                )
            if bold_terms:
                text = _bold_jd_tools(text, bold_terms, config.max_bold_per_bullet)
            if entry.item_style == "braced":
                lines.append(f"  {entry.item_command}{{{text}}}")
            else:
                lines.append(f"  {entry.item_command} {text}")
        lines.append(entry.list_end)
        return "\n".join(lines)

    section_chunks: list[str] = []
    for section_name in catalog.entry_sections:
        section_entries = [e for e in ordered_entries if e.section == section_name]
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
        if config.strong_aggressive:
            tail = _rewrite_skills_for_jd(
                catalog.tail, jd_inject_tools, selection.keywords, catalog.skill_anchors,
                max_items=config.max_skill_items_per_line,
            )
            all_rendered = "\n\n".join(section_chunks)
            tail = _reconcile_skills_and_bullets(tail, all_rendered, jd_inject_tools)
        else:
            tail = reorder_skills_for_keywords(
                catalog.tail,
                selection.keywords,
                config.max_skill_items_per_line,
            )
            if jd_inject_tools:
                tail = _inject_jd_tools_into_skills(tail, jd_inject_tools, catalog.skill_anchors)
        parts += ["", tail]
    parts += ["", "\\end{document}", ""]
    document = "\n".join(parts)

    visible_ids = {entry.entry_id for entry in included}
    report = RenderReport(
        visible_entries=[entry.entry_id for entry in included],
        hidden_entries=[
            entry.entry_id for entry in catalog.entries if entry.entry_id not in visible_ids
        ],
        visible_bullet_count=total,
        tailored_bullets_used=tailored_used,
        canonical_fallbacks=fallbacks,
        excluded_entries=excluded_entries,
        ignored_exclusions=ignored_exclusions,
        fidelity_findings=lint_render_fidelity(
            catalog, document, included, flexible_headers=flexible_headers
        ),
        rewrite_scope=config.rewrite_scope,
        header_tech_applied=header_tech_applied,
        header_tech_rejected=header_tech_rejected,
    )
    return document, report


def _strip_comment_blocks(text: str) -> str:
    """Remove \\begin{comment}...\\end{comment} blocks from template text."""
    return re.sub(
        r"\\begin\{comment\}.*?\\end\{comment\}",
        "",
        text,
        flags=re.DOTALL,
    )


# Matches an entry-shaped block: a line containing \textbf{...} (the header)
# followed (within a few lines) by a bullet list.  Used to detect entries that
# the tag-aware parser missed because they lack a % [tag] line.
_ENTRY_HEADER_PATTERN = re.compile(
    r"^([ \t]*\\textbf\{[^}]+\}.*)$",
    re.MULTILINE,
)


def _infer_tag(title: str, existing_tags: set[str]) -> str:
    """Derive a short category slug from an entry title, avoiding collisions."""
    # Drop trailing punctuation and common suffixes
    cleaned = re.sub(r"\s*[-–—,|].*", "", title).strip()
    words = re.findall(r"[a-z]+", cleaned.lower())
    # Skip generic leading words
    skip = {"senior", "junior", "lead", "intern", "co", "op", "coop", "student"}
    candidate_words = [w for w in words if w not in skip]
    slug = candidate_words[0] if candidate_words else (words[0] if words else "entry")
    if slug not in existing_tags:
        return slug
    # Try two-word slug
    if len(candidate_words) >= 2:
        two_word = f"{candidate_words[0]}_{candidate_words[1]}"
        if two_word not in existing_tags:
            return two_word
    # Append numeric suffix
    n = 2
    while f"{slug}{n}" not in existing_tags:
        return f"{slug}{n}"
    while True:
        n += 1
        candidate = f"{slug}{n}"
        if candidate not in existing_tags:
            return candidate


def _extract_bold_tools(bullets_text: str) -> list[str]:
    """Extract \\textbf{...} tool names from bullet text."""
    tools: list[str] = []
    for match in re.finditer(r"\\textbf\{([^}]+)\}", bullets_text):
        tool = match.group(1).strip().rstrip(",")
        if tool and len(tool) < 40:
            tools.append(tool)
    return tools


def _infer_baseinfo_section(section_name: str) -> str:
    """Map a template \\section name to a baseinfo == SECTION == header."""
    name_lower = section_name.lower().strip()
    if "experience" in name_lower or "work" in name_lower:
        return "EXPERIENCE"
    if "project" in name_lower:
        return "PROJECTS"
    if "volunteer" in name_lower:
        return "VOLUNTEER"
    if "education" in name_lower:
        return "EDUCATION"
    return "EXPERIENCE"


def _detect_untagged_entries(
    template_text: str,
    parsed_entry_titles: set[str],
) -> list[tuple[str, str, str]]:
    """Find entry-shaped blocks in the template that the parser missed.

    Returns [(header_line, title, section_name), ...] for each untagged entry
    that is not inside a comment block and was not already parsed.
    """
    uncommented = _strip_comment_blocks(template_text)
    begin_doc = uncommented.find("\\begin{document}")
    end_doc = uncommented.rfind("\\end{document}")
    if begin_doc == -1 or end_doc == -1:
        return []
    body = uncommented[begin_doc:end_doc]

    sections = list(SECTION_PATTERN.finditer(body))
    if not sections:
        return []

    # Build a map of line positions to section names
    section_ranges: list[tuple[int, int, str]] = []
    for i, sec_match in enumerate(sections):
        sec_end = sections[i + 1].start() if i + 1 < len(sections) else len(body)
        section_ranges.append((sec_match.start(), sec_end, sec_match.group(1).strip()))

    untagged: list[tuple[str, str, str]] = []
    # Normalise parsed titles for fuzzy matching
    parsed_normalised = {re.sub(r"[^a-z0-9]+", "", t.lower()) for t in parsed_entry_titles}

    for header_match in _ENTRY_HEADER_PATTERN.finditer(body):
        header_line = header_match.group(1).strip()
        # Check if this header is followed by a bullet list within 5 lines
        after = body[header_match.end():header_match.end() + 500]
        has_bullets = False
        for list_begin_pat, _, _, _ in LIST_CONVENTIONS:
            if re.search(list_begin_pat, after[:300]):
                has_bullets = True
                break
        if not has_bullets:
            continue

        title = _extract_entry_title(header_line)
        title_normalised = re.sub(r"[^a-z0-9]+", "", title.lower())
        if title_normalised in parsed_normalised:
            continue

        # Check the line before this header for a tag
        preceding = body[:header_match.start()].rstrip()
        last_line = preceding.rsplit("\n", 1)[-1].strip() if preceding else ""
        if CATEGORY_TAG_PATTERN.match(last_line):
            continue

        # Determine which section this belongs to
        section_name = "Experience"
        for sec_start, sec_end, sec_name in section_ranges:
            if sec_start <= header_match.start() < sec_end:
                section_name = sec_name
                break

        untagged.append((header_line, title, section_name))

    return untagged


def sync_template_baseinfo(
    template_path: Path,
    baseinfo_path: Path,
    template_text: str,
    baseinfo_text: str,
    parsed_entry_titles: set[str],
) -> tuple[str, str, list[str]]:
    """Detect and fix untagged template entries and missing baseinfo blocks.

    Writes fixes to disk and returns (new_template_text, new_baseinfo_text,
    log_messages).  Returns the original texts unchanged when nothing needs
    fixing.
    """
    messages: list[str] = []
    untagged = _detect_untagged_entries(template_text, parsed_entry_titles)
    if not untagged:
        return template_text, baseinfo_text, messages

    existing_tags: set[str] = set()
    for m in CATEGORY_TAG_PATTERN.finditer(template_text):
        for part in m.group(1).split(","):
            existing_tags.add(part.strip())

    baseinfo_blocks_existing = set(parse_baseinfo_blocks(baseinfo_text)[0].keys())
    new_template = template_text
    new_baseinfo = baseinfo_text

    for header_line, title, section_name in untagged:
        tag = _infer_tag(title, existing_tags)
        existing_tags.add(tag)

        # Insert % [tag] above the header line in template
        new_template = new_template.replace(
            header_line,
            f"% [{tag}]\n{header_line}",
            1,
        )
        messages.append(f"Auto-tagged entry {title!r} as [{tag}]")

        # Add baseinfo block if missing
        if tag not in baseinfo_blocks_existing:
            baseinfo_section = _infer_baseinfo_section(section_name)
            # Find the section in baseinfo and append the block there
            section_pattern = re.compile(
                rf"^==\s*{re.escape(baseinfo_section)}\s*==\s*$",
                re.MULTILINE | re.IGNORECASE,
            )
            section_match = section_pattern.search(new_baseinfo)
            if section_match:
                # Find the next == section or end of text to insert before it
                next_section = re.search(
                    r"^==\s",
                    new_baseinfo[section_match.end():],
                    re.MULTILINE,
                )
                if next_section:
                    insert_pos = section_match.end() + next_section.start()
                    new_baseinfo = (
                        new_baseinfo[:insert_pos].rstrip()
                        + f"\n\n[{tag}] {title}\nNotes:\n\n"
                        + new_baseinfo[insert_pos:]
                    )
                else:
                    # Before the next top-level section (like SKILL ANCHORS)
                    remaining = new_baseinfo[section_match.end():]
                    next_top = re.search(r"^==\s", remaining, re.MULTILINE)
                    if next_top:
                        insert_pos = section_match.end() + next_top.start()
                        new_baseinfo = (
                            new_baseinfo[:insert_pos].rstrip()
                            + f"\n\n[{tag}] {title}\nNotes:\n\n"
                            + new_baseinfo[insert_pos:]
                        )
                    else:
                        new_baseinfo = (
                            new_baseinfo.rstrip()
                            + f"\n\n[{tag}] {title}\nNotes:\n"
                        )
            else:
                new_baseinfo = (
                    new_baseinfo.rstrip()
                    + f"\n\n== {baseinfo_section} ==\n\n[{tag}] {title}\nNotes:\n"
                )
            baseinfo_blocks_existing.add(tag)
            messages.append(f"Added baseinfo stub for [{tag}] {title}")

        # Extract bold tools from the entry's bullets and add to SKILL ANCHORS
        # Find the entry's bullets in the uncommented template
        uncommented = _strip_comment_blocks(template_text)
        entry_start = uncommented.find(header_line)
        if entry_start != -1:
            entry_chunk = uncommented[entry_start:entry_start + 3000]
            tools = _extract_bold_tools(entry_chunk)
            if tools:
                existing_anchors = set(parse_skill_anchors(new_baseinfo))
                new_tools = [
                    t for t in tools
                    if t.lower() not in existing_anchors
                    and len(t) > 1
                ]
                if new_tools:
                    anchor_match = re.search(
                        r"==\s*SKILL ANCHORS\s*==",
                        new_baseinfo,
                        re.IGNORECASE,
                    )
                    if anchor_match:
                        tools_line = f"From {title.split(',')[0].strip()}: {', '.join(new_tools)}\n"
                        new_baseinfo = new_baseinfo.rstrip() + "\n" + tools_line
                        messages.append(
                            f"Added skill anchors from {title!r}: {', '.join(new_tools)}"
                        )

    if new_template != template_text:
        template_path.write_text(new_template, encoding="utf-8")
    if new_baseinfo != baseinfo_text:
        baseinfo_path.write_text(new_baseinfo, encoding="utf-8")

    return new_template, new_baseinfo, messages


def load_structured_profile(
    template_path: Path,
    baseinfo_path: Path,
) -> TemplateCatalog | None:
    """Load the catalog for a profile, or None if unsupported.

    A profile is structured when its template parses under the `% [category]`
    entry convention — nothing else gates it. Also picks up two optional
    per-profile files next to template.tex:
    - structured_config.json: rendering knobs (bullet budget, rewrite scope)
    - instructions.txt: free-text prompt guidance (voice, seniority). Legacy
      (non-structured) profiles also read this file, but for a full step-by-step
      LaTeX-authoring prompt instead — the two uses are mutually exclusive per
      profile since a profile is either structured or legacy, never both.
    """
    try:
        template_text = template_path.read_text(encoding="utf-8")
        baseinfo_text = baseinfo_path.read_text(encoding="utf-8")
    except OSError:
        return None
    catalog = parse_template_catalog(template_text)
    if catalog is None:
        return None

    # Auto-sync: detect entries the parser missed (no % [tag]) and fix on disk.
    parsed_titles = {entry.title for entry in catalog.entries}
    template_text, baseinfo_text, sync_messages = sync_template_baseinfo(
        template_path, baseinfo_path, template_text, baseinfo_text, parsed_titles,
    )
    if sync_messages:
        for msg in sync_messages:
            print(f"[profile-sync] {msg}")
        catalog = parse_template_catalog(template_text)
        if catalog is None:
            return None

    catalog.skill_anchors = parse_skill_anchors(baseinfo_text)
    catalog.skill_anchor_display = parse_skill_anchor_display(baseinfo_text)
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

    return catalog
