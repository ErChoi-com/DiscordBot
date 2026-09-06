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

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

# Bullet-count window required by the profile instructions ("fill the page").
# The ceiling is 16 rather than the instructions' historical 18: sample compiles
# showed 17 bullets across 6 entries overflow to a second page. Profiles can
# override these via structured_config.json (see RenderConfig).
MIN_VISIBLE_BULLETS = 14
MAX_VISIBLE_BULLETS = 16
MAX_BULLET_CHARS = 400
# Strong-aggressive may return fewer bullets than an entry has slots, but only
# as a trade for heavier ones. The trade is measured in TOTAL characters, not
# mean bullet length: a mean-length check clears at 1.15x while three bullets
# replace five slots and the entry loses a third of its text. The written set
# must carry at least this multiple of the text in the canonical slots it
# stands in for; anything thinner keeps the trailing slots so the page fills.
STRONG_DEPTH_LENGTH_RATIO = 1.0
# Entry headers a page carries before strong-aggressive starts charging them
# against the character budget (see _estimated_chars).
STRONG_FREE_ENTRY_HEADERS = 6
# Page-fit model for strong-aggressive: visible characters that fit on one
# rendered bullet line, and the number of bullet lines a single page holds.
# Calibrated 2026-08-26 against 28 compiled builds (see _estimated_bullet_lines).
BULLET_LINE_CHARS = 95
MAX_BULLET_LINES = 31
# LaTeX markup that takes no width on the page.
_LATEX_MARKUP_PATTERN = re.compile(r"\\[a-zA-Z]+\*?|[{}]")
# Appended "…, demonstrating <posting phrase>" clauses are keyword padding in a
# trench coat. If the canonical bullet did not use the construction, the tail
# clause is stripped and the rest of the rewrite kept (rejecting the whole
# bullet threw away otherwise-good tailoring). The second group covers the
# empty purpose-clause tails HR review flagged ("…, ensuring clean, performant
# rendering", "…, driving successful customer relations"): gerunds that assert
# an outcome without adding a fact.
# The connector set includes bare "to": review of real builds found the same
# empty tail arriving as a purpose clause ("… to support office service
# delivery") as often as a gerund one, and the verb alternation therefore
# covers the infinitive as well as the -ing form.
FILLER_CLAUSE_PATTERN = re.compile(
    r"(?:,|\band|\bwhile|\bto)\s+("
    r"demonstrat(?:ing|e)|showcas(?:ing|e)|highlight(?:ing)?|underscor(?:ing|e)"
    r"|evidenc(?:ing|e)|exemplif(?:ying|y)|ensur(?:ing|e)|enabl(?:ing|e)"
    r"|support(?:ing)?|driv(?:ing|e)|prioriti[sz](?:ing|e)|facilitat(?:ing|e)"
    r"|streamlin(?:ing|e)|empower(?:ing)?|exercis(?:ing|e)|align(?:ing)?"
    r"|uphold(?:ing)?|leverag(?:ing|e)|reflect(?:ing)?|embod(?:ying|y)"
    r"|provid(?:ing|e)|contribut(?:ing|e)|foster(?:ing)?|promot(?:ing|e)"
    # "serve" is deliberately absent: "to serve model inference" / "serving
    # 10k users" names real function far more often than empty purpose.
    r"|help(?:ing)?|assist(?:ing)?|allow(?:ing)?"
    r")\b",
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
# Carries the same identifier guard as STANDALONE_NUMBER_PATTERN: without it
# the menu offered "STM32" and "HCS12" to the model as metrics it could
# reproduce, which is prompt budget spent on part numbers.
MENU_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
)
# Unit suffixes that keep a number a metric ("50ms", "75th", "10k", "500+").
# Any OTHER letter or hyphenated word glued to the digits makes it a compound
# name, not a figure: "3D terrain", "7-segment display", "4-month term".
MENU_UNIT_SUFFIX_PATTERN = re.compile(
    r"(?:ms|s|m|h|k|b|x|%|\+|st|nd|rd|th|gb|mb|kb|tb|hz|khz|mhz|ghz|fps|px|pt|"
    r"min|hr|hrs|sec|secs)(?![A-Za-z0-9])",
    re.I,
)
# Math spans are structural notation ("$4 \times 16$ decoder"), never metrics.
MATH_SPAN_PATTERN = re.compile(r"(?<!\\)\$[^$]*(?<!\\)\$")
# Where a metric menu snippet should start: the clause the number lives in.
# Punctuation is the strongest break, but a metric clause is just as often
# joined by a conjunction with no comma ("… on the board and trimmed memory
# by 12%"), and cutting mid-phrase there reads as a fragment.
MENU_CLAUSE_BREAK_PATTERN = re.compile(
    r"[,;:]\s|(?<![A-Za-z])(?:and|or|but|then|while|which|that)\s", re.I
)
# A clause cut can leave a dangling conjunction ("and trimmed memory by 12%").
MENU_LEADING_CONJUNCTION_PATTERN = re.compile(r"^(?:and|or|but|then|which|that)\s+", re.I)
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
# Openers that describe proximity to work rather than doing it. Recruiter
# guidance is consistent that these are the weakest way to start a bullet, and
# OWNERSHIP_VERB_SWAPS only guards the opposite failure (a rewrite claiming
# MORE than the grounding supports). Observed live: canonical "Maintained clear
# LaTeX documentation…" came back as "Supported operational management by
# maintaining clear LaTeX documentation…" — strictly weaker for the same facts.
WEAK_OPENING_VERBS = frozenset(
    {
        "assisted", "aided", "helped", "worked", "participated", "supported",
        "used", "utilized", "utilised", "learned", "gained", "observed",
        "shadowed", "attended", "engaged",
    }
)

# Present/imperative resume openers and their past forms. The tense guard used
# to work only by stem-matching the canonical opener, which silently missed a
# bullet whenever the model returned its bullets in a different order — index
# n's rewrite is then compared against a different canonical bullet. Observed
# live: "Integrate frontend web services…" and "Build production-grade web
# applications…" both reached a rendered page. Irregular forms are listed
# explicitly because naive suffixing produces "Builded".
PRESENT_TO_PAST_OPENERS = {
    "build": "Built", "write": "Wrote", "lead": "Led", "run": "Ran",
    "make": "Made", "drive": "Drove", "teach": "Taught", "hold": "Held",
    "bring": "Brought", "grow": "Grew", "meet": "Met", "send": "Sent",
    "integrate": "Integrated", "maintain": "Maintained", "track": "Tracked",
    "validate": "Validated", "design": "Designed", "develop": "Developed",
    "deploy": "Deployed", "create": "Created", "manage": "Managed",
    "test": "Tested", "analyze": "Analyzed", "analyse": "Analysed",
    "implement": "Implemented", "automate": "Automated", "support": "Supported",
    "coordinate": "Coordinated", "collaborate": "Collaborated",
    "document": "Documented", "monitor": "Monitored", "configure": "Configured",
    "optimize": "Optimized", "optimise": "Optimised", "engineer": "Engineered",
    "prototype": "Prototyped", "present": "Presented", "review": "Reviewed",
    "debug": "Debugged", "refactor": "Refactored", "migrate": "Migrated",
}

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
    # Floor on rewrite substance. A rewrite that collapses an achievement into
    # a tool stub ("Deployed \textbf{Tableau} and \textbf{Power BI}
    # dashboards.") spends a line of a one-page resume naming products and
    # stating no outcome — strictly worse than the canonical text it replaced,
    # and the shape recruiter guidance calls neither measurable nor
    # searchable. Below this many characters a rewrite is rejected so the
    # bullet falls back to canonical, UNLESS it carries a real figure: a short
    # quantified bullet ("…, reducing coordination delays by 14%") is the best
    # line on the page, not the worst. 0 disables the floor entirely.
    min_rewrite_chars: int = 95
    # Reject a rewrite that opens with a proximity verb (Supported, Assisted,
    # Worked on) when the canonical bullet did not — the fallback text is the
    # stronger line. Never fires when the canonical opens that way itself.
    reject_weak_openers: bool = True
    # Whether a rewrite may state a figure the profile never supplied.
    # Enabled by profile-owner decision (2026-08-25): invented metrics are
    # acceptable on this bot's output. The duplicate-metric guard still
    # applies — two bullets claiming the same number reads as an error
    # regardless of where the number came from.
    allow_invented_metrics: bool = True
    # Bolding beyond this many \textbf{} per bullet reads as keyword farming;
    # extras are demoted to plain text (listing-relevant bolds kept first).
    max_bold_per_bullet: int = 4
    # Ceiling on bolds inside ONE clause. The per-bullet cap cannot see
    # distribution, so a bullet within budget could still stack three tool
    # names into a single clause and read as a parts list.
    max_bold_per_clause: int = 2
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
    # How lean the Skills section is, as a PERCENTAGE of each line's own
    # length: 100 keeps every item, 50 keeps half, 25 is aggressively lean.
    # This is the dial to turn for "less/more stripping"; it exists alongside
    # max_skill_items_per_line because an absolute item count cannot serve
    # lines of different natural length. On this profile one cap of 8 leaves
    # the 12-item Languages line nearly intact while cutting 12 of 20 Relevant
    # Courses — the courses line is not 60% noisier, it is just longer.
    # A percentage self-scales, so one number moves the whole section
    # proportionally. Both limits apply when both are set (the stricter wins);
    # every line keeps at least one item, and listing-matched items still
    # survive either cap exactly as before.
    skills_generosity: int = 100
    # [floor, ceiling] for the coursework line, which opts it OUT of the two
    # caps above and sizes it by relevance instead: it renders as many courses
    # as actually fit the listing, clamped into these bounds. None (the
    # default) leaves coursework under the normal caps, where its length is a
    # fixed number that says nothing about the posting.
    course_item_bounds: tuple[int, int] | None = None
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
    # Category -> domain-signal terms (lowercase). An entry tagged with one of
    # these categories only earns ranking-eligibility when at least one of its
    # category's terms appears as a whole token in the JD text. This targets
    # narrow specialist entries (e.g. hardware/electrical work) that the
    # ranking prompt's own guidance ("generic process-word overlap does NOT
    # count") is supposed to rank low but sometimes doesn't (found live: an
    # avionics entry ranked mid-page on an unrelated AI-intern listing via a
    # thin "automation"/"data handling" bridge). Gated entries are still
    # admitted through the existing capacity rollback if the page cannot reach
    # min_visible_bullets without them — same safety net as any other
    # exclusion, not a hard category bar.
    specialist_categories: dict[str, tuple[str, ...]] = field(default_factory=dict)

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
        # Clamped rather than accepted verbatim by the loop above: values over
        # 100 are meaningless (a line cannot keep more items than it has) and
        # 0 would empty the section, so it floors at 1 — "leanest" is one item
        # per line, never a blank Skills block.
        generosity = payload.get("skills_generosity")
        if isinstance(generosity, int) and not isinstance(generosity, bool):
            config.skills_generosity = max(1, min(100, generosity))
        bounds = payload.get("course_item_bounds")
        if (
            isinstance(bounds, list)
            and len(bounds) == 2
            and all(isinstance(b, int) and not isinstance(b, bool) and b > 0 for b in bounds)
        ):
            # Sorted rather than rejected when reversed: [10, 4] is a typo with
            # an obvious intent, and refusing it would silently fall back to
            # the fixed cap the profile was trying to leave.
            low, high = sorted(bounds)
            config.course_item_bounds = (low, high)
        scope = payload.get("rewrite_scope")
        if isinstance(scope, str) and scope.strip().lower() in ("full", "limited", "selection"):
            config.rewrite_scope = scope.strip().lower()
        leeway_flag = payload.get("adjacent_tool_leeway")
        if isinstance(leeway_flag, bool):
            config.adjacent_tool_leeway = leeway_flag
        audit_flag = payload.get("grounding_audit")
        if isinstance(audit_flag, bool):
            config.grounding_audit = audit_flag
        specialist_raw = payload.get("specialist_categories")
        if isinstance(specialist_raw, dict):
            parsed_specialist: dict[str, tuple[str, ...]] = {}
            for category, terms in specialist_raw.items():
                if not isinstance(category, str) or not isinstance(terms, list):
                    continue
                cleaned = tuple(
                    term.strip().lower()
                    for term in terms
                    if isinstance(term, str) and term.strip()
                )
                if cleaned:
                    parsed_specialist[category.strip().lower()] = cleaned
            if parsed_specialist:
                config.specialist_categories = parsed_specialist
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
            #
            # The minimum was -5 (a 9-bullet page), which assumed the freed
            # count would come back as length. Measured 2026-08-26 over the
            # 14-JD corpus it did not: strong-aggressive settled at exactly
            # the 9-bullet floor carrying ~1,200-1,500 characters against
            # normal mode's ~2,170 — a visibly half-empty page. -2 keeps the
            # depth trade available while holding the page's text volume at
            # roughly normal mode's, since each bullet may also run 120
            # characters longer.
            #
            # 2026-08-26, second pass: the floor is where these pages actually
            # LAND, because the keywordless-bullet filter drops bullets down to
            # it. So it is not a safety net here, it is the page length. The
            # character budget — which the same trials showed sitting ~500
            # characters under its ceiling at 14-15 bullets — is what protects
            # page fit, and it keeps its own lower floor (min - 6) for the
            # genuinely long-bullet builds. Fill the page like every other
            # mode and let the budget trim back when it must.
            max_bullet_chars=config.max_bullet_chars + 120,
            # Skills stay MORE generous here than in normal mode, but not
            # unbounded. Every bullet is written from the JD and the prompt
            # requires each bolded keyword to also appear in Skills, so a lean
            # section fights the mode's own goal — the profile's cap of 8 cuts
            # a 20-item courses line to 8 and takes matched-but-unbolded
            # vocabulary with it. Both limits are therefore loosened rather
            # than switched off: the absolute cap gains headroom for injected
            # JD tools, and the percentage sits above the profile's own so the
            # section still scales with each line's length instead of running
            # off the page. Listing-matched items survive either cap anyway,
            # so the loosening only ever protects unmatched vocabulary.
            max_skill_items_per_line=(
                config.max_skill_items_per_line + 3
                if config.max_skill_items_per_line > 0
                else 0
            ),
            skills_generosity=max(config.skills_generosity, 75),
            # Aggressive modes exist specifically to permit cross-domain
            # reframing (per profile owner: "domain fabrication is allowed,
            # I'm just stress testing it") — the specialist domain-fit gate
            # would fight that goal, so it's off here.
            specialist_categories={},
        )
    if not config.aggressive:
        return config
    return replace(
        config,
        adjacent_tool_leeway=True,
        grounding_audit=False,
        max_bold_per_bullet=config.max_bold_per_bullet + 2,
        # Scale the per-clause ceiling with the per-bullet one: leaving it
        # fixed would let this mode raise the bullet's budget and then throttle
        # it back at the clause, which is the density aggressive exists for.
        # Still a ceiling, not a licence — a bare run of names reads as a parts
        # list at any keyword budget.
        # 0 means "disabled" for this knob, and scaling must not resurrect a
        # cap the profile explicitly switched off.
        max_bold_per_clause=(
            config.max_bold_per_clause + 1 if config.max_bold_per_clause > 0 else 0
        ),
        specialist_categories={},
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
    tail: str  # non-entry sections AFTER the first tagged one, verbatim
    # Non-entry sections BEFORE the first tagged one (commonly Education on
    # Jake-style templates). Rendered between the header block and the entry
    # sections so they keep their place at the top of the page.
    head_sections: str = ""
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
    domain_gated_entries: list[str] = field(default_factory=list)  # specialist entries with no JD signal
    fidelity_findings: list[str] = field(default_factory=list)  # template-structure drift
    rewrite_scope: str = "full"  # which rewrite dial rendered this document
    header_tech_applied: list[str] = field(default_factory=list)  # entries with retargeted \textit lists
    header_tech_rejected: list[str] = field(default_factory=list)  # "entry_id: reason"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "entry"


# Font-size and shape declarations carry no meaning in a title, but they sit
# INSIDE the braces on Jake-style headers ({\textbf{\normalsize Project Name}}),
# so they leaked into the entry title and its slugged entry_id — the model was
# shown "normalsize Accelerometer-Based Range of Motion Detector" and asked to
# rank it.
_TITLE_FONT_MACRO_PATTERN = re.compile(
    r"\\(?:tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge"
    r"|bfseries|itshape|slshape|scshape|upshape|mdseries|rmfamily|sffamily|ttfamily)"
    r"(?![a-zA-Z])"
)


def _clean_entry_title(title: str) -> str:
    return " ".join(_TITLE_FONT_MACRO_PATTERN.sub(" ", title).split()).strip().rstrip(",")


def _extract_entry_title(header: str) -> str:
    match = re.search(r"\\textbf\{([^}]*)\}", header)
    if match:
        title = _clean_entry_title(match.group(1))
        brace = re.search(r"\\textbf\{[^}]*\}\s*\{([^}]*)\}", header)
        if brace and brace.group(1).strip():
            title = f"{title} {_clean_entry_title(brace.group(1))}"
        return title.strip()
    # Macro-style headers (\resumeSubheading{Title}{...}{...}{...}): the first
    # braced argument is the title.
    macro = re.search(r"\\[a-zA-Z]+\s*\{([^{}]*)\}", header)
    if macro and macro.group(1).strip():
        return _clean_entry_title(macro.group(1))
    return _clean_entry_title(header)


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
    # Non-entry sections (Skills, Education, Awards…) are kept verbatim and
    # split by whether they precede the first tagged section. A single
    # contiguous "everything from here to \end{document}" tail was wrong twice
    # over on the common Jake-style layout, where Skills sits BETWEEN entry
    # sections: the tail swallowed every later entry section and rendered it a
    # second time, while any non-entry section before the first tagged one
    # (Education, on that layout) was dropped from the document entirely.
    head_section_texts: list[str] = []
    tail_section_texts: list[str] = []

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
        elif entry_sections:
            tail_section_texts.append(section_text.strip("\n"))
        else:
            head_section_texts.append(section_text.strip("\n"))

    if not entries:
        return None

    # Interleaved non-entry sections collapse to the end rather than holding
    # their original position: entry sections render grouped, so preserving
    # exact interleaving would mean re-ordering the entries around them.
    tail = "\n\n".join(text for text in tail_section_texts if text)
    head_sections = "\n\n".join(text for text in head_section_texts if text)

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
        head_sections=head_sections,
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

# Below this much real posting body, "tailoring" is guesswork off the job
# title. Observed on a Glassdoor listing whose scrape returned nothing but the
# metadata header, where the build still reported "14 bullets, 10 tailored" —
# ten bullets rewritten toward a posting the model had never seen.
THIN_JOB_DESCRIPTION_CHARS = 400


def usable_job_description_chars(description: str) -> int:
    """Length of the real posting body, ignoring the scraper's metadata header.

    A description that is only "Source site: …/Posting URL: …/Title: …" is
    ~200 characters of plumbing and zero characters of job, so raw ``len()``
    cannot tell a thin scrape from a short posting.
    """
    body: list[str] = []
    for line in (description or "").splitlines():
        if not line.strip():
            continue
        header = LISTING_METADATA_LINE_PATTERN.match(line)
        if header:
            # Only the description label carries body text worth counting;
            # the title is what the thin-scrape case already has.
            if header.group(1).lower() == "description":
                body.append(header.group(2))
            continue
        body.append(line)
    return len(URL_IN_TEXT_PATTERN.sub(" ", " ".join(body)).strip())

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
    # 2026-08-20, found live via a real strong-aggressive trial: a Greenhouse
    # posting's "build the engineering backbone that scales AI-assisted
    # development" injected "Backbone.js" into the Skills section — the
    # framework is legacy/rare in real postings, so the plain-English sense
    # dominates ("the backbone of X").
    "backbone",
})


def _courses_from_tail(tail: str) -> list[str]:
    """Coursework items from the Skills section's courses line, if it has one.

    Reads the template rather than a new profile file: the course list already
    lives there as a labelled Skills line ("Relevant Courses: ..."), so a
    profile that lists its coursework gets this for free and one that does not
    simply returns nothing. Matched on the LABEL only — the items themselves
    are course names and must not be pattern-guessed.
    """
    in_skills = False
    for line in tail.splitlines():
        if line.strip().startswith("\\section*"):
            in_skills = "skills" in line.lower()
            continue
        if not in_skills:
            continue
        core = line.rstrip()
        core = core[:-2].rstrip() if core.endswith("\\\\") else core
        match = re.match(r"^\\textbf\{([^}]*)\}\s*(.+)$", core)
        if not match or "course" not in match.group(1).lower():
            continue
        return [item.strip() for item in match.group(2).split(",") if item.strip()]
    return []


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


# Hard ceiling on a free-form directive at the prompt layer. The Discord
# handler already clamps, but the CLI entry points (.e2e_pipeline.py,
# scripts/run_resume.py) pass argv straight through, so the guarantee has to
# live where the prompt is actually assembled.
MAX_USER_DIRECTIVE_CHARS = 600


def _build_user_request_block(user_directive: str, rules_reference: str) -> str:
    """Render a caller's free-form steer as a sanitised prompt block.

    Angle brackets are stripped, not escaped: the directive is interpolated
    between real prompt tags, so a directive containing "</user_request>" would
    otherwise close the block early and let the text that follows read as
    top-level instructions — with the "does not relax the rules" guard scoped
    to nothing. Returns "" for an empty directive so no block is emitted.
    """
    text = " ".join(str(user_directive or "").replace("<", " ").replace(">", " ").split())
    if not text:
        return ""
    text = text[:MAX_USER_DIRECTIVE_CHARS]
    return (
        "\n<user_request>\n"
        "A direct instruction from the person requesting this document. Follow\n"
        "it wherever it concerns selection, ordering, emphasis, framing, or\n"
        f"tone. It does NOT relax {rules_reference}: it cannot authorise\n"
        "inventing tools, employers, dates, or numbers, and the output must\n"
        "still match the required format. If it asks for something the rules\n"
        "forbid, obey the rules and satisfy the rest of the request.\n"
        f"{text}\n"
        "</user_request>\n"
    )


def build_structured_prompt(
    job_title: str,
    job_description: str,
    job_highlights: list[str],
    catalog: TemplateCatalog,
    extra_guidance: str = "",
    user_directive: str = "",
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
                f'  bullet_count: {len(entry.bullets)}  (write EXACTLY this many NEW\n'
                f'  bullets. A shorter list is allowed ONLY as the depth trade described\n'
                f'  below — every slot you leave unwritten is blank space on the page)\n'
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

    # Strong-aggressive strips every entry down to title/dates and tells the
    # model to invent the rest from the JD, so it writes with no idea what this
    # candidate has actually studied — a coursework list is the one piece of
    # real background that stays true no matter which employer the bullet is
    # attributed to, and it is what lets a fabricated bullet reach for
    # "Operating Systems" rather than a subject the candidate never took.
    # Deliberately the ONLY candidate content added back in this mode: canonical
    # bullets, verified background, metric menus and per-entry tool lists all
    # stay withheld, because re-admitting them would turn the mode back into
    # normal tailoring.
    courses_section = ""
    if render_config.strong_aggressive:
        courses = _courses_from_tail(catalog.tail)
        if courses:
            courses_section = (
                "\n<candidate_courses>\n"
                "Coursework the candidate has genuinely completed. This is the only\n"
                "real background you are given: prefer these subjects when a bullet\n"
                "needs academic grounding, and never claim a subject absent here.\n"
                f"{', '.join(courses)}\n"
                "</candidate_courses>\n"
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
            "  - HARD FLOOR: the finished page must carry at least "
            f"{render_config.min_visible_bullets} bullets in\n"
            "    total across all included entries. Short lists that would drop the\n"
            "    page below that floor are refilled from canonical text instead — you\n"
            "    lose the tailoring rather than gaining depth, so never go under it.\n"
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
            "  - Reuse the profile's real metrics at their stated values, and\n"
            "    where a bullet has no figure, supply a modest believable one\n"
            "    (student/intern scope, right kind of number for the work,\n"
            "    defensible in an interview, never repeated across bullets)."
        )
        combined_guidance = f"{combined_guidance}\n\n{aggressive_note}" if combined_guidance else aggressive_note
        if jd_tools_block:
            combined_guidance += jd_tools_block
    guidance_section = f"\n<profile_guidance>\n{combined_guidance}\n</profile_guidance>\n" if combined_guidance else ""

    # A per-build steer typed by the person running the command, e.g.
    # ".resumebuild (lead with the embedded work, this listing is firmware)".
    # Kept in its own block rather than folded into <profile_guidance>: that
    # file is durable profile config, this is one request. The precedence note
    # matters — the directive ranks and emphasises, it never licenses a claim,
    # because everything downstream (grounding, numbers, headers) still
    # validates the output regardless of what was asked for here.
    user_directive_section = _build_user_request_block(
        user_directive, "any rule above, and the response must still be valid JSON"
    )

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
                "   Beyond <skill_anchors>, you MAY name a tool this listing asks for when\n"
                "   the entry's real work could plausibly have used it — a deployment bullet\n"
                "   can carry \\textbf{Docker}, a data bullet \\textbf{Pandas}, a ticket-\n"
                "   tracking bullet \\textbf{Jira}. You do not need proof the candidate used\n"
                "   it; you need a reader to find it unremarkable that they did. The limits\n"
                "   are the claims that would collapse under one interview question or that\n"
                "   a reader would call a lie: never a certification or degree the candidate\n"
                "   lacks, never an employer or job title they never held, never equipment a\n"
                "   student would have no access to, and never a technology from a stack this\n"
                "   entry plainly never touched (no \\textbf{ServiceNow} in a rocketry\n"
                "   soldering bullet, no \\textbf{Kubernetes} on a course assignment). When a\n"
                "   tool has no plausible home, leave it for the Skills line.\n"
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
        f"{courses_section}"
        f"{anchors_section}"
        f"{listing_language_section}"
        f"{guidance_section}\n"
        f"{user_directive_section}"
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
        "   strengthen the bullet — which is almost always. Prefer a real figure from\n"
        "   'metrics on offer' whenever one fits.\n"
        "   WHERE NO REAL FIGURE EXISTS, SUPPLY A PLAUSIBLE ONE. Every bullet should\n"
        "   end on something measurable. An invented figure must be:\n"
        "     - modest and believable for a student/intern scope — 'cut review time\n"
        "       35%', 'across 4 services', 'for 200+ users'; never '10x', never\n"
        "       '$2M', never a company-wide claim;\n"
        "     - the right KIND of number for the work (throughput for a pipeline,\n"
        "       accuracy for a model, coverage for a test suite, turnaround for a\n"
        "       process);\n"
        "     - answerable in an interview — you must be able to say how it was\n"
        "       measured, so prefer figures a student could actually observe;\n"
        "     - stated once. Never repeat the same number in two bullets.\n"
        "   Light extrapolation from what an entry clearly implies is\n"
        "   fine (deployment implies configuration; testing implies debugging) — named\n"
        "   tools, employers, dates, and certifications are not extrapolation.\n"
        + tool_rule
        +
        "3b. EVERY BULLET NAMES ITS POINT. A bullet that describes architecture and\n"
        "   stops ('Built a bot in Python that scrapes and deduplicates postings\n"
        "   through REST APIs') tells the reader what was assembled and never why it\n"
        "   mattered. Close each bullet on the consequence: the figure it moved, the\n"
        "   thing it made possible, the problem it removed, the scale it reached.\n"
        "   Where the entry shows 'metrics on offer', SPEND THEM — at least one bullet\n"
        "   in that entry must carry one of its real figures, digit-for-digit. When an\n"
        "   entry genuinely has no figure, name the concrete result in words instead\n"
        "   ('… so releases stopped needing a manual tagging pass'). What is banned is\n"
        "   the empty gesture at importance ('supporting business objectives') — that\n"
        "   is not an outcome, it is filler, and it will be stripped.\n"
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
        "   may wrap ONLY a PROPER NAME — a named tool, language, platform, piece of\n"
        "   equipment, or certification. Never an activity or a field of work\n"
        "   ('analysis', 'debugging', 'scripting', 'technical', 'hardware\n"
        "   verification'), never soft phrases from the posting ('analysis\n"
        "   experience', 'client relationships'), never business vocabulary, verbs,\n"
        "   or metrics. Wrap the NAME ONLY, never the noun trailing it: write\n"
        "   \\textbf{Quartus II} simulation and \\textbf{PCB} circuits, NOT\n"
        "   \\textbf{Quartus II simulation} or \\textbf{PCB circuits}. SPREAD, don't\n"
        "   cram: carry the tools this listing wants, but never stack three of them\n"
        "   into one clause — 'scrapes postings through \\textbf{REST APIs},\n"
        "   \\textbf{Playwright} and \\textbf{SQLite}' is a parts list, not a sentence.\n"
        "   Give each name its own grammatical role across the bullet, and read the\n"
        "   result aloud in your head: if it does not sound like a person describing\n"
        "   their work, redistribute the names or drop one. Bolding an ordinary word\n"
        "   wastes the cue.\n"
        "6. Keep each bullet a single concise sentence (under 260 characters, and not\n"
        "   below ~95 unless it carries a number: a bullet that names tools and stops\n"
        "   ('Deployed Tableau and Power BI dashboards.') states no outcome and is\n"
        "   dropped in favour of the canonical text). Lead with\n"
        "   a strong PAST-TENSE active verb (Developed, Built — never the listing's\n"
        "   imperative mood) calibrated to the listing's seniority. Never open with a\n"
        "   proximity verb — Supported, Assisted, Helped, Worked on, Used — which\n"
        "   describes being NEAR the work instead of doing it; such a bullet is\n"
        "   dropped for the canonical text. Do NOT open two\n"
        "   bullets with the same verb — five bullets starting 'Built' read\n"
        "   machine-generated no matter how good each one is. Do not reuse the\n"
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
        "    this listing's language values (scale, cost, reliability, adoption).\n"
        "    Where the entry offers no figure for the category this listing cares\n"
        "    about, supply a believable one under rule 2's constraints rather than\n"
        "    leaving the bullet without an outcome.\n"
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
    seen_snippets: set[str] = set()
    for source in sources:
        plain = source.replace("\\%", "%")
        # Math spans go before command stripping: "$4 \times 16$" would
        # otherwise survive as the bare digits "4 16" and read as a metric.
        plain = MATH_SPAN_PATTERN.sub(" ", plain)
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
            if not _digits_read_as_metric(plain, match.end()):
                continue
            seen.add(number)
            start = _menu_snippet_start(plain, match.start())
            end = plain.find(" ", match.end() + 30)
            snippet = plain[start : end if end != -1 else len(plain)].strip(" ,;.")
            snippet = MENU_LEADING_CONJUNCTION_PATTERN.sub("", snippet)
            if label and start > label.end():
                snippet = f"{label.group(0)} ... {snippet}"
            # Two numbers in one clause ("206,775 postings (120,989 URLs)")
            # produce the same window twice; one menu line already shows both.
            # Containment, not equality: the same claim reached from a bullet
            # and from its baseinfo Notes line differs only by a trailing
            # phrase, and two menu lines saying one thing waste the budget.
            normalized = snippet.lower()
            if any(
                normalized in prior or prior in normalized for prior in seen_snippets
            ):
                continue
            seen_snippets.add(normalized)
            menu.append(snippet)
            if len(menu) >= 6:
                return menu
    return menu


def _digits_read_as_metric(text: str, number_end: int) -> bool:
    """False when the digits are glued to a word rather than a unit.

    "50ms"/"75th"/"500+" stay metrics; "3D terrain" and "7-segment display"
    are compound names whose digits the candidate cannot claim as a figure.
    """
    tail = text[number_end:]
    if not tail:
        return True
    if tail[0] == "-" and re.match(r"-[A-Za-z]", tail):
        return False
    if not tail[0].isalpha():
        return True
    return bool(MENU_UNIT_SUFFIX_PATTERN.match(tail))


def _menu_snippet_start(text: str, number_start: int) -> int:
    """Start offset for a menu snippet: the clause the number sits in.

    A fixed character lookback cut mid-phrase ("and C for a mobile robot",
    "Suite , reducing coordination delays"), so the model saw fragments where
    it needed a readable claim. Prefer the last clause break within the
    lookback budget, falling back to the old word-boundary cut.
    """
    window_start = max(0, number_start - 70)
    last_break = None
    for match in MENU_CLAUSE_BREAK_PATTERN.finditer(text, window_start, number_start):
        last_break = match.end()
    if last_break is not None:
        return last_break
    if window_start == 0:
        return 0
    return text.rfind(" ", 0, max(0, number_start - 40)) + 1


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
    posting postings read reading please carefully submit submitted submitting note notes
    everyone everybody people together talent growth community trust firm culture inclusive
    diverse diversity belonging mission values vision passion passionate excited exciting
    opportunity awaits welcome welcoming committed commitment thrive succeed success
    information email emails full max maximum minimum considered consideration resumes
    resume cover letter letters deadline deadlines interview interviews transcript
    transcripts eligible eligibility deemed selected candidates
    avec pour dans notre votre vous nous sont être etre cette celles ceux aux par plus tout tous toute toutes comme
    ainsi afin chez sous entre aussi leur leurs elle elles vos ils mais fait faire sans autres autre bien très tres
    sera seront doit devra devrez poste emploi équipe equipe stagiaire stage candidat candidats candidate candidature
    exigences compétences competences expérience expériences connaissances milieu travail journée langue français
    francais anglais veuillez développement developpement déveloper travailler titre lieu heures semaine semaines
    salaire avantages télétravail teletravail""".split()
)


# The scraper prefixes every description with these metadata lines.
LISTING_METADATA_LINE_PATTERN = re.compile(
    r"^\s*(Source site|Posting URL|Apply URL|Title|Company|Location|Description)\s*:\s*(.*)$",
    re.IGNORECASE,
)
URL_IN_TEXT_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Two- and three-letter all-caps acronyms, which the main token scan (minimum
# four characters) cannot see. Case-sensitive on purpose: "IT" is a listing
# term, "it" is not.
ACRONYM_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,3}(?![A-Za-z0-9])")

# Dates and postings' season words are never resume vocabulary, but they are
# frequent and title-weighted enough to crowd out the real terms.
_LISTING_CALENDAR_WORDS = frozenset(
    """january february march april june july august september october november december
    jan feb mar apr jun jul aug sept sep oct nov dec monday tuesday wednesday thursday
    friday saturday sunday janvier fevrier février mars avril juin juillet aout août
    septembre octobre novembre decembre décembre""".split()
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
    # The scraped description opens with a metadata header ("Source site: …",
    # "Company: Saatchi & Saatchi Canada", "Location: Toronto, Ontario"). Its
    # tokens are posting plumbing, but they scored like listing vocabulary and
    # reached the model's rule-15 checklist as terms to work into bullets —
    # observed output offered "Johns", "Saatchi", "Tesla" and "Toronto".
    body_lines: list[str] = []
    suppressed: set[str] = set()
    title_lower = ""
    for line in job_text.splitlines():
        if not line.strip():
            continue
        header = LISTING_METADATA_LINE_PATTERN.match(line)
        if header:
            field, value = header.group(1).lower(), header.group(2)
            if field == "title":
                title_lower = value.lower()
                body_lines.append(value)
            elif field == "description":
                # The body often rides on the same line as its label; dropping
                # the line whole would discard the entire posting.
                if value.strip():
                    body_lines.append(value)
            elif field in {"company", "location"}:
                suppressed.update(re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*", value.lower()))
            continue
        # URLs tokenize into path fragments ("KE18", "IC2281069"), but a
        # posting body is often ONE long line that happens to contain a link —
        # dropping the line would discard the whole description. Excise the
        # URL, keep the prose.
        body_lines.append(URL_IN_TEXT_PATTERN.sub(" ", line))
    if not title_lower and body_lines:
        title_lower = body_lines[0].lower()
    job_text = "\n".join(body_lines)

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
    # Short all-caps acronyms are the densest listing vocabulary there is —
    # SEM, ETL, CAD, PLC, ERP, CRM, SQL — and the token scan below never sees
    # them: its minimum length is four characters. Score them explicitly.
    for token in ACRONYM_TOKEN_PATTERN.findall(job_text):
        low = token.lower()
        if low in _LISTING_STOPWORDS or low in suppressed:
            continue
        score = scores.get(low, (0, token))[0] + 3
        if low in title_lower:
            score += 2
        scores[low] = (score, token)
    # Accented letters stay inside a token — bilingual Canadian postings
    # otherwise shed fragments like "veloppement" (from "développement") into
    # the keyword list shown to the model.
    for token in re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9.+#/-]{3,}", job_text
    ):
        low = token.lower().strip("-/.")
        if not low or low in _LISTING_STOPWORDS or low in suppressed:
            continue
        if low in _LISTING_CALENDAR_WORDS:
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
        # The all-caps bonus is for acronyms ("SEM", "SQL", "ETL"), not for a
        # posting shouting its instructions: "PLEASE", "CAREFULLY", "EMAIL"
        # and "INFORMATION" were outscoring the listing's real vocabulary.
        acronym_shaped = token.isupper() and len(token) <= 4
        tech_shaped = (
            bool(re.search(r"[0-9.+#/]", token))
            or acronym_shaped
            or (token[:1].isupper() and not token.istitle() and not token.isupper())
        )
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


# Emphasis is a scanning aid: a recruiter's eye should land on a \textbf and
# hit a NAME every time. The grounding checks in validate_tailored_bullet prove
# the candidate really did the work — they never prove the bolded span names
# anything, so any ordinary word the entry happens to contain can carry bold.
# Found live (2026-08-11, City of Winnipeg "Technical Assistant" listing):
# \textbf{technical} twice and \textbf{CITY}, all three grounded in the entry's
# own canonical text, all three plain English. Words like these are demoted to
# plain text — the word stays, the emphasis goes — so what survives is only
# what is worth stopping on. Listed lowercase; matching is case-insensitive, so
# "Technical" and "CITY" are covered by the same entry.
_GENERIC_BOLD_WORDS = frozenset({
    "ability", "accuracy", "ai", "algorithm", "algorithms", "analysis",
    "analytics", "application", "applications", "architecture", "automation",
    "backend", "best practices", "board", "boards", "circuit", "circuits",
    "city", "collaboration", "communication", "company", "compliance",
    "customer", "customers", "data", "database", "databases", "debugging",
    "deployment", "deployments", "design", "development", "documentation",
    "documents", "efficiency", "electrical", "electronics", "engineer",
    "engineering", "equipment", "experience", "frontend", "fullstack",
    "hardware", "important", "infrastructure", "innovation", "integration",
    "integrations", "knowledge", "leadership", "maintenance", "management",
    "manufacturing", "method", "methods", "ml", "mobile", "model", "models",
    "network", "networking", "operations", "optimization", "performance",
    "pipeline", "pipelines", "platform", "platforms", "procedures",
    "process", "processes", "production", "productivity", "programming",
    "project", "projects", "quality", "reliability", "reporting", "reports",
    "requirements", "research", "safety", "scalability", "scripting",
    "security", "service", "services", "simulation", "skills", "software",
    "solution", "solutions", "specifications", "standards", "strategy",
    "support", "system", "systems", "team", "teams", "teamwork", "technical",
    "technologies", "technology", "test", "testing", "tests", "tooling",
    "tools", "training", "troubleshooting", "validation", "verification",
    "web", "workflow", "workflows",
})

# Real tool names that carry no shape signal — all lowercase, no digits, no
# interior capitals — so _word_names_a_tool would read them as prose. Kept
# separate from _KNOWN_TECH_TERMS on purpose: that set also drives aggressive
# JD keyword injection, and this one must only ever decide emphasis.
_BOLDWORTHY_LOWERCASE_TOOLS = frozenset({
    "asyncio", "bazel", "clang", "cmake", "conda", "cuda", "eslint", "fastapi",
    "ffmpeg", "flask", "gcc", "gdb", "glsl", "gradle", "gunicorn", "i2c",
    "jquery", "jtag", "matplotlib", "maven", "modbus", "mpi", "mypy", "npm",
    "numpy", "openmp", "opengl", "pandas", "pip", "poetry", "postgres",
    "pygame", "ros", "rtos", "ruff", "scipy", "seaborn", "sklearn", "spi",
    "sqlite", "tailwind", "tkinter", "tox", "uart", "unittest", "uvicorn",
    "valgrind", "verilog", "vhdl", "webgl", "webgpu", "yarn",
})


def _word_names_a_tool(word: str) -> bool:
    """True when a single token is shaped like a product name, not prose."""
    core = word.strip().strip(".,;:()[]{}'\"")
    if not core or core.lower() in _GENERIC_BOLD_WORDS:
        return False
    if re.search(r"[0-9#+]", core):
        return True  # C++, C#, HTML5, STM32
    if re.search(r"\.[A-Za-z0-9]", core):
        return True  # Node.js, .NET, monday.com
    if core[1:] != core[1:].lower():
        return True  # PyTorch, JavaScript, OpenGL, SQL, PCB, APIs
    lowered = core.lower()
    if lowered in _KNOWN_TECH_TERMS or lowered in _BOLDWORTHY_LOWERCASE_TOOLS:
        return True  # Spring, Redis — real names that also read as English
    if lowered.endswith(("ed", "ing")):
        return False  # Designed, Building — a capital letter alone is not a name
    return core[0].isupper()  # Docker, Excel, Kafka, Tableau


def _is_boldworthy(phrase: str, skill_anchors: tuple[str, ...]) -> bool:
    """True when a bolded phrase names a tool, system, language, or platform.

    Anchors win first: a skill the profile owner explicitly declared is theirs
    to emphasise even when it reads like prose ("power distribution"). Past
    that, a phrase earns bold by being a known tool name or by every one of its
    words being shaped like one.
    """
    normalized = phrase.strip().replace("\\", "").lower()
    if not normalized:
        return False
    # Ahead of the anchor check: an anchor like "database design" must not
    # license a bold on "design" alone just by containing the word.
    if normalized in _GENERIC_BOLD_WORDS:
        return False
    # Deliberately tighter than _matches_skill_anchor, which also accepts a
    # phrase that merely CONTAINS an anchor — right for grounding, wrong here:
    # "PCB circuits" contains the anchor "pcb", but the bold belongs on "PCB"
    # alone, so let it fall through to _narrow_bold_span instead. The match
    # into the anchor is whole-token so "ver" does not ride in on "verilog".
    if any(
        normalized == anchor
        or re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", anchor)
        for anchor in skill_anchors
    ):
        return True
    if normalized in _KNOWN_TECH_TERMS or normalized in _BOLDWORTHY_LOWERCASE_TOOLS:
        return True
    words = [w for w in re.split(r"[\s/]+", phrase.strip()) if w]
    return bool(words) and all(_word_names_a_tool(w) for w in words)


def _narrow_bold_span(phrase: str) -> tuple[int, int] | None:
    """Offsets of the sub-span of `phrase` that deserves bold, or None.

    Models routinely wrap a real tool together with a trailing noun —
    \\textbf{Quartus II simulation}, \\textbf{PCB circuits}. Dropping the whole
    bold would throw away a legitimate emphasis, so trim the prose off the
    edges and keep the tool. Only edge trimming: a phrase whose interior is
    prose ("Arithmetic and Logic Unit") is a description, not a name, and
    loses its bold entirely.
    """
    # Split on slashes as well as spaces, matching _is_boldworthy's tokenizer:
    # "Testing/QA" must narrow to QA rather than pass through whole. Spans run
    # first-kept to last-kept, so an interior "/" stays inside ("CI/CD").
    tokens = [m.span() for m in re.finditer(r"[^\s/]+", phrase)]
    if not tokens:
        return None
    lo, hi = 0, len(tokens)
    while lo < hi and not _word_names_a_tool(phrase[slice(*tokens[lo])]):
        lo += 1
    while hi > lo and not _word_names_a_tool(phrase[slice(*tokens[hi - 1])]):
        hi -= 1
    if lo >= hi:
        return None
    if not all(_word_names_a_tool(phrase[slice(*span)]) for span in tokens[lo:hi]):
        return None
    return tokens[lo][0], tokens[hi - 1][1]


def _demote_unworthy_bolds(text: str, skill_anchors: tuple[str, ...]) -> str:
    """Strip \\textbf from spans that do not name a tool, keeping the words."""
    for phrase in re.findall(r"\\textbf\{([^}]*)\}", text):
        # A leftover "{" means the [^}]* capture stopped at an INNER brace —
        # nested math ($x^{2}$) or a nested \textbf. The span is truncated, so
        # editing it would move the bold onto the wrong words; leave it alone.
        if not phrase.strip() or "{" in phrase:
            continue
        if _is_boldworthy(phrase, skill_anchors):
            continue
        span = _narrow_bold_span(phrase)
        if span is None:
            replacement = phrase
        else:
            start, end = span
            replacement = (
                f"{phrase[:start]}\\textbf{{{phrase[start:end]}}}{phrase[end:]}"
            )
        text = text.replace(f"\\textbf{{{phrase}}}", replacement)
    return text


def _cap_bold_density(
    text: str, listing_keywords: tuple[str, ...] | list[str], max_bold: int
) -> str:
    """Demote all but the `max_bold` most listing-relevant bolds in a bullet.

    Past this ceiling emphasis reads as keyword farming rather than emphasis.
    Keyword matches are kept first, then earliest mention.
    """
    bold_spans = re.findall(r"\\textbf\{([^}]*)\}", text)
    if len(bold_spans) <= max_bold:
        return text
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
    keep = set(ranked[:max_bold])
    for i, phrase in enumerate(bold_spans):
        if i not in keep:
            text = text.replace(f"\\textbf{{{phrase}}}", phrase, 1)
    return text


# Text that can sit BETWEEN two bolded names without the pair ceasing to be a
# bare enumeration: list punctuation and conjunctions only. Anything else
# (a preposition, a verb, any real words) means the names are doing separate
# grammatical work, which is exactly the distribution we want to preserve.
_BOLD_ENUMERATION_GAP = re.compile(r"^[\s,;]*(?:and|&|or|plus)?[\s,;]*$", re.I)


def _cap_bold_per_clause(
    text: str, listing_keywords: tuple[str, ...] | list[str], max_per_clause: int
) -> str:
    """Demote bolds past `max_per_clause` inside one bare enumeration.

    The per-bullet ceiling cannot see distribution: three tool names run
    together ("through \\textbf{REST APIs}, \\textbf{Playwright} and
    \\textbf{SQLite}") reads as a parts list rather than a sentence, even when
    the bullet's total is within budget. Detection is by enumeration run, not
    by clause — the commas that make it a list would also split it into
    "clauses" of one or two names each, hiding the very pattern being caught.
    Names separated by real words are left alone: "\\textbf{Kafka} and
    \\textbf{Redis} into the \\textbf{Airflow} scheduler" is prose, not a list.
    Keeps the listing-relevant names, same ranking rule as the density cap.
    """
    if max_per_clause <= 0:
        return text
    keyword_lowered = [k.lower() for k in listing_keywords if k and k.strip()]

    def _listing_relevant(phrase: str) -> bool:
        p = phrase.strip().lower()
        return any(
            p == k or _whole_token_search(k, p) or _whole_token_search(p, k)
            for k in keyword_lowered
        )

    matches = list(re.finditer(r"\\textbf\{([^}]*)\}", text))
    if len(matches) <= max_per_clause:
        return text

    runs: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = []
    for match in matches:
        if current and _BOLD_ENUMERATION_GAP.match(
            text[current[-1].end() : match.start()]
        ):
            current.append(match)
        else:
            if current:
                runs.append(current)
            current = [match]
    if current:
        runs.append(current)

    demote: list[str] = []
    for run in runs:
        if len(run) <= max_per_clause:
            continue
        phrases = [m.group(1) for m in run]
        ranked = sorted(
            range(len(phrases)),
            key=lambda i: (0 if _listing_relevant(phrases[i]) else 1, i),
        )
        keep = set(ranked[:max_per_clause])
        demote.extend(phrase for i, phrase in enumerate(phrases) if i not in keep)

    for phrase in demote:
        text = text.replace(f"\\textbf{{{phrase}}}", phrase, 1)
    return text


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

    # Emphasis quality gate. Runs after the grounding checks (which decide
    # whether a claim is TRUE) and before the density cap (which decides how
    # MANY bolds survive): this one decides whether a bold is worth spending
    # the reader's attention on at all. Aggressive mode is included on purpose
    # — its own prompt already says "Bold the TOOL, not the concept", so this
    # enforces what the model was told rather than loosening with the mode.
    # It must also run AFTER the restore above, which otherwise reinstates the
    # template's own descriptive bolds (\textbf{Arithmetic and Logic Unit
    # (ALU)}) that this gate just cleared.
    text = _demote_unworthy_bolds(text, skill_anchors)
    text = _cap_bold_density(text, listing_keywords, limits.max_bold_per_bullet)
    text = _cap_bold_per_clause(
        text, listing_keywords, limits.max_bold_per_clause
    )

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
        # A tail carrying a REAL figure is an outcome, not padding — ",
        # ensuring 99.9% uptime" is the quantified result the bullet exists to
        # state, and stripping it threw away the strongest thing on the line.
        # Grounded-only on purpose: a fabricated figure that lives just in the
        # tail must stay strippable, or the number guard below rejects the
        # whole rewrite instead of salvaging its good front half.
        if filler:
            tail_numbers = STANDALONE_NUMBER_PATTERN.findall(
                text[filler.start() :].replace(",", "")
            )
            if limits.allow_invented_metrics:
                # Provenance is irrelevant when invented figures are permitted,
                # and requiring grounding here silently undid the metric
                # injection pass: it phrases outcomes as "…, supporting 4
                # concurrent rooms", whose number is invented by construction,
                # so the exemption failed and the guard stripped the very
                # metric the pass had just added.
                if tail_numbers:
                    filler = None
            else:
                tail_grounding = set(
                    STANDALONE_NUMBER_PATTERN.findall(grounding.replace(",", ""))
                )
                if tail_numbers and all(
                    number in tail_grounding for number in tail_numbers
                ):
                    filler = None
        if filler and not FILLER_CLAUSE_PATTERN.search(canonical):
            stripped = text[: filler.start()].rstrip(" ,;")
            # The "to" connector can sit mid-clause rather than at a clause
            # boundary ("…, analogous to supporting client communications"),
            # and cutting there orphans the adjective that introduced it —
            # "…in clear, non-technical terms, analogous." reached a rendered
            # PDF. If what survives ends in a stub clause, drop that too.
            head, separator, tail = stripped.rpartition(",")
            remainder = tail.strip()
            # A stub is one or two orphaned words ("analogous", "similar to"),
            # never a real clause — and never one carrying a figure, which is
            # an outcome the bullet exists to state.
            if (
                separator
                and len(remainder.split()) <= 2
                and not STANDALONE_NUMBER_PATTERN.search(remainder)
            ):
                stripped = head.rstrip(" ,;")
            if len(stripped) < 30:
                return None, f"appended filler clause {filler.group(1).lower()!r}"
            if not stripped.endswith((".", "!", "?")):
                stripped += "."
            text = stripped

    # Tense repair. Rule 6 tells the model never to borrow the listing's
    # imperative mood, and nothing enforced it: a role that ended in 2024 came
    # back with "Maintain clear documentation…", "Track project milestones…",
    # "Validate system changes…". Detected by exact stem match against the
    # canonical opener rather than by reading the entry's dates, so it only
    # fires on an unambiguous downgrade of the same verb — "Maintained" ->
    # "Maintain" — and never on a legitimately different word choice.
    #
    # REPAIRED, not rejected: the stem match has already proven the correct
    # past form, so restoring it costs one word and keeps the rewrite's
    # JD-specific tailoring. Rejecting here instead measurably pushed the
    # tailored-bullet rate down (56% -> 42% across the corpus) for a defect
    # whose fix is mechanical.
    rewrite_lead = re.match(r"(?:\\(?:textbf|textit|emph)\{)?([A-Za-z]+)", text)
    if rewrite_lead:
        present = rewrite_lead.group(1).lower()
        restored = PRESENT_TO_PAST_OPENERS.get(present)
        if restored is None and canonical:
            # Verbs outside the table still get caught when the canonical
            # opener is the same word's past form.
            canonical_lead = re.match(
                r"(?:\\(?:textbf|textit|emph)\{)?([A-Za-z]+)", canonical
            )
            if canonical_lead and canonical_lead.group(1).lower() in {
                f"{present}ed",
                f"{present}d",
            }:
                restored = canonical[canonical_lead.start(1) : canonical_lead.end(1)]
        if restored:
            text = (
                text[: rewrite_lead.start(1)]
                + restored
                + text[rewrite_lead.end(1) :]
            )

    # Weak-opener guard. Same principle as the substance floor and the metric
    # rules: a rewrite may not lose a quality the canonical bullet had. If the
    # canonical already opens this way it is the profile's own voice and
    # stands — this only catches the rewrite DOWNGRADING a strong opener.
    if limits.reject_weak_openers and canonical:
        rewrite_opener = re.match(r"(?:\\(?:textbf|textit|emph)\{)?([A-Za-z]+)", text)
        canonical_opener = re.match(
            r"(?:\\(?:textbf|textit|emph)\{)?([A-Za-z]+)", canonical
        )
        if (
            rewrite_opener
            and rewrite_opener.group(1).lower() in WEAK_OPENING_VERBS
            and not (
                canonical_opener
                and canonical_opener.group(1).lower() in WEAK_OPENING_VERBS
            )
        ):
            return None, f"weak opener {rewrite_opener.group(1)!r}"

    # Substance floor. Runs after filler stripping, so it also catches a
    # bullet whose only content WAS the padding. A rewrite carrying a real
    # figure is exempt: "Tracked milestones in \textbf{Microsoft Suite},
    # reducing coordination delays by 14%" is short and excellent, while
    # "Deployed \textbf{Tableau} and \textbf{Power BI} dashboards." is short
    # and empty. Rejection means canonical fallback — the longer, outcome-
    # bearing text — which is the better line either way.
    floor = limits.min_rewrite_chars
    if floor > 0 and len(text) < floor and len(canonical) >= floor:
        quantified = any(
            len(number.replace(".", "")) >= DUP_METRIC_MIN_DIGITS
            for number in STANDALONE_NUMBER_PATTERN.findall(text.replace(",", ""))
        )
        if not quantified:
            return None, "thin rewrite (names tools, states no outcome)"

    # Numbers are strict on every profile and every mode this code path sees:
    # a significant standalone number in a rewrite must already exist in the
    # entry's grounding (bullets, header, baseinfo block). Runs AFTER filler
    # stripping so a fabricated figure living only in a stripped tail doesn't
    # reject the salvageable rest of the bullet. Commas are stripped on both
    # sides so "206,775" and "206775" ground each other; digits embedded in
    # identifiers (STM32, HCS12) are neither claims nor grounding.
    #
    # Gated on allow_invented_metrics, which the profile owner turned on for
    # every command (2026-08-25). With it on, a figure the profile never
    # supplied is permitted; the duplicate-metric guard in
    # enforce_cross_bullet_consistency still runs, so the same number cannot
    # be claimed twice on one page.
    if not limits.allow_invented_metrics:
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


# Domain-fit audit: a general-purpose companion to the deterministic
# specialist_categories gate (which needs a hand-curated term list per
# category, currently only "electrical"). This judge needs no per-domain
# configuration — it works from the entry's own content and the listing's
# text, so it covers any narrow specialist domain (healthcare, construction,
# culinary, legal, ...) a profile owner never anticipated. Both gates feed
# the SAME excluded_ids/capacity-rollback path in render_structured_resume,
# so an LLM outage or a wrong verdict degrades exactly like a model exclude:
# rolled back automatically if the page can't be filled without the entry.
DOMAIN_FIT_AUDIT_PROMPT_HEADER = (
    "You are checking whether resume entries belong on this specific job listing.\n"
    "\n"
    "You will see the job listing once, then a list of the candidate's resume\n"
    "entries (projects and experience). For EACH entry, decide whether it is a\n"
    "clear DOMAIN MISMATCH: the entry's core subject matter belongs to a narrow\n"
    "specialist field that has essentially nothing to do with the listing's\n"
    "field. This applies to ANY domain, not just one industry — a domain\n"
    "mismatch could be an electrical/hardware/avionics entry on a marketing or\n"
    "generalist software listing, a clinical/nursing entry on a construction\n"
    "listing, a construction/trades entry on a legal listing, a culinary entry\n"
    "on a finance listing, or any other pairing where the entry's field and the\n"
    "listing's field simply do not overlap.\n"
    "\n"
    "Flag ONLY when the entry's core work — its equipment, procedures,\n"
    "licensure, or subject-matter discipline — genuinely does not belong in\n"
    "the listing's field. Do NOT flag because of:\n"
    "- generic shared process words (automation, testing, documentation, data\n"
    "  handling, process improvement, communication, teamwork, troubleshooting,\n"
    "  inspection, installation, maintenance, \"hands-on\", equipment) — these\n"
    "  appear in every technical/physical field and are NOT domain signal on\n"
    "  their own. \"Equipment troubleshooting\" in a mining listing does not\n"
    "  match \"hardware debugging\" in an avionics entry just because both\n"
    "  involve physical equipment — the equipment itself is unrelated\n"
    "- the entry being less impressive, less senior, or less directly relevant\n"
    "  than other entries — that is a ranking decision, not a domain mismatch\n"
    "- the listing not naming any specific field at all (a truly generic\n"
    "  listing with no named discipline accepts any entry) — but this is rare:\n"
    "  most listings DO name a specific field (AI, data science, mining\n"
    "  engineering, nursing, ...) even when their tone is casual or their\n"
    "  wording is broad (\"curious and technical\", \"eager to learn\", \"fast-\n"
    "  paced\"). Casual tone is NOT the same as an unnamed field — judge the\n"
    "  field the listing actually names, not how informally it's worded\n"
    "- the employer's stated industry or business (what the COMPANY does) —\n"
    "  judge by what the LISTING asks the person to actually DO day to day,\n"
    "  not the industry vertical the employer operates in. An AI/data role at\n"
    "  a power-equipment or electrical-manufacturing company is still an\n"
    "  AI/data role; it does not make an electrical/hardware entry relevant\n"
    "  just because the employer's business happens to touch electrical work\n"
    "- transferable skills (programming languages, project tools, soft skills)\n"
    "  that legitimately cross domains\n"
    "\n"
    "A broad shared category label is NOT enough on its own — the SPECIFIC\n"
    "sub-discipline must actually correspond, not just the umbrella term:\n"
    "\"engineering\", \"STEM\", \"technical\", and \"data-related\" are umbrella\n"
    "terms, not domain matches. A mining/civil/mechanical engineering listing\n"
    "and an electrical/avionics entry are both \"engineering\" but are\n"
    "different sub-disciplines — still a mismatch. An AI, automation, or data\n"
    "science listing and a hardware/PCB/embedded-firmware entry are both\n"
    "\"technical\" but are different sub-disciplines — still a mismatch, even\n"
    "if the entry mentions software tools (Python, scripting) in service of\n"
    "the hardware work.\n"
    "\n"
    "When unsure, do NOT flag — a false mismatch hides real, relevant\n"
    "experience from the resume. Only flag confident, obvious mismatches.\n"
    "\n"
    'Return ONLY a JSON object: {"verdicts": [{"id": <int>, "mismatch":\n'
    'true|false, "reason": "<short reason, or empty>"}]}\n'
    "One verdict per item, using the same ids as given.\n"
)


def build_domain_fit_audit_items(catalog: TemplateCatalog) -> tuple[list[dict], dict[int, str]]:
    """Judge-call payload: one summary per catalog entry (canonical content,
    not tailored bullets — domain identity does not change with tailoring).
    """
    items: list[dict] = []
    id_map: dict[int, str] = {}
    for entry in catalog.entries:
        background = " ".join(catalog.baseinfo_blocks.get(tag, "") for tag in entry.categories)
        summary = " ".join(f"{entry.header} {' '.join(entry.bullets)} {background}".split()).strip()
        item_id = len(items)
        id_map[item_id] = entry.entry_id
        items.append({"id": item_id, "section": entry.section, "summary": summary})
    return items, id_map


def domain_fit_audit_prompt(job_title: str, job_description: str, items: list[dict]) -> str:
    listing_text = f"Title: {job_title}\n{job_description}".strip()
    return (
        DOMAIN_FIT_AUDIT_PROMPT_HEADER
        + "\nJob listing:\n"
        + listing_text
        + "\n\nResume entries:\n"
        + json.dumps(items, ensure_ascii=True, indent=1)
    )


def apply_domain_fit_verdicts(
    id_map: dict[int, str], verdicts: Any
) -> tuple[list[str], list[str]]:
    """Returns (gated_entry_ids, reasons). Malformed verdicts (bad ids,
    missing fields, wrong types) are ignored — the audit must never be able
    to break a build."""
    gated: list[str] = []
    reasons: list[str] = []
    if not isinstance(verdicts, list):
        return gated, reasons
    for verdict in verdicts:
        if not isinstance(verdict, dict) or not verdict.get("mismatch"):
            continue
        try:
            entry_id = id_map.get(int(verdict.get("id")))
        except (TypeError, ValueError):
            entry_id = None
        if entry_id is None or entry_id in gated:
            continue
        gated.append(entry_id)
        reason = " ".join(str(verdict.get("reason") or "").split())[:80]
        reasons.append(f"{entry_id}: {reason}" if reason else entry_id)
    return gated, reasons


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


# Skills rewrite: a real LLM pass over the Skills section (strong-aggressive
# only), replacing the deterministic reorder/inject path
# (_rewrite_skills_for_jd below) with judgement calls the regex-keyword
# matcher can't make — e.g. recognizing a skill is relevant to the listing
# even without literal keyword overlap, or that an owned tool fits a
# different category line than it currently sits in. The one rule that never
# bends: every returned item must come from the candidate's OWNED
# skill_anchors. apply_llm_skills_rewrite re-validates that against the
# model's output regardless of what the prompt asked for, so a model that
# ignores the instruction still can't fabricate a skill onto the resume.
# Strong-aggressive is the only mode that runs this rewrite, and per the
# profile owner (2026-08-28) its Skills section is fabricated from the listing
# exactly as its bullets already are. The ownership rule that used to be "the
# single hard rule" here is therefore gone — deliberately, not by oversight.
# Coursework is the one exception and is never sent to this call at all.
SKILLS_REWRITE_PROMPT_HEADER = (
    "You are rewriting the Skills section of a resume to match one specific\n"
    "job listing as closely as possible.\n"
    "\n"
    "You will see: the job listing, the skills the candidate already lists,\n"
    "the resume content those skills support, and the Skills section's\n"
    "current category labels.\n"
    "\n"
    "For EACH label shown, return an ordered list of items for that line:\n"
    "- Lead with the technologies, tools and platforms THIS LISTING names,\n"
    "  written the way the listing writes them, in whichever line's category\n"
    "  each one fits. These matter more than anything already on the line.\n"
    "- Then fill the line out with the candidate's existing items that suit\n"
    "  this listing, most relevant first.\n"
    "- Aim for roughly 6-9 items per line: enough to look like a real\n"
    "  engineer's toolkit, not so many that the relevant ones get buried.\n"
    "- Drop an existing item when it points at a different discipline than\n"
    "  this listing (hardware tools on a web role, and the reverse). A\n"
    "  shorter line beats a line that argues for the wrong job.\n"
    "- Only name a technology a working engineer in this role would plausibly\n"
    "  use. No invented product names, no tools that do not exist.\n"
    "- Never place the same item on two different lines.\n"
    "- Do not change the label text itself, only the items after it.\n"
    "\n"
    'Return ONLY a JSON object: {"skills": {"<label exactly as given>":\n'
    '["item1", "item2", ...]}}, with one key per label shown below, using the\n'
    "labels verbatim.\n"
)


def gather_rewritten_bullet_text(catalog: TemplateCatalog, selection: StructuredSelection) -> str:
    """Flat text of every entry's effective bullets — the model's tailored
    text where provided, canonical text otherwise. Cheap stand-in for 'what
    the resume will actually say' for prompts that need that context before
    the deterministic render has run (skills rewrite needs to see this to
    judge relevance; waiting for the full render would mean rendering twice).
    """
    parts: list[str] = []
    for entry in catalog.entries:
        provided = selection.bullets.get(entry.entry_id)
        bullets = entry.bullets
        if provided:
            bullets = tuple(
                provided[i] if i < len(provided) and str(provided[i]).strip() else entry.bullets[i]
                for i in range(len(entry.bullets))
            )
        text = " ".join(bullets).strip()
        if text:
            parts.append(f"{entry.header}: {text}")
    return "\n".join(parts)


def _skill_label_key(label_text: str) -> str:
    return label_text.strip().lower().rstrip(":").strip()


def build_skills_rewrite_prompt(
    catalog: TemplateCatalog,
    job_title: str,
    job_description: str,
    rendered_bullets: str,
    keywords: list[str],
    jd_inject_tools: tuple[str, ...] = (),
) -> str | None:
    """Prompt for the strong-aggressive skills rewrite. Returns None when
    there is nothing safe to rewrite — no owned skill_anchors to choose from,
    or no recognizable `\\textbf{Label:} a, b, c` lines in the tail — so the
    caller falls back to the deterministic reorder/inject path instead.

    `jd_inject_tools` (from _extract_jd_tools) is the SAME curated,
    JD-mentioned tool list the deterministic path (_rewrite_skills_for_jd)
    already injects in aggressive modes — a bounded "real tech term the
    listing names" allowance, not open invention. It has to be offered here
    too: without it, this rewrite's stricter "only owned skills" instruction
    would silently strip that already-sanctioned injection back out wherever
    it covers a label.
    """
    if not catalog.skill_anchors:
        return None

    labels: list[str] = []
    in_skills = False
    for line in catalog.tail.splitlines():
        if line.strip().startswith("\\section*"):
            in_skills = "skills" in line.lower()
            continue
        if not in_skills:
            continue
        core = line.rstrip()
        core = core[:-2].rstrip() if core.endswith("\\\\") else core
        label_match = re.match(r"^\\textbf\{([^}]*)\}\s*(.+)$", core)
        if not label_match or "," not in label_match.group(2):
            continue
        # Coursework is the one line that stays the candidate's real record,
        # so it is never offered to this call.
        if "course" in label_match.group(1).lower():
            continue
        labels.append(label_match.group(1))
    if not labels:
        return None

    owned = ", ".join(catalog.skill_anchor_display.get(a, a) for a in catalog.skill_anchors)
    # Narrower budget than the main structured prompt (4500): this call only
    # picks/orders among a known skill list rather than rewriting bullets, so
    # it needs the requirements section, not the full posting — trimming
    # keeps a second per-build LLM call from doubling the JD token cost.
    listing_text = f"Title: {job_title}\n{excerpt_job_description(job_description, 2500)}".strip()
    keywords_line = (
        "\nListing hard-skill keywords (from an earlier pass): " + ", ".join(keywords) + "\n"
        if keywords
        else ""
    )
    bullets_block = (
        "\n\nResume content this Skills section supports (context only — do\n"
        "not copy sentences from here, only judge which owned skills matter\n"
        "most given what the candidate actually did):\n" + rendered_bullets[:4000]
        if rendered_bullets
        else ""
    )
    jd_tools_block = (
        "\n\nThe listing also specifically names these tools, which are NOT in\n"
        "the candidate's owned list above. You MAY additionally use one of\n"
        "these — and ONLY these — in whichever line's category it fits, on\n"
        "top of the owned skills; do not add any other item from outside\n"
        "either list:\n" + ", ".join(jd_inject_tools)
        if jd_inject_tools
        else ""
    )
    return (
        SKILLS_REWRITE_PROMPT_HEADER
        + "\nJob listing:\n"
        + listing_text
        + keywords_line
        + "\n\nCandidate's owned skills (the ONLY allowed output items):\n"
        + owned
        + "\n\nCurrent Skills section labels:\n"
        + json.dumps(labels, ensure_ascii=True)
        + bullets_block
        + jd_tools_block
    )


def normalize_skills_rewrite_payload(
    payload: Any,
    skill_anchors: tuple[str, ...],
    skill_anchor_display: dict[str, str],
    jd_inject_tools: tuple[str, ...] = (),
    allow_unowned: bool = False,
) -> dict[str, list[str]]:
    """Validate + clean a raw `{"skills": {...}}` payload down to only items
    the candidate actually owns PLUS the same curated jd_inject_tools list
    the deterministic path is already allowed to inject in aggressive modes
    (see _extract_jd_tools) — never anything outside that union, keyed by
    normalized label.

    This is the ONE place that decides whether a model's skills-rewrite
    response is usable at all: an empty result means every item it proposed
    was either malformed or outside both allowances (i.e. invented), so the
    caller should treat the response as a failure and try another provider —
    not silently accept a no-op rewrite as "ok". Shared by
    apply_llm_skills_rewrite (render time) and listing._llm_rewrite_skills
    (provider-acceptance time) so both judge the same response the same way.

    `allow_unowned` drops the ownership check entirely. Strong-aggressive sets
    it (profile owner, 2026-08-28: "let everything in skills be fabricated but
    courses"), which makes Skills JD-driven the same way that mode's bullets
    already are — the section stops being a record of what the candidate owns
    and becomes an answer to what the listing asked for. Off by default and off
    in every other mode, where the ownership union stays the real guard.

    Coursework is NOT covered by this: apply_llm_skills_rewrite refuses the
    model's items for any label containing "course" regardless of this flag, so
    the courses line stays the candidate's real classes.
    """
    if not isinstance(payload, dict):
        return {}
    anchor_set = {a.lower() for a in skill_anchors}
    jd_display = {t.lower(): t for t in jd_inject_tools}
    allowed = set() if allow_unowned else anchor_set | set(jd_display)

    normalized: dict[str, list[str]] = {}
    for label, items in payload.items():
        if not isinstance(label, str) or not isinstance(items, list):
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in items:
            if not isinstance(raw, str):
                continue
            item = raw.strip()
            if not item:
                continue
            lowered = item.lower()
            if allowed and lowered not in allowed:
                continue  # neither owned nor a sanctioned JD-tool injection — dropped, never rendered
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(skill_anchor_display.get(lowered) or jd_display.get(lowered) or item)
        if cleaned:
            normalized[_skill_label_key(label)] = cleaned
    return normalized


def apply_llm_skills_rewrite(
    tail: str,
    payload: Any,
    skill_anchors: tuple[str, ...],
    skill_anchor_display: dict[str, str],
    max_items_per_line: int = 0,
    jd_inject_tools: tuple[str, ...] = (),
    generosity: int = 100,
    min_items_per_line: int = 0,
    allow_unowned: bool = False,
) -> str:
    """Overlay the model's per-label item lists onto `tail`, keyed by
    normalized label text. Every item is re-validated against skill_anchors
    (plus the sanctioned jd_inject_tools allowance — see
    normalize_skills_rewrite_payload) here regardless of what the model
    returned — this is the actual invention guard, not the prompt wording. A
    label the model didn't cover, or covered with nothing but
    invented/duplicate items, is left untouched (the caller runs this over an
    already deterministically-reordered tail, so "untouched" still means
    "reordered", never "unrendered").

    `jd_inject_tools` gets a second, stronger guarantee beyond validation:
    _rewrite_skills_for_jd (which produces the `tail` this function receives)
    unconditionally injects every JD tool that matches a line's category, and
    that injection survives its own item cap — a hard guarantee elsewhere in
    this pipeline, not a suggestion. If the model's own response for that
    line omits a tool already sitting there from that injection, it is kept
    anyway; only the model's own choice of items is capped, so a guaranteed
    tool can never be pushed out by ordinary items either.
    """
    normalized = normalize_skills_rewrite_payload(
        payload, skill_anchors, skill_anchor_display, jd_inject_tools,
        allow_unowned=allow_unowned,
    )
    if not normalized:
        return tail
    jd_lower = {t.lower() for t in jd_inject_tools}

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
        items = normalized.get(_skill_label_key(label_match.group(2))) if label_match else None
        # A coursework line is a credential list, not a skills list: the items
        # are classes the candidate sat, and the model has no standing to
        # revise them. Found live 2026-08-28 on a strong-aggressive build,
        # E-Commerce Website Developer Intern (Vallari Decor): the model
        # answered the "Relevant Courses" label with the single item "RESTful
        # APIs" — not a course, and it replaced all twenty real ones. The
        # deterministic reorder still runs on this line, so JD-relevant
        # courses lead; only the model's substitution is refused.
        if label_match and "course" in label_match.group(2).lower():
            items = None
        if label_match and items:
            items = list(items)
            existing_lower = {i.lower() for i in items}
            baseline_items = [i.strip() for i in label_match.group(3).split(",") if i.strip()]
            guaranteed = [
                i for i in baseline_items
                if i.lower() in jd_lower and i.lower() not in existing_lower
            ]
            # Generosity is a percentage of the line the candidate actually
            # has, so it reads off the baseline items rather than the model's
            # proposed list — otherwise a short model response would shrink
            # its own cap and compound the trim.
            cap = skills_line_cap(len(baseline_items), max_items_per_line, generosity)
            if cap > 0 and len(items) + len(guaranteed) > cap:
                items = items[: max(0, cap - len(guaranteed))]
            items = items + guaranteed
            # Floor, applied after the cap: the model answers this call with
            # the items it judges JD-relevant, and on a narrow listing that is
            # one or two — so a 12-item Languages line renders as "Python" and
            # a 20-item courses line as a single course. Measured 2026-08-28
            # over 16 strong-aggressive builds: every build had at least one
            # line cut to a single item, Languages averaging 4.6 of 12.
            # Refill from the profile's own items, in the profile's order,
            # never past the cap — the model's choices keep their lead
            # position, they just stop being the whole line.
            if min_items_per_line > 0:
                floor = min(min_items_per_line, len(baseline_items))
                if cap > 0:
                    floor = min(floor, cap)
                if len(items) < floor:
                    have = {i.lower() for i in items}
                    for candidate in baseline_items:
                        if len(items) >= floor:
                            break
                        if candidate.lower() not in have:
                            items.append(candidate)
                            have.add(candidate.lower())
            rebuilt = f"{label_match.group(1)} {', '.join(items)}"
            out_lines.append(rebuilt + (" \\\\" if has_break else ""))
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


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


def skills_line_cap(item_count: int, max_items: int, generosity: int = 100) -> int:
    """Effective item cap for ONE Skills line, or 0 for "no cap".

    The single place the two skills limits are combined, shared by every path
    that trims a skills line (deterministic reorder, strong-aggressive
    rewrite, LLM-rewrite overlay) so they cannot drift apart and strip
    differently for the same profile.

    `max_items` is the absolute per-line cap (0 = unlimited, legacy default);
    `generosity` is the percentage of the line's OWN length to keep. Both are
    honoured when both are set and the stricter one wins. The percentage
    floors at one item so no line can be emptied outright, and 100 with no
    absolute cap reproduces the pre-existing reorder-only behaviour exactly.
    """
    caps = [cap for cap in (max_items,) if cap > 0]
    if generosity < 100 and item_count > 0:
        caps.append(max(1, (item_count * generosity) // 100))
    return min(caps) if caps else 0


# Ranking tiers 0-2 are the ones backed by real evidence: the listing names
# the course's subject, the course shares a resume entry with something the
# listing named, or the posting's own prose matches it. Tier 3 is the loose
# shared-significant-word tier the ranking comments already call "the last
# resort before canonical filler" — it is what put circuits coursework on
# backend postings via the word "systems", so it does not count as relevant
# for sizing even though it still orders the survivors.
_COURSE_RELEVANT_TIER = 2


def _course_line_length(
    items: list[str],
    rank: Callable[[str], tuple[int, int]],
    bounds: tuple[int, int],
) -> int:
    """How many courses to keep: as many as genuinely fit THIS listing.

    A coursework line is prose, not an ATS keyword surface, so a fixed count
    is the wrong instrument — it shows eight courses whether the posting has
    eight worth showing or two. This sizes the line off the ranking that
    already ordered it: every course at `_COURSE_RELEVANT_TIER` or better is
    kept, and the count therefore moves with the listing.

    `bounds` are a floor and ceiling, not a target. The floor keeps a
    near-miss posting from rendering a one-course line (the section still has
    to look like a resume); the ceiling stops a broad software listing, where
    most of a CE course list legitimately matches, from spending half the page
    on coursework. A line shorter than the floor is returned whole rather than
    padded with items it does not have.
    """
    low, high = bounds
    relevant = sum(1 for item in items if rank(item)[0] <= _COURSE_RELEVANT_TIER)
    return max(1, min(len(items), high, max(low, relevant)))


def _rewrite_skills_for_jd(
    tail: str,
    jd_tools: tuple[str, ...],
    keywords: list[str],
    skill_anchors: tuple[str, ...],
    max_items: int = 0,
    generosity: int = 100,
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

    # Seeded with every item the section ALREADY carries, on any line, so a JD
    # tool the profile lists under a different label is never injected a second
    # time. Per-line dedup alone could not see this: "pytest" is categorised
    # here as a tool but sits on the profile's Frameworks line, and a
    # strong-aggressive build rendered it in both places (found 2026-09-03).
    # Seeding `placed` also keeps such a tool out of the end-of-section
    # fallback append below, which is the other way a duplicate got in.
    placed: set[str] = {item.lower() for item in _collect_skills_items(tail)}
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
                    if tl not in seen_lower and tl not in placed:
                        jd_added.append(tool)
                        seen_lower.add(tl)
                    placed.add(tl)

            kept = relevant + jd_added
            cap = skills_line_cap(len(items), max_items, generosity)
            if cap:
                kept += rest[: max(0, cap - len(kept))]
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


def _skill_key(item: str) -> str:
    """Identity of a Skills item for equality comparisons.

    Punctuation and spacing are noise between two spellings of the SAME tool
    ("Node.js" / "node js" / "NodeJS"), so they are stripped, and a trailing
    "js" is dropped so the framework suffix cannot make one spelling look
    like a different skill. Deliberately does NOT collapse containment:
    "React" and "React Native" are different skills, and "Java" is not
    "JavaScript" — the containment cases that ARE legitimate go through
    _skill_item_backed instead.
    """
    key = re.sub(r"[^a-z0-9+#]", "", item.lower())
    if key.endswith("js") and len(key) > 4:
        key = key[:-2]
    return key


def _skill_item_backed(item: str, candidates: tuple[str, ...]) -> bool:
    """True when a Skills item is the same tool as one of `candidates`.

    Equality is on _skill_key, plus containment in EITHER direction at
    non-alphanumeric boundaries only — "PyTorch" backs the item "PyTorch
    Lightning" and vice versa, but "JavaScript" does not back "Java" and
    "SQL" does not back "MySQL". That boundary rule is the whole point of
    not reusing _matches_skill_anchor here: its bare `normalized in anchor`
    substring test both KEEPS unbacked items (a bullet bolding "JavaScript"
    silently justified a "Java" skills claim) and SUPPRESSES real additions
    (a bolded "JavaScript" counted as already present because the line said
    "Java").
    """
    item_key = _skill_key(item)
    if not item_key:
        return False
    item_l = item.strip().lower()
    for candidate in candidates:
        cand_l = candidate.strip().lower()
        if not cand_l:
            continue
        if _skill_key(candidate) == item_key:
            return True
        if len(item_l) >= 3 and _boundary_contains(cand_l, item_l):
            return True
        if len(cand_l) >= 3 and _boundary_contains(item_l, cand_l):
            return True
    return False


def _boundary_contains(haystack: str, needle: str) -> bool:
    """`needle` occurs in `haystack` delimited by non-alphanumerics.

    Unlike _whole_token_search, a "." counts as a boundary on both sides:
    "react" must be found inside "react.js" (same skill, two spellings)
    while staying out of "javascript" (different skill).
    """
    return bool(
        re.search(
            rf"(?<![a-z0-9+#]){re.escape(needle)}(?![a-z0-9+#])",
            haystack,
            re.IGNORECASE,
        )
    )


def _skill_already_listed(term: str, existing: tuple[str, ...]) -> bool:
    """True when a bolded bullet term is already covered by the Skills line.

    Deliberately one-directional where _skill_item_backed is symmetric: an
    existing "PyTorch Lightning" already covers a bolded "PyTorch", but an
    existing "React" does NOT cover a bolded "React Native" — that is a
    second, real skill, and suppressing it was the add direction's own
    version of the substring bug.
    """
    term_key = _skill_key(term)
    if not term_key:
        return False
    term_l = term.strip().lower()
    for item in existing:
        item_l = item.strip().lower()
        if not item_l:
            continue
        if _skill_key(item) == term_key:
            return True
        if len(term_l) >= 3 and _boundary_contains(item_l, term_l):
            return True
    return False


def _is_jd_relevant_skill(item: str, jd_terms: tuple[str, ...]) -> bool:
    """True when a Skills item is one the listing itself asked for.

    Same whole-token, both-directions test _rewrite_skills_for_jd uses to
    decide which owned items lead the line — reused here so the prune cannot
    delete the very item that pass just promoted.
    """
    il = item.strip().lower()
    if not il:
        return False
    return any(
        il == term
        or _whole_token_search(il, term) is not None
        or _whole_token_search(term, il) is not None
        for term in jd_terms
    )


def _reconcile_skills_and_bullets(
    tail: str,
    rendered_sections: str,
    jd_tools: tuple[str, ...],
    skill_anchors: tuple[str, ...] = (),
    keywords: list[str] | None = None,
    min_items_per_line: int = 0,
    allow_unowned: bool = False,
) -> str:
    """Ensure bidirectional consistency between bolded bullet terms and skills:
    a term bolded in a bullet but missing from Skills gets added, AND a
    Skills item that is never bolded anywhere in the actual bullets gets
    dropped — an isolated skill claim the resume text never backs up is
    exactly the kind of thin, unsupported line-item this pass exists to
    prevent. A line whose PRE-reconciliation content was itself empty is
    removed entirely rather than left as a bare `\\textbf{Label:} \\\\`.

    Floor: if pruning would empty a line that DID have content going in, the
    line falls back to that pre-prune (already relevance-ranked) content
    instead of vanishing — found live: the skills-rewrite LLM and the
    bullet-bolding pass can pick slightly different terms on the same run,
    and losing an entire category to that mismatch is a coverage gap, not a
    fabrication.

    The ADD direction is gated to skill_anchors + jd_tools when either is
    given — same universe the LLM skills rewrite is validated against (see
    normalize_skills_rewrite_payload). Without this, a bullet could bold a
    phrase that only LOOKS tool-shaped (title-case words passing
    _is_boldworthy's style heuristic, e.g. a JD phrase like "Agentic
    Harness" sourced from the model's open-ended `keywords` field) and this
    pass would add it straight into Skills with no ownership check at all —
    found live via a real strong-aggressive trial (2026-08-20). PRUNE has no
    such gate: it only ever removes, so it can't introduce a new claim.

    Every comparison here (already-present, owned/jd-tool, bolded-backing)
    uses _matches_skill_anchor's whole-token/fragment matching, not byte
    equality — a bullet bolding "React.js" must be recognized as backing a
    "React" skills-line item (and vice versa), or a real skill silently
    falls out of Skills just because the bullet and the skills line phrased
    it differently.

    `keywords` (the listing's own hard-skill terms) exempts an item from the
    prune: an owned skill the LISTING explicitly asks for is the single most
    important thing the section can say, and dropping it because no bullet
    happened to bold it undoes the promotion _rewrite_skills_for_jd just
    made — the two passes were contradicting each other.

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

    existing_tail_items: list[str] = []
    in_skills_scan = False
    for line in tail.splitlines():
        if line.strip().startswith("\\section*"):
            in_skills_scan = "skills" in line.lower()
            continue
        if not in_skills_scan:
            continue
        core = line.rstrip()
        core = core[:-2].rstrip() if core.endswith("\\\\") else core
        m = re.match(r"^\\textbf\{([^}]*)\}\s*(.+)$", core)
        if m:
            existing_tail_items.extend(i.strip().lower() for i in m.group(2).split(",") if i.strip())
    existing_tail_items_t = tuple(existing_tail_items)

    jd_terms = tuple(t.lower().strip() for t in jd_tools if t.strip()) + tuple(
        k.lower().strip() for k in (keywords or []) if k.strip()
    )

    add_allowed = tuple(a.lower() for a in skill_anchors) + tuple(t.lower() for t in jd_tools)
    if allow_unowned:
        # Strong-aggressive: the bullets are written from the listing, so a term
        # they bold is a listing term. Blocking it from Skills would leave the
        # page bolding vocabulary the Skills section never claims -- the exact
        # mismatch this pass exists to close.
        add_allowed = ()
    missing: dict[str, list[str]] = {}
    for term, original in original_for.items():
        already_present = _skill_already_listed(term, existing_tail_items_t)
        owned_or_jd = not add_allowed or _matches_skill_anchor(term, add_allowed)
        if not already_present and owned_or_jd:
            cat = category_for.get(term, "tools")
            missing.setdefault(cat, []).append(original)

    out_lines: list[str] = []
    in_skills = False
    for line in tail.splitlines():
        if line.strip().startswith("\\section*"):
            in_skills = "skills" in line.lower()
            out_lines.append(line)
            continue
        if not in_skills:
            out_lines.append(line)
            continue
        stripped = line.rstrip()
        has_break = stripped.endswith("\\\\")
        core = stripped[:-2].rstrip() if has_break else stripped
        label_match = re.match(r"^(\\textbf\{([^}]*)\})\s*(.+)$", core)
        if not label_match:
            out_lines.append(line)
            continue

        label_full = label_match.group(1)
        label_text = label_match.group(2).lower().rstrip(":")
        items = [i.strip() for i in label_match.group(3).split(",") if i.strip()]

        # Add: any bolded bullet term whose category matches this line and
        # isn't already present (fuzzy — two different bullets can bold
        # "React" and "React.js" as separate original_for entries; without
        # this they'd both get appended as near-duplicates of each other).
        for cat, terms in list(missing.items()):
            if cat in label_text or label_text in cat:
                existing_lower: list[str] = [i.lower() for i in items]
                for term_original in terms:
                    if not _skill_already_listed(term_original, tuple(existing_lower)):
                        items.append(term_original)
                        existing_lower.append(term_original.lower())
                del missing[cat]
                break

        # Prune: a skill with nothing bolded for it anywhere in the actual
        # bullets is an isolated claim the resume text never backs up —
        # UNLESS the listing itself asked for it, in which case keeping it is
        # the entire job of this section. Matched against the bolded spans at
        # token boundaries (see _skill_item_backed) so a "React" skills item
        # survives a bullet that bolded "React.js", while a "Java" item is
        # not rescued by an unrelated "JavaScript" bold.
        pre_prune_items = items
        bolded_terms = tuple(original_for.keys())
        items = [
            i for i in items
            if _skill_item_backed(i, bolded_terms) or _is_jd_relevant_skill(i, jd_terms)
        ]
        if not items:
            # Floor: this line's bolding just didn't line up with this
            # particular bullet rewrite (common when the skills-rewrite LLM
            # and the bullet-bolding pass pick slightly different terms on a
            # given run) — that's a coverage gap, not a fabrication. Losing
            # the whole category is worse than keeping its already
            # relevance-ranked pre-reconciliation content, so fall back to
            # it instead of deleting the line outright.
            items = pre_prune_items
        elif min_items_per_line > 0 and len(items) < min_items_per_line:
            # Same floor, raised from "not empty" to a real minimum, and only
            # in strong-aggressive (the caller passes 0 everywhere else, so
            # normal and --aggressive keep the strict prune).
            #
            # Why the strict prune is wrong for THIS mode: a Skills section is
            # a summary of competencies, not an index of the bullets. Real
            # resumes list an owned skill whether or not a bullet happens to
            # bold it, and requiring bullet backing produced lines reading
            # "Tools: Git" — measured 2026-08-28 across two slices, build 006
            # (Power Platform Developer Intern) rendered Frameworks, Tools and
            # Platforms at one item each from a profile owning 14, 9 and 5.
            # Every item refilled here is the candidate's own, already
            # relevance-ranked by _rewrite_skills_for_jd, so this adds no
            # claim the profile did not already make.
            #
            # The prune still decides ORDER: backed and JD-relevant items were
            # filtered into `items` first and keep the lead position, which is
            # what a skimming recruiter and an ATS both read first. The cap
            # still decides the ceiling. This only stops the floor falling to
            # one.
            for candidate in pre_prune_items:
                if len(items) >= min_items_per_line:
                    break
                if candidate not in items:
                    items.append(candidate)
        if not items:
            continue  # pre-reconciliation content was itself empty — nothing to show

        out_lines.append(f"{label_full} {', '.join(items)}" + (" \\\\" if has_break else ""))

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


# Opening-verb synonym groups for the diversification pass. Every alternative
# is the same tense, register and claim strength as its siblings: the swap is
# pure style, so it must never change what the bullet asserts. Ownership verbs
# (Led/Owned/Managed…) are deliberately absent in BOTH directions — they are
# grounding-gated by OWNERSHIP_VERB_SWAPS and a style pass must not smuggle one
# in, nor rewrite one that the grounding check already approved.
OPENING_VERB_SYNONYMS: tuple[tuple[str, ...], ...] = (
    # The build family needs the most alternatives: six of the profile's 33
    # canonical bullets open with "Built", and a 15-bullet page exhausted a
    # six-word group by bullet nine (measured), falling back to a repeat.
    (
        "Built", "Developed", "Created", "Engineered", "Implemented",
        "Constructed", "Programmed", "Prototyped", "Authored", "Assembled",
    ),
    ("Deployed", "Shipped", "Released", "Rolled out", "Launched", "Published"),
    ("Integrated", "Connected", "Wired", "Linked", "Bridged"),
    ("Designed", "Architected", "Modelled", "Drafted", "Specified", "Mapped"),
    ("Automated", "Scripted", "Streamlined"),
    ("Tested", "Validated", "Verified", "Debugged"),
    ("Maintained", "Documented", "Sustained"),
    ("Analyzed", "Assessed", "Evaluated", "Investigated"),
    ("Optimized", "Tuned", "Refined", "Improved"),
    ("Tracked", "Monitored", "Logged", "Measured"),
    ("Taught", "Coached", "Mentored", "Trained"),
    ("Communicated", "Presented", "Explained", "Briefed"),
    ("Performed", "Conducted", "Ran", "Carried out"),
)
_OPENING_VERB_GROUP: dict[str, tuple[str, ...]] = {
    word.lower(): group for group in OPENING_VERB_SYNONYMS for word in group
}
# Matches the bullet's first word, seeing through a bolded opener.
_LEAD_VERB_PATTERN = re.compile(r"^(?:\\(?:textbf|textit|emph)\{)?([A-Za-z]+)")


def diversify_opening_verbs(
    ordered_keys: list[tuple[str, int]],
    resolved: dict[tuple[str, int], str],
) -> list[str]:
    """Swap repeated opening verbs for unused synonyms, in document order.

    Six of the profile's 33 canonical bullets open with "Built", so a rendered
    page reliably repeated an opener 25-36% of the time — which reads
    machine-generated however good each line is. Prompt instructions did not
    move this (measured: no change across seven builds), and the existing
    cross-bullet guard's remedy is revert-to-canonical, which cannot help when
    the canonical text is itself the source of the repetition.

    Only the leading verb token changes: no tool, number, or claim is touched,
    so nothing the validator established can be undone here. Returns a list of
    human-readable notes for the render report.
    """
    notes: list[str] = []
    used: set[str] = set()
    for key in ordered_keys:
        text = resolved.get(key, "")
        match = _LEAD_VERB_PATTERN.match(text)
        if not match:
            continue
        verb = match.group(1)
        low = verb.lower()
        if low not in used:
            used.add(low)
            continue
        group = _OPENING_VERB_GROUP.get(low)
        if not group:
            continue  # unknown verb (or an ownership verb): leave it alone
        replacement = next(
            (word for word in group if word.lower() not in used), None
        )
        if replacement is None:
            continue  # every synonym already spent; a repeat beats a wrong word
        used.add(replacement.lower())
        resolved[key] = (
            text[: match.start(1)] + replacement + text[match.end(1) :]
        )
        notes.append(f"{key[0]}[{key[1]}]: opening verb {verb} -> {replacement}")
    return notes


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


# Course names and other prose-shaped Skills items ("Object Oriented Eng
# Analysis and Design") never match the tool vocabulary, and entry adjacency
# cannot see them because they do not appear in bullet text. Measuring the
# word overlap that actually exists (2026-08-24, over the real sample
# listings) showed why naive overlap ranking is a trap: EVERY hit was a
# generic word — "design" on a Shopify backend posting, "systems" on an AMD
# posting, "analysis" on an ML posting. Ranking on those would have promoted
# "Microprocessor Systems" onto a backend web listing purely because the JD
# said "systems". So a shared word only counts when it is not generic, reusing
# _GENERIC_BOLD_WORDS — the list already curated from live trials for exactly
# this "looks technical, is ordinary English" problem.
_COURSE_STOPWORDS = frozenset({
    "and", "the", "for", "with", "eng", "engineering", "introduction",
    "intro", "advanced", "principles", "fundamentals", "topics", "course",
    "courses", "study", "studies",
    # "Computer Science degree" is boilerplate in essentially every software
    # posting, so "computer" carries no signal about WHICH posting this is.
    # Measured 2026-08-24: it was the only word hitting on all four sample
    # listings, which floated Computer Networks and Computer Organization to
    # the front of every software render regardless of subject.
    "computer",
})


def _item_match_parts(item_lowered: str) -> list[str]:
    """The fragments of a Skills item that may stand in for the whole item.

    Splitting exists for tool spellings: the resume line says
    "JavaScript/Node.js" and the posting says "Node.js". Only "/" and ","
    split, because only they mean "the same skill, written another way".
    " and " does NOT split: it is prose conjunction, and treating it as a
    spelling variant is what let "Signals and Systems" match on the fragment
    "signals" and "Object Oriented Eng Analysis and Design" match on
    "design". Those fragments hit ordinary words in postings and in the
    candidate's own bullets, which promoted whole courses to the tool tiers —
    found live 2026-08-24, once the real TMU course list was loaded: the ML
    entry says "signals" in a bullet, so a Signals and Systems course joined
    that entry's tool adjacency set and led the Cohere render.

    A surviving fragment is additionally dropped when it is a generic word or
    a course stopword, so a comma- or slash-separated prose fragment cannot
    sneak back in. Returns [] when nothing survives, which correctly means
    "this item cannot be matched by fragment".

    Prose items are not stranded by this: they still rank through
    _prose_item_matches_jd, which scores distinctive WORDS and is guarded
    against exactly the generic vocabulary this splitter refuses to trust.
    """
    parts = [p.strip() for p in re.split(r"[/,]", item_lowered) if p.strip()]
    if len(parts) < 2:
        return [item_lowered] if item_lowered.strip() else []
    return [
        part
        for part in parts
        if part not in _GENERIC_BOLD_WORDS and part not in _COURSE_STOPWORDS
    ]


# Words that are domain signal in one phrase and marketing filler in another,
# mapped to the phrases that make them count. A flat entry in
# _GENERIC_BOLD_WORDS cannot express this: banning "digital" outright stops
# "Digital Systems" promoting on an electronics posting that says "debug
# digital logic", which is exactly the promotion that ranking exists for.
# Added 2026-08-28 after an aggressive-mode slice put "Digital Systems" and
# "Digital Systems Engineering" — VLSI coursework — at the FRONT of the
# courses line on a full-stack AI posting (prose use of the word) and an
# e-commerce web posting ("digital marketing").
_CONTEXT_DEPENDENT_JD_WORDS = {
    "digital": (
        "digital logic", "digital design", "digital circuit", "digital circuits",
        "digital system", "digital systems", "digital electronics",
        "digital signal", "digital hardware",
    ),
}


def _word_counts_as_subject(word: str, jd_lower: str) -> bool:
    """Does this word carry subject signal FOR THIS POSTING?

    Shared by the two tiers that rank on words, so they cannot disagree about
    what counts as subject matter -- they already did. The prose tier learned
    that "digital" only counts next to digital logic/circuits/electronics; the
    word-overlap tier kept promoting "Digital Systems" anyway, because it
    compares against the model's KEYWORDS and never consulted the rule.
    Measured 2026-08-28 across two aggressive slices: the keywords "Digital
    Forensics" (financial crimes), "digital PR" and "digital customer
    experience" (two marketing postings) each dragged semiconductor coursework
    to the front of the courses line.
    """
    if word in _COURSE_STOPWORDS or word in _GENERIC_BOLD_WORDS:
        return False
    phrases = _CONTEXT_DEPENDENT_JD_WORDS.get(word)
    return any(phrase in jd_lower for phrase in phrases) if phrases else True


def _prose_item_matches_jd(item_lowered: str, jd_lower: str) -> bool:
    """A multi-word, non-tool item shares a DISTINCTIVE word with the listing.

    Scoped to prose-shaped items on purpose: a known tool already matches by
    name at the top rank, and letting it match by word would promote "Power
    BI" on any listing that says "power".
    """
    if not jd_lower or item_lowered in _KNOWN_TECH_TERMS:
        return False
    words = [w for w in re.findall(r"[a-z][a-z0-9+#.]*", item_lowered) if len(w) >= 4]
    if len(words) < 2:
        return False
    return any(
        _whole_token_search(jd_lower, word) is not None
        and _word_counts_as_subject(word, jd_lower)
        for word in words
    )


def _collect_skills_items(tail: str) -> list[str]:
    """Every comma-separated item on the Skills section's labelled lines."""
    items: list[str] = []
    in_skills = False
    for line in tail.splitlines():
        if line.strip().startswith("\\section*"):
            in_skills = "skills" in line.lower()
            continue
        if not in_skills:
            continue
        core = line.rstrip()
        core = core[:-2].rstrip() if core.endswith("\\\\") else core
        match = re.match(r"^\\textbf\{[^}]*\}\s*(.+)$", core)
        if match:
            items.extend(i.strip() for i in match.group(1).split(",") if i.strip())
    return items


def _entries_adjacent_items(
    resume_text: str,
    skills_items: list[str],
    is_listing_matched: Callable[[str], bool],
) -> set[str]:
    """Skills items sharing a resume entry with a listing-matched item.

    The rendered document is split on the `% [category]` markers that head
    every entry, so each chunk is one project or job. An entry that already
    names something the listing asked for is evidence that THIS entry is the
    work the listing cares about — so the other tools named in it are the
    candidate's nearest relevant skills, ahead of canonical filler from an
    unrelated part of their background.

    Entries COMPETE rather than each clearing an absolute bar. Measured on
    this profile (2026-08-24): "Python" appears in 5 of 12 entries, so any
    fixed ubiquity threshold generous enough to keep real matches still let a
    lone Python hit heat the ML project on a FRONTEND listing — which handed
    that resume scikit-learn and PyTorch Lightning. Scoring each entry by how
    many listing-matched items it contains and heating only the strongest
    separates them cleanly: the frontend internship scores 5 (React,
    TypeScript, Flask, Docker, Python) and the ML entry scores 1, so only the
    frontend entry contributes. The floor of 2 matters just as much — when
    every entry scores 1, nothing is distinctive and no entry is hot, so
    adjacency stays silent instead of inventing a relevance signal out of one
    ubiquitous language.

    Returns lowercase items. Empty when there is no resume text (callers
    outside the render path), which restores the old two-tier behavior.
    """
    if not resume_text or not skills_items:
        return set()

    chunks = [c for c in re.split(r"^\s*%\s*\[", resume_text, flags=re.MULTILINE) if c.strip()]
    if not chunks:
        return set()

    def _present_in(chunk: str, item_lowered: str) -> bool:
        # Match the item's parts too: an entry writing "Node.js" should
        # count for the skills line's "JavaScript/Node.js". _item_match_parts
        # drops generic fragments, without which a bullet containing the word
        # "systems" made "Signals and Systems" look like part of that entry's
        # toolset (2026-08-24, seen on the Cohere and AMD renders).
        return any(
            len(part) >= 3 and _whole_token_search(chunk, part) is not None
            for part in _item_match_parts(item_lowered)
        )

    per_chunk: list[list[str]] = []
    for chunk in chunks:
        per_chunk.append([i.lower() for i in skills_items if _present_in(chunk, i.lower())])

    scores = [
        len({item for item in present if is_listing_matched(item)})
        for present in per_chunk
    ]
    top = max(scores, default=0)
    # max(2, ...) is the floor described above; top - 1 keeps a close second
    # entry in play, since a listing routinely matches two related projects.
    threshold = max(2, top - 1)

    adjacent: set[str] = set()
    for present, score in zip(per_chunk, scores):
        if score >= threshold:
            adjacent.update(present)
    return adjacent


def reorder_skills_for_keywords(
    tail: str,
    keywords: list[str],
    max_items_per_line: int = 0,
    jd_text: str = "",
    resume_text: str = "",
    generosity: int = 100,
    course_bounds: tuple[int, int] | None = None,
) -> str:
    """Reorder items within Skills-section lines so listing-relevant tools lead.

    Format-preserving: same labels, same line breaks — only the order of
    comma-separated items within each `\\textbf{Label:} a, b, c \\\\` line
    changes. Items matching a listing keyword move to the front, then
    everything else in canonical order. Nothing is ever added, so there is no
    invention risk. When `max_items_per_line` > 0, unmatched trailing items
    beyond that cap are cut (matched items always survive, even past the cap).

    `jd_text` is the listing's own body, and it is the difference between a
    Skills section that is tailored and one that only looks tailored. The
    `keywords` list is the selection pass's top 8-10 terms, so a tool the
    posting genuinely asks for but that placed 11th never moved at all —
    across six real trial renders (2026-08-24: ABB electronics, ZTR
    mechanical, CMiC software, a math tutor posting) the emitted Skills
    lines were nearly IDENTICAL, because most items tied at the bottom rank
    and the stable sort just kept canonical order. Matching against the full
    posting text breaks those ties with the listing's own vocabulary.

    `resume_text` is the rendered entries, and it supplies the ADJACENCY rank
    that sits between "the listing named this" and "canonical filler". Once
    the listing-matched items are placed, the cap used to fill its remaining
    slots in canonical order, which is how a real trial (2026-08-24, the
    Cohere ML listing) kept OpenGL, WebRTC, JavaFX and React on an
    ML-engineering resume while CUTTING the owned PyTorch Lightning and
    NumPy. Adjacency is read off the candidate's own entries rather than a
    hand-curated domain table: an entry that already contains a
    listing-matched tool is a "hot" entry, and the other tools appearing in
    that same entry are the ones whose work this listing actually cares
    about. That keeps the Skills line consistent with the bullets a
    recruiter is reading directly above it, and it needs no per-domain
    configuration to work on a profile nobody anticipated.

    Promotion from `jd_text` is deliberately narrower than the keyword match:
    an item earns it only by occurring as a whole token in the posting, and
    only when the item name cannot be ordinary English (`_SKIP_JD_INJECT`
    plus a length floor) — "Go" or "C" appearing in prose must not outrank a
    tool the listing actually named.
    """
    # Nothing to order by — but a trim is still owed when a cap is configured.
    # Ordering and trimming are independent: a profile that asks for a lean
    # Skills section must get one even on a posting that yielded no keywords,
    # otherwise the section silently renders at full length exactly when the
    # tailoring is weakest.
    if not keywords and not jd_text:
        if max_items_per_line <= 0 and generosity >= 100:
            return tail

    keyword_lowered = [k.lower().strip() for k in keywords if k.strip()]

    def _significant_words(text: str) -> set[str]:
        """Words that carry subject signal, for the loosest matching tier.

        The word-overlap tier is the last resort before canonical filler, and
        without this filter it fires on whatever generic noun two multi-word
        phrases happen to share. Measured 2026-08-24 with production-shaped
        keywords: "distributed systems" and "systems programming" overlap
        "Embedded Systems Design" and "Digital Systems" on the single word
        "systems", which promoted two circuits-adjacent courses onto a
        backend-web posting and a GPU-driver posting. Same generic-word
        table the stricter tiers already use, so all four tiers now agree on
        what counts as subject matter.
        """
        return {
            word
            for word in re.findall(r"[a-z0-9.+#]+", text)
            if word not in _GENERIC_BOLD_WORDS and word not in _COURSE_STOPWORDS
        }

    keyword_words = [_significant_words(k) for k in keyword_lowered]

    # Whole-token matching everywhere: substring matching would let the item
    # "C" float to the front for any keyword containing the letter c
    # ("react", "recruitment").
    def _contains_whole(haystack: str, needle: str) -> bool:
        return bool(
            re.search(rf"(?<![a-z0-9.+#]){re.escape(needle)}(?![a-z0-9.+#])", haystack)
        )

    jd_lower = jd_text.lower()

    def _named_in_jd(item_lowered: str) -> bool:
        """The posting itself names this exact tool.

        Guarded three ways because the JD is prose, not a tool list. Items
        whose plain-English sense dominates real postings are excluded by the
        same table the JD-injection path uses; a 2-character item ("C", "R",
        "Go") only counts when its own punctuation makes it unambiguous
        ("C++", "C#"); and a generic word never counts at this tier at all.

        That last guard exists because this function is also applied to the
        PARTS of a multi-word item, which is right for "JavaScript/Node.js"
        but was catastrophic for prose: splitting "Object Oriented Eng
        Analysis and Design" on " and " handed "design" to this function,
        and a posting saying "design" then ranked that course at the TOP
        tier — jumping the whole generic-word guard built for exactly this.
        Found live (2026-08-24) once the real TMU course list was loaded:
        "Signals and Systems" led the Cohere ML and AMD renders because its
        "systems" fragment matched ordinary JD prose.
        """
        if not jd_lower or item_lowered in _SKIP_JD_INJECT:
            return False
        if item_lowered in _GENERIC_BOLD_WORDS or item_lowered in _COURSE_STOPWORDS:
            return False
        if len(item_lowered) < 3 and item_lowered.isalnum():
            return False
        # _whole_token_search, not the local _contains_whole: the latter puts
        # "." inside the token character class (right for keyword matching,
        # where "node" must not match inside "node.js"), which also means a
        # posting's sentence-final "...written in Python." never matched the
        # item "Python" at all.
        return _whole_token_search(jd_lower, item_lowered) is not None

    def _listing_matched(item_lowered: str) -> bool:
        for keyword in keyword_lowered:
            if keyword == item_lowered:
                return True
            # The top tier was the one place that never consulted the generic
            # -word table the other three tiers share, so a bare extracted
            # keyword like "systems" or "data" matched every item containing
            # it. Measured 2026-09-03 over the lab's 98 real postings: an EHS
            # co-op ("systems" among its keywords) ranked NINE courses at this
            # tier -- Signals and Systems, Control Systems, Microprocessor
            # Systems and the rest -- and a warranty co-op eleven. A phrase
            # keyword is unaffected: "digital systems" is not itself a generic
            # word, only the bare noun is.
            if keyword in _GENERIC_BOLD_WORDS or keyword in _COURSE_STOPWORDS:
                continue
            if _contains_whole(item_lowered, keyword) or _contains_whole(keyword, item_lowered):
                return True
        # A multi-word item ("JavaScript/Node.js", "Microsoft Suite") is
        # named by the posting when any of its parts is: the listing says
        # "Node.js", the resume line says "JavaScript/Node.js".
        return any(_named_in_jd(part) for part in _item_match_parts(item_lowered))

    all_items = _collect_skills_items(tail)
    adjacent = _entries_adjacent_items(resume_text, all_items, _listing_matched)

    def item_rank(item: str) -> tuple[int, int]:
        item_lowered = item.lower()
        if _listing_matched(item_lowered):
            return (0, 0)
        if item_lowered in adjacent:
            return (1, 0)
        if _prose_item_matches_jd(item_lowered, jd_lower):
            return (2, 0)
        # Context-dependent words are dropped here too: a keyword like
        # "digital PR" must not carry semiconductor coursework onto a marketing
        # posting just by sharing the word "digital".
        item_words = {
            word for word in _significant_words(item_lowered)
            if _word_counts_as_subject(word, jd_lower)
        }
        for words in keyword_words:
            if item_words & words:
                return (3, 0)
        return (4, 0)

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
            is_course_line = "course" in label_match.group(1).lower()
            if course_bounds and is_course_line:
                items = items[: _course_line_length(items, item_rank, course_bounds)]
            else:
                cap = skills_line_cap(len(items), max_items_per_line, generosity)
                if cap > 0 and len(items) > cap:
                    matched = sum(1 for item in items if item_rank(item)[0] < 1)
                    items = items[: max(cap, matched)]
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
    llm_domain_gated: frozenset[str] = frozenset(),
    llm_rewritten_skills: Any = None,
) -> tuple[str, RenderReport]:
    """Deterministically render the resume from template parts + selection.

    `llm_rewritten_skills` is the strong-aggressive LLM skills-rewrite
    payload ({label: [items]]}), or None when that pass was skipped/failed —
    see structured.build_skills_rewrite_prompt / listing._llm_rewrite_skills.
    It only ever overlays labels it covers with owned-skill items; the
    deterministic reorder/inject path underneath always runs first, so a
    partial or missing payload still yields a fully reordered tail.

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

    # Domain-fit gate: a specialist-tagged entry only earns ranking-eligibility
    # when the JD text itself names its domain, not just a generic shared verb.
    # This runs BEFORE the budget loop and folds into excluded_ids so it gets
    # the exact same capacity-rollback safety net as a model-chosen exclusion.
    domain_gated_ids: set[str] = set()
    if jd_text:
        jd_lower = jd_text.lower()
        for entry in catalog.entries:
            if entry.entry_id in excluded_ids:
                continue
            terms = [
                term
                for category in entry.categories
                for term in config.specialist_categories.get(category, ())
            ]
            if terms and not any(_whole_token_search(jd_lower, term) for term in terms):
                domain_gated_ids.add(entry.entry_id)
    # General-purpose companion to the term-list gate above: entries the LLM
    # domain-fit audit flagged (see DOMAIN_FIT_AUDIT_PROMPT_HEADER), which
    # needs no curated category/term list and so covers domains the profile
    # owner never configured. Reported through the same domain_gated_entries
    # field — both sources get the identical capacity-rollback safety net.
    domain_gated_ids |= {eid for eid in llm_domain_gated if eid in known_ids}
    excluded_ids |= domain_gated_ids

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
            #
            # Charged only ABOVE the entry count a normal page already
            # carries. max_total_bullet_chars was calibrated on normal-mode
            # renders, which sit at ~5-6 entries and pay for those headers
            # implicitly — billing strong-aggressive for all of them again was
            # double-counting, and it is what held the mode to 9-10 bullets on
            # a page normal mode fills with 14-15 (measured 2026-08-26 over the
            # 14-JD corpus: ~1,500 characters of text against ~2,170).
            entry_count = sum(1 for e in included if kept_indices[e.entry_id])
            chars += 110 * max(0, entry_count - STRONG_FREE_ENTRY_HEADERS)
        return chars

    def _estimated_bullet_lines() -> int:
        """Rendered LINES, not characters — what the page actually runs out of.

        A character total treats a 214-character bullet as 1.4x a 156-character
        one; on the page it is 3 wrapped lines against 2, because every bullet
        wastes part of its last line. Measured 2026-08-26 across 28 compiled
        builds, that partial-line waste is the whole gap between the two
        models: every one-page build came in at or under 31 bullet lines, and
        the single build that spilled to a second page sat at 35 while still
        under the character budget.
        """
        lines = 0
        for entry in included:
            provided = effective_bullets.get(entry.entry_id, [])
            for index in kept_indices[entry.entry_id]:
                if index < len(provided) and provided[index].strip():
                    text = provided[index][: config.max_bullet_chars]
                else:
                    text = entry.bullets[index]
                visible = _LATEX_MARKUP_PATTERN.sub("", text)
                lines += max(1, -(-len(visible) // BULLET_LINE_CHARS))
        return lines

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
        and (
            _estimated_chars() > config.max_total_bullet_chars
            or (
                config.strong_aggressive
                and _estimated_bullet_lines() > MAX_BULLET_LINES
            )
        )
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
    if config.strong_aggressive:
        # Depth over breadth: in strong-aggressive a SHORTER bullets list is an
        # intentional "fewer, heavier bullets" choice — drop the unwritten
        # trailing slots instead of padding them with canonical text (canonical
        # filler is off-message on a fabricated page).
        #
        # The discount is conditional on actually paying for it. Measured
        # 2026-08-25: strong-aggressive returned 9 bullets per page against
        # normal mode's 14, at the SAME average length (~130 chars) — it took
        # the shorter list without writing heavier bullets, leaving a third of
        # the page empty. So honour the short list only when what was written
        # is genuinely denser than the canonical text it replaced.
        #
        # A per-bullet MEAN-length ratio was the wrong meter for that, and is
        # why the page still came back short: dropping 5 slots for 3 bullets
        # clears a 1.15x mean while writing 30% less text than the slots it
        # replaced. The trade is measured in total characters instead — the
        # written bullets must carry at least as much text as the canonical
        # slots they stand in for (plus the ratio's margin), so a shorter list
        # can never shrink the page's text volume.
        #
        # Volume parity alone is still not enough: it holds entry by entry
        # while the PAGE ends up with too few bullets to look filled, because
        # nothing was looking at the total. Trims are therefore collected
        # first, then applied heaviest-first only while the page stays at or
        # above min_visible_bullets.
        candidate_trims: list[tuple[float, str, list[int]]] = []
        for entry in ordered_entries:
            provided = effective_bullets.get(entry.entry_id, [])
            if not provided:
                continue
            trimmed_kept = [
                i for i in kept_indices[entry.entry_id] if i < len(provided)
            ]
            if not trimmed_kept or len(trimmed_kept) == len(kept_indices[entry.entry_id]):
                continue
            written_chars = sum(
                len(str(provided[i])) for i in trimmed_kept if str(provided[i]).strip()
            )
            canonical_chars = sum(
                len(entry.bullets[i]) for i in kept_indices[entry.entry_id]
            )
            if written_chars < canonical_chars * STRONG_DEPTH_LENGTH_RATIO:
                continue
            ratio = written_chars / canonical_chars if canonical_chars else 0.0
            candidate_trims.append((ratio, entry.entry_id, trimmed_kept))

        total_kept = sum(len(v) for v in kept_indices.values())
        for ratio, entry_id, trimmed_kept in sorted(
            candidate_trims, key=lambda item: item[0], reverse=True
        ):
            dropped = len(kept_indices[entry_id]) - len(trimmed_kept)
            if total_kept - dropped < config.min_visible_bullets:
                continue
            kept_indices[entry_id] = trimmed_kept
            total_kept -= dropped

    for entry in ordered_entries:
        provided = effective_bullets.get(entry.entry_id, [])
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

    # Runs AFTER the consistency pass: that pass reverts bullets to canonical
    # text, and the canonical text is where most of the opener repetition comes
    # from — diversifying earlier would be silently undone by every revert.
    diversify_opening_verbs(ordered_keys, resolved)

    # Canonical bullets never pass through validate_tailored_bullet, so without
    # this they render the template's raw emphasis — up to 6 bolds a bullet,
    # including descriptive spans like \textbf{PCB circuits} — right beside
    # tailored bullets just held to the tool-name policy. Two conventions on one
    # page is worse than either alone.
    #
    # This runs AFTER the consistency pass on purpose: that pass reverts
    # bullets to canonical text (`_revert` -> resolved[key] = canonicals[key]),
    # so normalising earlier gets silently undone. Found live (2026-08-11, CMiC
    # Software Engineer Co-op): a reverted bullet rendered with 5 bolds under a
    # cap of 3. Strong-aggressive is uncapped by design and keeps its own
    # JD-relevance pass in _render_entry.
    if not config.strong_aggressive:
        for key, is_tailored in tailored_flags.items():
            if is_tailored:
                continue
            text = _demote_unworthy_bolds(resolved[key], catalog.skill_anchors)
            text = _cap_bold_density(
                text, listing_keywords, config.max_bold_per_bullet
            )
            resolved[key] = _cap_bold_per_clause(
                text, listing_keywords, config.max_bold_per_clause
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
                # _unbold_non_jd_terms keeps any phrase that appears in the JD
                # text, and JDs are full of ordinary words — a live trial
                # (2026-08-11, CMiC Software Engineer Co-op) came back with
                # \textbf{enterprise} three times plus \textbf{debugging},
                # \textbf{refactoring}, \textbf{sprint ceremonies}. This mode
                # is uncapped in bold COUNT by design; that is not a licence to
                # bold prose, so the name test still applies here.
                text = _demote_unworthy_bolds(text, catalog.skill_anchors)
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
            # selection.keywords is whatever the model called a "hard skill",
            # which in practice includes plain process language — a live ABB
            # trial bolded "maintenance plans", "continuous improvement", and
            # "production engineering" this way, re-introducing exactly what
            # the gate above had just removed. Emphasis is added here, so the
            # name test has to apply here too.
            bold_terms = tuple(
                term for term in bold_terms if _is_boldworthy(term, catalog.skill_anchors)
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
    ]
    # Non-entry sections that preceded the first tagged section keep their
    # place at the top of the page (Education, on Jake-style templates).
    if catalog.head_sections:
        parts += ["", catalog.head_sections]
    parts += ["", "\n\n".join(section_chunks)]
    if catalog.tail:
        if config.strong_aggressive:
            tail = _rewrite_skills_for_jd(
                catalog.tail, jd_inject_tools, selection.keywords, catalog.skill_anchors,
                max_items=config.max_skill_items_per_line,
                generosity=config.skills_generosity,
            )
            if llm_rewritten_skills:
                tail = apply_llm_skills_rewrite(
                    tail,
                    llm_rewritten_skills,
                    catalog.skill_anchors,
                    catalog.skill_anchor_display,
                    max_items_per_line=config.max_skill_items_per_line,
                    jd_inject_tools=jd_inject_tools,
                    generosity=config.skills_generosity,
                    # Half the mode's own cap. Strong-aggressive requires every
                    # bolded keyword to also appear in Skills, so a section the
                    # model has pared to one item per line works against the
                    # rest of the mode.
                    min_items_per_line=max(4, config.max_skill_items_per_line // 2),
                    # Skills are fabricated from the listing in this mode, the
                    # same as its bullets. Courses stay real -- the label check
                    # inside this function refuses them regardless.
                    allow_unowned=True,
                )
            all_rendered = "\n\n".join(section_chunks)
            tail = _reconcile_skills_and_bullets(
                tail,
                all_rendered,
                jd_inject_tools,
                catalog.skill_anchors,
                keywords=selection.keywords,
                # Same floor the LLM overlay uses; without it the prune here
                # undoes that refill on the very next pass.
                min_items_per_line=max(4, config.max_skill_items_per_line // 2),
                allow_unowned=True,
            )
        else:
            tail = reorder_skills_for_keywords(
                catalog.tail,
                selection.keywords,
                config.max_skill_items_per_line,
                jd_text=jd_text,
                resume_text="\n\n".join(section_chunks),
                generosity=config.skills_generosity,
                course_bounds=config.course_item_bounds,
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
        domain_gated_entries=sorted(domain_gated_ids),
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
            # Bound the window to THIS entry. A fixed 3000-character slice ran
            # past the entry's own bullets into the next entries and harvested
            # their \textbf{} headers, so adding one untagged project inserted
            # "Frontend Development Intern", "Electrical Team Member" and
            # "Machine Learning Intern" into SKILL ANCHORS as if they were
            # tools — anchors that then feed tool grounding and bolding.
            body_start = entry_start + len(header_line)
            end_match = re.search(
                r"\\end\{itemize\}|\\resumeItemListEnd|^%\s*\[",
                uncommented[body_start : body_start + 3000],
                re.MULTILINE,
            )
            entry_chunk = uncommented[
                body_start : body_start + (end_match.end() if end_match else 3000)
            ]
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


# (template, baseinfo) paths -> (file-mtime stamp, parsed catalog). A single
# command resolves the same profile 2-3 times (structured-or-legacy check in
# handlers, then the rewrite, then cover/bootstrap), each of which used to
# re-read and re-parse from disk. Keyed by the mtimes of all four profile
# files, so any edit — including load's own on-disk auto-sync — invalidates.
_PROFILE_CATALOG_CACHE: dict[tuple[str, str], tuple[tuple, TemplateCatalog | None]] = {}


def _profile_cache_stamp(template_path: Path, baseinfo_path: Path) -> tuple:
    # mtime_ns + size, not float mtime: rapid successive writes (tests, the
    # auto-sync) can land inside float-mtime granularity and go stale.
    def _stat(path: Path) -> tuple[int, int]:
        try:
            st = path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return (-1, -1)

    return (
        _stat(template_path),
        _stat(baseinfo_path),
        _stat(template_path.parent / STRUCTURED_CONFIG_FILENAME),
        _stat(template_path.parent / STRUCTURED_GUIDANCE_FILENAME),
    )


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

    Results are cached by file mtimes. Returns a DEEP COPY of the cached
    catalog: callers mutate render_config in place (the aggressive flags), and
    a shared instance would leak one build's mode into the next.
    """
    cache_key = (str(template_path), str(baseinfo_path))
    stamp = _profile_cache_stamp(template_path, baseinfo_path)
    cached = _PROFILE_CATALOG_CACHE.get(cache_key)
    if cached is not None and cached[0] == stamp:
        return copy.deepcopy(cached[1])

    catalog = _load_structured_profile_uncached(template_path, baseinfo_path)
    # Re-stamp after loading: the auto-sync inside may have rewritten files.
    _PROFILE_CATALOG_CACHE[cache_key] = (
        _profile_cache_stamp(template_path, baseinfo_path),
        catalog,
    )
    return copy.deepcopy(catalog)


def _load_structured_profile_uncached(
    template_path: Path,
    baseinfo_path: Path,
) -> TemplateCatalog | None:
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


# ---------------------------------------------------------------------------
# Metric injection (2026-08-25)
#
# Quantified achievement is the strongest single signal in recruiter guidance,
# and the pipeline sat at ~38% of bullets carrying a figure. Permission alone
# did not move it: strong-aggressive has always allowed invented numbers and
# produced the FEWEST of them (11%), and enabling allow_invented_metrics plus
# rewriting three prompt rules shifted the corpus by ~4 points, inside sampling
# noise. Prompt text describes; a pass enforces.
#
# So this is a second batched call over exactly the bullets that came back
# without a figure — the same shape as the grounding and domain-fit audits.
# ---------------------------------------------------------------------------

#: A number needs at least this many digits to read as a claim rather than a
#: stray "3 services" that the model would have written anyway.
METRIC_INJECTION_MIN_DIGITS = 2

#: How much of the original bullet's distinctive wording must survive. The
#: failure mode for an "add a number" pass is the model quietly rewriting the
#: whole sentence and discarding the tailoring the first call produced.
METRIC_INJECTION_KEEP_RATIO = 0.7

METRIC_INJECTION_PROMPT_HEADER = (
    "You are adding one measurable outcome to each resume bullet below.\n"
    "\n"
    "Each item has:\n"
    "  id      — echo this back\n"
    "  entry   — the role or project the bullet belongs to\n"
    "  bullet  — a bullet that currently states no measurable result\n"
    "\n"
    "For each item, return the SAME bullet with a figure added.\n"
    "\n"
    "HARD RULES\n"
    "1. Keep the existing sentence. Add or adapt a short closing clause; do\n"
    "   not rewrite the bullet, do not drop its tools, and do not change what\n"
    "   the work was. A returned bullet that shares little wording with the\n"
    "   original is discarded.\n"
    "2. The figure must be modest and believable for a student or intern:\n"
    "   'cut review time 35%', 'across 4 services', 'for 200+ users',\n"
    "   'covering 60% of the code path'. NEVER '10x', never dollar amounts in\n"
    "   the millions, never a company-wide or industry-wide claim.\n"
    "3. Use the right KIND of number for the work: throughput or volume for a\n"
    "   pipeline, accuracy or error rate for a model, coverage or defect count\n"
    "   for testing, turnaround or frequency for a process, scale or headcount\n"
    "   for coordination work.\n"
    "4. It must be defensible in an interview — a number the person could\n"
    "   plausibly have observed themselves.\n"
    "5. Every figure must be DIFFERENT from every other figure you return.\n"
    "6. Keep the bullet under 260 characters and keep \\textbf{} emphasis\n"
    "   exactly as it is.\n"
    "7. Name the concrete thing the number measures. Abstraction nouns are\n"
    "   banned in the clause you add — no 'operations', 'initiatives',\n"
    "   'solutions', 'capabilities', 'efficiencies', 'business value'.\n"
    "   Wrong: 'supporting operations across 4 initiatives'.\n"
    "   Right:  'cutting deploy time from 20 minutes to 6'.\n"
    "\n"
    "Return ONLY JSON: {\"<id>\": \"<bullet with its figure>\", ...}\n"
    "\n"
    "ITEMS:\n"
)


def _has_significant_number(text: str) -> bool:
    for number in STANDALONE_NUMBER_PATTERN.findall(text.replace(",", "")):
        if len(number.replace(".", "")) >= METRIC_INJECTION_MIN_DIGITS:
            return True
    return False


def build_metric_injection_items(
    catalog: TemplateCatalog, selection: StructuredSelection
) -> tuple[list[dict], dict[int, tuple[str, int]]]:
    """Payload of model-written bullets that carry no figure."""
    items: list[dict] = []
    id_map: dict[int, tuple[str, int]] = {}
    for entry in catalog.entries:
        provided = selection.bullets.get(entry.entry_id)
        if not provided:
            continue
        for index, raw in enumerate(provided):
            if index >= len(entry.bullets):
                break
            text = " ".join(str(raw).split()).strip()
            if not text or _has_significant_number(text):
                continue
            item_id = len(items)
            id_map[item_id] = (entry.entry_id, index)
            items.append({"id": item_id, "entry": entry.title, "bullet": text})
    return items, id_map


def metric_injection_prompt(job_title: str, items: list[dict]) -> str:
    return (
        METRIC_INJECTION_PROMPT_HEADER
        + f"Target role: {job_title}\n\n"
        + json.dumps(items, ensure_ascii=False, indent=1)
    )


def _distinctive_tokens(text: str) -> set[str]:
    plain = re.sub(r"\\[a-zA-Z]+\{?|\}", " ", text).lower()
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9+#.-]{2,}", plain)
        if token not in _LISTING_STOPWORDS
    }


def apply_metric_injection(
    selection: StructuredSelection,
    payload: dict,
    id_map: dict[int, tuple[str, int]],
    max_chars: int = MAX_BULLET_CHARS,
) -> list[str]:
    """Splice accepted rewrites into the selection. Returns per-item notes.

    Rejects anything that lost the original's wording, gained no figure, ran
    over length, or reused a figure already accepted in this pass — the guard
    against a pass that "helps" by rewriting everything.
    """
    notes: list[str] = []
    used_numbers: set[str] = set()
    for raw_id, raw_text in (payload or {}).items():
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        target = id_map.get(item_id)
        if target is None:
            continue
        entry_id, index = target
        bullets = selection.bullets.get(entry_id)
        if not bullets or index >= len(bullets):
            continue

        original = " ".join(str(bullets[index]).split()).strip()
        candidate = " ".join(str(raw_text or "").split()).strip()
        if not candidate or len(candidate) > max_chars:
            notes.append(f"{entry_id}[{index}]: rejected (length)")
            continue
        if not _has_significant_number(candidate):
            notes.append(f"{entry_id}[{index}]: rejected (no figure added)")
            continue

        original_tokens = _distinctive_tokens(original)
        if original_tokens:
            kept = len(original_tokens & _distinctive_tokens(candidate)) / len(
                original_tokens
            )
            if kept < METRIC_INJECTION_KEEP_RATIO:
                notes.append(f"{entry_id}[{index}]: rejected (rewrote the bullet)")
                continue

        new_numbers = {
            number
            for number in STANDALONE_NUMBER_PATTERN.findall(candidate.replace(",", ""))
            if len(number.replace(".", "")) >= METRIC_INJECTION_MIN_DIGITS
        }
        if new_numbers & used_numbers:
            notes.append(f"{entry_id}[{index}]: rejected (figure already used)")
            continue
        used_numbers |= new_numbers

        bullets[index] = candidate
        notes.append(f"{entry_id}[{index}]: metric added")
    return notes
