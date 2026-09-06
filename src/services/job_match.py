"""Rank archived jobs by how well they fit a resume profile.

The bot already logs every scraped job to data/jba/jobs (via
services.jba.merge_data), and already keeps a structured resume profile per
user in resumes_cache/<key>/. This module joins the two: given a time window
(a day, a week) it returns the best-fitting postings for one profile.

Three stages: cheap filter, real descriptions, one LLM judgement
----------------------------------------------------------------
A week of archive is thousands of records (hundreds of thousands on a busy
host), so nothing expensive can run per record.

1. A lexical score over every job -- regex against the profile's own skill
   anchors, role terms, seniority and cities. Fast and explainable, but only
   string overlap: it cannot tell that "Avionics Firmware Co-op" suits an
   embedded student who has never used the word "avionics". Its job is purely
   to cut the window down to LLM_CANDIDATES.

2. Description enrichment for the strongest ENRICH_CANDIDATES of those. The
   archive holds what a search-results page exposes -- title, company,
   location -- so under 2% of records carry a description, and without this
   step the judge is scoring job titles. See enrich_descriptions().

3. One batched call handing the whole shortlist to the model with the profile,
   asking it to score and explain each. This is where the judgement happens.

This replaced a local sentence-transformers pass. Embedding a week of postings
took minutes and hundreds of MB of RSS -- more than the rest of the bot -- for
a weaker signal than one cheap model call.

Fail-open
---------
If no provider is configured or every one fails, the lexical ordering is
returned as-is and MatchReport.ranker records why. The command degrades to a
rougher ranking rather than to an error.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from services import capacity
from services import job_level

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Prefilter component weights. Renormalised over whichever components a run
#: actually produced, so these are ratios rather than absolutes.
WEIGHTS: dict[str, float] = {
    "skills": 0.32,
    "role": 0.28,
    "level": 0.25,
    "location": 0.15,
}

#: Matching this many solid anchors is already a perfect skills score; beyond
#: it, a job that name-drops twenty tools should not outrank a focused one.
ANCHOR_SATURATION: float = 4.0

#: Prefilter weights for the entry-aware path. A tailored resume is assembled
#: from a handful of entries, not from the whole profile, so the question is
#: not "does this profile match" but "is there one entry that carries this
#: posting (top1), and enough others behind it to fill a page (depth)".
BUILD_WEIGHTS: dict[str, float] = {
    "top1": 0.45,
    "depth": 0.25,
    "level": 0.15,
    "location": 0.15,
}

#: How many entries a one-page resume realistically holds. `depth` is the mean
#: fit of the best this many, so a profile with one strong entry and nothing
#: behind it scores below one that could actually fill the page.
DEPTH_ENTRIES: int = 4

#: Per-entry anchor saturation. Far below ANCHOR_SATURATION on purpose: a
#: single entry names a handful of tools, so requiring four to saturate would
#: make every entry look weak and flatten the ranking it is meant to sharpen.
ENTRY_ANCHOR_SATURATION: float = 2.0

#: Skills/role split inside one entry's fit.
ENTRY_WEIGHTS: dict[str, float] = {"skills": 0.55, "role": 0.45}

#: Chars of one entry's own text carried into the judge prompt. The entry pool
#: replaces the flat profile blob rather than adding to it, so this budget is
#: what keeps the structured document from costing more than the text it
#: supersedes.
ENTRY_SUMMARY_CHARS: int = 320

#: Ceiling on the assembled profile document handed to the judge. Sized so
#: ENTRY_SUMMARY_CHARS is actually reachable: the shrink loop below trades
#: per-entry detail for fit, and at a 4000-char ceiling every entry was
#: being cut to 200 characters -- all twelve present, but each too thin to
#: judge. The extra ~290 tokens buy back the detail the selection depends on.
PROFILE_DOC_CHARS: int = 6400

#: Chars of the SKILL ANCHORS section carried into that document. Kept as its
#: own budget because it is the one part that must never be cut: the anchors
#: are the candidate's actual toolset, and the prefilter scores against them
#: whether or not the judge can see them.
ANCHOR_SUMMARY_CHARS: int = 700

#: How many prefiltered jobs go into the judge prompt. Deliberately several
#: times `limit`: the model reorders substantially, so a shortlist barely
#: larger than the output would make its judgement cosmetic.
#:
#: Briefly cut to 45 to save tokens, then restored: an unjudged posting keeps
#: only its keyword score, and the fifteen that dropped out sat at 34.7-35.8%
#: against a kept range of 36.3-52.2% -- close enough to the survivors that
#: the judge was the only thing that could tell them apart. Fifteen title-only
#: rows cost ~410 tokens, which is a poor thing to trade judgement for once
#: the description encoding has already paid for itself several times over.
LLM_CANDIDATES: int = 60

#: The judge returns 0-100; this module works in 0..1 everywhere else.
_LLM_SCORE_SCALE: float = 100.0

#: How many entries the judge may name as the resume it would build. A page
#: holds about this many, and asking for more would invite it to list the
#: whole profile back -- which is the flat whole-resume answer this is
#: replacing.
MAX_PLAN_ENTRIES: int = 4

#: Cap on the "what the candidate cannot evidence" note.
PLAN_GAP_CHARS: int = 80

#: How many NETWORK FETCHES one run may spend. Cache hits are not counted
#: against it: a description already stored costs nothing to reuse, so the
#: whole shortlist is looked up and only the misses are rationed. Coverage
#: therefore compounds -- each run inherits everything earlier runs paid for
#: and spends its own budget on postings nobody has fetched yet.
ENRICH_CANDIDATES: int = 28

#: Wall-clock ceiling for the whole enrichment stage. A slow board must not be
#: able to hold a Discord command open indefinitely -- whatever has come back
#: when the budget expires is what the judge gets.
ENRICH_TIME_BUDGET_SECONDS: float = 40.0

#: Per-posting fetch timeout inside that budget.
ENRICH_FETCH_TIMEOUT_SECONDS: int = 12

#: How much fetched description is kept on the record. Deliberately more than
#: reaches the prompt: the excerpt below is applied at prompt-build time, so
#: improving the excerpting improves every description already in hand rather
#: than only the ones fetched afterwards.
ENRICH_DESCRIPTION_CHARS: int = 2000

#: How much of one description reaches the judge. Lower than the old flat
#: 1000-char head slice and strictly better content: measured over the real
#: archive, a head cut kept 53 of 144 requirement sentences while this budget,
#: spent through excerpt_job_description, keeps 96. Postings put marketing
#: first and qualifications last, so the head is the wrong half to keep.
JUDGE_DESCRIPTION_CHARS: int = 700

#: Hard ceiling on records held in memory for one query, scaled down on small
#: hosts the same way the dedup caches are.
MAX_WINDOW_RECORDS: int = 120_000

WINDOW_DAYS: dict[str, int] = {"day": 1, "week": 7}

DEFAULT_LIMIT: int = 10
MAX_LIMIT: int = 50


# ---------------------------------------------------------------------------
# Profile signal
# ---------------------------------------------------------------------------

# Junior-side and senior-side title markers. Matched with word boundaries
# against a lowercased title, so "internal" does not read as "intern" and
# "sriram" does not read as "sr".
_JUNIOR_MARKERS: tuple[str, ...] = (
    "intern", "internship", "co-op", "coop", "co op", "new grad", "new graduate",
    "newgrad", "entry level", "entry-level", "junior", "jr", "student",
    "university", "campus", "graduate program", "apprentice", "trainee",
    "undergraduate", "placement",
)
_SENIOR_MARKERS: tuple[str, ...] = (
    "senior", "sr", "staff", "principal", "lead", "manager", "director",
    "head of", "vp", "vice president", "architect", "chief", "expert",
    "iii", "iv", "10\\+ years",
)

#: Anchors this short carry little evidence on their own (a stray "c" in a
#: title is as likely punctuation as the language), so they count for less.
_SHORT_ANCHOR_CHARS = 3

#: Words that appear in nearly every job title and so cannot distinguish one
#: role from another. The level words (intern, junior, senior, ...) are in here
#: deliberately: they are scored by score_level, and leaving them in would let
#: "Senior Sales Intern" claim a role match against "Machine Learning Intern"
#: on the strength of the word "intern" alone.
_ROLE_STOPWORDS: frozenset[str] = frozenset({
    "and", "or", "the", "a", "an", "of", "for", "in", "at", "to", "with",
    "job", "jobs", "role", "position", "opportunity", "career", "careers",
    "full", "time", "part", "hiring", "we", "are", "team", "new", "remote",
    "intern", "internship", "co-op", "coop", "junior", "jr", "senior", "sr",
    "student", "member", "graduate", "grad", "entry", "level", "i", "ii",
})

#: Profile [tags] that name no domain at all -- keeping them would let any
#: title containing the word claim a full role match.
_GENERIC_TAGS: frozenset[str] = frozenset({"general", "misc", "other", "various"})

#: Below this, a shared word is coincidence rather than a domain match, and
#: naming the role in the output would mislead more than it explains.
_ROLE_REPORT_THRESHOLD: float = 0.5

#: Credit for a one-word role term (a bare [tag]) found in a title. Short of a
#: full match on purpose: "Project Management Systems & Tools" and "Technical
#: Support Analyst" both contain the word "systems", and letting a [systems]
#: tag claim a perfect role hit off that put both above real embedded and
#: electrical postings in an archive run. A two-word title match ("machine
#: learning") is evidence; one common noun is a coincidence worth partial
#: credit, no more.
_SINGLE_WORD_ROLE_CREDIT: float = 0.6

_WORD_RE = re.compile(r"[a-z0-9+#.]+")
_ENTRY_TAG_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
#: "Engineering Instructor, Obotz Robotics - Markham ON, Sep 2022 - Present"
#: The city/province sits between the dash and the trailing date.
#:
#: The city is one or two capitalised words, joined by a single space or a
#: single hyphen ("Trois-Rivieres QC"). It deliberately cannot span " - ",
#: which is three characters: a looser class lets the match run backwards
#: through the separator and swallow the employer, yielding locations like
#: "rocket society - toronto on" that then match nothing.
_LOCATION_RE = re.compile(
    r"\b([A-Z][a-zA-Z.]+(?:[ -][A-Z][a-zA-Z.]+)?\s"
    r"(?:ON|BC|AB|QC|NS|MB|SK|NB|NL|PE|YT|NT|NU))\b"
)


@dataclass(slots=True)
class ProfileEntry:
    """One selectable unit of the profile -- a job or a project.

    A tailored resume is assembled from four or five of these, not from the
    profile as a whole, so this is the unit the ranking has to reason about:
    "which of these would I put on the page for this posting" is a different
    question from "does my resume match", and only the first one has a useful
    answer.
    """

    entry_id: str
    title: str
    #: The entry's own [tag] labels, minus the ones that name no domain.
    categories: tuple[str, ...] = ()
    #: Collapsed block text, header included.
    text: str = ""
    #: Profile skill anchors this entry actually evidences.
    anchors: tuple[str, ...] = ()
    #: Role phrases this entry can claim: its tags plus its own title.
    role_terms: tuple[str, ...] = ()


@dataclass(slots=True)
class ProfileSignal:
    """Everything the scorer needs from one resume profile."""

    profile_key: str
    #: Lowercase tools from the SKILL ANCHORS section.
    anchors: tuple[str, ...] = ()
    #: Lowercase multi-word role phrases drawn from experience/project headers
    #: and their [category] tags.
    role_terms: tuple[str, ...] = ()
    #: Lowercase place names the candidate has actually worked in.
    locations: tuple[str, ...] = ()
    #: "student" when the profile shows an in-progress degree, else "professional".
    seniority: str = "professional"
    #: Profile text handed to the judge as the candidate description.
    document_text: str = ""
    #: The profile's selectable entries, newest-first as written. Empty for a
    #: prose-only or unseeded profile, which is what keeps those on the flat
    #: whole-document path instead of failing them.
    entries: tuple[ProfileEntry, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.anchors or self.role_terms)


def _lower_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _anchor_pattern(anchor: str) -> re.Pattern[str]:
    """Word-boundary matcher for one anchor.

    ``re.escape`` then a manual boundary, because ``\\b`` behaves wrongly next
    to the non-word characters that are common in tool names: ``\\bc\\+\\+\\b``
    can never match, since ``+`` is not a word character so there is no
    boundary after it.
    """
    escaped = re.escape(anchor)
    left = r"(?<![a-z0-9])" if anchor[0].isalnum() else r""
    right = r"(?![a-z0-9])" if anchor[-1].isalnum() else r""
    return re.compile(left + escaped + right)


@lru_cache(maxsize=512)
def _compiled_anchor(anchor: str) -> re.Pattern[str]:
    return _anchor_pattern(anchor)


@lru_cache(maxsize=1024)
def _term_words(term: str) -> frozenset[str]:
    """The meaningful words of one role term.

    Cached because these are profile constants compared against every posting
    in the window: recomputing the split per posting was the single hottest
    call in the prefilter, running tens of thousands of times to produce a few
    dozen distinct answers.
    """
    return frozenset(_lower_words(term)) - _ROLE_STOPWORDS


@lru_cache(maxsize=1024)
def _term_is_phrase(term: str) -> bool:
    """Whether a role term carries more than one meaningful word."""
    return len(_lower_words(term)) > 1


def _names(text: str, term: str) -> bool:
    """Word-boundary containment, with a plain substring test in front.

    The boundary regexes carry lookarounds and are the hot path of the whole
    prefilter -- a full window is ~60 of them per posting across a hundred
    thousand postings. The overwhelming majority of anchors do not appear in a
    given posting at all, and `in` is a C-level scan that settles those cases
    an order of magnitude faster. The regex still decides every case it
    reaches, so the result is unchanged: this only skips work whose answer was
    already known to be False.
    """
    return term in text and _compiled_anchor(term).search(text) is not None


def _marker_pattern(markers: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z])(?:" + "|".join(markers) + r")(?![a-z])")


_JUNIOR_RE = _marker_pattern(_JUNIOR_MARKERS)
_SENIOR_RE = _marker_pattern(_SENIOR_MARKERS)


def _parse_role_terms(baseinfo_text: str) -> tuple[str, ...]:
    """Role phrases from baseinfo entry headers and their [category] tags.

    A header reads ``[ml] Machine Learning Intern, Goopter ... - Remote, Sep``.
    The part before the first comma is the role; the tags are the profile's own
    shorthand for the domain ("ml", "embedded", "fullstack") and are just as
    usable as query terms.
    """
    terms: list[str] = []
    for raw_line in baseinfo_text.splitlines():
        match = _ENTRY_TAG_RE.match(raw_line.strip())
        if not match:
            continue
        for tag in match.group(1).split(","):
            tag = tag.strip().lower()
            if tag and tag not in _GENERIC_TAGS:
                terms.append(tag)
        head = match.group(2)
        # Projects read "eebot Mobile Robot System (Assembly, C)" -- drop the
        # parenthesised tool list, it is already covered by the anchors.
        head = head.split("(")[0]
        role = head.split(",")[0].strip().lower()
        if role and len(role) > 2:
            terms.append(role)
    return tuple(dict.fromkeys(terms))


def _parse_locations(baseinfo_text: str) -> tuple[str, ...]:
    found = [match.group(1).strip().lower() for match in _LOCATION_RE.finditer(baseinfo_text)]
    if re.search(r"(?<![a-z])remote(?![a-z])", baseinfo_text, re.IGNORECASE):
        found.append("remote")
    # A "Markham ON" entry should also match a job listed as plain "Markham".
    expanded: list[str] = []
    for place in found:
        expanded.append(place)
        city = place.rsplit(" ", 1)[0].strip()
        if city and city != place:
            expanded.append(city)
    return tuple(dict.fromkeys(expanded))


#: One baseinfo block: a "[tags] Header" line and everything up to the next
#: block or section rule. The same convention the resume template uses, so the
#: entries this yields line up with the ones .resumebuild selects between.
_ENTRY_BLOCK_RE = re.compile(
    r"^\[([a-z][a-z0-9_, ]*)\]\s*(.+?)(?=^\[[a-z]|^==\s|\Z)",
    re.MULTILINE | re.DOTALL,
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _entry_slug(title: str) -> str:
    return _SLUG_RE.sub("-", title.lower()).strip("-") or "entry"


def _parse_profile_entries(
    baseinfo_text: str, anchors: tuple[str, ...]
) -> tuple[ProfileEntry, ...]:
    """Split baseinfo into the units a tailored resume actually picks from.

    Deliberately parsed here rather than through
    ``structured.parse_baseinfo_blocks``: that function keys blocks by tag and
    concatenates everything sharing one, which is right for grounding a render
    and wrong here -- two projects tagged ``[embedded]`` are two things the
    candidate can choose between, and collapsing them hides exactly the depth
    this scorer exists to measure.
    """
    entries: list[ProfileEntry] = []
    used_ids: set[str] = set()
    for match in _ENTRY_BLOCK_RE.finditer(baseinfo_text):
        categories = tuple(
            tag for tag in (part.strip().lower() for part in match.group(1).split(","))
            if tag and tag not in _GENERIC_TAGS
        )
        body = match.group(2).strip()
        if not body:
            continue
        head = body.splitlines()[0]
        title = head.split("(")[0].split(",")[0].strip()
        if not title:
            continue
        text = " ".join(body.split())

        entry_id = _entry_slug(title)
        base = entry_id
        suffix = 2
        while entry_id in used_ids:
            entry_id = f"{base}-{suffix}"
            suffix += 1
        used_ids.add(entry_id)

        lowered = text.lower()
        entry_anchors = tuple(anchor for anchor in anchors if _names(lowered, anchor))
        role_terms = tuple(dict.fromkeys(categories + ((title.lower(),) if len(title) > 2 else ())))
        entries.append(
            ProfileEntry(
                entry_id=entry_id,
                title=title,
                categories=categories,
                text=text,
                anchors=entry_anchors,
                role_terms=role_terms,
            )
        )
    return tuple(entries)


def _skill_anchor_section(baseinfo_text: str) -> str:
    """The SKILL ANCHORS block, collapsed. Empty when the profile has none."""
    match = re.search(
        r"==\s*SKILL ANCHORS\s*==(.*?)(?:\n==\s|\Z)",
        baseinfo_text,
        re.DOTALL | re.IGNORECASE,
    )
    return " ".join(match.group(1).split()) if match else ""


def _allocate_entry_budget(lengths: list[int], budget: int) -> list[int]:
    """Share `budget` characters across entries, shortest need first.

    A flat per-entry cap wastes the difference on every entry shorter than it
    while starving the ones above it -- on a real profile that meant five
    entries leaving ~380 characters unused and the largest project keeping 320
    of its 1,062. Filling the small needs first and redistributing what is left
    over the entries still short of complete spends the same budget with less
    of the profile lost.
    """
    count = len(lengths)
    if count == 0:
        return []
    allowance = [0] * count
    remaining = max(0, budget)
    # Shortest first: settling a small need frees its surplus for the rest.
    for index in sorted(range(count), key=lambda i: lengths[i]):
        pending = sum(1 for position in range(count) if allowance[position] == 0)
        share = remaining // max(1, pending)
        take = min(lengths[index], share)
        allowance[index] = take
        remaining -= take
    # Anything still spare goes to whoever is still truncated.
    for index in sorted(range(count), key=lambda i: lengths[i], reverse=True):
        if remaining <= 0:
            break
        short_by = lengths[index] - allowance[index]
        if short_by > 0:
            extra = min(short_by, remaining)
            allowance[index] += extra
            remaining -= extra
    return [max(0, size) for size in allowance]


def _trim_at_boundary(text: str, size: int) -> str:
    """`text` cut to `size`, preferring a sentence then a word boundary.

    A mid-word cut ("monitoring five ATS provi") reads as corrupted text to the
    model rather than as an entry that simply ends.
    """
    if size >= len(text):
        return text
    if size <= 0:
        return ""
    window = text[:size]
    stop = max(window.rfind(". "), window.rfind("; "))
    if stop > size * 0.6:
        return window[: stop + 1]
    space = window.rfind(" ")
    return (window[:space] if space > size * 0.6 else window).rstrip()


def build_profile_document(
    baseinfo_text: str,
    entries: tuple[ProfileEntry, ...],
    anchor_text: str,
) -> str:
    """The candidate description handed to the judge.

    Previously this was the raw file collapsed and cut at a fixed character
    count, which on a real profile silently dropped the last three entries and
    the entire SKILL ANCHORS section -- the judge was inferring the candidate's
    toolset from prose while the prefilter scored against an explicit list the
    judge could not see.

    Assembling it from parts fixes that within the same budget: every entry
    reaches the model, the anchors get a reserved allowance, and the structure
    tells the model what it is looking at -- a pool to select from, not a
    document to match against.
    """
    if not entries:
        # Prose-only and unseeded profiles keep the old behaviour exactly.
        return " ".join(baseinfo_text.split())[:PROFILE_DOC_CHARS]

    first_section = re.search(r"^==\s", baseinfo_text, re.MULTILINE)
    head = baseinfo_text[: first_section.start()] if first_section else ""
    preamble = " ".join(head.split())[:500]

    fixed: list[str] = []
    if preamble:
        fixed.append(f"Candidate: {preamble}")
    if anchor_text:
        fixed.append(f"Verified toolset: {anchor_text[:ANCHOR_SUMMARY_CHARS]}")
    fixed.append(
        "Resume entries (any one resume uses about "
        f"{DEPTH_ENTRIES} of these; the rest are simply left off):"
    )
    fixed_len = sum(len(line) + 1 for line in fixed)

    budget = PROFILE_DOC_CHARS - fixed_len - sum(
        len(entry.entry_id) + 5 for entry in entries
    )
    allowance = _allocate_entry_budget([len(entry.text) for entry in entries], budget)
    body = [
        f"- [{entry.entry_id}] {_trim_at_boundary(entry.text, size)}"
        for entry, size in zip(entries, allowance)
    ]
    return "\n".join(fixed + body)[:PROFILE_DOC_CHARS]


def _infer_seniority(baseinfo_text: str, today: date) -> str:
    """A degree whose end year has not passed yet means the candidate is still
    a student, and so wants junior postings."""
    for match in re.finditer(r"\b(19|20)\d{2}\s*[-–—]\s*((?:19|20)\d{2})\b", baseinfo_text):
        if int(match.group(2)) >= today.year:
            return "student"
    if re.search(r"\b(present|current)\b", baseinfo_text, re.IGNORECASE) and re.search(
        r"\b(bachelor|b\.?eng|b\.?sc|undergrad|degree)\b", baseinfo_text, re.IGNORECASE
    ):
        return "student"
    return "professional"


def build_profile_signal(
    profile_dir: Path,
    profile_key: str | None = None,
    today: date | None = None,
) -> ProfileSignal:
    """Read one profile directory into a ProfileSignal.

    Reads baseinfo.txt directly rather than going through
    ``structured.load_structured_profile``: that loader parses (and may rewrite)
    template.tex to build a rendering catalog, which is far more work than a
    read-only ranking query needs, and it returns ``None`` for legacy profiles
    that this scorer handles fine.
    """
    from services.resumes.structured import parse_skill_anchors

    key = profile_key or profile_dir.name
    try:
        baseinfo_text = (profile_dir / "baseinfo.txt").read_text(encoding="utf-8")
    except OSError:
        baseinfo_text = ""

    anchors = tuple(
        anchor for anchor in parse_skill_anchors(baseinfo_text)
        if len(anchor) >= 1 and not anchor.isdigit()
    )
    role_terms = _parse_role_terms(baseinfo_text)
    entries = _parse_profile_entries(baseinfo_text, anchors)
    return ProfileSignal(
        profile_key=key,
        anchors=anchors,
        role_terms=role_terms,
        locations=_parse_locations(baseinfo_text),
        seniority=_infer_seniority(baseinfo_text, today or date.today()),
        document_text=build_profile_document(
            baseinfo_text, entries, _skill_anchor_section(baseinfo_text)
        ),
        entries=entries,
    )


def resolve_profile_dir(profile_key: str | int | None, cache_root: Path | None = None) -> Path:
    from services.resumes.resume import RESUMES_CACHE_ROOT, resolve_profile_seed_dir

    root = cache_root or RESUMES_CACHE_ROOT
    return resolve_profile_seed_dir(root, profile_key)


# ---------------------------------------------------------------------------
# Window loading
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ArchivedJob:
    title: str
    company: str
    location: str
    link: str
    site_label: str
    date_posted: str
    description: str = ""
    #: Stamped by the ATS scrape, where the platform's own employment-type field
    #: was available. Empty for records written before that, and for the watcher
    #: send path -- score_level falls back to classifying the title.
    level: str = ""
    employment_type: str = ""

    def match_text(self) -> str:
        return " ".join(
            part for part in (self.title, self.company, self.location, self.description) if part
        ).lower()


def _normalize_record(record: dict[str, Any]) -> ArchivedJob | None:
    """Flatten either archive record shape into one struct.

    Two producers write to the archive: the ATS scraper (``job_url``,
    ``_source_site``) and the watcher send path (``link``, ``site_label``, and
    a title that already has "(company, location)" appended). Both are read
    here; a record with no title is dropped as unusable.
    """
    from services.job_service import fix_text_encoding

    # Archived titles carry mojibake from the source boards ("développement"
    # stored as "dA©veloppement"), which would reach both the judge prompt and
    # the user's screen.
    title = fix_text_encoding(record.get("title") or "").strip()
    # Collapse runs of whitespace here rather than in either producer: the send
    # path already normalizes, the ATS scraper does not, so 74 of the real
    # archive's titles render as "Associate Medical Editor  - US Students".
    # Doing it at read time fixes the rows already stored too.
    title = " ".join(title.split())
    if not title:
        return None

    link = str(
        record.get("link") or record.get("job_url") or record.get("job_url_direct") or ""
    ).strip()
    site_label = str(
        record.get("site_label") or record.get("_source_site") or record.get("site") or ""
    ).strip()
    # The send path stores a presentation label ("LinkedIn"); the ATS scraper
    # stores the raw site key, so the same report listed "LinkedIn" next to
    # "lever" and "icims". Map the key back to its label at read time.
    from services.job_service import JOBSPY_SITE_LABELS, _readable_company

    site_label = JOBSPY_SITE_LABELS.get(site_label.lower(), site_label)
    description = str(
        record.get("description") or record.get("snippet") or record.get("summary") or ""
    ).strip()
    return ArchivedJob(
        title=title,
        # ATS records carry the board's slug ("abnormalsecurity", "2k") where the
        # send path carries a real name ("General Motors"); _readable_company
        # opens the slug up and leaves an already-readable name untouched.
        company=_readable_company(fix_text_encoding(record.get("company") or "").strip()),
        location=fix_text_encoding(record.get("location") or "").strip(),
        link=link,
        site_label=site_label or "unknown",
        date_posted=str(record.get("date_posted") or record.get("scraped_at") or "")[:10],
        description=description[:2000],
        # Carried through rather than re-derived: the scrape had the platform's
        # own employment-type field, which the title alone cannot replace.
        # Absent on older rows and on the send path, and score_level falls back
        # to classifying the title when it is.
        level=str(record.get("level") or "").strip(),
        employment_type=str(record.get("employment_type") or "").strip(),
    )


def window_dates(days: int, end_date: date | None = None) -> list[str]:
    """ISO dates in the window, newest first. ``days=1`` is just ``end_date``."""
    end = end_date or date.today()
    span = max(1, int(days))
    return [(end - timedelta(days=offset)).isoformat() for offset in range(span)]


#: Tokens that a job board appends to a company name without changing which
#: company it is: legal form, corporate-structure, and country qualifiers. The
#: same employer reaches the archive as "Jp2g Consultants Inc." on Indeed and
#: "Jp2g Consultants" on Glassdoor, as "Soucy Group" and "Soucy", as "PwC
#: Canada" and "PwC".
#:
#: Deliberately excludes identity-bearing words like "Technologies", "Labs",
#: "Solutions" and "Services": those distinguish real companies from each
#: other, so stripping them would merge unrelated employers.
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(incorporated|inc|llc|ltd|limited|corp|corporation|company|co|plc|gmbh"
    r"|pty|llp|group|holdings?|canada|usa|international|global)\b"
)

#: Everything that is not a letter or digit. Boards differ on punctuation,
#: casing and whitespace for the same string ("Non-CPA" / "Non CPA").
_KEY_NOISE_RE = re.compile(r"[^a-z0-9]+")


def _squash(text: str) -> str:
    """Casefold and reduce every run of punctuation/whitespace to one space."""
    return _KEY_NOISE_RE.sub(" ", str(text or "").lower()).strip()


def _normalize_title_key(title: str) -> str:
    """Identity form of a job title.

    Punctuation and casing only. Any trailing parenthetical is deliberately
    KEPT: on an ATS record it is the disambiguator, not noise -- "Senior Combat
    Designer (Encounters)" and "Senior Combat Designer (AI archetype)" are two
    real openings at one studio, as are "(Hybrid)" and "(Remote)" variants of
    one engineering role. Stripping it collapsed those into each other.

    The send path's "Role (Company, City)" suffix is handled before this, by
    `_display_fields`, which knows the parenthetical is a company there and
    hands the two halves over separately.
    """
    return _squash(title)


def _normalize_company_name(company: str) -> str:
    """Identity form of a company name, ignoring board-specific decoration.

    Suffixes are stripped repeatedly, because they stack ("Soucy Group Inc.").
    If stripping consumes the whole name -- a company genuinely called "Group"
    or "Co" -- the squashed original is kept instead, since an empty company is
    no identity at all and would collapse unrelated postings.
    """
    base = _squash(company)
    current, previous = base, None
    while previous != current:
        previous = current
        current = re.sub(r"\s+", " ", _COMPANY_SUFFIX_RE.sub(" ", current)).strip()
    return current or base


def _company_matches(left: str, right: str) -> bool:
    """Whether two normalized company strings name the same employer.

    Equal, or one a leading-token prefix of the other. Boards abbreviate the
    same employer differently and suffix-stripping alone does not close the
    gap: over a week of archive, "Lumentum" and "Lumentum Operations", "BMO"
    and "BMO Financial", "Zurich" and "Zurich Insurance", "BBA" and "BBA
    Consultants" were each one job listed twice.

    A prefix and not a shared word, deliberately: "Hitachi Energy" and
    "Hitachi Rail" are different companies that happen to share a first token,
    and collapsing them would hide a real posting. A prefix keeps them apart
    because neither leads the other.

    Only ever consulted for postings that already share a normalized title, so
    two employers must collide on both the job title and the leading words of
    their name before this can merge anything.
    """
    if left == right:
        return True
    if not left or not right:
        # An empty name is a prefix of everything, which would make one
        # nameless posting absorb every other job with the same title.
        # `_job_identity` already withholds a title identity in that case; this
        # is here so the rule cannot be misused from anywhere else.
        return False
    left_tokens, right_tokens = left.split(), right.split()
    return (
        left_tokens[:len(right_tokens)] == right_tokens
        or right_tokens[:len(left_tokens)] == left_tokens
    )


def _job_identity(job: ArchivedJob) -> tuple[tuple[str, ...], str, str]:
    """``(url_keys, title, company)`` -- everything dedup compares.

    Two duplications happen, so there are two identities:

    * Same URL on different days. A posting is re-logged every day it stays
      live, so a week's window would otherwise show one job seven times.
    * Same title at the same company. One job listed once per city, and one job
      syndicated to several boards -- the case a URL can never catch, because
      every board mints its own URL.

    The title and company come from `_display_fields`, which already knows how
    to undo the send path's "Role (Company, City)" fold, so the two archive
    record shapes reach dedup in one form rather than through two parallel
    code paths that had to be kept in step.

    An empty company yields no title identity: a bare title is far too weak to
    collapse unrelated postings on, and "Finance Intern" alone would merge
    Acceldata with Laurentian Bank.
    """
    from services.job_service import canonicalize_job_link

    urls: tuple[str, ...] = ()
    if job.link:
        urls = ((canonicalize_job_link(job.link) or job.link).lower(),)
    title, company, _ = _display_fields(job)
    company = _normalize_company_name(company)
    if not company:
        return urls, "", ""
    return urls, _normalize_title_key(title), company


def _split_trailing_paren(title: str) -> tuple[str, str]:
    """``(head, inner)`` for a title ending in a parenthetical, else ``("", "")``.

    Scans back from the end tracking depth, so a nested group comes back whole:
    "Actuarial intern (iA Financial Group (Industrial Alliance), Quebec,
    Canada)" is one suffix. The regex this replaced used ``[^()]*`` and matched
    no such title at all, so those postings lost both their dedup identity and
    their company.
    """
    text = title.rstrip()
    if not text.endswith(")"):
        return "", ""
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        if text[index] == ")":
            depth += 1
        elif text[index] == "(":
            depth -= 1
            if depth == 0:
                return text[:index].strip(), text[index + 1:-1]
    return "", ""


@lru_cache(maxsize=8192)
def _country_from_location(location: str) -> str | None:
    """ISO country code for a location string, or None when it does not say.

    Delegates to the geo parser the scraper already uses, which is what makes
    this safe: "CA" is the single most common tail in the archive and means
    Canada in "Toronto, ON, CA" but California in "San Leandro, CA". Splitting
    on the trailing comma-segment would get that exactly backwards; the parser
    resolves both correctly.

    Cached because a window is mostly repeated locations, and this runs per
    record -- about a second at the 120k cap, versus a minute uncached.
    """
    if not location:
        return None
    try:
        from services.jba.geo_db import country_for_city
        from services.jba.geolocation import parse_job_location

        parsed = parse_job_location(location)
        country = parsed.get("country")
        if country:
            return country
        # The parser needs a country or admin token to place a location, so it
        # gives up on the many records that name only a city ("Basingstoke",
        # "Dublin"). The geo table places those by population, which recovers
        # 283 of the 707 it could not resolve over a week of archive.
        return country_for_city(parsed.get("city") or "")
    except Exception:  # a geo-data problem must not take the command down
        return None


def _job_country(job: ArchivedJob) -> str | None:
    """The country a posting is in, from whichever field carries its location.

    A send-path record has an empty location field -- the watcher folded it
    into the title -- so the remainder of that fold is used instead.
    """
    location = job.location
    if not location:
        head, inner = _split_trailing_paren(job.title)
        if head and "," in inner:
            location = inner.partition(",")[2].strip()
    return _country_from_location(location)


def _matches_channel_region(job: ArchivedJob, allow_north_america: bool | None) -> bool:
    """Same policy as job_service.filter_rows_by_region: Canada is always
    kept, the US only when the channel allows North America broadly, and a
    location the geo parser could not place is kept regardless.

    That last rule diverges from filter_rows_by_region, which keeps an
    unresolved location only for ATS-sourced rows. Deliberate, but not because
    the archive is ATS-only -- it is not (_normalize_record reads two
    producers). It holds because both producers already passed the live region
    gate before reaching the archive.

    ``None`` means no channel context was given at all (e.g. a caller outside
    the Discord command path) and skips the gate entirely, as opposed to
    ``False`` which is a channel actively scoped to Canada only.
    """
    if allow_north_america is None:
        return True
    country = _job_country(job)
    if country == "CA":
        return True
    if country == "US":
        return allow_north_america
    return country is None


def _matches_channel_scope(
    job: ArchivedJob,
    role_filters: Iterable[str],
    exclusion_terms: Iterable[str],
    allow_north_america: bool | None,
) -> bool:
    """Whether ``job`` is something this channel's job-watcher settings would
    actually surface -- otherwise .bestjobs ranks the entire cross-channel
    archive, including postings for a completely different search than the
    channel invoking the command cares about.

    Deliberately does not filter on the channel's free-text ``keywords``: a
    single-word match ("developer") is too loose (an unrelated "Frontend
    Developer" posting would pass a "Python Developer" channel) and requiring
    every word is too strict (a genuine "Python Software Engineer" match would
    be silently dropped for lacking the literal word "developer"). Domain
    relevance is left to the resume-fit scoring below, which matches against
    the profile's own terms instead of a coarse channel setting.
    """
    from services import job_service

    if not job_service.matches_role_filters(job.title, list(role_filters)):
        return False
    if job_service.matches_exclusion_terms(
        {"title": job.title, "company": job.company}, list(exclusion_terms)
    ):
        return False
    return _matches_channel_region(job, allow_north_america)


def _seen_before(
    seen_titles: dict[str, dict[str, list[tuple[str, set[str | None]]]]],
    title: str,
    company: str,
    country: str | None,
) -> bool:
    """Whether this posting duplicates one already kept, recording it if not.

    Employers under a title are scanned rather than looked up, because boards
    abbreviate the same name and `_company_matches` decides what counts as the
    same one.

    The scan is bucketed by the company's first word, which costs nothing in
    accuracy: `_company_matches` is a leading-token prefix, so two names that
    match necessarily share a first token, and names in other buckets could
    never have matched anyway. Without it the scan is linear in the employers
    sharing a title and the whole pass is quadratic in them -- 20,000 employers
    under one title took a minute. Real data peaks at four, but the archive cap
    is 120,000 records and nothing structurally prevents a common title from
    drawing a long list.

    A known company in a new country is kept and its country remembered against
    the same employer, so a third record with an unresolved country still
    recognises both.
    """
    # A nameless company cannot reach here -- `_job_identity` withholds the
    # title -- but an IndexError would take the whole command down, so the
    # bucket key degrades instead of indexing an empty split.
    bucket = (company.split() or [""])[0]
    employers = seen_titles.setdefault(title, {}).setdefault(bucket, [])
    for index, (known_company, countries) in enumerate(employers):
        if not _company_matches(known_company, company):
            continue
        # Keep the more specific of the two names. A bare "Hitachi" seen before
        # "Hitachi Energy" and "Hitachi Rail" would otherwise absorb both --
        # every extension matches the stump, so two distinct employers would
        # collapse into one on scrape order alone. Narrowing the entry to the
        # first full name it meets means the second is compared against
        # "Hitachi Energy", does not match, and survives.
        if len(company) > len(known_company):
            employers[index] = (company, countries)
        if _country_absorbs(countries, country):
            return True
        countries.add(country)
        return False
    employers.append((company, {country}))
    return False


def _country_absorbs(kept: set[str | None] | None, country: str | None) -> bool:
    """Whether a title+company already kept covers this posting's country.

    Country only ever *splits* a title+company group; an unresolved country
    never splits one. About 77% of archived records resolve to a country (62%
    before the geo table began placing the city-only ones), so treating
    "unknown" as its own value would strand the remaining quarter -- and would
    undo real cross-board merges, since boards disagree on how much location
    they print ("Ottawa, ON, CA" on Indeed against a bare "Ottawa" on
    Glassdoor for one posting). Unknown therefore matches anything.

    The cost is order-dependence: an unknown-country record seen first absorbs
    later ones from any country. Days are walked newest-first, so the surviving
    copy is still the most recent, and the alternative -- splitting duplicates
    back apart on missing data -- is the worse failure for a top-10 list.
    """
    if kept is None:
        return False
    return country is None or None in kept or country in kept


def load_window_jobs(
    days: int,
    end_date: date | None = None,
    max_records: int | None = None,
) -> tuple[list[ArchivedJob], list[str]]:
    """Load and dedupe every archived job in the window.

    Returns ``(jobs, dates_scanned)``. Days are walked newest-first, and the
    first sighting of a job wins, so the surviving copy is the most recent one.
    """
    from services.jba.merge_data import load_daily_log

    cap = max_records if max_records is not None else capacity.scaled_cap(
        MAX_WINDOW_RECORDS, minimum=5_000
    )
    dates = window_dates(days, end_date)
    #: URL identities. A shared URL is the same posting whatever the location
    #: says, so these stay exact.
    seen_urls: set[str] = set()
    #: title -> first company word -> [(company, countries kept under it)].
    #: Lists rather than a dict of companies because they are matched by
    #: `_company_matches`, not by equality; bucketed by first word because a
    #: prefix match always shares one.
    seen_titles: dict[str, dict[str, list[tuple[str, set[str | None]]]]] = {}
    jobs: list[ArchivedJob] = []

    for date_key in dates:
        try:
            records = load_daily_log(date_key)
        except Exception as exc:  # a corrupt zip must not take the command down
            print(f"[job-match] failed to read {date_key}: {exc}")
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            job = _normalize_record(record)
            if job is None:
                continue
            url_keys, title, company = _job_identity(job)
            if any(key in seen_urls for key in url_keys):
                continue
            if title and _seen_before(seen_titles, title, company, _job_country(job)):
                continue
            seen_urls.update(url_keys)
            jobs.append(job)
            if len(jobs) >= cap:
                return jobs, dates
    return jobs, dates


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class JobScore:
    total: float
    components: dict[str, float] = field(default_factory=dict)
    matched_anchors: tuple[str, ...] = ()
    matched_roles: tuple[str, ...] = ()
    #: (entry_id, fit) for the profile entries that actually fit this posting,
    #: best first. Empty on the flat path. This is the answer to "which parts
    #: of my resume would I use here", which is a more useful thing to show a
    #: user than a single percentage.
    entry_fits: tuple[tuple[str, float], ...] = ()
    #: Set once the judge has scored this job. ``total`` then holds the judge's
    #: score, and ``components`` stays as the prefilter's own reasoning.
    llm_score: float | None = None
    llm_reason: str = ""
    #: Entry ids the judge would actually put on a page for this posting, best
    #: first. Only ever populated for a posting whose description was fetched:
    #: a title alone is not enough to choose entries from, and a plan invented
    #: off one would read as knowledge the model does not have.
    plan_entries: tuple[str, ...] = ()
    #: The one requirement the judge found nothing in the profile to evidence.
    plan_gap: str = ""


@dataclass(slots=True)
class JobMatch:
    job: ArchivedJob
    score: JobScore


def score_skills(text: str, signal: ProfileSignal) -> tuple[float, tuple[str, ...]]:
    """Saturating weighted count of profile tools named by the posting."""
    if not signal.anchors:
        return 0.0, ()
    matched: list[str] = []
    weight = 0.0
    for anchor in signal.anchors:
        if _names(text, anchor):
            matched.append(anchor)
            weight += 0.5 if len(anchor) <= _SHORT_ANCHOR_CHARS else 1.0
    return min(1.0, weight / ANCHOR_SATURATION), tuple(matched)


def _best_role_overlap(
    title_lower: str, title_words: set[str], terms: Iterable[str]
) -> tuple[float, tuple[str, ...]]:
    """Best overlap between a job title and any of `terms`.

    A full phrase hit ("machine learning" inside "Machine Learning Intern")
    scores 1.0; otherwise the fraction of the term's meaningful words present
    in the title, which lets "frontend developer" partially match "Frontend
    Engineer".
    """
    best = 0.0
    matched: list[str] = []
    for term in terms:
        if _names(title_lower, term):
            credit = 1.0 if _term_is_phrase(term) else _SINGLE_WORD_ROLE_CREDIT
            if credit >= _ROLE_REPORT_THRESHOLD:
                matched.append(term)
            best = max(best, credit)
            continue
        term_words = _term_words(term)
        if not term_words:
            continue
        overlap = len(term_words & title_words) / len(term_words)
        if overlap >= _ROLE_REPORT_THRESHOLD:
            matched.append(term)
        best = max(best, overlap)
    return best, tuple(dict.fromkeys(matched))


def score_role(title: str, signal: ProfileSignal) -> tuple[float, tuple[str, ...]]:
    """Best overlap between the job title and any of the profile's own roles."""
    if not signal.role_terms:
        return 0.0, ()
    title_lower = title.lower()
    return _best_role_overlap(
        title_lower, set(_lower_words(title_lower)) - _ROLE_STOPWORDS, signal.role_terms
    )


def score_entry(
    text: str, title_lower: str, title_words: set[str], entry: ProfileEntry
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    """How well one resume entry, on its own, fits one posting.

    The whole point of the entry-aware path: this asks "would I put this entry
    on the page for this job", which is the question a tailored resume actually
    answers. An entry that does not fit costs nothing here -- it is simply left
    off, exactly as it would be left off the page.
    """
    weight = 0.0
    matched: list[str] = []
    for anchor in entry.anchors:
        if _names(text, anchor):
            matched.append(anchor)
            weight += 0.5 if len(anchor) <= _SHORT_ANCHOR_CHARS else 1.0
    skills = min(1.0, weight / ENTRY_ANCHOR_SATURATION)
    role, matched_roles = _best_role_overlap(title_lower, title_words, entry.role_terms)
    fit = ENTRY_WEIGHTS["skills"] * skills + ENTRY_WEIGHTS["role"] * role
    return fit, tuple(matched), matched_roles


def score_level(title: str, signal: ProfileSignal, description: str = "",
                employment_type: str = "", level: str = "") -> float:
    """How well the posting's seniority matches the profile's.

    Neutral is 0.5: most titles say nothing about level, and an unlabelled
    posting should neither be rewarded nor punished. A title carrying both
    signals ("Senior Intern Program Lead") also lands neutral -- job_level
    reports that as `conflict`, which is checked before the level itself so the
    contract is preserved exactly.

    Backed by job_level rather than a junior/senior regex pair, because that
    pair got two whole classes wrong, both silently and both against a student:

      * "Campus Recruiter" scored **1.0**, a perfect match. It is a staff job
        whose subject happens to be students. job_level's program-role override
        exists for exactly this.
      * "Software Developer (Winter 2027)" scored 0.5, neutral. It is a student
        posting, and the term is the only thing that says so -- no level word
        appears in the title at all.

    *level* is the value stamped at scrape time, where the platform's own
    employment-type field was available; it is trusted when present rather than
    re-derived from the title alone.
    """
    verdict = None
    name = str(level or "").strip()
    if name not in job_level.LEVELS:
        verdict = job_level.classify(title, description, employment_type)
        name = verdict.level
        # A title pulling both ways is not evidence either way.
        if verdict.conflict:
            return 0.5

    early = name in job_level.EARLY_CAREER or name == job_level.JUNIOR
    if name == job_level.MID:
        return 0.5
    if signal.seniority == "student":
        return 1.0 if early else 0.1
    return 0.2 if early else 1.0


def score_location(location: str, signal: ProfileSignal) -> float:
    if not signal.locations:
        return 0.5
    lowered = location.lower()
    if not lowered:
        return 0.5
    for place in signal.locations:
        if place and place in lowered:
            return 1.0
    if "remote" in lowered or "anywhere" in lowered:
        return 0.9
    return 0.3


def _blend(components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Weighted mean over whichever components this run produced."""
    table = weights if weights is not None else WEIGHTS
    total_weight = sum(table[name] for name in components)
    if total_weight <= 0:
        return 0.0
    return sum(components[name] * table[name] for name in components) / total_weight


def score_job(job: ArchivedJob, signal: ProfileSignal) -> JobScore:
    """Prefilter score for one job.

    Two paths. A profile with parsed entries is scored on the resume it could
    build -- the best single entry (`top1`) plus whether enough others stand
    behind it to fill a page (`depth`). A profile without them (prose-only,
    legacy, unseeded) keeps the flat whole-document blend, unchanged.

    Why depth matters: a union of skill anchors cannot tell a profile with one
    embedded project from a profile with five, because the union is identical.
    Only "how good is the fourth-best entry" separates them, and that is
    exactly what decides whether a tailored resume can be assembled at all.
    """
    text = job.match_text()
    level = score_level(job.title, signal, job.description,
                        job.employment_type, job.level)
    location = score_location(job.location, signal)

    if signal.entries:
        title_lower = job.title.lower()
        title_words = set(_lower_words(title_lower)) - _ROLE_STOPWORDS
        fits: list[tuple[str, float]] = []
        anchors: list[str] = []
        roles: list[str] = []
        for entry in signal.entries:
            fit, matched_anchors, matched_roles = score_entry(
                text, title_lower, title_words, entry
            )
            fits.append((entry.entry_id, fit))
            if fit > 0:
                anchors.extend(matched_anchors)
                roles.extend(matched_roles)
        fits.sort(key=lambda pair: pair[1], reverse=True)
        top1 = fits[0][1] if fits else 0.0
        # Denominator is the pool size when the pool is small, so a profile
        # with three entries is not permanently capped at three quarters of
        # the depth score a profile with four can reach.
        span = min(DEPTH_ENTRIES, len(fits))
        depth = sum(fit for _, fit in fits[:span]) / span if span else 0.0
        components = {
            "top1": top1,
            "depth": depth,
            "level": level,
            "location": location,
        }
        return JobScore(
            total=_blend(components, BUILD_WEIGHTS),
            components=components,
            matched_anchors=tuple(dict.fromkeys(anchors)),
            matched_roles=tuple(dict.fromkeys(roles)),
            entry_fits=tuple(pair for pair in fits if pair[1] > 0)[:DEPTH_ENTRIES],
        )

    skills, matched_anchors = score_skills(text, signal)
    role, matched_roles = score_role(job.title, signal)
    components = {"skills": skills, "role": role, "level": level, "location": location}
    return JobScore(
        total=_blend(components),
        components=components,
        matched_anchors=matched_anchors,
        matched_roles=matched_roles,
    )


# ---------------------------------------------------------------------------
# Description enrichment
# ---------------------------------------------------------------------------

def _failure_status(exc: BaseException) -> str:
    """Classify a failed fetch so the cache knows how long to remember it.

    scrape_job_posting never returns None -- it re-raises the last transport
    error once every rung of its ladder is exhausted -- so the exception is the
    only signal available. A 404 is a fact about the posting; a timeout is a
    fact about the moment, and the two deserve very different lifetimes.
    """
    from services.jba import description_cache

    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 404 or status_code == 410:
        return description_cache.STATUS_DEAD
    if status_code in (401, 403, 429):
        return description_cache.STATUS_BLOCKED
    return description_cache.STATUS_ERROR


def enrich_descriptions(
    jobs: list[ArchivedJob],
    time_budget: float = ENRICH_TIME_BUDGET_SECONDS,
    fetch_timeout: int = ENRICH_FETCH_TIMEOUT_SECONDS,
    max_fetches: int | None = None,
    use_cache: bool = True,
) -> int:
    """Fetch the real posting text for `jobs`, in place. Returns how many were
    filled in.

    The archive stores what the watchers sent -- a title, a company, a location
    -- because that is all a search-results page exposes. Fewer than 2% of
    records carry a description, so without this the judge is scoring job
    titles. This fetches the actual posting through the same ladder the resume
    commands use (listing.scrape_job_posting: per-platform APIs first, browser
    last), which is why no new per-board fetching lives here.

    Deliberately best-effort. A posting that 404s, rate-limits, blocks the
    client, or simply takes too long is left with its title, and the judge is
    told which entries came with a description. A ranking that mostly works is
    worth far more than one that fails because one board was down.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from services.jba import description_cache

    pending = [job for job in jobs if job.link and not job.description]
    if not pending:
        return 0

    filled = 0

    # Cache first. Hits are free, so they are never rationed -- only the misses
    # consume the fetch budget. That is what makes coverage compound: every run
    # starts with everything previous runs paid for and spends its whole budget
    # on postings nobody has fetched yet.
    if use_cache:
        cached = description_cache.lookup(job.link for job in pending)
        if cached:
            still_pending: list[ArchivedJob] = []
            for job in pending:
                entry = cached.get(description_cache.url_key(job.link))
                if entry is None:
                    still_pending.append(job)
                    continue
                status, body = entry
                if status == description_cache.STATUS_OK:
                    if not body:
                        # An ok row with no text should not exist -- store_many
                        # cannot write one -- but a corrupt or truncated row
                        # must not make the posting silently disappear from
                        # both the cache path and the fetch queue.
                        still_pending.append(job)
                        continue
                    job.description = body[:ENRICH_DESCRIPTION_CHARS]
                    filled += 1
                # A remembered failure is skipped, not retried: re-attempting a
                # dead or blocking posting every run spends the scarcest budget
                # in the command rediscovering what is already known.
            pending = still_pending

    if max_fetches is not None:
        pending = pending[:max_fetches]
    if not pending:
        return filled

    outcomes: list[tuple[str, str, str]] = []

    def fetch(job: ArchivedJob) -> tuple[ArchivedJob, str, str]:
        from services.jba import description_cache
        from services.resumes.listing import scrape_job_posting

        try:
            scraped = scrape_job_posting(
                job.link,
                timeout_seconds=fetch_timeout,
                # Bulk pass: never queue behind the single shared browser context.
                allow_browser=False,
            )
        except Exception as exc:
            return job, "", _failure_status(exc)
        body = _posting_body(getattr(scraped, "description", "") or "")
        return job, body, (
            description_cache.STATUS_OK if body else description_cache.STATUS_EMPTY
        )

    # Modest pool: these are outbound calls to a handful of job boards, and
    # hammering one of them is the fastest way to get the bot rate-limited.
    workers = capacity.workers(6, minimum=2, maximum=len(pending))
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(fetch, job) for job in pending]
        # The budget belongs on as_completed, not on a check inside the loop:
        # as_completed blocks until the next future finishes, so a per-iteration
        # deadline check is never reached while a slow fetch is outstanding --
        # exactly the case the budget is for.
        try:
            for future in as_completed(futures, timeout=time_budget):
                try:
                    job, description, status = future.result()
                except Exception:
                    continue
                # Capped at the same length the record keeps: a cache hit is
                # read back through the same truncation, so anything longer
                # would be stored forever and never read.
                outcomes.append(
                    (job.link, description[:ENRICH_DESCRIPTION_CHARS], status)
                )
                if description:
                    # Stored long, excerpted at prompt-build time.
                    job.description = description[:ENRICH_DESCRIPTION_CHARS]
                    filled += 1
        except TimeoutError:
            # Whatever came back before the deadline is still worth keeping;
            # the fetches still in flight are simply not recorded.
            pass
    finally:
        # wait=False, not a `with` block: the context manager's exit calls
        # shutdown(wait=True), which blocks on any fetch still running and so
        # would sail straight past the deadline -- the exact thing the budget
        # exists to prevent. cancel_futures drops the ones not yet started; a
        # thread already in a socket read cannot be killed from outside, so it
        # is left to finish into a result nobody reads, bounded by
        # fetch_timeout.
        pool.shutdown(wait=False, cancel_futures=True)

    if use_cache and outcomes:
        # Written after the pool is released so a slow database cannot extend
        # the fetch budget the shutdown above just enforced.
        description_cache.store_many(outcomes)

    return filled


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


#: scrape_job_posting() returns the body behind a fixed metadata preamble
#: ("Source site: / Posting URL: / Title: / Company: / Location: / Description:").
#: Every one of those fields is already a column in the judge prompt, so keeping
#: the preamble would spend ~150 of each posting's character budget restating
#: what the model was just told.
_POSTING_BODY_RE = re.compile(r"^.*?\bDescription:\s*\n", re.DOTALL)


#: Board markup that survives scraping: escaped punctuation ("$20\\.00") and
#: markdown emphasis/bullets. Only ~2% of a description, but it is pure noise
#: in a prompt and free to remove.
_MARKUP_ESCAPE_RE = re.compile(r"\\([.\-*()#+&$%!])")
_MARKUP_BULLET_RE = re.compile(r"(?:^|\s)\*(?=\s)")


def compact_description(description: str, limit: int = JUDGE_DESCRIPTION_CHARS) -> str:
    """The part of a posting worth spending prompt budget on.

    A plain head slice is the wrong half. Postings open with the company pitch
    and close with the qualifications, so cutting at a fixed length keeps the
    marketing and drops the requirements -- measured over the real archive, a
    1000-char head kept 53 of 144 requirement sentences. Routing the same text
    through excerpt_job_description at 700 keeps 96, in fewer characters.
    """
    if not description:
        return ""
    from services.resumes.structured import excerpt_job_description

    cleaned = _MARKUP_ESCAPE_RE.sub(r"\1", description).replace("**", "")
    cleaned = " ".join(_MARKUP_BULLET_RE.sub(" ", cleaned).split())
    return excerpt_job_description(cleaned, limit)


def _posting_body(raw: str) -> str:
    """The posting's own text, without the scraper's metadata preamble."""
    from services.job_service import fix_text_encoding

    stripped = _POSTING_BODY_RE.sub("", raw, count=1)
    # No marker means an unexpected shape; the whole blob still beats nothing.
    return normalize_space(fix_text_encoding(stripped or raw))


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT_HEADER = """You rank job postings against one candidate's resume profile.

For each posting, decide how well it fits THIS candidate specifically, and
return a score from 0 to 100:

  85-100  strong fit: right field, right seniority, skills the candidate has
  60-84   plausible fit: adjacent field or partly-matching skills
  30-59   weak fit: same broad industry, wrong specialty or wrong level
  0-29    not a fit: different field, or a seniority the candidate cannot hold

Weigh seniority heavily. A posting that requires years of professional
experience is a poor fit for a student, however well the technical keywords
line up, and an internship is a poor fit for a senior professional.

Weigh location too. Cap a posting the candidate plausibly cannot take -- a
different country, or one demanding a language the profile gives no evidence
of -- well below a comparable role near where they have actually worked, or a
remote one. Do not cap a posting merely for being in another city.

Judge only from what is written. Every posting carries a title, a company and a
location; some also carry the posting's own description text. Do not invent
requirements that are not shown, and do not reward a posting for a vague title.

A missing description is missing evidence, not bad evidence. Score those on
their title as best you can -- never rank a posting below another simply
because it has less text attached.

The candidate profile below is a POOL of resume entries, each labelled with a
bracketed id. Any one resume uses only about four of them. So score the best
resume this candidate could BUILD for the posting, not how well the whole
profile matches it: entries that do not fit are simply left off the page and
must not count against the posting.

For every posting that carries a description, also return:
  "use"     the entry ids you would put on that page, best first, at most 4
  "missing" the one requirement nothing in the pool evidences, max 8 words

Omit "use" and "missing" for a posting with no description -- a title alone is
not enough to choose entries from, and guessing would misreport what you know.

Return JSON only, no prose:
{"ranked": [{"id": <int>, "score": <int 0-100>, "reason": "<max 12 words>",
             "use": ["<entry id>", ...], "missing": "<max 8 words>"}]}

Include every posting id exactly once. The reason must say what drove the
score for that specific posting -- never a generic phrase.
"""

#: Postings go in as delimited rows rather than pretty-printed JSON. The field
#: names were being repeated once per posting, which cost ~4,800 characters of
#: punctuation in a real 60-posting prompt -- around 12% of the whole call, for
#: no information. One header line says what the columns are instead.
POSTING_TABLE_HEADER = (
    "Postings, one per line, fields separated by | --\n"
    "id|title|company|location|description (description may be absent):"
)


def _cell(text: str, limit: int) -> str:
    """One table cell: single-line, and never carrying the field separator."""
    return " ".join(str(text or "").split()).replace("|", "/")[:limit]


def build_judge_prompt(signal: ProfileSignal, jobs: list[ArchivedJob]) -> str:
    """Profile + numbered candidate rows. Ids are list indices, so the response
    maps back positionally and a hallucinated id is simply dropped."""
    rows: list[str] = []
    for index, job in enumerate(jobs):
        cells = [
            str(index),
            _cell(job.title, 200),
            _cell(job.company, 80),
            _cell(job.location, 80),
        ]
        description = compact_description(job.description)
        if description:
            cells.append(_cell(description, JUDGE_DESCRIPTION_CHARS))
        rows.append("|".join(cells))
    return (
        JUDGE_PROMPT_HEADER
        + "\nCandidate profile:\n"
        + signal.document_text
        + "\n\n"
        + POSTING_TABLE_HEADER
        + "\n"
        + "\n".join(rows)
    )


@dataclass(slots=True)
class JudgeVerdict:
    """One posting's verdict. `entries`/`gap` are empty whenever the model did
    not answer with a build plan, which is the whole-profile behaviour this
    replaced -- so an older or terser response degrades instead of failing."""

    score: float
    reason: str = ""
    entries: tuple[str, ...] = ()
    gap: str = ""


def parse_judge_response(
    text: str, count: int, entry_ids: frozenset[str] = frozenset()
) -> dict[int, JudgeVerdict] | None:
    """Parse the judge payload into ``{index: JudgeVerdict}``.

    Tolerant on purpose -- a malformed entry is skipped rather than failing the
    whole call, since one bad row should not cost the user their ranking. Out
    of range ids and unparseable scores are dropped; ``None`` means the
    response had no usable rankings at all, which advances to the next provider.
    """
    from services.resumes.structured import extract_json_object

    payload = extract_json_object(text)
    ranked = payload.get("ranked") if isinstance(payload, dict) else None
    if not isinstance(ranked, list):
        return None

    result: dict[int, JudgeVerdict] = {}
    for entry in ranked:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry["id"])
            score = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= index < count or index in result:
            continue
        reason = " ".join(str(entry.get("reason") or "").split())[:120]
        result[index] = JudgeVerdict(
            score=max(0.0, min(1.0, score / _LLM_SCORE_SCALE)),
            reason=reason,
            entries=_parse_plan_entries(entry.get("use"), entry_ids),
            gap=" ".join(str(entry.get("missing") or "").split())[:PLAN_GAP_CHARS],
        )

    return result or None


def _parse_plan_entries(raw: Any, entry_ids: frozenset[str]) -> tuple[str, ...]:
    """The ``use`` list, kept only where it names real entries.

    An id the profile does not contain is dropped rather than shown: the plan's
    entire value is that the user can act on it, and an invented entry id is
    worse than no plan at all. Everything here is optional by design -- a
    response with no ``use`` key parses exactly as it did before build plans
    existed, because a validator that rejected it would send the whole prompt
    to the next provider and pay for the call twice.
    """
    if not isinstance(raw, list) or not entry_ids:
        return ()
    named: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        candidate = item.strip().strip("[]").lower()
        if candidate in entry_ids and candidate not in named:
            named.append(candidate)
    return tuple(named[:MAX_PLAN_ENTRIES])


def judge_jobs(
    signal: ProfileSignal,
    jobs: list[ArchivedJob],
    settings: Any,
    client_factory: Any | None = None,
) -> tuple[dict[int, JudgeVerdict] | None, str]:
    """One batched judge call over the shortlist.

    Returns ``(rankings, status)``; ``rankings`` is None when no provider
    produced a usable response, and ``status`` explains what happened either
    way. Never raises -- the caller falls back to the prefilter ordering.
    """
    if not jobs or not signal.document_text:
        return None, "skipped: nothing to judge"

    from services.resumes.listing import generate_validated_with_providers

    prompt = build_judge_prompt(signal, jobs)
    entry_ids = frozenset(entry.entry_id for entry in signal.entries)
    errors: list[str] = []
    try:
        rankings, provider = generate_validated_with_providers(
            prompt,
            settings,
            lambda text: parse_judge_response(text, len(jobs), entry_ids),
            client_factory,
            json_response=True,
            # Deliberately the bottom of the provider chain. Ranking an
            # archive is bulk work that nobody is blocked on the way they are
            # blocked on .resumebuild, so it should not spend the primary
            # providers' quota; the chain still falls back upward if the
            # last-resort provider is unconfigured or failing.
            lowest_priority_first=True,
            errors=errors,
        )
    except Exception as exc:  # transport bugs must not take the command down
        return None, f"failed: {exc}"

    if rankings is None:
        return None, "failed: " + ("; ".join(errors) if errors else "no providers configured")
    return rankings, f"ok:{provider}"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MatchReport:
    matches: list[JobMatch]
    scanned: int
    dates: list[str]
    profile_key: str
    #: "ok:<provider>" when the judge ranked these, otherwise why it did not.
    ranker: str = "skipped"
    #: The profile carried no text at all (no baseinfo.txt, or an empty one).
    #: Surfaced rather than swallowed: with nothing to match against, a printed
    #: ranking would read as a real answer.
    #:
    #: Deliberately narrower than ProfileSignal.is_empty. A profile written as
    #: prose, with no `== SKILL ANCHORS ==` section and no `[tag]` entries, has
    #: no prefilter signal but is still perfectly rankable by the judge -- only
    #: the shortlisting degrades, so that is not a case worth refusing.
    profile_empty: bool = False
    #: How many of the judged postings had their real description fetched.
    #: Reported because it is the difference between a ranking made on job
    #: titles and one made on the postings themselves.
    enriched: int = 0

    @property
    def judged(self) -> bool:
        return self.ranker.startswith("ok:")


def rank_jobs(
    signal: ProfileSignal,
    jobs: Iterable[ArchivedJob],
    settings: Any = None,
    limit: int = DEFAULT_LIMIT,
    dates: list[str] | None = None,
    client_factory: Any | None = None,
    enrich: bool = True,
    llm_candidates: int | None = None,
    enrich_candidates: int | None = None,
) -> MatchReport:
    """Prefilter every job, then have the judge score the shortlist.

    Jobs the judge did not score keep their prefilter score and sort below the
    judged ones: an unscored job has not been shown to be better than one the
    judge actually looked at, so it must not displace it.
    """
    job_list = list(jobs)
    scored = [(job, score_job(job, signal)) for job in job_list]
    scored.sort(key=lambda pair: pair[1].total, reverse=True)

    capped = max(1, min(int(limit), MAX_LIMIT))
    ranker = "skipped: no LLM settings"
    enriched = 0
    # A quota-limited caller judges a shorter shortlist and fetches fewer
    # descriptions; both default to the full allowance so nothing changes for
    # a caller with no quota.
    shortlist_size = max(1, llm_candidates if llm_candidates is not None else LLM_CANDIDATES)
    fetch_budget = max(
        1, enrich_candidates if enrich_candidates is not None else ENRICH_CANDIDATES
    )
    if settings is not None and scored:
        shortlist = scored[:shortlist_size]
        if enrich:
            # The whole shortlist is offered to the cache -- a hit costs nothing
            # -- while ENRICH_CANDIDATES rations only the network round trips.
            # A job that was not enriched still reaches the judge, so it is
            # never dropped for it.
            enriched = enrich_descriptions(
                [job for job, _ in shortlist], max_fetches=fetch_budget
            )
        rankings, ranker = judge_jobs(
            signal, [job for job, _ in shortlist], settings, client_factory
        )
        if rankings is not None:
            judged: list[tuple[ArchivedJob, JobScore]] = []
            unjudged: list[tuple[ArchivedJob, JobScore]] = []
            for index, (job, score) in enumerate(shortlist):
                verdict = rankings.get(index)
                if verdict is None:
                    unjudged.append((job, score))
                    continue
                score.llm_score = verdict.score
                score.llm_reason = verdict.reason
                score.plan_entries = verdict.entries
                score.plan_gap = verdict.gap
                score.total = verdict.score
                judged.append((job, score))
            judged.sort(key=lambda pair: pair[1].total, reverse=True)
            scored = judged + unjudged + scored[shortlist_size:]

    return MatchReport(
        matches=[JobMatch(job=job, score=score) for job, score in scored[:capped]],
        scanned=len(job_list),
        dates=dates or [],
        profile_key=signal.profile_key,
        ranker=ranker,
        profile_empty=not signal.document_text,
        enriched=enriched,
    )


def best_jobs(
    profile_key: str | int | None,
    window: str = "day",
    limit: int = DEFAULT_LIMIT,
    settings: Any = None,
    end_date: date | None = None,
    cache_root: Path | None = None,
    enrich: bool = True,
    llm_candidates: int | None = None,
    enrich_candidates: int | None = None,
    role_filters: Iterable[str] = (),
    exclusion_terms: Iterable[str] = (),
    allow_north_america: bool | None = None,
) -> MatchReport:
    """Top-``limit`` archived jobs for ``profile_key`` over ``window``.

    ``role_filters``/``exclusion_terms``/``allow_north_america`` scope the
    archive to one channel's job-watcher settings first -- without them this
    ranks the entire cross-channel archive, so a channel searching for one
    thing would see its best matches diluted by postings for whatever every
    other channel on the bot searches for. The channel's free-text
    ``keywords`` setting is deliberately not used for this -- see
    _matches_channel_scope for why a lexical match on it is worse than no
    filter at all.

    Blocking (archive reads plus one LLM call) -- call it through the priority
    scheduler, not straight off the event loop.
    """
    days = WINDOW_DAYS.get(str(window).lower(), WINDOW_DAYS["day"])
    profile_dir = resolve_profile_dir(profile_key, cache_root)
    signal = build_profile_signal(profile_dir, profile_key=str(profile_key or profile_dir.name))
    jobs, dates = load_window_jobs(days, end_date=end_date)
    jobs = [
        job for job in jobs
        if _matches_channel_scope(job, role_filters, exclusion_terms, allow_north_america)
    ]
    return rank_jobs(
        signal,
        jobs,
        settings=settings,
        limit=limit,
        dates=dates,
        enrich=enrich,
        llm_candidates=llm_candidates,
        enrich_candidates=enrich_candidates,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# orphan-ok: format_match_report deliberately stopped printing this line (see
# the note above its listing loop) -- five-line entries buried the links the
# command exists to surface. The plan is still computed and still on the
# JobScore; this is the agreed rendering for any surface that wants it, kept
# next to _reason_text so the two stay consistent.
def plan_text(score: JobScore) -> str:
    """The "which parts would I use" line, when there is a real answer.

    Prefers the judge's plan over the prefilter's keyword-level entry fits: the
    judge read the posting text, the prefilter matched strings against a title.
    """
    entries = score.plan_entries or tuple(
        entry_id for entry_id, _ in score.entry_fits[:3]
    )
    if not entries:
        return ""
    line = "use: " + ", ".join(entries)
    if score.plan_gap:
        line += f" | missing: {score.plan_gap}"
    return line


def _reason_text(score: JobScore) -> str:
    """The judge's own reason when it scored this job, else the prefilter's."""
    if score.llm_reason:
        return score.llm_reason
    parts: list[str] = []
    if score.matched_anchors:
        parts.append("skills: " + ", ".join(score.matched_anchors[:6]))
    if score.matched_roles:
        parts.append("role: " + ", ".join(score.matched_roles[:3]))
    level = score.components.get("level", 0.5)
    if level >= 0.9:
        parts.append("level fit")
    elif level <= 0.2:
        parts.append("level mismatch")
    if score.components.get("location", 0.5) >= 0.9:
        parts.append("location fit")
    return " | ".join(parts)


def _display_fields(job: ArchivedJob) -> tuple[str, str, str]:
    """``(title, company, location)`` for one line of the report.

    The archive holds two record shapes. An ATS record has a real ``company``
    field, a real ``location`` field, and a title that means what it says --
    its trailing parenthetical is part of the role ("Outreach Coordinator
    (Cantonese/Mandarin)") and must survive. A send-path record has neither
    field filled, because the watcher folded both into the title as "Role
    (Company, City, Region, Country)"; printing that raw is what named the
    company twice.

    Splitting the fold at the first comma separates the two halves: the company
    goes to the front of the line, the rest stays as the location.

    The presence of a ``company`` field is what tells the two shapes apart, and
    over a week of real archive it separates them exactly. The comma is
    required as a second guard, so an aggregator title ending in a plain
    "(Co-op/Intern)" is left alone rather than read as a company.
    """
    if job.company:
        # Strip the trailing parenthetical only when it is exactly the location,
        # so a send-path record keys the same as the older shape of the same job
        # and an ATS title's own "(Cantonese/Mandarin)" is left alone.
        head, inner = _split_trailing_paren(job.title)
        if head and job.location and inner.strip().casefold() == job.location.strip().casefold():
            return head, job.company, job.location
        return job.title, job.company, job.location
    head, inner = _split_trailing_paren(job.title)
    # A record with a known location and no company has nothing to recover: the
    # parenthetical is the location, and splitting it would report the city as
    # the company ("Data Intern (Toronto, ON)" -> company "Toronto").
    if head and job.location and inner.strip().casefold() == job.location.strip().casefold():
        return head, "", job.location
    if head and "," in inner:
        company, _, location = inner.partition(",")
        if company.strip():
            return head, company.strip(), location.strip()
    return job.title, "", job.location


def format_match_report(report: MatchReport, window: str) -> str:
    """Plain-text report for a Discord channel (chunked by the caller)."""
    if not report.dates:
        span = window
    elif len(report.dates) == 1:
        span = report.dates[0]
    else:
        span = f"{report.dates[-1]} to {report.dates[0]}"

    if report.judged:
        ranked_by = f"ranked by {report.ranker[len('ok:'):]}"
    else:
        # Say so plainly: these scores mean something weaker than a judged run,
        # and silently serving the prefilter order would misrepresent them.
        ranked_by = f"keyword ranking only ({report.ranker})"
    header = (
        f"Best matches for profile `{report.profile_key}` - {window} ({span})\n"
        f"Scanned {report.scanned:,} archived job(s); {ranked_by}"
        + (f", {report.enriched} with full descriptions." if report.enriched else ".")
    )
    if report.profile_empty:
        return (
            f"Profile `{report.profile_key}` has no resume content to match against "
            f"-- its `baseinfo.txt` is missing or empty. Seed the profile first."
        )
    if not report.matches:
        return f"{header}\n\nNo archived jobs in that window."

    # Listing and score only. The judge's reasoning, the build plan and the
    # posting date are all still computed and still on the JobScore -- they are
    # what the ranking is made of, and callers and tests can read them -- but a
    # channel full of five-line entries buries the one thing the command is for,
    # which is which postings to open.
    lines = [header, ""]
    for rank, match in enumerate(report.matches, start=1):
        job = match.job
        title, company, location = _display_fields(job)
        # Board and employer lead the line: they are what the eye scans for,
        # and they are the same shape on every row, unlike the titles.
        tag = "/".join(
            part for part in (job.site_label, company)
            if part and part.lower() != "unknown"
        )
        lines.append(
            f"{rank}. [{match.score.total * 100:.0f}%] "
            + (f"[{tag}] " if tag else "")
            + title
            + (f" ({location})" if location else "")
        )
        if job.link:
            lines.append(f"   {job.link}")
    return "\n".join(lines).rstrip()
