"""Seniority classification for job *titles*.

`_matches_keywords` in ats_service is a bare OR over word-boundary title
tokens, so a search for "software engineer intern" matches "Staff Software
Engineer" on the word `engineer` alone, and misses "Software Engineer, Co-op
(Winter 2027)" entirely because no query word appears in it. job_match's
`_infer_seniority` does this properly, but only for the *resume* side; the job
side had no counterpart.

This module is the job-side counterpart. It reads the title and returns a
decision, not a score: job_match.score_level returns a neutral 0.5 when a title
looks both junior and senior, which is right for ranking and wrong for
filtering. A filter has to choose, and for "Senior Intern Program Manager" the
safe choice is to exclude -- one staff role in a co-op feed costs more than one
missed match.

Deliberately its own module rather than living in ats_service (2,377 lines, and
every scraper would drag in the ranking machinery) or job_match (whose markers
are tuned for scoring against a resume, not for gating a scrape).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Ordered most junior to most senior. `min` over the hits picks the level, so
# the order here is load-bearing.
COOP = "coop"
INTERN = "intern"
CAMPUS = "campus"
NEWGRAD = "newgrad"
JUNIOR = "junior"
MID = "mid"
SENIOR = "senior"
STAFF = "staff"

LEVELS: tuple[str, ...] = (COOP, INTERN, CAMPUS, NEWGRAD, JUNIOR, MID, SENIOR, STAFF)
TIER: dict[str, int] = {name: i for i, name in enumerate(LEVELS)}

# Everything an early-career search should keep. Exposed so callers do not
# hardcode the list and drift from it.
EARLY_CAREER: frozenset[str] = frozenset({COOP, INTERN, CAMPUS, NEWGRAD})

# Suffixes are an explicit bounded set, never `intern\w*`: that form matches
# "Internal Auditor", "International Tax Analyst" and "Interventional
# Radiologist", which is the highest-volume false positive available here. Same
# reasoning for co-op, which must not reach "Co-operative".
_MARKERS: tuple[tuple[str, str], ...] = (
    (COOP, r"co[-\s]?op(?:s)?"),
    (COOP, r"co[-\s]?operative\s+education"),
    (INTERN, r"intern(?:s|ship|ships)?"),
    (INTERN, r"summer\s+analyst"),
    (INTERN, r"(?:industrial\s+)?placement\s+student"),
    (INTERN, r"stagiaire"),
    (CAMPUS, r"campus"),
    (CAMPUS, r"(?:university|student)\s+program(?:me)?"),
    (CAMPUS, r"apprentice(?:ship)?"),
    (CAMPUS, r"trainee"),
    (CAMPUS, r"rotational\s+program(?:me)?"),
    (CAMPUS, r"early\s+(?:careers?|talent)"),
    (CAMPUS, r"emerging\s+talent"),
    (NEWGRAD, r"new\s?grad(?:uate)?"),
    (NEWGRAD, r"grad(?:uate)?\s+program(?:me)?"),
    (NEWGRAD, r"graduate\s+(?:engineer|developer|analyst)"),
    (NEWGRAD, r"entry[-\s]level"),
    (NEWGRAD, r"university\s+grad(?:uate)?"),
    (NEWGRAD, r"e\.?i\.?t\.?"),
    (JUNIOR, r"junior"),
    (JUNIOR, r"jr\.?"),
    (SENIOR, r"senior"),
    (SENIOR, r"sr\.?"),
    (SENIOR, r"\d+\+?\s*years"),
    (STAFF, r"staff"),
    (STAFF, r"principal"),
    (STAFF, r"distinguished"),
    (STAFF, r"fellow"),
    (STAFF, r"architect"),
    (STAFF, r"director"),
    (STAFF, r"head\s+of"),
    (STAFF, r"v\.?p\.?"),
    (STAFF, r"vice\s+president"),
    (STAFF, r"chief"),
    (STAFF, r"c(?:eo|fo|to|oo|io|so)"),
    (STAFF, r"manager"),
)

# Level suffixes only count directly after a role noun. A bare `\bI\b` or
# `\bV\b` matches initials, roman numerals in product names and the pronoun
# "I"; anchoring to the noun is what makes this safe.
_ROLE_NOUN = r"(?:engineer|developer|analyst|scientist|designer|specialist|consultant)"
_ROLE_RANK = re.compile(rf"\b{_ROLE_NOUN}\s+(i{{1,3}}|iv|[1-4])\b")
_RANK_TIER: dict[str, str] = {
    "i": NEWGRAD, "1": NEWGRAD,
    "ii": MID, "2": MID,
    "iii": SENIOR, "3": SENIOR,
    "iv": SENIOR, "4": SENIOR,
}

# "Lead" is a role noun half the time ("Lead Generation Specialist") and a
# level the other half, so it only counts leading the title or after "tech".
_LEAD = re.compile(r"^(?:tech(?:nical)?\s+)?lead\b(?!\s+generation)")

# The family every large employer posts, and the reason a bare `intern` token
# match is unusable on its own: these are staff roles that *recruit* students.
_PROGRAM_ROLE = re.compile(
    r"\b(?:intern(?:ship)?|co[-\s]?op|campus|student|university|graduate|early\s+careers?)\b"
    r"[\w\s,&/-]{0,20}?"
    # "program" itself is deliberately NOT in this list: "Technology Campus
    # Program" is the student program, not a staff role that runs one. The
    # staff versions all name a person -- coordinator, manager, recruiter.
    r"\b(?:recruit\w*|coordinator|manager|lead|director|host|"
    r"mentor|supervisor|partner|relations)\b"
)

# "Associate" is junior alone and senior beside a senior noun, so it is handled
# apart from _MARKERS rather than given a fixed tier.
_ASSOCIATE = re.compile(r"\bassociate\b")
_ASSOCIATE_SENIOR = re.compile(
    r"\bassociate\s+(?:director|principal|partner|vice\s+president|v\.?p\.?|manager)\b"
)

_TERM = re.compile(r"\b(winter|summer|fall|autumn|spring)\s*/?\s*((?:20)\d{2})\b")
_TERM_REVERSED = re.compile(r"\b((?:20)\d{2})\s+(winter|summer|fall|autumn|spring)\b")
_PAREN_YEAR = re.compile(r"[(\[]\s*(20[2-9]\d)\s*[)\]]")

_DASHES = re.compile(r"[‐-―−]")
_WS = re.compile(r"\s+")

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (level, re.compile(r"\b" + pat + r"\b")) for level, pat in _MARKERS
)


@dataclass(frozen=True)
class LevelVerdict:
    level: str
    term: tuple[str, str] | None = None
    evidence: tuple[str, ...] = field(default_factory=tuple)
    conflict: bool = False

    @property
    def tier(self) -> int:
        return TIER[self.level]

    @property
    def is_early_career(self) -> bool:
        return self.level in EARLY_CAREER


def normalise(title: str) -> str:
    """Casefold, unify dashes, collapse whitespace -- and nothing else.

    Punctuation survives on purpose: `co-op` needs its hyphen and "(Winter
    2027)" needs its parentheses.
    """
    text = _DASHES.sub("-", str(title or "")).casefold()
    return _WS.sub(" ", text).strip()


def extract_term(title: str) -> tuple[str, str] | None:
    """Pull ("winter", "2027") out of a title, in either word order.

    A bare parenthesised year yields ("", "2027") -- enough to tell two cycles
    of the same evergreen requisition apart, which is what the term is for.
    """
    text = normalise(title)
    m = _TERM.search(text)
    if m:
        return (m.group(1), m.group(2))
    m = _TERM_REVERSED.search(text)
    if m:
        return (m.group(2), m.group(1))
    m = _PAREN_YEAR.search(text)
    if m:
        return ("", m.group(1))
    return None


def classify(title: str, description: str = "", employment_type: str = "") -> LevelVerdict:
    """Classify a posting's level from its title.

    *description* is consulted for exactly one thing -- deciding co-op vs intern
    for a title that carries a term but no level word. Letting it *create* a
    level would classify every posting whose blurb says "our interns love it
    here", and Greenhouse descriptions run to tens of kilobytes.

    *employment_type* is the platform's own field (Lever calls it `commitment`,
    Ashby and schema.org `employmentType`). "Intern" there is evidence of the
    same weight as a title match, and it is the only signal on a posting titled
    "Software Developer (Winter 2027)".
    """
    text = normalise(title)
    term = extract_term(title)
    if not text:
        return LevelVerdict(MID, term=term)

    hits: list[tuple[int, str, str]] = []
    for level, pattern in _COMPILED:
        m = pattern.search(text)
        if m:
            hits.append((TIER[level], level, m.group(0)))

    m = _ROLE_RANK.search(text)
    if m:
        rank_level = _RANK_TIER[m.group(1)]
        hits.append((TIER[rank_level], rank_level, m.group(0)))
    if _LEAD.search(text):
        hits.append((TIER[STAFF], STAFF, "lead"))
    if _ASSOCIATE.search(text):
        if _ASSOCIATE_SENIOR.search(text):
            hits.append((TIER[STAFF], STAFF, "associate"))
        elif not any(h[1] == STAFF for h in hits):
            hits.append((TIER[JUNIOR], JUNIOR, "associate"))

    emp = normalise(employment_type)
    if emp and re.search(r"\bintern(?:ship)?\b", emp):
        hits.append((TIER[INTERN], INTERN, f"employmentType={emp}"))

    # A term with no level word is itself the signal: "Software Developer,
    # Winter 2027" is a student posting, and the OR-over-tokens matcher never
    # saw it because none of its query words appear.
    if term and not hits:
        termed = COOP if re.search(r"\bco[-\s]?op\b", normalise(description)) else INTERN
        hits.append((TIER[termed], termed, f"term={term[0] or 'year'} {term[1]}"))

    if not hits:
        # Checked before the early return: a program role need not contain any
        # level marker ("University Relations Partner"), and returning MID here
        # would let it through an early-career filter that excludes staff.
        if _PROGRAM_ROLE.search(text):
            return LevelVerdict(STAFF, term=term, evidence=("program-role-override",))
        return LevelVerdict(MID, term=term)

    hits.sort()
    level = hits[0][1]
    evidence = tuple(h[2] for h in hits)
    conflict = hits[0][0] <= TIER[NEWGRAD] and hits[-1][0] >= TIER[SENIOR]

    # "Senior Intern Program Manager", "Campus Recruiting Coordinator": staff
    # roles whose subject happens to be students. Overriding after the fact --
    # rather than by never matching -- keeps the evidence visible.
    if conflict or _PROGRAM_ROLE.search(text):
        return LevelVerdict(
            STAFF, term=term,
            evidence=evidence + ("program-role-override",),
            conflict=conflict,
        )

    return LevelVerdict(level, term=term, evidence=evidence, conflict=False)


def allowed(level: LevelVerdict | str, wanted: list[str] | tuple[str, ...] | None) -> bool:
    """Does this level pass a channel's level filter? An empty filter allows all.

    An unrecognised name in *wanted* must not silently widen the filter, so
    membership is tested directly rather than by tier comparison.
    """
    if not wanted:
        return True
    name = level.level if isinstance(level, LevelVerdict) else str(level)
    return name in set(wanted)


# A channel's role filter offers five coarse names; the classifier resolves
# eight. The mapping is deliberately not one-to-one: "entry" is what a person
# means by new-grad *and* campus/apprenticeship intake, and "senior" is what
# they mean by senior *and* staff/principal, which is how the old keyword list
# already treated "lead", "staff" and "principal".
ROLE_FILTER_LEVELS: dict[str, frozenset[str]] = {
    "internship": frozenset({COOP, INTERN}),
    "entry": frozenset({NEWGRAD, CAMPUS}),
    "junior": frozenset({JUNIOR}),
    "mid": frozenset({MID}),
    "senior": frozenset({SENIOR, STAFF}),
}


def matches_role_filter(
    title: str,
    role_filters: list[str] | tuple[str, ...] | None,
    description: str = "",
    employment_type: str = "",
) -> bool:
    """Does this posting satisfy a channel's role filter, judged by level?

    The filter was a substring search over the title: "internship" looked for
    "intern", "co-op" and "coop". That cannot see a posting titled "Software
    Developer (Winter 2027)", which carries no level word at all -- the term and
    the platform's own employment_type are the only signals, and they are
    exactly what classify() reads. A student searching for internships was
    never shown that job.

    Widening only. A caller unions this with the old keyword check, so a
    posting that matched before still matches; this can add results and cannot
    remove them.

    Conflicted titles ("Senior Intern Program Lead") need no handling here:
    classify() already resolves them to STAFF via its program-role override, so
    they simply fail the membership test below. An explicit conflict branch was
    tried and removed -- mutation testing showed it could never change the
    outcome, because the only case it caught was one the membership test
    already rejected.
    """
    if not role_filters:
        return True

    wanted: set[str] = set()
    for name in role_filters:
        wanted |= ROLE_FILTER_LEVELS.get(str(name).strip().lower(), frozenset())
    if not wanted:
        # Every name was unrecognised. Returning True would silently widen the
        # filter to "everything"; the caller's keyword pass is the authority on
        # names this mapping does not know.
        return False

    verdict = classify(title, description=description, employment_type=employment_type)
    return verdict.level in wanted
