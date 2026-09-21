"""Split a job description into one-idea pieces.

Stored descriptions are flattened to a single line for ~70% of postings
(ats_service._html_to_text and job_service.normalize_description_text replace
every tag and line break with a space), so headers, bullets and label: value
fields run together: "Why Work With Us? Flexible hours Work from home ABOUT THE
ROLE As a ...". Classifying such a chunk as a whole forces one label onto
several different things. This recovers the pieces with rules only.

Measured with scripts/segment_eval.py against 1,800 postings that kept their
real line breaks (flattened, then re-split), 2026-09-17:

    split on . ! ? only    missed line breaks 53.3%   mid-sentence cuts 0.0 per 100 splits
    split_description      missed line breaks 34.9%   mid-sentence cuts 2.2 per 100 splits

The remaining misses are mostly bare noun lists ("Paid Time Off Retirement
Savings") and non-English text, where no rule is safe. Real line breaks, when a
description has them, are always kept.
"""
from __future__ import annotations

import re

__all__ = ["split_description", "split_points"]

_HEADERS = [
    "about the role", "about the job", "about the position", "about the team", "about the opportunity",
    "about us", "about you", "who we are", "who you are", "the role", "the opportunity", "the team",
    "role overview", "position overview", "job summary", "job description", "position summary", "overview",
    "summary", "what you'll do", "what you will do", "what you’ll do", "what you'll own", "what you’ll own",
    "your role", "your impact", "in this role you will", "responsibilities", "key responsibilities",
    "duties", "job duties", "main responsibilities", "primary responsibilities", "day to day",
    "a day in the life", "typical day-to-day tasks include",
    "qualifications", "required qualifications", "minimum qualifications", "basic qualifications",
    "preferred qualifications", "desired qualifications", "requirements", "job requirements",
    "what we're looking for", "what we’re looking for", "what we are looking for", "who we're looking for",
    "what you bring", "what you'll bring", "what you’ll bring", "you bring", "you have", "skills",
    "required skills", "preferred skills", "skills and experience", "experience", "education",
    "nice to have", "nice-to-have", "bonus points", "assets", "preferred", "must have", "must-haves",
    "what we offer", "what's in it for you", "what’s in it for you", "benefits", "perks", "perks include",
    "compensation", "compensation & benefits", "compensation and benefits", "why join us",
    "why work with us", "reasons to apply", "our benefits", "salary", "pay range",
    "location", "work location", "schedule", "working conditions", "work environment", "hours",
    "how to apply", "application process", "interview process", "our process", "next steps",
    "equal opportunity employer", "eeo statement", "accommodation", "accessibility", "please note",
    "additional information", "additional details", "other information", "disclaimer",
]
_HEADER_RE = re.compile(
    r"(?<![\w’'])(" + "|".join(re.escape(h) for h in sorted(set(_HEADERS), key=len, reverse=True))
    + r")(?=\s*[:?]?\s+[A-Z0-9“\"(])",
    re.IGNORECASE,
)
# Single common words are headers only when marked as one (a colon, or ALL
# CAPS); Title Case alone is just a sentence start. Matching them loosely made
# 87% of all mid-sentence cuts ("5+ years of | experience in ...").
_AMBIGUOUS_HEADERS = {
    "experience", "skills", "education", "preferred", "assets", "location", "hours", "schedule",
    "salary", "compensation", "summary", "overview", "duties", "accommodation", "accessibility",
    "benefits", "perks", "requirements", "qualifications", "responsibilities", "you have",
    "the role", "the team", "the opportunity", "you bring", "your role", "your impact",
}
# Field labels keep their value on the same piece ("Location: Colindale").
_FIELD_LABELS = {
    "location", "work location", "salary", "pay", "pay range", "compensation", "job type", "shift",
    "schedule", "hours", "department", "reports to", "closing date", "start date", "duration",
    "employment type", "position type", "job id", "req id", "requisition id", "travel", "remote",
    "work model", "work arrangement", "term", "wage", "rate", "hourly rate", "positions", "team",
    "level", "category", "division", "position title", "job title", "title", "status",
}
_SMALL = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "at", "you", "we", "us", "it"}
# A determiner or preposition always continues its phrase: "pursuing a |
# Bachelor's", "3+ Years of | Experience" are never line breaks.
_GLUE_WORDS = {
    "a", "an", "the", "of", "to", "for", "in", "with", "and", "or", "on", "at", "by", "from", "as",
    "your", "our", "their", "its", "is", "are", "be", "will", "can", "into", "about", "including", "such",
}
# Words that begin a real line far more often than they continue one, taken
# from the line starts of 8,044 postings that kept their line breaks (245k
# lines). Function words ("The", "We", "You") are excluded: they start clauses
# mid-paragraph too.
_STARTERS = set("""Experience Strong Work Build Ability Own Lead Partner Support Familiarity Develop Excellent
Comfortable Maintain Design Collaborate Proven Drive Demonstrated Hands Join Desire Comprehensive Manage Preferred
Identify Help Flexible Weekly Comfort Completing Proficiency Use Contribute Assist Coordinate Conduct Working High
Must Training Self Take Create Provide Deep Background Ongoing Additional Mentorship Define Ready Knowledge Track
Passionate Perform Ensure Apply Current Conducting Assessing Advancement Previous Typical Full Bachelor Bachelor's
Master's Degree Understanding Solid Effective Exposure Prior Minimum Participate Prepare Review Analyze
Implement Monitor Write Test Troubleshoot Deliver Communicate Operate Install Inspect Document Establish Evaluate
Oversee Plan Research Resolve Respond Schedule Train Update Verify Able Willingness Eligible Currently Enrolled
Pursuing Valid Proficient Knowledgeable Attention Detail Great Inspire Competitive Generous Health Dental Paid
Remote Hybrid Career Professional Opportunities Opportunity Growth Access Employee Medical Life RRSP Tuition
Collaborating Developing Designing Building Supporting Managing Leading Creating Writing Testing Performing
Providing Maintaining Ensuring Assisting Coordinating Reviewing Analyzing Preparing Monitoring Identifying""".split())
_STARTER_RE = re.compile(
    r"(?<=[A-Za-z0-9%)\]’'&])\s+(?=(" + "|".join(sorted(_STARTERS, key=len, reverse=True)) + r")\b[\s’'-])"
)
_CAPS_WORD = re.compile(r"\b[A-Z][A-Z&’'/-]*[A-Z]\b(?![a-z])")
_ALLCAPS_RUN = re.compile(r"(?<=[a-z.!?:)])\s+(?=[A-Z][A-Z&’' ]{6,}[A-Z](?:\s|:|$))")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[\"“(]?[A-Z0-9•])")
_LABEL = re.compile(
    r"(?<=[a-z0-9)%$.])\s+(?=((?:[A-Z][a-z][\w’'&/-]*)(?:\s+(?:[A-Z][a-z][\w’'&/-]*|of|&|and|to|per)){0,3}):\s+\S)"
)
# Legal notices glued onto the last requirement bullet, with no punctuation
# between: "...release schedules Amazon is an equal opportunity employer",
# "...testing lifecycle. *We may use artificial intelligence tools". Found in
# 5 of the first 10 EEO pieces the labeller saw (2026-09-17).
_NOTICE_START = re.compile(
    r"(?<=[A-Za-z0-9).,;])\s+(?="
    r"(?:[A-Z][\w&.’'-]*\s+){1,4}(?:is|are)\s+(?:an?\s+)?(?:proud\s+)?equal\s+(?:employment\s+)?opportunit"
    r"|[*\"“]?We\s+(?:may\s+)?use\s+(?:AI|artificial\s+intelligence)\b"
    r")"
)
_BULLET_GLYPH = re.compile(r"[•▪●◦·]")
_EDGE_CHARS = " \t\n:-–*•▪●◦·"


def _words_before(text: str, index: int, span: int = 60) -> list[str]:
    return re.findall(r"[\w’'&/-]+", text[max(0, index - span):index])


def split_points(text: str) -> set[int]:
    """Indices in `text` where a new piece starts."""
    points: set[int] = set()

    def add(i: int) -> None:
        while i < len(text) and text[i] in " \t\n":
            i += 1
        if 0 < i < len(text):
            points.add(i)

    for m in re.finditer(r"\n+", text):
        add(m.end())
    for m in _BULLET_GLYPH.finditer(text):
        add(m.end())

    for m in _HEADER_RE.finditer(text):
        header = m.group(1)
        colon = text[m.end(1):m.end(1) + 2].lstrip()[:1] in (":", "?")
        caps = header.isupper()
        title = all(w[:1].isupper() for w in re.findall(r"[A-Za-z][\w’']*", header) if w.lower() not in _SMALL)
        if not (caps or title):
            continue
        if header.lower() in _AMBIGUOUS_HEADERS and not (colon or caps):
            continue
        start, end = m.start(1), m.end(1)
        if caps:
            # One ALL CAPS header, however long ("REQUIREMENTS AND PREFERRED
            # EXPERIENCE:"), is one piece -- extended by whole caps words only.
            for w in reversed(list(_CAPS_WORD.finditer(text, max(0, start - 80), start))):
                if text[w.end():start].strip():
                    break
                start = w.start()
            for w in _CAPS_WORD.finditer(text, end):
                if text[end:w.start()].strip():
                    break
                end = w.end()
        add(start)
        if header.lower() in _AMBIGUOUS_HEADERS and colon and not caps:
            continue  # a field label keeps its value
        while end < len(text) and text[end] in ":?":
            end += 1
        add(end)

    for m in _ALLCAPS_RUN.finditer(text):
        add(m.end())
    for m in _NOTICE_START.finditer(text):
        add(m.end())
    for m in _SENTENCE.finditer(text):
        add(m.end())

    for m in _LABEL.finditer(text):
        words = _words_before(text, m.start())
        if words and (words[-1].lower() in _GLUE_WORDS or re.fullmatch(r"[A-Z][a-z][\w’'&/-]*", words[-1])):
            continue  # mid-phrase, or this match began inside a longer label
        add(m.end())

    # "Key Responsibilities: Coordinate ..." -- a colon closing a short heading
    # that starts its own piece. Field labels stay joined to their value.
    starts = sorted(points | {0})
    for m in re.finditer(r":\s+(?=[A-Z0-9\"“(])", text):
        prev = max((s for s in starts if s <= m.start()), default=0)
        head = text[prev:m.start()].strip()
        words = head.split()
        if not (1 <= len(words) <= 5) or head.lower() in _FIELD_LABELS:
            continue
        if all(w[:1].isupper() or w.lower() in _SMALL or not w[:1].isalpha() for w in words):
            add(m.end())

    for m in _STARTER_RE.finditer(text):
        # No "/" in these words: "Outgoing/friendly/patient Detail" is a bullet
        # break, "A/R Lead" is a job title.
        before = re.findall(r"[\w’'&-]+", text[max(0, m.start() - 30):m.start()])
        after = re.match(r"\S+\s+([\w’'&-]+)", text[m.end():m.end() + 40])
        if before:
            last = before[-1]
            # Title Case before it = inside a phrase ("Equal Employment |
            # Opportunity"); an acronym ("GD&T", "SQL") ends a bullet like any word.
            if last[:1].isupper() and not re.fullmatch(r"[A-Z0-9&/+.#-]{2,}", last):
                continue
            if last.lower() in _GLUE_WORDS:
                continue
        if after and after.group(1)[:1].isupper() and after.group(1).lower() not in _SMALL:
            continue  # starts a Title Case phrase ("Paid Time Off")
        add(m.end())
    return points


def split_description(text: str) -> list[str]:
    """Pieces of `text` in order, edges trimmed, empty fragments dropped."""
    text = text or ""
    cuts = [0, *sorted(split_points(text)), len(text)]
    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        piece = text[a:b].strip(_EDGE_CHARS)
        if len(re.sub(r"\W", "", piece)) >= 2:
            pieces.append(piece)
    return pieces
