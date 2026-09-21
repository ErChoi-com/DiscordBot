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
    (NEWGRAD, r"graduate\s+(?:(?:software|systems?|backend|frontend|data|ml|ai|qa|cloud)\s+)?(?:engineer|developer|analyst)"),
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


_SENIORITY_MODEL_CACHE = None
_SENIORITY_MODEL_LOADED = False


def _get_fine_tuned_seniority_model():
    global _SENIORITY_MODEL_CACHE, _SENIORITY_MODEL_LOADED
    if _SENIORITY_MODEL_LOADED:
        return _SENIORITY_MODEL_CACHE

    _SENIORITY_MODEL_LOADED = True
    try:
        import json
        from pathlib import Path
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        root = Path(__file__).resolve().parents[2]
        model_dir = root / "models" / "seniority_encoder"
        pt_path = model_dir / "seniority_encoder.pt"
        meta_path = model_dir / "meta.json"
        if not pt_path.is_file() or not meta_path.is_file():
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = meta.get("labels", [INTERN, NEWGRAD, JUNIOR, MID, SENIOR, STAFF])
        base_model = meta.get("backbone") or meta.get("base") or "BAAI/bge-small-en-v1.5"
        prefix = meta.get("prefix", "")
        feature_dim = int(meta.get("embedding_dim", 384))
        max_len = int(meta.get("max_len", 64))

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        class DualSeniorityClassifier(nn.Module):
            def __init__(self, base: str, num_classes: int, feat_dim: int):
                super().__init__()
                self.feat_dim = feat_dim
                self.encoder = AutoModel.from_pretrained(base, trust_remote_code=True)
                self.drop = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.feat_dim, num_classes)

            def forward(self, input_ids, attention_mask):
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                if pooled.shape[-1] > self.feat_dim:
                    pooled = pooled[:, :self.feat_dim]
                pooled = nn.functional.normalize(pooled, p=2, dim=-1)
                return self.classifier(self.drop(pooled))

        model = DualSeniorityClassifier(base_model, len(labels), feature_dim)
        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()

        _SENIORITY_MODEL_CACHE = (model, tokenizer, labels, prefix, max_len)
        return _SENIORITY_MODEL_CACHE
    except Exception:
        return None


def classify_title_neural(title: str) -> tuple[str, float] | None:
    """Classify title level using fine-tuned seniority encoder or pure Nomic SemanticEngine."""
    if not title or not title.strip():
        return None

    # 1. Try fine-tuned seniority model if available
    ft = _get_fine_tuned_seniority_model()
    if ft is not None:
        try:
            import torch
            model, tokenizer, labels, prefix, max_len = ft
            inp = f"{prefix}{title.strip()}"
            enc = tokenizer(inp, return_tensors="pt", truncation=True, max_length=max_len)
            with torch.no_grad():
                logits = model(enc["input_ids"], enc["attention_mask"])
                probs = torch.softmax(logits, dim=-1)[0]
                idx = int(torch.argmax(probs))
                return labels[idx], float(probs[idx])
        except Exception:
            pass

    # 2. Fall back to pure Nomic SemanticEngine
    try:
        from services.semantic.engine import get_semantic_engine
        return get_semantic_engine().classify_seniority(title)
    except Exception:
        return None


def _classify_rule_based(title: str, description: str = "", employment_type: str = "") -> LevelVerdict:
    """Deterministic regex classification fallback."""
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

    if term and not hits:
        termed = COOP if re.search(r"\bco[-\s]?op\b", normalise(description)) else INTERN
        hits.append((TIER[termed], termed, f"term={term[0] or 'year'} {term[1]}"))

    if not hits:
        if _PROGRAM_ROLE.search(text):
            return LevelVerdict(STAFF, term=term, evidence=("program-role-override",))
        return LevelVerdict(MID, term=term)

    hits.sort()
    level = hits[0][1]
    evidence = tuple(h[2] for h in hits)
    conflict = hits[0][0] <= TIER[NEWGRAD] and hits[-1][0] >= TIER[SENIOR]

    if conflict or _PROGRAM_ROLE.search(text):
        return LevelVerdict(
            STAFF, term=term,
            evidence=evidence + ("program-role-override",),
            conflict=conflict,
        )

    return LevelVerdict(level, term=term, evidence=evidence, conflict=False)


def classify(
    title: str,
    description: str = "",
    employment_type: str = "",
    use_neural: bool = False,
) -> LevelVerdict:
    """Classify a posting's level from its title, with optional neural SemanticEngine pass."""
    text = normalise(title)
    term = extract_term(title)
    if not text:
        return LevelVerdict(MID, term=term)

    # 1. Collect initial pattern signals for conflict and evidence tracking
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

    hits.sort()
    conflict = bool(hits and hits[0][0] <= TIER[NEWGRAD] and hits[-1][0] >= TIER[SENIOR])
    is_program_role = bool(_PROGRAM_ROLE.search(text))

    # Conflict or student program manager/recruiter always resolves to STAFF
    if conflict or is_program_role:
        evidence = tuple(h[2] for h in hits)
        if is_program_role and "program-role-override" not in evidence:
            evidence = evidence + ("program-role-override",)
        return LevelVerdict(STAFF, term=term, evidence=evidence, conflict=conflict)

    # Platform's own employment_type (e.g. Lever commitment / Ashby employmentType)
    has_intern_emp = bool(emp and re.search(r"\bintern(?:ship)?\b", emp))
    if has_intern_emp:
        return LevelVerdict(INTERN, term=term, evidence=tuple(h[2] for h in hits), conflict=False)

    # If title has an explicit student term and no hits, it's INTERN
    if term and not hits:
        termed = COOP if re.search(r"\bco[-\s]?op\b", normalise(description)) else INTERN
        hits.append((TIER[termed], termed, f"term={term[0] or 'year'} {term[1]}"))
        hits.sort()

    if hits:
        level = hits[0][1]
        evidence = tuple(h[2] for h in hits)
        return LevelVerdict(level, term=term, evidence=evidence, conflict=False)

    # 2. Neural classification pass for titles without explicit keyword markers
    if use_neural:
        neural = classify_title_neural(text)
        if neural is not None:
            pred, conf = neural
            # Guard: Lead generation specialist is mid, not staff/senior
            if "lead generation" in text and pred in (SENIOR, STAFF):
                return LevelVerdict(
                    MID, term=term,
                    evidence=(f"neural:{pred}:{conf:.2f}", "lead-generation-guard"),
                    conflict=False,
                )
            return LevelVerdict(pred, term=term, evidence=(f"neural:{conf:.2f}",), conflict=False)

    # 3. Fallback to rule-based
    return _classify_rule_based(title, description, employment_type)


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
