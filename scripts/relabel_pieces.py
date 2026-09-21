"""Audit and relabel pieces in .resume_lab/boilerplate/pieces.jsonl.

Ensures high label fidelity and strict exact-span terms consistency:
1. Corrects section headings against the rubric using context.
2. Reclassifies mislabeled boilerplate (EEO, benefits, accommodations, privacy, legal notices)
   wrongly labeled as SKILL_DUTY or ROLE_FACTS.
3. Restores content false positives (real responsibilities/skills incorrectly flagged as BOILERPLATE).
4. Cleans mojibake / text glitches.
5. Verifies and enriches exact-substring terms according to _KNOWN_TECH_TERMS, _KNOWN_DOMAIN_TERMS,
   and technical acronym shapes, while removing generic stopwords.

Usage:
    .venv/Scripts/python.exe scripts/relabel_pieces.py [--apply]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from services.resumes.structured import (
    _KNOWN_TECH_TERMS,
    _KNOWN_DOMAIN_TERMS,
    _BOLDWORTHY_LOWERCASE_TOOLS,
    _DOMAIN_AMBIGUOUS,
    _SKIP_JD_INJECT,
    ACRONYM_TOKEN_PATTERN,
)

INPUT_PATH = ROOT / ".resume_lab" / "boilerplate" / "pieces.jsonl"
OUTPUT_PATH = ROOT / ".resume_lab" / "boilerplate" / "pieces.jsonl"
BACKUP_PATH = ROOT / ".resume_lab" / "boilerplate" / "pieces_pre_relabel_backup.jsonl"
REPORT_PATH = ROOT / ".resume_lab" / "boilerplate" / "relabel_audit_report.json"

LABELS = ("SKILL_DUTY", "ROLE_FACTS", "COMPANY_CONTEXT", "BOILERPLATE")

_CORE_TECH_TERMS = frozenset({
    "python", "java", "c++", "c#", "typescript", "javascript", "sql", "html", "css",
})

# Standard section headings
HEADING_MAP = {
    # SKILL_DUTY
    "requirements": "SKILL_DUTY",
    "required qualifications": "SKILL_DUTY",
    "minimum qualifications": "SKILL_DUTY",
    "basic qualifications": "SKILL_DUTY",
    "preferred qualifications": "SKILL_DUTY",
    "desired qualifications": "SKILL_DUTY",
    "qualifications": "SKILL_DUTY",
    "skills": "SKILL_DUTY",
    "required skills": "SKILL_DUTY",
    "preferred skills": "SKILL_DUTY",
    "skills and qualifications": "SKILL_DUTY",
    "responsibilities": "SKILL_DUTY",
    "key responsibilities": "SKILL_DUTY",
    "duties": "SKILL_DUTY",
    "job duties": "SKILL_DUTY",
    "what you'll do": "SKILL_DUTY",
    "what you’ll do": "SKILL_DUTY",
    "what you will do": "SKILL_DUTY",
    "what you bring": "SKILL_DUTY",
    "what you'll bring": "SKILL_DUTY",
    "what you’ll bring": "SKILL_DUTY",
    "what we're looking for": "SKILL_DUTY",
    "what we’re looking for": "SKILL_DUTY",
    "who you are": "SKILL_DUTY",
    "must have": "SKILL_DUTY",
    "nice to have": "SKILL_DUTY",
    "bonus points": "SKILL_DUTY",
    
    # ROLE_FACTS
    "role overview": "ROLE_FACTS",
    "job summary": "ROLE_FACTS",
    "position overview": "ROLE_FACTS",
    "position summary": "ROLE_FACTS",
    "job details": "ROLE_FACTS",
    "location": "ROLE_FACTS",
    "work location": "ROLE_FACTS",
    "compensation": "ROLE_FACTS",
    "salary": "ROLE_FACTS",
    "pay range": "ROLE_FACTS",
    "schedule": "ROLE_FACTS",
    "hours": "ROLE_FACTS",
    "work arrangement": "ROLE_FACTS",
    
    # COMPANY_CONTEXT
    "about us": "COMPANY_CONTEXT",
    "about the company": "COMPANY_CONTEXT",
    "about our team": "COMPANY_CONTEXT",
    "who we are": "COMPANY_CONTEXT",
    "our team": "COMPANY_CONTEXT",
    "company overview": "COMPANY_CONTEXT",
    
    # BOILERPLATE
    "benefits": "BOILERPLATE",
    "what we offer": "BOILERPLATE",
    "our benefits": "BOILERPLATE",
    "perks": "BOILERPLATE",
    "perks include": "BOILERPLATE",
    "equal opportunity employer": "BOILERPLATE",
    "eeo statement": "BOILERPLATE",
    "equal employment opportunity": "BOILERPLATE",
    "accommodation": "BOILERPLATE",
    "accessibility": "BOILERPLATE",
    "privacy notice": "BOILERPLATE",
    "privacy policy": "BOILERPLATE",
    "how to apply": "BOILERPLATE",
    "application process": "BOILERPLATE",
    "additional information": "BOILERPLATE",
}

# Strong boilerplate patterns
RE_EEO = re.compile(
    r"(?i)(?:\bequal (?:employment )?opportunit|affirmative action"
    r"|without regard to (?:race|color|religion|sex|national origin|age|disability|veteran|sexual orientation)"
    r"|protected veteran|sexual orientation|gender identity|reasonable accommodation|accessibility"
    r"|\be-verify\b|drug-free workplace|criminal background check|privacy policy|data protection"
    r"|applicant tracking|only (?:shortlisted|selected) candidates will be contacted"
    r"|thank you for your interest|apply directly at\b|sign up for job alerts)",
)

RE_BENEFITS = re.compile(
    r"(?i)(?:health, dental|dental and vision|medical, dental|401\(k\)|paid time off|\bpto\b"
    r"|wellness stipend|parental leave|tuition reimbursement|life insurance|employee assistance program"
    r"|unlimited vacation|commuter benefits|flexible spending account|\bfsa\b|\bhsa\b)",
)

# Text cleanup regex
RE_MOJIBAKE = [
    (re.compile(r"Thats"), "That's"),
    (re.compile(r"youll"), "you'll"),
    (re.compile(r"were"), "we're"),
    (re.compile(r"whats"), "what's"),
    (re.compile(r"its"), "it's"),
    (re.compile(r"[\ufffd\u2018\u2019]+"), "'"),
    (re.compile(r"[\u201c\u201d]+"), '"'),
    (re.compile(r"'+"), "'"),
    (re.compile(r"\s+"), " "),
]

GENERIC_TERM_STOPWORDS = {
    "team", "experience", "communication", "skill", "skills", "company", "work", "responsibilities",
    "candidate", "candidates", "role", "position", "ability", "knowledge", "working", "opportunity",
    "support", "tools", "systems", "processes", "environment", "project", "projects", "people",
    "development", "requirements", "qualifications", "education", "years", "degree", "field",
    "job", "duties", "understanding", "strong", "excellent", "proven", "proficient", "familiarity",
    "level", "office", "services", "solutions", "success", "business", "growth", "culture",
    "high", "fast", "paced", "fast-paced", "collaborative", "passionate", "motivated", "detail-oriented",
    "self-starter", "problem", "solving", "problem-solving", "interpersonal", "written", "verbal",
    "bonus", "plus", "required", "preferred", "minimum", "ideal", "successful", "responsible",
    # Uppercase grammatical stopwords leaked from capitalized headers
    "on", "you", "to", "the", "for", "do", "in", "and", "ll", "be", "we", "or", "at", "are",
    "our", "of", "an", "is", "re", "can", "get", "not", "with", "by", "as", "all", "if", "so",
    "tc", "aa", "dot", "eeo", "your", "my", "me", "it", "its", "from", "that", "this", "these",
}


def clean_text(text: str) -> str:
    cleaned = text
    for pat, repl in RE_MOJIBAKE:
        cleaned = pat.sub(repl, cleaned)
    return cleaned.strip()


def normalize_heading(text: str) -> str:
    norm = " ".join(text.lower().split()).strip(" :.-_#*")
    return norm


# Precompile valid terms into a single combined regex for O(1) matching per piece
_ALL_VALID = (_KNOWN_TECH_TERMS | _KNOWN_DOMAIN_TERMS | _BOLDWORTHY_LOWERCASE_TOOLS | _CORE_TECH_TERMS)
_VALID_TERMS = sorted(
    [t for t in _ALL_VALID if len(t) >= 2 and t not in _DOMAIN_AMBIGUOUS and t not in _SKIP_JD_INJECT],
    key=len,
    reverse=True,
)
_COMBINED_TERMS_RE = re.compile(
    r"(?<![a-zA-Z0-9])(" + "|".join(re.escape(t) for t in _VALID_TERMS) + r")(?![a-zA-Z0-9])",
    re.IGNORECASE,
)


def extract_terms_for_piece(text: str, label: str) -> list[str]:
    """Extract concrete terms from piece consistent with structured.py."""
    if label == "BOILERPLATE" or not text:
        return []
    
    found: list[str] = []
    seen: set[str] = set()
    
    # 1. Precompiled known tech & domain terms matching
    for m in _COMBINED_TERMS_RE.finditer(text):
        span = m.group(1)
        low = span.lower()
        if low not in seen and low not in GENERIC_TERM_STOPWORDS:
            seen.add(low)
            found.append(span)
                
    # 2. Acronyms & technical shapes
    for token in ACRONYM_TOKEN_PATTERN.findall(text):
        token_strip = token.strip(" ,.;:()[]{}'\"")
        token_lower = token_strip.lower()
        if len(token_strip) >= 2 and token_lower not in seen and token_lower not in GENERIC_TERM_STOPWORDS:
            if token_lower not in _DOMAIN_AMBIGUOUS and token_lower not in _SKIP_JD_INJECT:
                seen.add(token_lower)
                idx = text.find(token_strip)
                if idx != -1:
                    found.append(token_strip)
                    
    return found


def audit_and_relabel(rows: list[dict]) -> tuple[list[dict], dict]:
    stats = collections.Counter()
    relabeled: list[dict] = []
    
    # Group by posting id (pid)
    by_pid: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_pid[r["pid"]].append(r)
        
    for pid, p_rows in by_pid.items():
        p_rows.sort(key=lambda x: x.get("idx", 0))
        current_section = None
        
        for r in p_rows:
            old_label = r.get("label", "SKILL_DUTY")
            old_terms = r.get("terms", [])
            text = clean_text(r.get("text", ""))
            r["text"] = text
            
            norm_head = normalize_heading(text)
            exact_head = HEADING_MAP.get(norm_head)
            if exact_head:
                current_section = exact_head
                new_label = exact_head
                reason = "Heading matched rubric"
            elif RE_EEO.search(text):
                new_label = "BOILERPLATE"
                reason = "EEO / Legal / Application boilerplate detected"
            elif re.search(r"(?i)\b(?:how to (?:submit|apply)|please submit|in one document please submit|submit your (?:application|resume)|application deadline|closing date)\b", text):
                new_label = "BOILERPLATE"
                reason = "Application logistics / submission instruction"
            elif RE_BENEFITS.search(text) and not any(k in text.lower() for k in ["degree", "bachelor", "experience with", "responsible for"]):
                new_label = "BOILERPLATE"
                reason = "Benefits / perks list without requirements"
            elif re.search(r"(?i)^(?:about [A-Z]|headquartered in|proudly headquartered|founded in \d{4}|who we are\b)", text):
                new_label = "COMPANY_CONTEXT"
                reason = "Company introduction / background"
            elif old_label == "BOILERPLATE" and any(k in text.lower() for k in ["must have", "bachelor", "master's", "years of experience", "proficiency in", "responsible for designing", "responsible for developing"]):
                new_label = "SKILL_DUTY"
                reason = "Requirement text rescued from boilerplate"
            elif old_label in LABELS:
                new_label = old_label
                reason = "Retained valid label"
            else:
                new_label = current_section or "BOILERPLATE"
                reason = "Fallback to section context"
                
            if new_label != old_label:
                stats[f"label_change_{old_label}_to_{new_label}"] += 1
            else:
                stats["label_unchanged"] += 1
                
            # Key term extraction
            # Reconcile terms: combine existing valid exact-substring terms with newly detected terms
            extracted = extract_terms_for_piece(text, new_label)
            merged_terms: list[str] = []
            seen_spans = set()
            
            # Keep valid old terms that are exact substrings and not stopwords
            if new_label != "BOILERPLATE":
                for ot in old_terms:
                    ot_clean = ot.strip(" ,.;:()[]{}'\"")
                    if ot_clean and ot_clean in text and ot_clean.lower() not in GENERIC_TERM_STOPWORDS and ot_clean.lower() not in seen_spans:
                        seen_spans.add(ot_clean.lower())
                        merged_terms.append(ot_clean)
                        
            # Add newly extracted
            for nt in extracted:
                if nt.lower() not in seen_spans:
                    seen_spans.add(nt.lower())
                    merged_terms.append(nt)
                    
            r["label"] = new_label
            r["terms"] = merged_terms
            relabeled.append(r)
            stats[f"label_{new_label}"] += 1
            stats["total_terms"] += len(merged_terms)
            if merged_terms:
                stats["pieces_with_terms"] += 1
                
    return relabeled, dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Overwrite pieces.jsonl with relabeled data")
    args = parser.parse_args()
    
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        sys.exit(1)
        
    print(f"Reading {INPUT_PATH}...")
    raw = [json.loads(line) for line in INPUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(raw)} pieces across {len({r['pid'] for r in raw})} postings.")
    
    relabeled, stats = audit_and_relabel(raw)
    
    print("\nRelabeling & Audit Results:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
        
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")
    
    if args.apply:
        print(f"Backing up original pieces to {BACKUP_PATH}...")
        BACKUP_PATH.write_text(INPUT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Writing {len(relabeled)} relabeled pieces to {OUTPUT_PATH}...")
        OUTPUT_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in relabeled) + "\n", encoding="utf-8")
        print("Applied successfully.")
    else:
        print("Run with --apply to apply changes to pieces.jsonl.")


if __name__ == "__main__":
    main()
