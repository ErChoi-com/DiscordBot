"""Mine hard negatives and boundary edge-cases for active learning relabeling.

Active learning strategies:
1. Distractor Keyword Traps: Real responsibilities/skills containing boilerplate keywords
   ('compliance', 'benefit', 'equal', 'opportunity', 'diversity', 'accommodate', etc.).
2. Subtle Boilerplate Traps: Application friction, recruiter warnings, background checks,
   ATS instructions, and scraper junk that lack standard EEO/benefits keywords.
3. Model Boundary Uncertainty: Pieces where the current classifier has high entropy
   (0.30 <= P(BOILERPLATE) <= 0.70).

Output:
    .resume_lab/agent_batches/hard_negatives/prompt.txt
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from services.resumes.posting_segments import split_description

DISTRACTOR_KEYWORDS = re.compile(
    r"\b(compliance|comply|complies|benefit|benefits|equal|opportunity|opportunities|diversity|inclusion|"
    r"accommodat\w+|disabilit\w+|veteran[s]?|compensation|salary|apply|applying|applicant[s]?|policy|policies|"
    r"regulations?|standards?|legal|confidentiality)\b",
    re.IGNORECASE,
)

DUTY_SYNTAX = re.compile(
    r"^(architect|build|design|develop|lead|manage|ensure|collaborate|oversee|create|maintain|implement|"
    r"support|drive|analyze|coordinate|provide|execute|conduct|perform|direct|evaluate|review|monitor|"
    r"investigate|audit|partner|establish|deliver|troubleshoot|optimize|configure|author|write|prepare|"
    r"experience\s+with|ability\s+to|knowledge\s+of|proficient\s+in|responsible\s+for|skilled\s+in|"
    r"proven\s+track\s+record|demonstrated|bachelor|master|phd|degree|years\s+of)\b",
    re.IGNORECASE,
)

SUBTLE_BOILERPLATE_CUES = re.compile(
    r"\b(unsolicited\s+(resumes?|submissions?|profiles?)|recruitment\s+agenc\w+|staffing\s+agenc\w+|"
    r"click\s+(here|below|the\s+link)|portal|ats|workday|taleo|greenhouse|lever|icims|"
    r"status\s+of\s+your\s+application|closing\s+date|close\s+at\s+any\s+time|pre-employment|"
    r"background\s+check|drug\s+screen|right\s+to\s+work|e-verify|privacy\s+policy|terms\s+of\s+use|"
    r"all\s+rights\s+reserved|copyright|\xa9|share\s+this\s+job|connect\s+with\s+us)\b",
    re.IGNORECASE,
)

STANDARD_EEO_PHRASES = re.compile(
    r"\b(equal\s+opportunity\s+employer|race,\s*color,\s*religion|sexual\s+orientation,\s*gender\s+identity|"
    r"reasonable\s+accommodation\s+to\s+qualified\s+individuals|affirmative\s+action)\b",
    re.IGNORECASE,
)


def load_candidates_from_db(db_path: Path, max_postings: int = 1500) -> list[dict]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, data FROM jobs ORDER BY RANDOM() LIMIT ?", (max_postings,))
    rows = cur.fetchall()
    postings = []
    for jid, raw in rows:
        try:
            d = json.loads(raw)
            desc = d.get("description", "")
            title = d.get("title", "Untitled")
            company = d.get("company", "Unknown")
            if desc and len(desc) > 100:
                postings.append({
                    "id": f"db-{jid}",
                    "title": title,
                    "company": company,
                    "description": desc,
                })
        except Exception:
            continue
    return postings


def mine_hard_negatives(postings: list[dict], existing_texts: set[str], target_count: int = 600) -> list[dict]:
    distractor_duties: list[dict] = []
    subtle_bps: list[dict] = []
    seen_texts = set(existing_texts)

    for p in postings:
        pieces = split_description(p["description"])
        for piece in pieces:
            clean = piece.strip()
            if len(clean) < 25 or len(clean) > 350:
                continue
            clean_lower = clean.lower()
            if clean_lower in seen_texts:
                continue

            # Case A: Real duty with distractor keywords
            if DISTRACTOR_KEYWORDS.search(clean) and DUTY_SYNTAX.search(clean):
                seen_texts.add(clean_lower)
                distractor_duties.append({
                    "posting_id": p["id"],
                    "title": p["title"],
                    "company": p["company"],
                    "text": clean,
                    "strategy": "DUTY_WITH_BOILERPLATE_KEYWORD",
                })

            # Case B: Subtle boilerplate without standard EEO phrases
            elif SUBTLE_BOILERPLATE_CUES.search(clean) and not STANDARD_EEO_PHRASES.search(clean):
                seen_texts.add(clean_lower)
                subtle_bps.append({
                    "posting_id": p["id"],
                    "title": p["title"],
                    "company": p["company"],
                    "text": clean,
                    "strategy": "SUBTLE_BOILERPLATE_NO_EEO",
                })

    print(f"Mined {len(distractor_duties)} duty-distractor traps and {len(subtle_bps)} subtle-boilerplate traps.")
    
    # Balance and sample
    half = target_count // 2
    selected = distractor_duties[:half] + subtle_bps[:half]
    import random
    random.Random(42).shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=600)
    args = parser.parse_args()

    pieces_path = ROOT / ".resume_lab" / "boilerplate" / "pieces.jsonl"
    existing_texts = set()
    if pieces_path.exists():
        for line in pieces_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing_texts.add(json.loads(line)["text"].strip().lower())
                except Exception:
                    pass

    db_path = ROOT / "data" / "jba" / "jobs" / "jobs.db"
    postings = load_candidates_from_db(db_path, max_postings=2000)
    mined = mine_hard_negatives(postings, existing_texts, target_count=args.count)

    out_dir = ROOT / ".resume_lab" / "agent_batches" / "hard_negatives"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "candidates.json"
    out_file.write_text(json.dumps(mined, indent=2), encoding="utf-8")
    print(f"Saved {len(mined)} hard-negative candidates to {out_file}")


if __name__ == "__main__":
    main()

