"""Mine a diverse corpus of job titles from jobs.db for seniority labeling."""
from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "jba" / "jobs" / "jobs.db"
OUT_DIR = ROOT / ".resume_lab" / "seniority" / "batches"

GUIDE_TEXT = """=== SENIORITY LABELING RUBRIC (8 CLASSES) ===

You must classify each job title into exactly ONE of the following 8 classes based on the semantic seniority of the target candidate being hired:

1. coop:
   - Co-op work-terms, university cooperative education.
   - Examples: "Software Engineering Co-op", "Fall 2026 Co-op - Mechanical Engineering", "Electrical Co-op Student".

2. intern:
   - Internships, summer analysts, student industrial placements, stagiaires, work placements.
   - Examples: "Data Science Intern", "Summer 2026 Investment Banking Analyst", "Student Intern - IT", "Stagiaire en génie logiciel".
   - Date-term roles for students: "Software Developer (Winter 2027)", "QA Tester (Summer 2026)".

3. campus:
   - Rotational campus programs, apprenticeships, student trainee programs, early-talent cohorts.
   - Examples: "Early Career Rotational Associate", "Campus Graduate Apprentice", "Student Engineering Trainee", "Emerging Talent Program".

4. newgrad:
   - Roles specifically targeting recent university graduates, entry-level candidates, or EITs.
   - Examples: "Associate Product Manager - New Grad", "Software Engineer - New Graduate 2026", "Entry Level Financial Analyst", "Junior EIT Civil Engineer".

5. junior:
   - Junior IC roles, 0-2 years, explicitly designated with Junior / Jr. or non-newgrad Associate IC titles.
   - Examples: "Junior Full Stack Developer", "Jr. Systems Administrator", "Associate Data Analyst", "Associate Software Engineer".

6. mid:
   - Standard professional IC roles without seniority modifiers (the bulk of normal industry jobs).
   - Examples: "Software Engineer", "Frontend Developer", "Accountant", "Registered Nurse", "Commercial Driver", "Product Manager", "Electrician".
   - Also false-positive keywords that look like intern/coop: "Internal Auditor", "International Trade Specialist", "Interventional Radiologist", "Co-operative Bank Teller".

7. senior:
   - Senior IC roles, technical specialists, team anchors (typically 5+ years experience).
   - Marked by: Senior, Sr., III, IV, Specialist, Lead technical contributor.
   - Examples: "Senior Backend Engineer", "Sr. DevOps Specialist", "Software Engineer III", "Data Scientist IV".

8. staff:
   - Staff, Principal, Distinguished, Fellow, Technical Architect, Director, Head of, VP, Vice President, Chief (CEO/CTO/CFO), People Managers.
   - ALSO roles that recruit, manage, or coordinate student programs (the employee is staff, not the student!):
     - Examples: "Senior Intern Program Manager", "Campus Talent Recruiter", "University Relations Lead", "Co-op Program Coordinator", "Director of Campus Programs".
   - Executive & People Management Examples: "Engineering Manager", "Director of Product", "VP of Sales", "CTO", "Principal Systems Architect", "General Manager".

CRITICAL DISAMBIGUATION RULES:
- If a title has "Associate Product Manager - New Grad", the hiring cohort IS "newgrad", so label it newgrad!
- If a title has "Senior Intern Program Manager", the person is a staff manager running an internship program, so label it staff!
- If a title has "Internal Auditor", "internal" is an accounting domain adjective, NOT an intern, so label it mid!
- If a title has "Software Engineer I" or "Developer 1", label it newgrad / junior!
- If a title has "Software Engineer II" or "Developer 2", label it mid!
- If a title has "Software Engineer III / IV", label it senior!
"""


def mine_titles(target_count: int = 1400) -> list[str]:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Query diverse slices
    slices = [
        "SELECT data FROM jobs WHERE id % 7 = 0 LIMIT 1500",
        "SELECT data FROM jobs WHERE id % 13 = 0 LIMIT 1500",
        "SELECT data FROM jobs WHERE id % 17 = 0 LIMIT 1500",
    ]

    raw_titles = set()
    for q in slices:
        for (payload,) in cur.execute(q):
            try:
                data = json.loads(payload)
                t = str(data.get("title") or "").strip()
                # Basic cleanup
                t = re.sub(r"\s+", " ", t)
                if len(t) >= 4 and len(t) <= 90 and not t.isdigit():
                    raw_titles.add(t)
            except Exception:
                continue
    conn.close()

    # Categorize into priority buckets to guarantee rich distribution
    coop_intern = []
    campus_newgrad = []
    senior_staff = []
    associates = []
    ranked = []
    standard = []

    for t in raw_titles:
        tl = t.lower()
        if re.search(r"\b(co[-\s]?op|intern(?:ship|s)?|summer\s+analyst|stagiaire)\b", tl):
            coop_intern.append(t)
        elif re.search(r"\b(campus|apprentice|rotational|trainee|early\s+career|new\s?grad|entry[-\s]level)\b", tl):
            campus_newgrad.append(t)
        elif re.search(r"\b(staff|principal|director|head\s+of|v\.?p\.?|chief|c[etfo]o|lead|manager|architect)\b", tl):
            senior_staff.append(t)
        elif re.search(r"\bassociate\b", tl):
            associates.append(t)
        elif re.search(r"\b(senior|sr\.?|junior|jr\.?|[iI]{1,3}|iv|\b[1-4]\b)\b", tl):
            ranked.append(t)
        else:
            standard.append(t)

    print(f"Mined pool: coop/intern={len(coop_intern)}, campus/newgrad={len(campus_newgrad)}, senior/staff={len(senior_staff)}, associate={len(associates)}, ranked={len(ranked)}, standard={len(standard)}")

    rng = random.Random(42)
    rng.shuffle(coop_intern)
    rng.shuffle(campus_newgrad)
    rng.shuffle(senior_staff)
    rng.shuffle(associates)
    rng.shuffle(ranked)
    rng.shuffle(standard)

    # Injected known hard traps to guarantee model robustness
    curated_traps = [
        "Associate Product Manager - New Grad",
        "Associate Product Manager",
        "Product Manager",
        "Senior Product Manager",
        "Senior Intern Program Manager",
        "Campus Talent Recruiter",
        "University Relations Coordinator",
        "Co-op Program Specialist",
        "Internal Auditor",
        "International Tax Analyst",
        "Interventional Radiologist",
        "Co-operative Bank Teller",
        "Software Developer (Winter 2027)",
        "Software Engineer, Fall 2026",
        "Summer 2026 Quantitative Analyst Intern",
        "Software Engineer I",
        "Software Engineer II",
        "Software Engineer III",
        "Staff Software Engineer",
        "Principal Systems Architect",
        "Lead Generation Specialist",
        "Technical Lead - Backend Infrastructure",
        "Early Career Rotational Associate",
        "Student Engineering Trainee",
        "Junior Full Stack Developer",
        "Associate Software Engineer",
        "Executive Assistant to CEO",
        "Director of Campus Relations",
        "Brand Manager - Entry Level",
        "Sales Associate - Part Time",
        "Commercial Driver",
        "Registered Nurse - ICU",
        "CTO & Co-Founder",
    ]

    selected = set(curated_traps)
    for pool, target in [
        (coop_intern, 250),
        (campus_newgrad, 200),
        (associates, 150),
        (ranked, 300),
        (senior_staff, 300),
        (standard, 250),
    ]:
        for item in pool:
            if len(selected) >= target_count:
                break
            selected.add(item)

    final_list = sorted(selected)
    rng.shuffle(final_list)
    return final_list[:target_count]


def main():
    titles = mine_titles(1400)
    print(f"Total selected titles for subagent labeling: {len(titles)}")

    batch_size = (len(titles) + 3) // 4
    for b_idx in range(4):
        batch = titles[b_idx * batch_size : (b_idx + 1) * batch_size]
        b_dir = OUT_DIR / f"batch_{b_idx + 1}"
        b_dir.mkdir(parents=True, exist_ok=True)

        (b_dir / "GUIDE.txt").write_text(GUIDE_TEXT, encoding="utf-8")
        lines = []
        for i, t in enumerate(batch):
            pid = f"st-{b_idx + 1}-{i:03d}"
            lines.append(f"[{pid}] {t}")
        (b_dir / "prompt.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {len(batch)} titles to {b_dir / 'prompt.txt'}")


if __name__ == "__main__":
    main()
