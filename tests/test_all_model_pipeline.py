"""End-to-end pipeline validation for all MiniLM / transformer-backed components:

1. Seniority & Tier Classification (job_level.classify)
2. Channel Routing & Role Filtering (job_level.matches_role_filter)
3. Semantic Job Deduplication (services.jba.semantic_dedup)
4. Resume-to-Job Seniority Scoring (services.job_match.score_level)
5. Resume Skill & Keyword Extraction (services.resumes.posting_classifier)
6. Scrape Boilerplate & EEO Rejection (services.resumes.posting_classifier)
7. Semantic Job Filtering & Search Matching (services.job_service.semantic_filter_items)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from services import job_level
from services import job_match
from services import job_service
from services.jba import semantic_dedup
from services.resumes import posting_classifier


def run_pipeline_test():
    total_checks = 0
    passed_checks = 0

    print("=" * 80)
    print("PIPELINE TEST 1: SENIORITY CLASSIFIER (job_level)")
    print("=" * 80)
    seniority_samples = [
        ("Software Engineer Intern", "intern"),
        ("New Graduate Software Engineer 2026", "newgrad"),
        ("Associate Software Engineer", "junior"),
        ("Full Stack Developer", "mid"),
        ("Senior Backend Engineer", "senior"),
        ("Principal Infrastructure Architect", "staff"),
    ]
    for title, expected in seniority_samples:
        total_checks += 1
        verdict = job_level.classify(title)
        ok = verdict.level == expected
        if ok:
            passed_checks += 1
            print(f"  [PASS] '{title}' -> {verdict.level} (evidence: {verdict.evidence})")
        else:
            print(f"  [FAIL] '{title}' -> GOT {verdict.level}, EXPECTED {expected}")

    print("\n" + "=" * 80)
    print("PIPELINE TEST 2: DISCORD CHANNEL ROUTING & ROLE FILTERING")
    print("=" * 80)
    routing_samples = [
        # (title, filter_name, expected_match, description)
        ("Software Engineering Intern", ["internship"], True, "SWE Intern should pass #internships"),
        ("Staff Software Engineer", ["internship"], False, "Staff SWE must NOT pass #internships"),
        ("Senior Intern Program Coordinator", ["internship"], False, "Program coordinator must NOT pass #internships"),
        ("Graduate Software Engineer", ["entry"], True, "New grad should pass #entry"),
        ("Junior QA Analyst", ["junior"], True, "Junior should pass #junior"),
        ("Director of Product Management", ["entry"], False, "Director must NOT pass #entry"),
        ("Staff Infrastructure Engineer", ["senior"], True, "Staff passes #senior"),
        ("Senior DevOps Engineer", ["senior"], True, "Senior passes #senior"),
        ("Co-op Software Developer", ["senior"], False, "Co-op must NOT pass #senior"),
    ]
    for title, r_filter, expected, note in routing_samples:
        total_checks += 1
        actual = job_level.matches_role_filter(title, r_filter)
        ok = actual == expected
        if ok:
            passed_checks += 1
            print(f"  [PASS] Filter={r_filter} | '{title}' -> Allowed: {actual} | {note}")
        else:
            print(f"  [FAIL] Filter={r_filter} | '{title}' -> GOT {actual}, EXPECTED {expected} | {note}")

    print("\n" + "=" * 80)
    print("PIPELINE TEST 3: SEMANTIC JOB DEDUPLICATION (services.jba.semantic_dedup)")
    print("=" * 80)
    sample_jobs = [
        {
            "id": "job-1",
            "title": "Machine Learning Engineer - NLP",
            "company": "Anthropic",
            "location": "San Francisco, CA",
            "description": "Train and evaluate large language models and reinforcement learning infrastructure in PyTorch.",
        },
        {
            "id": "job-2",
            "title": "Machine Learning Engineer - NLP",
            "company": "Anthropic",
            "location": "San Francisco, CA",
            "description": "Train and evaluate large language models and reinforcement learning infrastructure in PyTorch. EOE apply online.",
        },
        {
            "id": "job-3",
            "title": "Data Center Facilities Technician",
            "company": "Anthropic",
            "location": "San Francisco, CA",
            "description": "Maintain HVAC, electrical systems, and server racks in high-density data center facilities.",
        },
    ]
    dups = semantic_dedup.find_semantic_duplicates(sample_jobs, threshold=0.88)
    total_checks += 2
    # Check 1: job-1 and job-2 are identified as dups
    is_1_2_dup = any(
        (d[0] == 0 and d[1] == 1) or (d[0] == 1 and d[1] == 0) for d in dups
    )
    # Check 2: job-3 is NOT duplicate of job-1 or job-2
    is_3_dup = any(d[0] == 2 or d[1] == 2 for d in dups)

    if is_1_2_dup:
        passed_checks += 1
        print("  [PASS] Duplicate Requisitions (job-1, job-2) correctly detected!")
    else:
        print("  [FAIL] Duplicate Requisitions (job-1, job-2) were missed!")

    if not is_3_dup:
        passed_checks += 1
        print("  [PASS] Distinct Requisition (job-3) cleanly preserved without false collision.")
    else:
        print("  [FAIL] False positive duplicate detected with job-3!")

    print("\n" + "=" * 80)
    print("PIPELINE TEST 4: RESUME-TO-JOB SENIORITY SCORING (services.job_match)")
    print("=" * 80)
    student_signal = job_match.ProfileSignal(
        profile_key="test_student",
        seniority="student",
        anchors=("python", "pytorch"),
        role_terms=("software engineer",),
        locations=("san francisco",),
    )
    experienced_signal = job_match.ProfileSignal(
        profile_key="test_experienced",
        seniority="senior",
        anchors=("python", "kubernetes"),
        role_terms=("software engineer",),
        locations=("san francisco",),
    )

    match_cases = [
        # (title, signal, expected_min_score, note)
        ("Avionics Firmware Co-op (Summer 2026)", student_signal, 0.9, "Student gets 1.0 for co-op"),
        ("Campus Recruiter", student_signal, 0.2, "Student gets 0.1 for staff campus recruiter"),
        ("Senior Intern Program Lead", student_signal, 0.5, "Student gets <=0.5 for conflicted/staff role"),
        ("Staff Infrastructure Engineer", student_signal, 0.2, "Student gets 0.1 for staff role"),
        ("Staff Infrastructure Engineer", experienced_signal, 0.9, "Experienced gets 1.0 for staff role"),
        ("Software Engineering Intern", experienced_signal, 0.3, "Experienced gets 0.2 for intern role"),
    ]
    for title, signal, exp_thresh, note in match_cases:
        total_checks += 1
        score = job_match.score_level(title=title, signal=signal)
        # Check direction
        if exp_thresh >= 0.8:
            ok = score >= 0.8
        else:
            ok = score <= exp_thresh
        if ok:
            passed_checks += 1
            print(f"  [PASS] Profile={signal.seniority:11s} | Title='{title}' -> Score: {score:.2f} | {note}")
        else:
            print(f"  [FAIL] Profile={signal.seniority:11s} | Title='{title}' -> Score: {score:.2f} | {note}")

    print("\n" + "=" * 80)
    print("PIPELINE TEST 5: RESUME SKILL & KEYWORD EXTRACTION (posting_classifier)")
    print("=" * 80)
    resume_bullet = (
        "Architected real-time streaming pipelines in Python and Go using Apache Kafka, "
        "PostgreSQL, and Docker containers deployed to AWS EKS clusters."
    )
    extracted = posting_classifier.extract_resume_skills(resume_bullet)
    total_checks += 1
    expected_skills = {"Kafka", "PostgreSQL", "Docker", "AWS", "EKS"}
    found_intersection = expected_skills.intersection(set(extracted))
    if len(found_intersection) >= 3:
        passed_checks += 1
        print(f"  [PASS] Successfully extracted core technical skills: {found_intersection}")
    else:
        print(f"  [FAIL] Failed to extract expected skills: {extracted}")

    print("\n" + "=" * 80)
    print("PIPELINE TEST 6: SCRAPER NOISE & BOILERPLATE REJECTION")
    print("=" * 80)
    raw_scraping = (
        "Develop scalable APIs in FastAPI and PostgreSQL.\n"
        "Equal Opportunity Employer: We do not discriminate based on race, color, religion, gender identity, or national origin.\n"
        "COVID-19 vaccination is mandatory for all on-site personnel per corporate health policies.\n"
        "We offer comprehensive 401(k) matching, health benefits, and dental coverage."
    )
    cleaned = posting_classifier.retain_resume_relevant_text(raw_scraping)
    total_checks += 2
    if "Equal Opportunity" not in cleaned:
        passed_checks += 1
        print("  [PASS] Legal EEO boilerplate successfully stripped.")
    else:
        print("  [FAIL] Legal EEO boilerplate remained in description.")

    if "FastAPI and PostgreSQL" in cleaned:
        passed_checks += 1
        print("  [PASS] Essential technical responsibilities preserved.")
    else:
        print("  [FAIL] Technical duties accidentally dropped.")

    print("\n" + "=" * 80)
    print("PIPELINE TEST 7: SEMANTIC SEARCH & WATCHER FILTER (services.job_service)")
    print("=" * 80)
    search_test_items = [
        {
            "id": "item-1",
            "title": "Computer Vision Research Engineer",
            "company": "Robotics Corp",
            "location": "Remote",
            "description": "Develop deep learning algorithms for 3D object detection using PyTorch, OpenCV, and CUDA.",
        },
        {
            "id": "item-2",
            "title": "Payroll and Benefits Administrator",
            "company": "Finance Inc",
            "location": "Remote",
            "description": "Process payroll spreadsheets, administer employee benefits, and reconcile bank statements.",
        },
    ]

    # Filter items searching for computer vision and pytorch
    filtered = job_service.semantic_filter_items(
        items=search_test_items,
        keywords="computer vision pytorch deep learning",
        location="Remote",
        role_filters=["mid"],
        threshold=None,
    )
    total_checks += 2
    kept_ids = {it["id"] for it in filtered}
    if "item-1" in kept_ids:
        passed_checks += 1
        score_1 = next(it["semantic_score"] for it in filtered if it["id"] == "item-1")
        print(f"  [PASS] Relevant Computer Vision posting matched (semantic score: {score_1:.4f})")
    else:
        print("  [FAIL] Relevant Computer Vision posting was incorrectly rejected.")

    if "item-2" not in kept_ids:
        passed_checks += 1
        print("  [PASS] Irrelevant Payroll posting was cleanly rejected by semantic filter.")
    else:
        score_2 = next(it["semantic_score"] for it in filtered if it["id"] == "item-2")
        print(f"  [FAIL] Irrelevant Payroll posting was matched with score: {score_2:.4f}")

    print("\n" + "=" * 80)
    print(f"OVERALL PIPELINE RESULTS: {passed_checks}/{total_checks} CHECKS PASSED ({passed_checks/total_checks:.1%})")
    print("=" * 80)
    return passed_checks == total_checks


if __name__ == "__main__":
    success = run_pipeline_test()
    sys.exit(0 if success else 1)
