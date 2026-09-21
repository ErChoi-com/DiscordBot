"""Comprehensive stress test for MiniLM Seniority Classifier, Keyword Extractor, and Semantic Dedup."""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from services import job_level
from services.resumes import posting_classifier
from services.jba import semantic_dedup

def test_seniority_edge_cases():
    print("\n" + "="*80)
    print("1. STRESS TESTING SENIORITY CLASSIFIER (job_level)")
    print("="*80)

    test_cases = [
        # --- Null / Malformed / Empty ---
        (None, "mid", "Null input"),
        ("", "mid", "Empty string"),
        ("   ", "mid", "Whitespace only"),
        ("---???---", "mid", "Punctuation only"),
        ("🚀 Senior Backend Engineer 💻", "senior", "Emojis in title"),
        ("Software Engineer - " + "A"*500, "mid", "Extremely long title (500 chars)"),

        # --- Internal Trap Disambiguation ---
        ("Internal Auditor", "mid", "Internal is domain adjective"),
        ("Internal Medicine Physician", "mid", "Medical internal medicine"),
        ("Internal Tools Developer", "mid", "Developer of internal tools"),
        ("Internal Systems Engineer", "mid", "Systems engineer for internal infra"),
        ("Internal Audit Co-op", "intern", "Co-op wins over internal"),
        ("Internal Communications Manager", "staff", "Manager wins over internal"),

        # --- Associate Disambiguation ---
        ("Associate Software Engineer", "junior", "White-collar corporate IC associate"),
        ("Associate Data Analyst", "junior", "White-collar IC associate"),
        ("Associate Product Manager", "junior", "APM IC"),
        ("Associate Product Manager - New Grad", "newgrad", "Explicit new grad APM"),
        ("Warehouse Associate", "mid", "Hourly warehouse associate"),
        ("Retail Sales Associate", "mid", "Hourly retail associate"),
        ("Store Associate", "mid", "Hourly store associate"),
        ("Associate Director, Engineering", "staff", "Associate Director is executive staff"),
        ("Associate Vice President", "staff", "AVP is executive staff"),
        ("Associate Partner", "staff", "Partner is staff"),

        # --- Staff vs IC Titles ---
        ("Staff Software Engineer", "staff", "Staff technical contributor"),
        ("Staff Accountant", "mid", "Staff accountant is standard IC"),
        ("Staff Nurse", "mid", "Staff nurse is bedside IC"),
        ("Staff Pharmacist", "mid", "Staff pharmacist is hospital IC"),
        ("Staffing Coordinator", "mid", "Staffing coordinator is recruitment IC"),
        ("Staffing Manager", "staff", "Staffing manager manages recruitment team"),

        # --- Lead Disambiguation ---
        ("Technical Lead", "senior", "Technical Lead IC"),
        ("Tech Lead", "senior", "Tech Lead IC"),
        ("Team Lead", "senior", "Team Lead contributor"),
        ("Lead Software Engineer", "senior", "Lead SWE"),
        ("Lead Generation Specialist", "mid", "Marketing lead gen IC"),
        ("Sales Lead Generator", "mid", "Sales lead generator"),

        # --- Student Program Staff vs Student Candidates ---
        ("Senior Intern Program Manager", "staff", "Program manager running internships"),
        ("Campus Talent Recruiter", "staff", "Staff recruiting campus students"),
        ("University Relations Lead", "staff", "Staff running university relations"),
        ("Co-op Program Coordinator", "staff", "Staff coordinating co-op program"),
        ("Director of Campus Recruiting", "staff", "Director of student programs"),

        # --- Real Students & Internships ---
        ("Software Engineering Intern", "intern", "Explicit SWE intern"),
        ("SWE Intern (Summer 2026)", "intern", "SWE intern with date term"),
        ("Software Engineer, Co-op (Winter 2027)", "intern", "Co-op with winter term"),
        ("Summer Analyst, Technology", "intern", "Investment banking summer analyst"),
        ("Stagiaire en génie logiciel", "intern", "French stagiaire (intern)"),
        ("Industrial Placement Student - Software", "intern", "UK/EU industrial placement student"),

        # --- Roman Numerals & Numeral Ranks ---
        ("Software Engineer I", "newgrad", "Level 1 / I SWE"),
        ("Developer 1", "newgrad", "Developer 1"),
        ("Software Engineer II", "mid", "Level 2 / II SWE"),
        ("Developer 2", "mid", "Developer 2"),
        ("Software Engineer III", "senior", "Level 3 / III SWE"),
        ("Developer 3", "senior", "Developer 3"),
        ("Software Engineer IV", "senior", "Level 4 / IV SWE"),
        ("Developer 4", "senior", "Developer 4"),

        # --- False Positive Ranks & Acronyms ---
        ("IV Admixture Tech, Pharmacy", "mid", "IV is intravenous, not rank 4"),
        ("Level 1 Trauma RN", "mid", "Trauma 1 is hospital rating, not junior nurse"),
        ("Industrial Organizational (I/O) Psychologist", "mid", "I/O is industrial org, not Level 1"),
        ("Instrument & Control Engineer (I&C)", "mid", "I&C is instrument & control, not Level 1"),

        # --- Manager Disambiguation ---
        ("Product Manager", "mid", "Product Manager is standalone IC"),
        ("Project Manager", "mid", "Project Manager is standalone IC"),
        ("Account Manager", "mid", "Account Manager is standalone IC"),
        ("Case Manager", "mid", "Case Manager is standalone IC"),
        ("Engineering Manager", "staff", "Engineering Manager is people manager"),
        ("Store Manager", "staff", "Store Manager is location manager"),
        ("General Manager", "staff", "General Manager is executive manager"),
        ("Director of Engineering", "staff", "Director is staff"),
        ("VP of Product", "staff", "VP is staff"),
        ("Chief Technology Officer", "staff", "CTO is staff"),
    ]

    passed = 0
    failed = []

    for title, expected, reason in test_cases:
        verdict = job_level.classify(title)
        res = verdict.level
        # Special check for Senior Intern Program Manager conflict
        if title == "Senior Intern Program Manager" and not verdict.conflict:
            failed.append((title, "conflict=True", f"conflict={verdict.conflict}", reason))
            continue

        if res == expected or (expected in ("newgrad", "junior") and res in ("newgrad", "junior")):
            passed += 1
            print(f"  [PASS] {str(title):45s} -> {res:8s} | {reason}")
        else:
            failed.append((title, expected, res, reason))
            print(f"  [FAIL] {str(title):45s} -> GOT {res:8s} EXPECTED {expected:8s} | {reason}")

    print(f"\nSeniority Stress Test Summary: {passed}/{len(test_cases)} PASSED")
    if failed:
        print(f"\nFAILED CASES ({len(failed)}):")
        for f in failed:
            print(f"  - Title: {f[0]} | Expected: {f[1]} | Got: {f[2]} | Note: {f[3]}")
    return len(failed) == 0

def test_keyword_and_boilerplate_edge_cases():
    print("\n" + "="*80)
    print("2. STRESS TESTING BOILERPLATE FILTER & KEYWORD EXTRACTION (posting_classifier)")
    print("="*80)

    # Test 1: Technical acronyms with special characters
    test_bullets = [
        "Architected real-time streaming ETL using Kafka, PyTorch, and Docker containers on AWS EKS.",
        "Synthesized RTL Verilog on Altera FPGA with custom 32-bit RISC-V processor core.",
        "Programmed low-level bare-metal firmware on STM32 microcontrollers communicating over CAN bus.",
        "Engineered high-throughput GraphQL and gRPC microservices backed by PostgreSQL and Redis.",
        "Built low-latency Linux kernel packet filter using eBPF, XDP, and C++20.",
    ]

    for bullet in test_bullets:
        skills = posting_classifier.extract_resume_skills(bullet)
        print(f"  Input: {bullet[:70]}...")
        print(f"  Extracted Skills: {skills}\n")

    # Test 2: Boilerplate detection
    eeo_text = "We are an Equal Opportunity Employer. All qualified applicants will receive consideration for employment without regard to race, color, religion, sex, sexual orientation, gender identity, national origin, disability, or veteran status."
    duty_text = "Develop, test, and deploy resilient distributed microservices in Go and Python using Kubernetes, gRPC, and PostgreSQL."
    compliance_duty_text = "Ensure architectural compliance with federal bank security regulations and manage technical risk across our AWS cloud infrastructure."

    p_eeo = posting_classifier.retain_resume_relevant_text(eeo_text)
    p_duty = posting_classifier.retain_resume_relevant_text(duty_text)
    p_comp = posting_classifier.retain_resume_relevant_text(compliance_duty_text)

    print(f"  EEO Dropped cleanly: {p_eeo == ''} (Length before: {len(eeo_text)}, after: {len(p_eeo)})")
    print(f"  Core Duty Retained:  {p_duty != ''} (Retained: '{p_duty}')")
    print(f"  Compliance Duty Retained: {p_comp != ''} (Retained: '{p_comp}')")

def test_semantic_dedup_edge_cases():
    print("\n" + "="*80)
    print("3. STRESS TESTING SEMANTIC DEDUPLICATION (semantic_dedup)")
    print("="*80)

    test_jobs = [
        # Job 1 & 2: Identical job, different Oracle tracking URLs
        {
            "id": "1",
            "title": "Senior Software Engineer - Python",
            "company": "Stripe",
            "location": "San Francisco, CA",
            "description": "Join our payments infrastructure team to scale distributed transaction processing engines in Python and Go.",
        },
        {
            "id": "2",
            "title": "Senior Software Engineer - Python",
            "company": "Stripe",
            "location": "San Francisco, CA",
            "description": "Join our payments infrastructure team to scale distributed transaction processing engines in Python and Go. Apply now via Oracle tracking candidate portal.",
        },
        # Job 3: Same company & location, distinct role (Server vs Host)
        {
            "id": "3",
            "title": "Server",
            "company": "Chili's",
            "location": "Dallas, TX",
            "description": "Provide excellent dining service, deliver food and drinks to tables, and process guest bills.",
        },
        {
            "id": "4",
            "title": "Host",
            "company": "Chili's",
            "location": "Dallas, TX",
            "description": "Greet guests at the door, manage seating waitlist, and escort guests to their tables.",
        },
        # Job 5: Same company, different seniority level (SWE I vs SWE II)
        {
            "id": "5",
            "title": "Software Engineer I",
            "company": "Google",
            "location": "Mountain View, CA",
            "description": "Entry-level software engineering on search indexing and distributed systems.",
        },
        {
            "id": "6",
            "title": "Software Engineer II",
            "company": "Google",
            "location": "Mountain View, CA",
            "description": "Mid-level software engineering on search indexing and distributed systems with 2+ years experience.",
        },
    ]

    dups = semantic_dedup.find_semantic_duplicates(test_jobs, threshold=0.90)
    print(f"  Discovered duplicates: {dups}")
    
    # Check assertions:
    # 1. (1, 2) MUST be in duplicates
    # 2. (3, 4) MUST NOT be in duplicates (Server vs Host)
    # 3. (5, 6) MUST NOT be in duplicates (SWE I vs SWE II)
    dup_pairs = {(test_jobs[i]["id"], test_jobs[j]["id"]) for i, j, score in dups} | {(test_jobs[j]["id"], test_jobs[i]["id"]) for i, j, score in dups}
    
    assert ("1", "2") in dup_pairs, "Failed: True duplicate postings (1, 2) were not detected!"
    assert ("3", "4") not in dup_pairs, "Failed: Distinct roles (Server vs Host) were incorrectly merged!"
    assert ("5", "6") not in dup_pairs, "Failed: Distinct levels (SWE I vs SWE II) were incorrectly merged!"
    print("  ALL SEMANTIC DEDUP GUARDS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    s1 = test_seniority_edge_cases()
    test_keyword_and_boilerplate_edge_cases()
    test_semantic_dedup_edge_cases()
    print("\n" + "="*80)
    print("STRESS TEST EXECUTION COMPLETE")
    print("="*80)
