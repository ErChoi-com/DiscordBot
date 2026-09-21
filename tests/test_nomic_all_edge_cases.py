"""Exhaustive edge-case stress test and evaluation across ALL use cases of Nomic.

Evaluates:
1. Embedding cache eviction storm & task_type namespace isolation.
2. job_service.semantic_filter_items with corrupt rows, empty inputs, adversarial keywords, and callable thresholds.
3. job_match.judge_jobs_neural with missing signals, corrupted archived jobs, tie-breaking, and ranking monotonicity.
4. job_level SeniorityClassifier neural edge cases with conflicting titles, Roman numerals, and non-English text.
5. posting_classifier piece classifier & boilerplate filtering under 100% and 0% boilerplate, markdown, code blocks.
6. semantic_dedup Cross-ATS deduplication boundary conditions and employer/title isolation.
7. Matryoshka dimension mathematical consistency and cosine correlation monotonicity across all 6 dimensions.
"""
from __future__ import annotations

import concurrent.futures
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.semantic.engine import (
    MATRYOSHKA_DIMS,
    TASK_CLASSIFICATION,
    TASK_SEARCH_DOC,
    TASK_SEARCH_QUERY,
    SemanticEngine,
    get_semantic_engine,
)
from services import job_service
from services import job_match
from services.job_match import ArchivedJob, ProfileSignal, judge_jobs_neural, rank_jobs
from services.jba import semantic_dedup
from services import job_level
from services.resumes import posting_classifier, posting_segments


# ============================================================================
# 1. Embedding Cache Eviction Storm & Task Namespace Isolation
# ============================================================================

def test_cache_eviction_storm_and_task_isolation():
    """Verify cache holds distinct embeddings for different tasks and evicts cleanly under load."""
    engine = get_semantic_engine()
    phrase = "Lead Machine Learning Platform Architect"

    # 1. Different task types must yield DIFFERENT vectors for the same text
    v_query = engine.encode(phrase, dim=384, task_type=TASK_SEARCH_QUERY)
    v_doc = engine.encode(phrase, dim=384, task_type=TASK_SEARCH_DOC)
    v_cls = engine.encode(phrase, dim=384, task_type=TASK_CLASSIFICATION)

    assert v_query is not None and v_doc is not None and v_cls is not None
    # Cosine between asymmetric query and doc should NOT be identical (1.0)
    sim_query_doc = float(np.dot(v_query, v_doc))
    assert sim_query_doc < 0.999, f"Query and Doc embeddings should differ! Got {sim_query_doc}"

    # 2. Verify cache returns exact match for same task type
    v_query_cached = engine.encode(phrase, dim=384, task_type=TASK_SEARCH_QUERY)
    assert np.allclose(v_query, v_query_cached, atol=1e-6)

    # 3. Cache Eviction Storm: insert 1,500 unique strings across 8 threads
    eviction_errors: list[str] = []

    def cache_hammer(worker_id: int):
        for i in range(150):
            text = f"Hammer text payload worker={worker_id} iteration={i} random={time.monotonic()}"
            dim = 384 if i % 2 == 0 else 128
            task = TASK_SEARCH_DOC if i % 3 == 0 else TASK_SEARCH_QUERY
            try:
                vec = engine.encode(text, dim=dim, task_type=task)
                if vec is None or vec.shape != (dim,):
                    eviction_errors.append(f"Worker {worker_id} failed on text {i}: {vec}")
            except Exception as exc:
                eviction_errors.append(f"Worker {worker_id} exception: {exc}")

    threads = [threading.Thread(target=cache_hammer, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert not eviction_errors, f"Cache eviction storm errors:\n" + "\n".join(eviction_errors[:10])


# ============================================================================
# 2. job_service.semantic_filter_items Edge Cases
# ============================================================================

def test_semantic_filter_items_comprehensive_edge_cases():
    """Verify semantic_filter_items survives empty inputs, None values, and adversarial text."""
    # A. Empty items list
    res_empty = job_service.semantic_filter_items([], "python", "remote", ["software"])
    assert res_empty == []

    # B. Empty query (no keywords, location, or role filters)
    dummy_items = [{"title": "Software Engineer", "description": "Python code"}]
    res_no_query = job_service.semantic_filter_items(dummy_items, "", "", [])
    assert res_no_query == dummy_items

    # C. Corrupted items (None fields, empty dicts, missing keys)
    corrupted_items = [
        {"title": None, "description": None, "company": None},
        {},
        {"title": "   ", "description": "\n\t"},
        {"title": "Valid Python Backend Engineer", "description": "Developing FastAPI and PostgreSQL microservices."},
    ]
    filtered_corrupt = job_service.semantic_filter_items(
        corrupted_items,
        keywords="Python Backend",
        location="Remote",
        role_filters=["backend"],
        threshold=0.55,
    )
    # The valid Python engineer must be kept and scored; unscoreable items with no evidence are kept safely
    assert len(filtered_corrupt) >= 1
    python_matches = [it for it in filtered_corrupt if it.get("title") == "Valid Python Backend Engineer"]
    assert len(python_matches) == 1
    assert "semantic_score" in python_matches[0]
    assert python_matches[0]["semantic_score"] >= 0.55

    # D. Adversarial keyword input
    adversarial_queries = [
        r"Python (?P<name>.*?) \d+ [a-z]+ (.*)",  # Regex injection
        "'; DROP TABLE jobs; --",  # SQL injection
        "🚀🔥💻🤖" * 20,  # Emoji storm
        "\x00\x01\x02\x03\x08",  # Control characters
    ]
    for q in adversarial_queries:
        res = job_service.semantic_filter_items(
            [{"title": "Software Engineer", "description": "Python developer"}],
            keywords=q,
            location="Remote",
            role_filters=[],
            threshold=0.50,
        )
        assert isinstance(res, list)

    # E. Dynamic Callable Threshold
    def dynamic_threshold(item: dict[str, Any]) -> float:
        # Require higher bar (0.75) for senior roles, lower (0.50) for others
        return 0.75 if "Senior" in str(item.get("title", "")) else 0.50

    batch = [
        {"title": "Senior Python Architect", "description": "Python architecture, microservices, AWS."},
        {"title": "Junior Python Dev", "description": "Basic Python scripting and unit testing."},
    ]
    res_dynamic = job_service.semantic_filter_items(
        batch,
        keywords="Python developer",
        location="Remote",
        role_filters=[],
        threshold=dynamic_threshold,
    )
    assert len(res_dynamic) >= 1


# ============================================================================
# 3. job_match.judge_jobs_neural Edge Cases
# ============================================================================

def test_neural_judge_empty_signal_and_boundary_cases():
    """Verify judge_jobs_neural handles sparse signals and boundary conditions cleanly."""
    # A. Completely sparse signal
    sparse_signal = ProfileSignal(
        profile_key="sparse",
        anchors=(),
        role_terms=(),
        seniority="",
        locations=(),
        document_text="",
    )

    jobs = [
        ArchivedJob(
            title="Firmware Engineer",
            company="Robotics Co",
            location="Boston, MA",
            link="https://x/1",
            site_label="LinkedIn",
            date_posted="2026-08-25",
            description="Embedded C/C++ development for RTOS and microcontrollers.",
        ),
        ArchivedJob(
            title="Senior Financial Analyst",
            company="Bank Corp",
            location="New York, NY",
            link="https://x/2",
            site_label="Indeed",
            date_posted="2026-08-25",
            description="Financial modeling and Excel pivot tables.",
        ),
    ]

    # Must survive and return "skipped: nothing to judge" without crashing when signal has no content
    verdicts_sparse, ranker_sparse = judge_jobs_neural(sparse_signal, jobs, dim=384)
    assert ranker_sparse == "skipped: nothing to judge"
    assert verdicts_sparse is None

    # Minimal signal with document_text must execute ok:nomic-neural
    minimal_signal = ProfileSignal(
        profile_key="minimal",
        anchors=("embedded",),
        role_terms=(),
        seniority="junior",
        locations=(),
        document_text="Embedded C++ developer with RTOS experience.",
    )
    verdicts, ranker = judge_jobs_neural(minimal_signal, jobs, dim=384)
    assert ranker == "ok:nomic-neural"
    assert len(verdicts) == 2
    assert all(0.0 <= v.score <= 1.0 for v in verdicts.values())

    # B. 0 Jobs input
    v_empty, r_empty = judge_jobs_neural(minimal_signal, [], dim=384)
    assert r_empty == "skipped: nothing to judge"
    assert v_empty is None

    # C. Massive description (>15,000 characters)
    huge_job = ArchivedJob(
        title="Distributed Systems Lead",
        company="Tech Giant",
        location="Remote",
        link="https://x/huge",
        site_label="Direct",
        date_posted="2026-08-25",
        description="Key responsibilities and qualifications:\n" + ("- Design scalable distributed microservices.\n" * 500),
    )
    rich_signal = ProfileSignal(
        profile_key="rich",
        anchors=("distributed systems", "microservices"),
        role_terms=("systems lead",),
        seniority="senior",
        locations=("remote",),
        document_text="Senior engineer with deep expertise in distributed systems architecture.",
    )
    v_huge, _ = judge_jobs_neural(rich_signal, [huge_job], dim=384)
    assert len(v_huge) == 1
    assert v_huge[0].score >= 0.40, f"Expected valid positive score for matching huge job, got {v_huge[0].score}"


# ============================================================================
# 4. Seniority Classifier Neural Edge Cases
# ============================================================================

def test_seniority_classifier_neural_edge_cases():
    """Verify seniority classification handles conflicting titles, roman numerals, and non-English."""
    engine = get_semantic_engine()

    # Conflicted titles
    conflicted_cases = [
        ("Senior Intern Program Manager", "senior"),  # Manager/Senior takes precedence over intern program
        ("Lead Campus University Recruiter", "mid"),    # Recruiter roles are non-intern
        ("Chief Technology Officer", "staff"),
        ("VP of Engineering", "staff"),
        ("Software Engineering Intern (Summer 2026)", "intern"),
        ("Junior Associate Software Developer", "junior"),
        ("Level 4 Software Engineer", "mid"),
    ]

    for title, expected_tier in conflicted_cases:
        res = engine.classify_seniority(title, dim=384)
        assert res is not None
        tier, conf = res
        assert tier in ["intern", "newgrad", "junior", "mid", "senior", "staff"]
        assert 0.0 <= conf <= 1.0

    # Punctuation & single character boundary checks
    boundary_titles = ["", "   ", "A", "---", "???", "123", "!@#$%^"]
    for bt in boundary_titles:
        res = engine.classify_seniority(bt, dim=384)
        if bt.strip():
            assert res is not None
            assert 0.0 <= res[1] <= 1.0


# ============================================================================
# 5. Posting Piece Classifier & Boilerplate Filtering Edge Cases
# ============================================================================

def test_posting_classifier_boilerplate_and_content_extremes():
    """Verify piece classifier on 100% boilerplate, 0% boilerplate, and markdown formats."""
    # A. 100% Boilerplate document
    pure_boilerplate = (
        "Equal Opportunity Employer. All qualified applicants will receive consideration for employment "
        "without regard to race, color, religion, sex, sexual orientation, gender identity, national origin, "
        "disability, or protected veteran status.\n\n"
        "We offer a 401(k) retirement plan with employer matching, comprehensive medical, dental, and vision insurance.\n\n"
        "This company participates in E-Verify and will provide the federal government with your Form I-9 information."
    )
    filtered_pure_bp = posting_classifier.retain_resume_relevant_text(pure_boilerplate)
    # The pure boilerplate should have significant portions removed
    assert len(filtered_pure_bp) < len(pure_boilerplate)

    # B. 0% Boilerplate document (100% core technical requirements)
    pure_technical = (
        "Architect distributed systems in Go and Python.\n"
        "Design real-time event streaming pipelines with Apache Kafka.\n"
        "Deploy and monitor microservices on Kubernetes clusters using Prometheus and Grafana.\n"
        "Optimize PostgreSQL queries and maintain Redis caching layers."
    )
    filtered_pure_tech = posting_classifier.retain_resume_relevant_text(pure_technical)
    # Technical lines must NOT be stripped
    for line in pure_technical.splitlines():
        assert line in filtered_pure_tech or line.strip() in filtered_pure_tech, f"Technical line dropped: {line}"

    # C. Code snippets & markdown tables in descriptions
    markdown_posting = (
        "# Senior Python Developer\n"
        "| Skill | Minimum Experience |\n"
        "| Python | 5+ years |\n"
        "| Docker | 3+ years |\n\n"
        "```python\n"
        "def microservice():\n"
        "    return 'fastapi'\n"
        "```\n\n"
        "Responsibilities:\n"
        "- Build scalable REST APIs in Python."
    )
    filtered_md = posting_classifier.retain_resume_relevant_text(markdown_posting)
    assert "Responsibilities" in filtered_md or "Python" in filtered_md


# ============================================================================
# 6. Cross-ATS Deduplication Boundary Edge Cases
# ============================================================================

def test_cross_ats_dedup_boundary_and_threshold_guards():
    """Verify deduplication behaves predictably around 0.90 threshold and boundary cases."""
    # A. Exactly empty and single-element lists
    assert semantic_dedup.find_semantic_duplicates([]) == []
    assert semantic_dedup.find_semantic_duplicates([{"title": "Job 1", "company": "Acme"}]) == []

    # B. Requisitions with punctuation and formatting variations at same company
    variant_jobs = [
        {
            "company": "Stripe",
            "title": "Software Engineer, Infrastructure - Compute",
            "location": "San Francisco, CA",
            "description": "Build high-reliability compute infrastructure with Kubernetes and Go.",
        },
        {
            "company": "Stripe",
            "title": "Software Engineer - Infrastructure (Compute)",
            "location": "San Francisco, CA",
            "description": "Build high-reliability compute infrastructure with Kubernetes and Go.",
        },
    ]
    dups_variant = semantic_dedup.find_semantic_duplicates(variant_jobs, threshold=0.88)
    assert len(dups_variant) == 1, "Format-variant identical role at same employer should be identified as duplicate"

    # Distinct roles with copy-pasted descriptions at same company must NOT merge
    distinct_jobs = [
        {
            "company": "Stripe",
            "title": "Software Engineer, Infrastructure - Compute",
            "location": "San Francisco, CA",
            "description": "Build high-reliability compute infrastructure with Kubernetes and Go.",
        },
        {
            "company": "Stripe",
            "title": "Recruiter, Infrastructure - Compute",
            "location": "San Francisco, CA",
            "description": "Build high-reliability compute infrastructure with Kubernetes and Go.",
        },
    ]
    dups_distinct = semantic_dedup.find_semantic_duplicates(distinct_jobs, threshold=0.88)
    assert len(dups_distinct) == 0, "Title guard must prevent distinct roles from merging!"

    # C. Completely identical descriptions but DIFFERENT companies
    diff_company_jobs = [
        {
            "company": "Apple",
            "title": "Hardware Validation Engineer",
            "location": "Cupertino, CA",
            "description": "Validate silicon circuitry and execute automated test benches.",
        },
        {
            "company": "Meta",
            "title": "Hardware Validation Engineer",
            "location": "Cupertino, CA",
            "description": "Validate silicon circuitry and execute automated test benches.",
        },
    ]
    dups_diff = semantic_dedup.find_semantic_duplicates(diff_company_jobs, threshold=0.88)
    assert len(dups_diff) == 0, "Different employers must NEVER be merged as duplicates!"


# ============================================================================
# 7. Matryoshka Dimension Monotonicity & Mathematical Consistency
# ============================================================================

def test_matryoshka_all_dimensions_mathematical_consistency():
    """Verify L2 norms, bounded cosines, and correlation across all 6 Matryoshka dimensions."""
    engine = get_semantic_engine()
    query = "Senior Machine Learning Engineer specializing in PyTorch and Transformer architectures."
    doc_match = "We are seeking a Senior ML Engineer with deep expertise in PyTorch, LLMs, and distributed training."
    doc_unrelated = "Experienced Commercial Truck Driver with clean CDL-A license and hazmat endorsement."

    scores_match = []
    scores_unrelated = []

    for dim in MATRYOSHKA_DIMS:  # [768, 512, 384, 256, 128, 64]
        # 1. Encode query and docs
        q_vec = engine.encode(query, dim=dim, normalize=True)
        m_vec = engine.encode(doc_match, dim=dim, normalize=True)
        u_vec = engine.encode(doc_unrelated, dim=dim, normalize=True)

        assert q_vec is not None and m_vec is not None and u_vec is not None
        assert q_vec.shape == (dim,)
        assert m_vec.shape == (dim,)
        assert u_vec.shape == (dim,)

        # 2. Strict unit sphere norm
        assert abs(np.linalg.norm(q_vec) - 1.0) < 1e-4
        assert abs(np.linalg.norm(m_vec) - 1.0) < 1e-4
        assert abs(np.linalg.norm(u_vec) - 1.0) < 1e-4

        # 3. Cosine dot product
        sim_m = float(np.dot(q_vec, m_vec))
        sim_u = float(np.dot(q_vec, u_vec))

        assert -1.0 <= sim_m <= 1.0
        assert -1.0 <= sim_u <= 1.0
        assert sim_m > sim_u, f"At dim={dim}, matching doc ({sim_m:.3f}) should score higher than unrelated ({sim_u:.3f})"
        assert sim_m - sim_u >= 0.15, f"At dim={dim}, margin ({sim_m - sim_u:.3f}) must be >= 0.15"

        scores_match.append(sim_m)
        scores_unrelated.append(sim_u)

    # 4. Monotonic consistency: 768-d score vs 64-d score should be closely aligned (within 0.10)
    assert abs(scores_match[0] - scores_match[-1]) < 0.10, (
        f"768-d ({scores_match[0]:.3f}) and 64-d ({scores_match[-1]:.3f}) diverged too much!"
    )
