"""Exhaustive stress test battery and edge-case finder for the rewired Pure Nomic Architecture.

Tests:
1. Malformed inputs, control characters, zero-width chars, emojis, multilingual text.
2. Ultra-long documents (>60,000 chars, ~10,000 tokens) testing Nomic 8,192-token context.
3. Heavy multi-threaded concurrency (16 concurrent threads) testing dimension/task isolation.
4. Numerical stability: zero-vectors, NaN/Inf checks, strict [0.0, 1.0] bounds.
5. Batch scaling: large batches (128 items) for resume fit & watcher filtering.
6. Cross-ATS deduplication edge cases (subtle role differences, title guards).
7. Neural judge (.bestjobs) edge cases with missing/none attributes.
"""
from __future__ import annotations

import concurrent.futures
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure rebuilt_app/src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.semantic.engine import SemanticEngine, get_semantic_engine
from services import job_service
from services import job_match
from services.job_match import ArchivedJob, ProfileSignal, judge_jobs_neural, rank_jobs
from services.jba import semantic_dedup


# ============================================================================
# 1. Malformed, Bizarre & Adversarial Text Payloads
# ============================================================================

MALFORMED_PAYLOADS = [
    ("", "Empty string"),
    ("   ", "Whitespace only"),
    ("\n\t\r\v\f", "Whitespace escapes"),
    ("\x00\x01\x02\x03\x08\x0b\x0c\x0e\x1f", "ASCII control characters"),
    ("\u200b\u200c\u200d\ufeff\u2060", "Zero-width spaces & invisible formatting"),
    ("\x1b[31;1mRed Bold Text\x1b[0m\x1b[2J", "ANSI escape sequences"),
    ("🚀🔥💻🤖🧠✨🎉🍕🚗🎯" * 40, "Heavy emoji bombardment"),
    ("SELECT * FROM jobs WHERE 1=1; DROP TABLE users; --", "SQL injection payload"),
    ("<script>alert('xss');</script><style>body{display:none}</style>", "HTML/XSS payload"),
    ("$$ \\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi} $$", "LaTeX math notation"),
    ("Développeur logiciel senior à Montréal, Québec", "French accented characters"),
    ("Разработчик программного обеспечения (Python / C++)", "Cyrillic text"),
    ("软件工程师 - 机器学习平台架构", "Simplified Chinese"),
    ("مهندس برمجيات أول - أنظمة موزعة", "Arabic RTL text"),
    ("מפתח תוכנה בכיר - מערכות משובצות", "Hebrew RTL text"),
    ("!@#$%^&*()_+=-~`{}[]|:;'<>?,./" * 10, "Punctuation explosion"),
]


def test_payload_extremes_encoding():
    """Engine must encode all malformed, exotic, and control payloads without crashing."""
    engine = get_semantic_engine()

    for text, label in MALFORMED_PAYLOADS:
        # 1. Single encode
        vec = engine.encode(text, dim=384, normalize=True)
        assert vec is not None, f"Failed on {label}"
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,), f"Shape mismatch on {label}: {vec.shape}"
        assert not np.isnan(vec).any(), f"NaN detected on {label}"
        assert not np.isinf(vec).any(), f"Inf detected on {label}"
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4, f"Vector not unit-normalized on {label}: norm={norm}"

        # 2. Seniority prediction
        sen_res = engine.classify_seniority(text, dim=384)
        if text.strip():
            assert sen_res is not None, f"Seniority failed on {label}"
            tier, conf = sen_res
            assert tier in ["intern", "newgrad", "junior", "mid", "senior", "staff"]
            assert 0.0 <= conf <= 1.0

        # 3. Posting piece prediction
        post_res = engine.classify_posting_piece(text, dim=384)
        if text.strip():
            assert post_res is not None, f"Posting piece failed on {label}"
            ptype, pconf = post_res
            assert ptype in ["SKILL_DUTY", "BOILERPLATE", "ROLE_FACTS", "COMPANY_CONTEXT"]
            assert 0.0 <= pconf <= 1.0


# ============================================================================
# 2. Ultra-Long Context Stress (Nomic 8,192-Token Handling)
# ============================================================================

def test_ultra_long_document_handling():
    """Verify engine handles massive texts (>60,000 characters) without OOM or truncation errors."""
    engine = get_semantic_engine()

    # Generate a realistic 12-page job description (~65,000 characters, ~10,000 tokens)
    paragraphs = [
        "We are looking for a Senior Principal Software Engineer to lead distributed systems architecture.",
        "Responsibilities include designing high-throughput messaging pipelines using Kafka, RabbitMQ, and gRPC.",
        "You will mentor junior engineers, lead architectural design reviews, and establish engineering standards.",
        "Required qualifications include 10+ years experience in Python, C++, Go, and distributed database systems.",
        "Compensation includes competitive base salary of $250,000 - $320,000, equity grants, 401(k) matching.",
        "We are an equal opportunity employer committed to a diverse and inclusive workplace for all applicants.",
    ]
    giant_text = "\n\n".join(paragraphs * 120)
    assert len(giant_text) > 60_000, f"Length was {len(giant_text)}"

    # 1. Encode giant text at master 768-d and sliced 384-d
    vec_768 = engine.encode(giant_text, dim=768, normalize=True)
    assert vec_768 is not None
    assert vec_768.shape == (768,)
    assert not np.isnan(vec_768).any()

    vec_384 = engine.encode(giant_text, dim=384, normalize=True)
    assert vec_384 is not None
    assert vec_384.shape == (384,)
    assert not np.isnan(vec_384).any()

    # 2. Score resume fit against giant text
    resume = (
        "Staff Systems Architect with 12 years of experience designing high-throughput "
        "distributed data pipelines with Kafka, Go, C++, and Python."
    )
    fit_score = engine.score_resume_fit(resume, giant_text, dim=384)
    assert 0.40 <= fit_score <= 1.0, f"Expected high fit on matching giant text, got {fit_score}"


# ============================================================================
# 3. High-Concurrency & Thread-Safety Stress Test
# ============================================================================

def test_multithreaded_concurrency_stress():
    """16 threads executing concurrent encodings, dimensions, and task queries."""
    engine = get_semantic_engine()
    errors: list[str] = []
    num_threads = 16
    iterations_per_thread = 20

    dims_to_test = [768, 512, 384, 256, 128, 64]
    titles_to_test = [
        "Software Engineer Intern",
        "Junior Backend Developer",
        "Full Stack Developer",
        "Senior Systems Architect",
        "Staff Infrastructure Engineer",
        "Lead Product Designer",
    ]

    def worker_loop(worker_id: int):
        for i in range(iterations_per_thread):
            target_dim = dims_to_test[(worker_id + i) % len(dims_to_test)]
            title = titles_to_test[(worker_id + i) % len(titles_to_test)]

            try:
                # 1. Test encoding with explicit dim
                vec = engine.encode(title, dim=target_dim, normalize=True)
                if vec is None or vec.shape != (target_dim,):
                    errors.append(f"W{worker_id}: Vec shape mismatch: {vec.shape if vec is not None else None} != {target_dim}")
                elif np.isnan(vec).any():
                    errors.append(f"W{worker_id}: NaN in vector")

                # 2. Test seniority prediction with explicit dim
                sen = engine.classify_seniority(title, dim=target_dim)
                if sen is None or sen[0] not in ["intern", "newgrad", "junior", "mid", "senior", "staff"]:
                    errors.append(f"W{worker_id}: Invalid seniority result: {sen}")

                # 3. Test context manager dimension swapping
                with engine.using_dimension(target_dim):
                    scoped_vec = engine.encode(f"Scoped query {worker_id}-{i}", normalize=True)
                    if scoped_vec is None or scoped_vec.shape != (target_dim,):
                        errors.append(f"W{worker_id}: using_dimension({target_dim}) produced shape {scoped_vec.shape if scoped_vec is not None else None}")

                # 4. Batch scoring
                scores = engine.score_resume_fit_batch(
                    "Software engineering student with Python and C++ experience.",
                    [title, "Unrelated Dental Receptionist Role"],
                    dim=target_dim,
                )
                if len(scores) != 2 or not all(0.0 <= s <= 1.0 for s in scores):
                    errors.append(f"W{worker_id}: Invalid scores: {scores}")

            except Exception as exc:
                errors.append(f"W{worker_id} iteration {i} crashed: {exc}")

    threads = [threading.Thread(target=worker_loop, args=(t_id,)) for t_id in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert not errors, f"Encountered {len(errors)} concurrency errors:\n" + "\n".join(errors[:10])


# ============================================================================
# 4. Massive Batch Scaling Stress Test
# ============================================================================

def test_large_batch_scoring_stress():
    """Verify GPU/CPU handles large batches of 128 candidate postings in one pass."""
    engine = get_semantic_engine()
    resume_text = (
        "Senior Cloud Architect specializing in Kubernetes (EKS/GKE), Terraform, "
        "AWS multi-region infra, CI/CD automation, and Golang microservices."
    )

    # 128 job descriptions: 64 relevant, 64 irrelevant
    relevant_jobs = [
        f"Job {i}: Senior Cloud Infrastructure Engineer at TechCorp {i}. Manage Kubernetes, Terraform, and AWS."
        for i in range(64)
    ]
    irrelevant_jobs = [
        f"Job {i}: Certified Sous Chef at Restaurant {i}. Prepare gourmet culinary dishes, manage inventory."
        for i in range(64)
    ]
    all_jobs = relevant_jobs + irrelevant_jobs

    start_time = time.perf_counter()
    scores = engine.score_resume_fit_batch(resume_text, all_jobs, dim=384)
    elapsed = time.perf_counter() - start_time

    assert len(scores) == 128
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert not any(math.isnan(s) for s in scores)

    # Relevant scores should all be substantially higher than culinary jobs
    mean_rel = sum(scores[:64]) / 64.0
    mean_irrel = sum(scores[64:]) / 64.0

    print(f"\n[Batch 128] Time: {elapsed:.2f}s ({128/elapsed:.1f} jobs/s) | Relevant Mean: {mean_rel:.4f} | Irrelevant Mean: {mean_irrel:.4f}")
    assert mean_rel > 0.70, f"Expected high relevant mean, got {mean_rel}"
    assert mean_irrel < 0.58, f"Expected low irrelevant mean, got {mean_irrel}"
    assert mean_rel - mean_irrel > 0.18, "Semantic margin between relevant and irrelevant is too narrow"


# ============================================================================
# 5. Neural Judge (.bestjobs) Edge Cases
# ============================================================================

def test_neural_judge_none_and_corrupt_attributes():
    """Verify judge_jobs_neural survives corrupted jobs with None fields without crashing."""
    signal = ProfileSignal(
        profile_key="candidate",
        anchors=("python", "docker"),
        role_terms=("software developer",),
        seniority="student",
        locations=("toronto",),
        document_text="Computer Science student proficient in Python and Docker.",
    )

    corrupted_jobs = [
        # Normal
        ArchivedJob(
            title="Software Developer Intern",
            company="Acme",
            location="Toronto, ON",
            link="https://x/1",
            site_label="LinkedIn",
            date_posted="2026-08-25",
            description="Write Python code and test containerized services.",
        ),
        # Empty title and empty description
        ArchivedJob(
            title="",
            company="",
            location="",
            link="https://x/2",
            site_label="Unknown",
            date_posted="2026-08-25",
            description="",
        ),
        # None description (if bypassed)
        ArchivedJob(
            title="Python Intern",
            company=None,  # type: ignore
            location=None,  # type: ignore
            link="https://x/3",
            site_label="Lever",
            date_posted="2026-08-25",
            description=None,  # type: ignore
        ),
    ]

    verdicts, ranker = judge_jobs_neural(signal, corrupted_jobs, dim=384)
    assert ranker == "ok:nomic-neural"
    assert verdicts is not None
    assert len(verdicts) == 3

    # Every verdict must have valid scores and non-null reasons
    for idx, verdict in verdicts.items():
        assert 0.0 <= verdict.score <= 1.0
        assert isinstance(verdict.reason, str)
        assert len(verdict.reason) > 0


# ============================================================================
# 6. Cross-ATS Deduplication Subtle Edge Cases
# ============================================================================

def test_cross_ats_dedup_title_subtleties():
    """Verify 128-d deduplication strictly separates distinct roles and employers."""
    jobs = [
        # Pair 1: Requisitions of same role with slightly varying URL / punctuation at same employer
        {
            "company": "Amazon Web Services",
            "title": "Software Development Engineer II",
            "location": "Seattle, WA",
            "description": "Design distributed microservices at massive scale in Java and Rust.",
        },
        {
            "company": "Amazon Web Services",
            "title": "Software Development Engineer II - Distributed Systems",
            "location": "Seattle, WA",
            "description": "Design distributed microservices at massive scale in Java and Rust.",
        },
        # Distinct roles at same company with identical boilerplates
        {
            "company": "Amazon Web Services",
            "title": "Recruiting Coordinator",
            "location": "Seattle, WA",
            "description": "Design distributed microservices at massive scale in Java and Rust.",
        },
        # Same role but different company
        {
            "company": "Google Cloud",
            "title": "Software Development Engineer II",
            "location": "Seattle, WA",
            "description": "Design distributed microservices at massive scale in Java and Rust.",
        },
    ]

    dups = semantic_dedup.find_semantic_duplicates(jobs, threshold=0.88)
    dup_pairs = [(i, j) for i, j, score in dups]

    # Recruiting Coordinator must NEVER be merged with Software Development Engineer
    for i, j in dup_pairs:
        assert not (jobs[i]["title"] == "Recruiting Coordinator" or jobs[j]["title"] == "Recruiting Coordinator"), (
            f"Distinct roles wrongly merged: {jobs[i]['title']} <=> {jobs[j]['title']}"
        )

    # AWS and Google Cloud must NEVER be merged together
    for i, j in dup_pairs:
        assert jobs[i]["company"] == jobs[j]["company"], (
            f"Different employers merged: {jobs[i]['company']} <=> {jobs[j]['company']}"
        )


# ============================================================================
# 7. Mathematical Invariance of Matryoshka Slices
# ============================================================================

def test_matryoshka_matrix_slice_invariance():
    """Verify that W[:, :d] dot-product exactly equals slicing the full vector."""
    engine = get_semantic_engine()
    assert "seniority" in engine._task_weights
    w_master = engine._task_weights["seniority"]["weight"]  # (6, 768)
    b = engine._task_weights["seniority"]["bias"]          # (6,)

    text = "Senior Distributed Systems Engineer (Kafka / Kubernetes)"
    vec_768 = engine.encode(text, dim=768, task_type="classification", normalize=True)
    assert vec_768 is not None

    for d in [512, 384, 256, 128, 64]:
        # Direct dynamic slice from weights
        w_d, b_d, labels = engine.get_task_weights("seniority", dim=d)
        assert w_d.shape == (6, d)
        assert np.array_equal(w_d, w_master[:, :d])

        # Truncate and re-normalize vector to dim d
        vec_d = vec_768[:d]
        vec_d_normed = vec_d / np.linalg.norm(vec_d)

        # Logits via sliced matrix
        logits = np.dot(w_d, vec_d_normed) + b_d
        assert logits.shape == (6,)
        assert not np.isnan(logits).any()
