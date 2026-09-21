"""Verification tests for the Pure Nomic Neural Judge in bestjobs.

Tests offline neural candidate ranking using 8,192-token Nomic embeddings,
verifying semantic ranking without requiring external paid LLM APIs.
"""
from __future__ import annotations

import pytest

from services import job_match
from services.job_match import ArchivedJob, ProfileSignal, judge_jobs_neural, rank_jobs


def _make_signal(text: str, seniority: str = "student") -> ProfileSignal:
    return ProfileSignal(
        profile_key="test_candidate",
        anchors=("c", "c++", "embedded", "firmware", "rtos", "stm32"),
        role_terms=("embedded software engineer", "firmware developer"),
        seniority=seniority,
        locations=("toronto", "ontario"),
        document_text=text,
    )


def _job(title: str, company: str = "Acme", location: str = "Toronto, ON", **kwargs) -> ArchivedJob:
    return ArchivedJob(
        title=title,
        company=company,
        location=location,
        link=kwargs.get("link", f"https://example.com/{abs(hash(title)) % 10**8}"),
        site_label=kwargs.get("site_label", "LinkedIn"),
        date_posted=kwargs.get("date_posted", "2026-08-24"),
        description=kwargs.get("description", ""),
    )


def test_judge_jobs_neural_ranks_relevant_job_higher():
    resume = (
        "Third-year Computer Engineering student with hands-on experience in C, C++, "
        "STM32 microcontrollers, FreeRTOS, device drivers, SPI, I2C, and embedded systems development."
    )
    signal = _make_signal(resume, seniority="student")

    jobs = [
        _job(
            title="Dental Hygienist",
            company="Smile Clinic",
            location="Toronto, ON",
            description="Perform teeth cleanings, dental examinations, patient care and dental charting.",
        ),
        _job(
            title="Junior Embedded Firmware Developer",
            company="Robotics Inc",
            location="Toronto, ON",
            description="Develop low-level C firmware for STM32 microcontrollers, configure peripheral registers, and test board bringup with oscilloscopes.",
        ),
    ]

    verdicts, ranker = judge_jobs_neural(signal, jobs, dim=384)

    assert ranker == "ok:nomic-neural"
    assert verdicts is not None
    assert len(verdicts) == 2

    dental_verdict = verdicts[0]
    embedded_verdict = verdicts[1]

    # Embedded job must significantly outscore dental job
    assert embedded_verdict.score > dental_verdict.score
    assert embedded_verdict.score >= 0.50
    assert "Nomic neural fit" in embedded_verdict.reason


def test_rank_jobs_with_neural_judge():
    resume = (
        "Software Engineering student experienced in Python, PyTorch, Pytest, backend development and REST APIs."
    )
    signal = _make_signal(resume, seniority="student")

    jobs = [
        _job(
            title="Landscape Laborer",
            company="Green Lawn",
            location="Toronto, ON",
            description="Mow lawns and maintain outdoor garden grounds.",
        ),
        _job(
            title="Software Engineering Intern",
            company="Tech Corp",
            location="Toronto, ON",
            description="Work with Python, PyTorch machine learning models, REST APIs, and automated test pipelines.",
        ),
    ]

    # Run rank_jobs with use_neural_judge=True and no LLM settings
    report = rank_jobs(signal, jobs, settings=None, limit=5, use_neural_judge=True)

    assert report.judged is True
    assert report.ranker == "ok:nomic-neural"
    assert len(report.matches) == 2

    # Top match must be the software engineering intern job
    top_match = report.matches[0]
    assert "Software Engineering Intern" in top_match.job.title
    assert top_match.score.llm_score is not None
    assert top_match.score.llm_score > 0.50
    assert "Nomic neural fit" in top_match.score.llm_reason

    # Format report check
    report_text = job_match.format_match_report(report, "day")
    assert "ranked by nomic-neural" in report_text
    assert "Software Engineering Intern" in report_text

