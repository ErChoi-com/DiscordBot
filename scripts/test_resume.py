#!/usr/bin/env python3
"""
Standalone smoke-test for the resume rewrite pipeline.

Picks a random sample job listing, runs the full LLM + LaTeX compile
pipeline for the xboxsignout._ profile, and writes a PDF to the
project root.

Usage (from rebuilt_app/):
    python test_resume.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.resumes.configkey import load_gemini_settings
from services.resumes.listing import JobContext, ScrapedJobPosting, generate_resume_rewrite
from services.resumes.resume import RESUMES_CACHE_ROOT, compile_latex_to_pdf

PROFILE_KEY = "xboxsignout._"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "test_resume_output.pdf"

SAMPLE_LISTINGS: list[ScrapedJobPosting] = [
    ScrapedJobPosting(
        title="Backend Software Engineering Intern",
        company="Shopify",
        location="Toronto, ON (Hybrid)",
        description="""
Backend Software Engineering Intern — Shopify, Toronto ON

About the role:
Join Shopify's developer platform team to build scalable backend services
that power millions of merchants worldwide.

Responsibilities:
- Design and ship production backend features in Ruby or Python
- Build and maintain RESTful and GraphQL APIs consumed by external developers
- Write comprehensive automated tests and participate in code review
- Work with PostgreSQL and distributed data systems
- Containerize services with Docker and contribute to CI/CD pipelines

Requirements:
- Enrolled in Computer Science, Software Engineering, or related degree
- Experience with Python, Ruby, Go, or similar backend language
- Familiarity with REST APIs, SQL databases, and Git
- Strong problem-solving skills and attention to detail
- Docker or containerization experience is a plus
""".strip(),
        highlights=[
            "Design and ship production backend features in Ruby or Python",
            "Build and maintain RESTful and GraphQL APIs",
            "Write comprehensive automated tests and participate in code review",
            "Experience with Python, Ruby, Go, or similar backend language",
            "Familiarity with REST APIs, SQL databases, and Git",
        ],
        source_url="https://www.shopify.com/careers/backend-software-engineering-intern",
    ),
    ScrapedJobPosting(
        title="Machine Learning Engineering Intern",
        company="Cohere",
        location="Toronto, ON",
        description="""
Machine Learning Engineering Intern — Cohere, Toronto ON

About the role:
Cohere builds large language models for the enterprise. You'll join the
ML infrastructure team, working on data pipelines and model evaluation tooling.

Responsibilities:
- Build and maintain ML data pipelines for training and evaluation
- Develop tooling for model performance analysis and reporting using Pandas and Python
- Integrate experiment tracking with internal dashboards (Tableau, custom tooling)
- Work with FastAPI-based microservices for inference endpoints
- Collaborate with research and engineering teams to ship improvements

Requirements:
- Pursuing a degree in Computer Science, Engineering, or related field
- Experience with Python and ML frameworks (TensorFlow, PyTorch, or scikit-learn)
- Familiarity with data pipeline tools (Kafka, Spark, or similar)
- Strong Python skills including async patterns, testing, and code quality
- Experience with containerized deployments (Docker) preferred
""".strip(),
        highlights=[
            "Build and maintain ML data pipelines for training and evaluation",
            "Develop tooling for model performance analysis and reporting",
            "Integrate experiment tracking with internal dashboards",
            "Experience with Python and ML frameworks (TensorFlow, PyTorch, or scikit-learn)",
            "Familiarity with data pipeline tools (Kafka, Spark, or similar)",
        ],
        source_url="https://cohere.com/careers/machine-learning-engineering-intern",
    ),
    ScrapedJobPosting(
        title="Frontend Software Developer Intern",
        company="Thomson Reuters",
        location="Toronto, ON (Hybrid)",
        description="""
Frontend Software Developer Intern — Thomson Reuters, Toronto ON

About the role:
You'll work on the Westlaw Edge product team, building responsive web
interfaces used by legal professionals across North America.

Responsibilities:
- Build and iterate on React and TypeScript components in a large production codebase
- Integrate frontend features with RESTful backend services (Python/Flask)
- Write unit and integration tests for UI components
- Participate in agile ceremonies and collaborate with UX designers
- Use Docker for local development environments

Requirements:
- Currently enrolled in Computer Science, Software Engineering, or related program
- Hands-on experience with React and TypeScript
- Understanding of REST APIs and how frontend integrates with backend services
- Familiarity with Git, code review workflows, and agile development
- Experience with containerization or CI/CD is a strong plus
""".strip(),
        highlights=[
            "Build and iterate on React and TypeScript components",
            "Integrate frontend features with RESTful backend services",
            "Write unit and integration tests for UI components",
            "Hands-on experience with React and TypeScript",
            "Understanding of REST APIs and how frontend integrates with backend services",
        ],
        source_url="https://thomsonreuters.com/careers/frontend-software-developer-intern",
    ),
    ScrapedJobPosting(
        title="Software Developer Intern – Systems & Infrastructure",
        company="AMD",
        location="Markham, ON",
        description="""
Software Developer Intern – Systems & Infrastructure — AMD, Markham ON

About the role:
Join AMD's driver and tools team to develop software that interfaces with
GPU hardware. You'll work on performance tooling and validation infrastructure
for AMD graphics products.

Responsibilities:
- Develop C++ tooling for GPU performance measurement and validation
- Write automated test suites to verify driver correctness and performance
- Optimize memory allocation and data throughput in graphics pipelines
- Document test procedures and validation results clearly
- Collaborate with hardware and firmware teams to reproduce and triage issues

Requirements:
- Enrolled in Computer Engineering, Electrical Engineering, or Computer Science
- Strong C++ skills with understanding of memory management and performance optimization
- Experience with OpenGL, Vulkan, or similar graphics APIs is a strong plus
- Familiarity with GPU programming (GLSL, HLSL, CUDA, or similar) preferred
- Solid debugging skills and attention to detail in low-level systems work
""".strip(),
        highlights=[
            "Develop C++ tooling for GPU performance measurement and validation",
            "Write automated test suites to verify driver correctness and performance",
            "Optimize memory allocation and data throughput in graphics pipelines",
            "Strong C++ skills with understanding of memory management and performance optimization",
            "Experience with OpenGL, Vulkan, or similar graphics APIs is a strong plus",
        ],
        source_url="https://amd.com/careers/software-developer-intern-systems-infrastructure",
    ),
]


def main() -> None:
    profile_dir = RESUMES_CACHE_ROOT / PROFILE_KEY
    if not profile_dir.exists():
        print(f"[error] Profile not found: {profile_dir}")
        sys.exit(1)

    template_path = profile_dir / "template.tex"
    if not template_path.exists():
        print(f"[error] template.tex missing from {profile_dir}")
        sys.exit(1)

    job_posting = random.choice(SAMPLE_LISTINGS)
    settings = load_gemini_settings()

    print("=" * 60)
    print("  Resume pipeline smoke test")
    print("=" * 60)
    print(f"  Profile : {PROFILE_KEY}")
    print(f"  Job     : {job_posting.title} @ {job_posting.company}")
    print(f"  Location: {job_posting.location}")
    print(f"  Model   : {settings.model}")
    print(f"  Gemini  : {'set' if settings.api_key else 'MISSING'}")
    print(f"  Groq    : {'set' if settings.groq_api_key else 'missing'}")
    print(f"  Output  : {OUTPUT_PATH}")
    print("=" * 60)

    job = JobContext(
        title=job_posting.title,
        posting_url=job_posting.source_url,
        apply_url=None,
        source_message=f"{job_posting.title}\n{job_posting.source_url}",
    )

    print("\n[1/2] Generating tailored resume via LLM...")
    t0 = time.monotonic()
    result = generate_resume_rewrite(
        settings=settings,
        job=job,
        cache_name=None,
        baseinfo_paths=[profile_dir / "baseinfo.txt"],
        support_paths=[profile_dir / "instructions.txt"],
        template_path=template_path,
        scraper=lambda _url: job_posting,
    )
    llm_elapsed = time.monotonic() - t0

    if result.status != "ok" or result.latex_document is None:
        print(f"[error] LLM step failed after {llm_elapsed:.1f}s: {result.message}")
        if result.prompt_preview:
            print(f"\n[prompt preview]\n{result.prompt_preview}")
        sys.exit(1)

    print(f"[ok]    Done in {llm_elapsed:.1f}s via {result.used_provider}")

    print("\n[2/2] Compiling LaTeX to PDF...")
    t1 = time.monotonic()
    compile_result = compile_latex_to_pdf(
        latex_document=result.latex_document,
        job_title=job_posting.title,
        template_path=template_path,
    )
    compile_elapsed = time.monotonic() - t1

    if compile_result.status != "ok" or compile_result.pdf_bytes is None:
        print(f"[error] Compile failed after {compile_elapsed:.1f}s: {compile_result.message}")
        if compile_result.log_excerpt:
            print(f"\n[log excerpt]\n{compile_result.log_excerpt}")
        sys.exit(1)

    OUTPUT_PATH.write_bytes(compile_result.pdf_bytes)

    print(f"[ok]    Done in {compile_elapsed:.1f}s")
    if compile_result.repairs_applied:
        print(f"        Repairs: {', '.join(compile_result.repairs_applied)}")
    if compile_result.lint_findings:
        print(f"        Lint   : {'; '.join(compile_result.lint_findings)}")

    print(f"\n{'=' * 60}")
    print(f"  PDF written to: {OUTPUT_PATH}")
    print(f"  Total time    : {llm_elapsed + compile_elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
