#!/usr/bin/env python3
"""
Run the resume pipeline against a direct company job posting URL.

Scrapes the listing directly (no job banks), generates a tailored PDF,
then compares it against the live job description.

Usage (from rebuilt_app/):
    python run_resume.py [URL]

    URL  — direct company career page (Lever, Greenhouse, Ashby, or company site)
           defaults to Nord Quantique's careers page (appeared in bot history)

Examples of direct company URLs (not job banks):
    https://nordquantique.ca/en/career/
    https://jobs.lever.co/clio/<job-id>
    https://jobs.ashbyhq.com/cohere/<job-id>
    https://boards.greenhouse.io/shopify/jobs/<job-id>
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from services.resumes.configkey import load_gemini_settings
from services.resumes.listing import JobContext, scrape_job_posting, generate_resume_rewrite
from services.resumes.resume import RESUMES_CACHE_ROOT, compile_latex_to_pdf

PROFILE_KEY = "xboxsignout._"
OUT_PDF = Path(__file__).resolve().parent / "resume_output.pdf"
OUT_TEX = Path(__file__).resolve().parent / "resume_output.tex"

# Nord Quantique: appeared in bot history, direct company site, no job bank
DEFAULT_URL = "https://nordquantique.ca/en/career/"

SEP = "-" * 64


# ── Keyword comparison ────────────────────────────────────────────────────────

_TECH_PATTERN = re.compile(
    r'\b('
    r'python|java(?:script)?|typescript|node\.?js|react|vue|angular|'
    r'flask|fastapi|django|spring|rails|'
    r'docker|kubernetes|k8s|terraform|ansible|'
    r'postgresql|mysql|mongodb|redis|sqlite|sql|nosql|'
    r'aws|gcp|azure|cloud|s3|lambda|'
    r'git|github|gitlab|ci[/-]cd|jenkins|'
    r'kafka|rabbitmq|celery|'
    r'tensorflow|pytorch|scikit[- ]?learn|pandas|numpy|'
    r'rest(?:ful)?|graphql|grpc|api|'
    r'linux|ubuntu|bash|shell|'
    r'c\+\+|c#|golang|go|rust|kotlin|swift|matlab|'
    r'html|css|sass|webpack|vite|'
    r'opengl|glsl|glfw|glut|webrtc|'
    r'fpga|vhdl|verilog|arduino|raspberry[- ]?pi|'
    r'llm|nlp|computer vision|reinforcement learning|deep learning|machine learning'
    r')\b',
    re.IGNORECASE,
)

_ROLE_PATTERN = re.compile(
    r'\b('
    r'data science|data engineering|data pipeline|'
    r'backend|front.?end|full.?stack|devops|platform engineering|'
    r'distributed systems|microservices|'
    r'performance optimization|scalab|reliability|'
    r'unit test|integration test|automated test|tdd|'
    r'agile|scrum|cross.functional|'
    r'technical documentation|technical writing|system design'
    r')\b',
    re.IGNORECASE,
)


def extract_keywords(text: str) -> list[str]:
    found: dict[str, str] = {}
    for m in _TECH_PATTERN.finditer(text):
        found[m.group(0).lower()] = m.group(0)
    for m in _ROLE_PATTERN.finditer(text):
        found[m.group(0).lower()] = m.group(0)
    return sorted(found.values(), key=lambda s: s.lower())


def strip_latex_comments(tex: str) -> str:
    active: list[str] = []
    in_comment_block = False
    for line in tex.splitlines():
        stripped = line.strip()
        if stripped == r"\begin{comment}":
            in_comment_block = True
            continue
        if stripped == r"\end{comment}":
            in_comment_block = False
            continue
        if in_comment_block or stripped.startswith("%"):
            continue
        active.append(line)
    return "\n".join(active)


def compare(listing_text: str, tex: str) -> None:
    listing_kw = extract_keywords(listing_text)
    if not listing_kw:
        print("  (no recognisable tech/role keywords found in listing)")
        return

    active_tex = strip_latex_comments(tex).lower()
    matched = [kw for kw in listing_kw if kw.lower() in active_tex]
    missing = [kw for kw in listing_kw if kw.lower() not in active_tex]

    pct = int(100 * len(matched) / len(listing_kw)) if listing_kw else 0
    print(f"  Listing keywords  : {len(listing_kw)}")
    print(f"  Matched in resume : {len(matched)}  ({pct}%)")
    print(f"  Not in resume     : {len(missing)}")
    if matched:
        print()
        print("  Matched:")
        for kw in matched:
            print(f"    [+] {kw}")
    if missing:
        print()
        print("  Missing (may be irrelevant or deliberately commented out):")
        for kw in missing:
            print(f"    [ ] {kw}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    profile_dir = RESUMES_CACHE_ROOT / PROFILE_KEY
    template_path = profile_dir / "template.tex"

    for required in [profile_dir, template_path]:
        if not required.exists():
            print(f"[error] Missing: {required}")
            sys.exit(1)

    settings = load_gemini_settings()

    print(SEP)
    print(f"  URL     : {url}")
    print(f"  Profile : {PROFILE_KEY}")
    print(f"  Model   : {settings.model}")
    print(f"  Gemini  : {'set' if settings.api_key else 'MISSING'}")
    print(SEP)

    # ── Step 1: scrape the posting directly ──────────────────────────────────
    print("\n[1/3] Scraping job posting...")
    t0 = time.monotonic()
    try:
        scraped = scrape_job_posting(url)
    except Exception as exc:
        print(f"[error] Scrape failed: {exc}")
        sys.exit(1)
    print(f"  Title   : {scraped.title}")
    print(f"  Company : {scraped.company or '(unknown)'}")
    print(f"  Location: {scraped.location or '(unknown)'}")
    print(f"  Desc len: {len(scraped.description)} chars")
    print(f"  Done in {time.monotonic() - t0:.1f}s")

    if len(scraped.description) < 200:
        print("[warn] Description is very short — the page may have blocked scraping.")

    # ── Step 2: LLM rewrite ──────────────────────────────────────────────────
    print(f"\n[2/3] Generating tailored resume via LLM...")
    job = JobContext(
        title=scraped.title,
        posting_url=url,
        apply_url=None,
        source_message=f"{scraped.title}\n{url}",
    )
    t1 = time.monotonic()
    rewrite = generate_resume_rewrite(
        settings,
        job,
        None,
        baseinfo_paths=[profile_dir / "baseinfo.txt"],
        support_paths=[profile_dir / "instructions.txt"],
        template_path=template_path,
        scraper=lambda _: scraped,
    )
    llm_elapsed = time.monotonic() - t1

    if rewrite.status != "ok" or not rewrite.latex_document:
        print(f"[error] LLM failed after {llm_elapsed:.1f}s: {rewrite.message}")
        if rewrite.prompt_preview:
            print(f"\n[prompt preview]\n{rewrite.prompt_preview}")
        sys.exit(1)

    OUT_TEX.write_text(rewrite.latex_document, encoding="utf-8")
    print(f"  Done in {llm_elapsed:.1f}s via {rewrite.used_provider}")
    print(f"  LaTeX  -> {OUT_TEX.name}")

    # ── Step 3: compile ───────────────────────────────────────────────────────
    print(f"\n[3/3] Compiling LaTeX to PDF...")
    t2 = time.monotonic()
    compiled = compile_latex_to_pdf(
        rewrite.latex_document,
        scraped.title,
        template_path,
    )
    compile_elapsed = time.monotonic() - t2

    if compiled.status != "ok" or not compiled.pdf_bytes:
        print(f"[error] Compile failed after {compile_elapsed:.1f}s: {compiled.message}")
        if compiled.log_excerpt:
            print(f"\n[log]\n{compiled.log_excerpt[-1000:]}")
        sys.exit(1)

    OUT_PDF.write_bytes(compiled.pdf_bytes)
    pdf_kb = len(compiled.pdf_bytes) // 1024
    print(f"  Done in {compile_elapsed:.1f}s  ({pdf_kb} KB)")
    if compiled.repairs_applied:
        print(f"  Repairs: {', '.join(compiled.repairs_applied)}")
    print(f"  PDF    -> {OUT_PDF.name}")

    # ── Comparison ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  LISTING vs RESUME COMPARISON")
    print(SEP)
    compare(scraped.description, rewrite.latex_document)

    total = llm_elapsed + compile_elapsed
    print(f"\n{SEP}")
    print(f"  Total: {total:.1f}s   PDF: {OUT_PDF}   LaTeX: {OUT_TEX}")
    print(SEP)


if __name__ == "__main__":
    main()
