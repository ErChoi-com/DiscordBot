"""End-to-end pipeline: scrape real Job Bank listings -> Gemini Pro -> LaTeX -> PDF.

Runs both rickydisappoints and xboxsignout._ profiles with real API calls.
Saves every generated PDF to .e2e_output/ and every raw .tex to .e2e_output/tex/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from services import job_service
from services.resumes.configkey import load_gemini_settings
from services.resumes.listing import JobContext, extract_job_context_from_message, generate_resume_rewrite
from services.resumes.resume import compile_latex_to_pdf, sanitize_latex_document_for_compile
from watchers.manager import WatcherManager

CACHE_ROOT = Path(__file__).parent / "src" / "services" / "resumes" / "resumes_cache"
FAILED_DIR = Path(__file__).parent / ".resume_cache" / "failed_latex"
LOG_DIR    = Path(__file__).parent / ".resume_cache" / "compile_logs"
OUT_DIR    = Path(__file__).parent / ".e2e_output"
TEX_DIR    = OUT_DIR / "tex"
PROFILES   = ["rickydisappoints", "xboxsignout._"]
RESULTS_WANTED = 2   # 2 real listings is enough for a solid E2E pass

SEP = "=" * 72


def scrape_listings() -> list[dict]:
    print(SEP)
    print("STEP 1  Scraping Job Bank Canada (keywords=software developer)...")
    items = job_service.scrape_job_postings(
        site_names=[job_service.JOBBANK_CANADA_SITE],
        keywords="software developer",
        location="Canada",
        hours_old=168,
        results_wanted=RESULTS_WANTED,
        radius_miles=50,
        country_indeed="CANADA",
    )
    print(f"  {len(items)} listing(s) returned:")
    for item in items:
        print(f"  [{item.get('site_label','?')}] {item['title'][:75]}")
        print(f"    {item['link']}")
    return items


def format_and_parse(items: list[dict]) -> list[JobContext]:
    print(SEP)
    print("STEP 2  Formatting as Discord messages and parsing job context...")
    jobs: list[JobContext] = []
    for item in items:
        msg = WatcherManager.format_job_watcher_message(item)
        job = extract_job_context_from_message(msg)
        if job is None:
            print(f"  WARN  Could not parse job context from message.")
            continue
        jobs.append(job)
        print(f"  OK    title='{job.title[:70]}'")
        print(f"        url={job.posting_url}")
    return jobs


def run_one(profile: str, job: JobContext, settings, idx: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    profile_dir   = CACHE_ROOT / profile
    template_path = profile_dir / "template.tex"
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in job.title[:45]).strip("-")
    label = f"[{profile}] [{idx}]"

    # ── LLM generation ────────────────────────────────────────────────────────
    t0 = time.monotonic()
    print(f"\n{label} '{job.title[:60]}'")
    print(f"  Calling LLM...")

    rewrite = generate_resume_rewrite(
        settings,
        job,
        None,
        [profile_dir / "baseinfo.txt"],
        [profile_dir / "instructions.txt"],
        template_path,
    )
    elapsed_llm = time.monotonic() - t0

    if rewrite.status != "ok" or not rewrite.latex_document:
        print(f"  LLM FAIL  ({elapsed_llm:.1f}s)  {rewrite.message[:120]}")
        return {"profile": profile, "job": job.title, "llm": "fail", "compile": "skip",
                "message": rewrite.message}

    provider = rewrite.used_provider or "unknown"
    latex_len = len(rewrite.latex_document)
    line_count = rewrite.latex_document.count("\n") + 1
    print(f"  LLM OK    ({elapsed_llm:.1f}s)  provider={provider}  latex={latex_len} chars / {line_count} lines")

    # ── Save raw .tex ─────────────────────────────────────────────────────────
    tex_name = f"{profile}-{idx}-{slug}.tex"
    tex_path = TEX_DIR / tex_name
    tex_path.write_text(rewrite.latex_document, encoding="utf-8")
    print(f"  Raw LaTeX saved -> {tex_path.relative_to(Path(__file__).parent)}")

    # ── Show first few lines of raw output ────────────────────────────────────
    raw_lines = rewrite.latex_document.replace("\\n", "\n").splitlines()
    preview = raw_lines[:5]
    print(f"  --- raw LaTeX preview (first 5 lines) ---")
    for ln in preview:
        print(f"    {ln[:100]}")

    # ── Show what sanitize produces (what actually gets compiled) ─────────────
    sanitized = sanitize_latex_document_for_compile(rewrite.latex_document)
    san_lines  = sanitized.splitlines()
    print(f"  --- sanitized preview (first 5 lines, {len(san_lines)} total lines) ---")
    for ln in san_lines[:5]:
        print(f"    {ln[:100]}")

    # ── LaTeX compilation ─────────────────────────────────────────────────────
    log_path = LOG_DIR / f"e2e-{profile}-{idx}-{slug}.log"
    t1 = time.monotonic()
    print(f"  Compiling LaTeX...")

    compile_result = compile_latex_to_pdf(
        rewrite.latex_document,
        job.title,
        template_path,
        log_path,
    )
    elapsed_compile = time.monotonic() - t1

    if compile_result.status == "ok" and compile_result.pdf_bytes:
        pdf_name = f"{profile}-{idx}-{slug}.pdf"
        pdf_path = OUT_DIR / pdf_name
        pdf_path.write_bytes(compile_result.pdf_bytes)
        pdf_kb = len(compile_result.pdf_bytes) // 1024
        print(f"  COMPILE OK  ({elapsed_compile:.1f}s)  {pdf_kb} KB  -> {pdf_path.relative_to(Path(__file__).parent)}")
        if compile_result.repairs_applied:
            print(f"    Auto-repairs: {compile_result.repairs_applied}")
        return {"profile": profile, "job": job.title, "llm": "ok", "compile": "ok",
                "pdf_kb": pdf_kb, "provider": provider}
    else:
        fail_name = f"FAILED-{profile}-{idx}-{slug}.tex"
        (FAILED_DIR / fail_name).write_text(rewrite.latex_document, encoding="utf-8")
        print(f"  COMPILE FAIL  ({elapsed_compile:.1f}s)  {compile_result.message[:100]}")
        if compile_result.log_excerpt:
            # Show the first fatal error, not just the tail
            log_text = compile_result.log_excerpt
            err_idx = log_text.find("! ")
            excerpt = log_text[max(0, err_idx - 40): err_idx + 300] if err_idx >= 0 else log_text[-300:]
            print(f"    log:\n{excerpt}")
        print(f"    Failed LaTeX saved -> .resume_cache/failed_latex/{fail_name}")
        return {"profile": profile, "job": job.title, "llm": "ok", "compile": "fail",
                "message": compile_result.message, "provider": provider}


def main() -> None:
    print(SEP)
    print("E2E RESUME PIPELINE  (real Gemini Pro -> real LaTeX -> real PDF)")
    print(SEP)

    settings = load_gemini_settings()
    print(f"  Primary model   : {settings.model}")
    print(f"  Gemini key      : {'YES' if settings.api_key else 'NO'}")
    print(f"  OpenRouter key  : {'YES' if settings.openrouter_api_key else 'NO'}")
    print(f"  Groq key        : {'YES' if settings.groq_api_key else 'NO'}")
    print(f"  Profiles        : {PROFILES}")
    print(f"  Listings wanted : {RESULTS_WANTED}")

    if not (settings.api_key or settings.openrouter_api_key or settings.groq_api_key):
        print("\nERROR: No LLM API key found.")
        sys.exit(1)

    items = scrape_listings()
    if not items:
        print("\nERROR: No job listings scraped from Job Bank.")
        sys.exit(1)

    jobs = format_and_parse(items)
    if not jobs:
        print("\nERROR: No parseable job contexts.")
        sys.exit(1)

    all_results: list[dict] = []
    for profile in PROFILES:
        print(f"\n{SEP}")
        print(f"STEP 3  Profile: {profile}")
        profile_dir = CACHE_ROOT / profile
        missing = [f for f in ["baseinfo.txt", "instructions.txt", "template.tex"]
                   if not (profile_dir / f).exists()]
        if missing:
            print(f"  SKIP: missing files {missing}")
            continue
        for idx, job in enumerate(jobs, 1):
            result = run_one(profile, job, settings, idx)
            all_results.append(result)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    ok      = [r for r in all_results if r.get("compile") == "ok"]
    cfail   = [r for r in all_results if r.get("compile") == "fail"]
    lfail   = [r for r in all_results if r.get("llm") == "fail"]
    print(f"  Total runs   : {len(all_results)}")
    print(f"  Compile OK   : {len(ok)}")
    print(f"  Compile FAIL : {len(cfail)}")
    print(f"  LLM FAIL     : {len(lfail)}")

    if ok:
        print("\n  Successful PDFs:")
        for r in ok:
            print(f"    [{r['profile']}] via {r.get('provider','?')} -> {r['job'][:55]}  ({r.get('pdf_kb','?')} KB)")

    if cfail:
        print("\n  Compile failures:")
        for r in cfail:
            print(f"    [{r['profile']}] {r['job'][:55]} : {r.get('message','')[:70]}")

    if lfail:
        print("\n  LLM failures:")
        for r in lfail:
            print(f"    [{r['profile']}] {r['job'][:55]} : {r.get('message','')[:70]}")

    print()
    if not cfail and not lfail:
        print("  ALL RUNS PASSED - PDFs are in .e2e_output/")
        sys.exit(0)
    else:
        print(f"  {len(cfail) + len(lfail)} failure(s). See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
