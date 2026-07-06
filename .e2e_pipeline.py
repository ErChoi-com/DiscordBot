"""e2e pipeline: read jobs-goon channel settings -> scrape listing -> resume -> PDF + listing txt.

Usage:
  python .e2e_pipeline.py              # auto-picks first enabled job channel
  python .e2e_pipeline.py <channel_id> # targets a specific channel's saved settings
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from services import job_service
from services.resumes.configkey import load_gemini_settings
from services.resumes.listing import extract_job_context_from_message, generate_resume_rewrite
from services.resumes.resume import compile_latex_to_pdf, infer_owner_profile_key, RESUMES_CACHE_ROOT
from state.store import JOB_DEFAULTS, RuntimeStore
from watchers.manager import WatcherManager

OUT_DIR = ROOT / ".e2e_output"
TEX_DIR = OUT_DIR / "tex"
LOG_DIR = ROOT / ".resume_cache" / "compile_logs"
SEP = "=" * 72


def pick_channel_settings(store: RuntimeStore, target_channel_id: int | None) -> tuple[int | None, dict]:
    if target_channel_id is not None:
        return target_channel_id, store.get_job_settings(target_channel_id)

    enabled = [
        (int(cid), store.get_job_settings(int(cid)))
        for cid, settings in store.channel_job_settings.items()
        if settings.get("enabled")
    ]
    if enabled:
        return enabled[0]

    if store.channel_job_settings:
        cid = int(next(iter(store.channel_job_settings)))
        return cid, store.get_job_settings(cid)

    return None, dict(JOB_DEFAULTS)


def main() -> None:
    target_channel_id: int | None = None
    profile_key_override: str | None = None
    for arg in sys.argv[1:]:
        try:
            target_channel_id = int(arg)
        except ValueError:
            profile_key_override = arg

    config = load_config(ROOT)
    settings = load_gemini_settings(config)

    print(SEP)
    print("E2E PIPELINE  jobs-goon settings -> 1 listing -> resume -> PDF")
    print(SEP)
    print(f"  Gemini key     : {'YES' if settings.api_key else 'NO'}")
    print(f"  OpenRouter key : {'YES' if settings.openrouter_api_key else 'NO'}")
    print(f"  Groq key       : {'YES' if settings.groq_api_key else 'NO'}")
    print(f"  Primary model  : {settings.model}")

    if not (settings.api_key or settings.openrouter_api_key or settings.groq_api_key):
        print("\nERROR: No LLM API key found in .env")
        sys.exit(1)

    # ── Load state -> channel job settings ───────────────────────────────────
    store = RuntimeStore(config.state_path)
    store.load()

    channel_id, job_settings = pick_channel_settings(store, target_channel_id)
    keywords    = str(job_settings.get("keywords")      or JOB_DEFAULTS["keywords"])
    location    = str(job_settings.get("location")      or JOB_DEFAULTS["location"])
    hours_old   = int(job_settings.get("hours_old")     or JOB_DEFAULTS["hours_old"])
    radius_miles = int(job_settings.get("radius_miles") or JOB_DEFAULTS["radius_miles"])
    country_indeed = str(job_settings.get("country_indeed") or JOB_DEFAULTS["country_indeed"])

    print(f"\n  Channel        : {channel_id or '(none — defaults)'}")
    print(f"  Keywords       : {keywords}")
    print(f"  Location       : {location}")
    print(f"  Hours old      : {hours_old}")
    print(f"  Radius (mi)    : {radius_miles}")

    # ── Owner profile ─────────────────────────────────────────────────────────
    from services.resumes.resume import EXAMPLE_PROFILE_KEY, profile_cache_seed_ready, profile_cache_dir
    if profile_key_override:
        profile_key = profile_key_override
    else:
        configured_key = config.main_user_profile_key
        if configured_key and configured_key != EXAMPLE_PROFILE_KEY and profile_cache_seed_ready(profile_cache_dir(configured_key, RESUMES_CACHE_ROOT)):
            profile_key = configured_key
        else:
            profile_key = infer_owner_profile_key(RESUMES_CACHE_ROOT)
    if profile_key is None:
        available = [p.name for p in sorted(RESUMES_CACHE_ROOT.iterdir()) if p.is_dir() and p.name != EXAMPLE_PROFILE_KEY and profile_cache_seed_ready(p)]
        print(f"\nERROR: Cannot infer owner profile. Available: {available}")
        print(f"  Pass a profile name as an arg: python .e2e_pipeline.py xboxsignout._")
        sys.exit(1)
    profile_dir   = RESUMES_CACHE_ROOT / profile_key
    template_path = profile_dir / "template.tex"
    print(f"  Profile        : {profile_key}")

    # ── STEP 1: Scrape one listing ────────────────────────────────────────────
    raw_sites = job_settings.get("sites") or ["all"]
    if "all" in raw_sites:
        expanded = job_service.all_supported_job_sites(config.jobspy_python_exe)
        site_names = [s for s in expanded if s != job_service.JOBBANK_CANADA_SITE]
    else:
        site_names = [s for s in raw_sites if s != job_service.JOBBANK_CANADA_SITE]
    if not site_names:
        site_names = [s for s in job_service.FALLBACK_JOBSPY_SITES if s != job_service.JOBBANK_CANADA_SITE]

    print(f"\n{SEP}")
    print(f"STEP 1  Scraping listing from: {', '.join(site_names)}...")
    items = job_service.scrape_job_postings(
        site_names=site_names,
        keywords=keywords,
        location=location,
        configured_python_exe=config.jobspy_python_exe,
        hours_old=hours_old,
        results_wanted=8,
        radius_miles=radius_miles,
        country_indeed=country_indeed,
    )
    if not items:
        print(f"ERROR: No listings returned from {', '.join(site_names)}.")
        sys.exit(1)

    # Prefer items with a description so the LLM has real content to tailor against.
    item = next((i for i in items if i.get("description")), items[0])
    print(f"\n--- listing ---")
    for key, value in item.items():
        print(f"  {key}: {value}")
    print(f"--- end listing ---")

    # ── STEP 2: Parse job context ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 2  Parsing job context from listing...")
    msg = WatcherManager.format_job_watcher_message(item)
    job = extract_job_context_from_message(msg)
    if job is None:
        print("ERROR: Could not parse job context from listing.")
        sys.exit(1)
    print(f"  title : {job.title[:80]}")
    print(f"  url   : {job.posting_url}")

    # ── STEP 3: Generate tailored resume ──────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 3  Generating tailored resume (LLM)...")
    t0 = time.monotonic()
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
        print(f"ERROR: LLM failed ({elapsed_llm:.1f}s): {rewrite.message}")
        sys.exit(1)

    print(f"  OK  ({elapsed_llm:.1f}s)  provider={rewrite.used_provider}  {len(rewrite.latex_document)} chars")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in job.title[:50]).strip("-")
    tex_path = TEX_DIR / f"{profile_key}-{slug}.tex"
    tex_path.write_text(rewrite.latex_document, encoding="utf-8")
    print(f"  Raw .tex -> {tex_path.relative_to(ROOT)}")

    # ── STEP 4: Compile to PDF ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("STEP 4  Compiling LaTeX to PDF...")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"e2e-{profile_key}-{slug}.log"
    t1 = time.monotonic()
    compile_result = compile_latex_to_pdf(
        rewrite.latex_document,
        job.title,
        template_path,
        log_path,
        config.resume_normalize_json_latex,
    )
    elapsed_compile = time.monotonic() - t1

    if compile_result.status != "ok" or not compile_result.pdf_bytes:
        print(f"ERROR: Compile failed ({elapsed_compile:.1f}s): {compile_result.message}")
        if compile_result.log_excerpt:
            err = compile_result.log_excerpt
            idx = err.find("! ")
            excerpt = err[max(0, idx - 40): idx + 400] if idx >= 0 else err[-400:]
            print(f"\n--- log ---\n{excerpt}")
        sys.exit(1)

    pdf_path = OUT_DIR / f"{profile_key}-{slug}.pdf"
    pdf_path.write_bytes(compile_result.pdf_bytes)
    pdf_kb = len(compile_result.pdf_bytes) // 1024
    print(f"  OK  ({elapsed_compile:.1f}s)  {pdf_kb} KB")
    if compile_result.repairs_applied:
        print(f"  Auto-repairs: {compile_result.repairs_applied}")

    print(f"\n{SEP}")
    print(f"PDF  ->  {pdf_path}")
    print(SEP)


if __name__ == "__main__":
    main()
