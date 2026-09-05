from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from services.resumes.resume import EXAMPLE_PROFILE_KEY


@dataclass(slots=True)
class AppConfig:
    # ── Paths & secrets ───────────────────────────────────────────────────────
    base_dir: Path
    env_path: Path
    state_path: Path
    pid_path: Path
    lock_path: Path
    discord_token: str | None
    main_user_id: int | None
    main_user_profile_key: str | None
    openrouter_key: str | None
    groq_api_key: str | None
    gemini_api_key: str | None
    jobspy_python_exe: str | None
    gemini_model: str = "gemini-2.5-flash"
    gemini_resume_cache_ttl_seconds: int = 86400
    resume_normalize_json_latex: bool = True
    resume_max_pages: int = 1
    resume_profiles_dir: Path = field(default_factory=Path)
    resume_cache_dir: Path = field(default_factory=Path)

    # ── Semantic plugin ───────────────────────────────────────────────────────
    semantic_enabled: bool = True
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_threshold: float = 0.30
    semantic_match_target: str = "description"
    semantic_description_char_limit: int = 2200

    # ── Deduplication ─────────────────────────────────────────────────────────
    dedup_months_threshold: int = 1
    dedup_max_fifo_files: int = 6
    dedup_max_entries_per_file: int = 500

    # ── Network timeouts (seconds) ────────────────────────────────────────────
    http_timeout_seconds: int = 20
    subprocess_scrape_timeout_seconds: int = 90

    # ── Concurrency ──────────────────────────────────────────────────────────
    site_concurrency_limit: int = 2
    site_semaphore_timeout_seconds: int = 120

    # ── Watcher housekeeping ──────────────────────────────────────────────────
    watcher_dedupe_seconds: int = 120
    discord_history_check_limit: int = 5

    # ── Scrape defaults ───────────────────────────────────────────────────────
    scrape_max_items: int = 20
    scrape_timeout_seconds: int = 20
    scrape_use_ai_cleanup: bool = True

    # ── ATS scraper ───────────────────────────────────────────────────────────
    # On since 2026-09-05, having been off because BambooHR "is
    # Cloudflare-gated and needs a headless browser". Measured against 60
    # random non-dead boards with plain requests: 58 answered 200, two 401,
    # zero challenge pages, 43 boards carrying 347 live postings. Every
    # response is served through Cloudflare's CDN, which is what the original
    # claim saw -- but being behind the CDN is not being challenged by it.
    #
    # It is the largest fleet of the sixteen (21,291 boards, 14,169 not
    # dead-marked), so this was the single biggest source of coverage left
    # switched off. At 30 workers it reaches roughly half to three-quarters of
    # its non-dead fleet inside the 600s per-platform budget -- comparable to
    # icims at 43% -- and the tail rotation moves the unreached remainder each
    # cycle rather than stranding the same boards.
    #
    # If sustained load does provoke gating, the per-platform refusal breaker
    # now stops the platform after 25 consecutive refusals instead of spending
    # the whole budget on them, so the downside is bounded in a way it was not
    # when this was first switched off.
    ats_bamboohr_enabled: bool = True

    # ── Job-watcher defaults ──────────────────────────────────────────────────
    job_default_keywords: str = "python developer"
    job_default_location: str = "Canada"
    job_default_radius_miles: int = 25
    job_default_hours_old: int = 72
    job_default_results_wanted: int = 999
    job_default_refresh_seconds: int = 900
    job_default_country_indeed: str = "AUTO"
    job_default_allow_north_america: bool = False

    # ── Reddit-watcher defaults ───────────────────────────────────────────────
    reddit_default_subreddit: str = "wallpapers"
    reddit_default_sort: str = "new"
    reddit_default_time_filter: str = "day"
    reddit_default_limit: int = 10
    reddit_default_refresh_seconds: int = 300


def _load_settings(settings_path: Path) -> dict:
    if not settings_path.exists():
        return {}
    try:
        with settings_path.open("rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:
        print(f"Warning: could not load settings.toml: {exc}")
        return {}


def load_env(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    raw = value.strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def _env_bool(name: str, env_values: dict[str, str], default: bool) -> bool:
    parsed = _optional_bool(os.getenv(name, env_values.get(name)))
    return default if parsed is None else parsed


def load_config(base_dir: Path | None = None) -> AppConfig:
    root = (base_dir or Path(__file__).resolve().parents[1]).resolve()
    env_path = root / ".env"
    env_values = load_env(env_path)
    # Push .env values into os.environ so service modules can reach them via os.getenv()
    for k, v in env_values.items():
        os.environ.setdefault(k, v)
    resumes_cache_root = root / "src" / "services" / "resumes" / "resumes_cache"

    # Secrets — from .env / OS environment only, never from settings.toml
    discord_token = os.getenv("discordtoken", env_values.get("discordtoken"))
    main_user_id = _optional_int(
        os.getenv(
            "MAIN_USER_ID",
            os.getenv("mainUserId", env_values.get("MAIN_USER_ID", env_values.get("mainUserId"))),
        )
    )
    main_user_profile_key = os.getenv(
        "MAIN_USER_PROFILE",
        os.getenv("mainUserProfile", env_values.get("MAIN_USER_PROFILE", env_values.get("mainUserProfile"))),
    )
    if main_user_profile_key:
        # Lowercased to match discord_profile_key(), which lowercases every key
        # it generates -- so every profile directory on disk is lowercase. On
        # Windows a mixed-case .env value resolved anyway because the filesystem
        # is case-insensitive; on Linux it would silently fail to find the
        # profile.
        main_user_profile_key = str(main_user_profile_key).strip().lower() or None
    if not main_user_profile_key:
        main_user_profile_key = EXAMPLE_PROFILE_KEY
    openrouter_key = os.getenv("openRouter", env_values.get("openRouter"))
    groq_api_key = os.getenv(
        "GROQ_API_KEY",
        os.getenv("groqAPI", env_values.get("GROQ_API_KEY", env_values.get("groqAPI"))),
    )
    gemini_api_key = os.getenv(
        "GOOGLE_AI_STUDIO_API_KEY",
        os.getenv(
            "GEMINI_API_KEY",
            os.getenv(
                "geminiAPI",
                env_values.get(
                    "GOOGLE_AI_STUDIO_API_KEY",
                    env_values.get("GEMINI_API_KEY", env_values.get("geminiAPI")),
                ),
            ),
        ),
    )
    jobspy_python_exe = os.getenv("JOBSPY_PYTHON_EXE", env_values.get("JOBSPY_PYTHON_EXE"))
    gemini_model = os.getenv("GEMINI_MODEL", env_values.get("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_resume_cache_ttl_seconds = int(
        os.getenv(
            "GEMINI_RESUME_CACHE_TTL_SECONDS",
            env_values.get("GEMINI_RESUME_CACHE_TTL_SECONDS", "86400"),
        )
    )
    resume_normalize_json_latex = _env_bool("RESUME_NORMALIZE_JSON_LATEX", env_values, True)
    try:
        resume_max_pages = max(1, int(os.getenv("RESUME_MAX_PAGES", env_values.get("RESUME_MAX_PAGES", "1"))))
    except (TypeError, ValueError):
        resume_max_pages = 1

    # Behavioral settings — from settings.toml
    s = _load_settings(root / "settings.toml")
    sem = s.get("semantic", {})
    dd = s.get("dedup", {})
    net = s.get("network", {})
    ats = s.get("ats", {})
    wat = s.get("watcher", {})
    sc = s.get("scrape_defaults", {})
    jd = s.get("job_defaults", {})
    rd = s.get("reddit_defaults", {})

    return AppConfig(
        base_dir=root,
        env_path=env_path,
        state_path=root / ".bot_state.json",
        pid_path=root / ".bot.pid",
        lock_path=root / ".bot.lock",
        discord_token=discord_token,
        main_user_id=main_user_id,
        main_user_profile_key=main_user_profile_key,
        openrouter_key=openrouter_key,
        groq_api_key=groq_api_key,
        gemini_api_key=gemini_api_key,
        jobspy_python_exe=jobspy_python_exe,
        gemini_model=gemini_model,
        gemini_resume_cache_ttl_seconds=gemini_resume_cache_ttl_seconds,
        resume_normalize_json_latex=resume_normalize_json_latex,
        resume_max_pages=resume_max_pages,
        resume_profiles_dir=root / "resumes",
        resume_cache_dir=root / ".resume_cache",
        # semantic
        semantic_enabled=bool(sem.get("enabled", True)),
        semantic_model=str(sem.get("model", "sentence-transformers/all-MiniLM-L6-v2")),
        semantic_threshold=float(sem.get("threshold", 0.30)),
        semantic_match_target=str(sem.get("match_target", "description")),
        semantic_description_char_limit=int(sem.get("description_char_limit", 2200)),
        # dedup
        dedup_months_threshold=int(dd.get("months_threshold", 1)),
        dedup_max_fifo_files=int(dd.get("max_fifo_files", 6)),
        dedup_max_entries_per_file=int(dd.get("max_entries_per_file", 500)),
        # network
        http_timeout_seconds=int(net.get("http_timeout_seconds", 20)),
        subprocess_scrape_timeout_seconds=int(net.get("subprocess_scrape_timeout_seconds", 90)),
        site_concurrency_limit=int(net.get("site_concurrency_limit", 2)),
        site_semaphore_timeout_seconds=int(net.get("site_semaphore_timeout_seconds", 120)),
        # watcher housekeeping
        watcher_dedupe_seconds=int(wat.get("dedupe_seconds", 120)),
        discord_history_check_limit=int(wat.get("discord_history_check_limit", 5)),
        # ats
        ats_bamboohr_enabled=bool(ats.get("bamboohr_enabled", True)),
        # scrape defaults
        scrape_max_items=int(sc.get("max_items", 20)),
        scrape_timeout_seconds=int(sc.get("timeout_seconds", 20)),
        scrape_use_ai_cleanup=bool(sc.get("use_ai_cleanup", True)),
        # job defaults
        job_default_keywords=str(jd.get("keywords", "python developer")),
        job_default_location=str(jd.get("location", "Canada")),
        job_default_radius_miles=int(jd.get("radius_miles", 25)),
        job_default_hours_old=int(jd.get("hours_old", 72)),
        job_default_results_wanted=int(jd.get("results_wanted", 10)),
        job_default_refresh_seconds=int(jd.get("refresh_seconds", 900)),
        job_default_country_indeed=str(jd.get("country_indeed", "AUTO")),
        job_default_allow_north_america=bool(jd.get("allow_north_america", False)),
        # reddit defaults
        reddit_default_subreddit=str(rd.get("subreddit", "wallpapers")),
        reddit_default_sort=str(rd.get("sort", "new")),
        reddit_default_time_filter=str(rd.get("time_filter", "day")),
        reddit_default_limit=int(rd.get("limit", 10)),
        reddit_default_refresh_seconds=int(rd.get("refresh_seconds", 300)),
    )
