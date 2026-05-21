from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from services.resumes.resume import infer_owner_profile_key


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
    resume_profiles_dir: Path = field(default_factory=Path)
    resume_cache_dir: Path = field(default_factory=Path)

    # ── Semantic plugin ───────────────────────────────────────────────────────
    semantic_enabled: bool = True
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_threshold: float = 0.30
    semantic_match_target: str = "description"
    semantic_description_char_limit: int = 2200

    # ── Job Bank ──────────────────────────────────────────────────────────────
    jobbank_max_search_pages: int = 5
    jobbank_user_agent: str = "Mozilla/5.0 (compatible; RebuiltJobWatcher/1.0)"

    # ── Deduplication ─────────────────────────────────────────────────────────
    dedup_months_threshold: int = 2
    dedup_max_fifo_files: int = 6
    dedup_max_entries_per_file: int = 500

    # ── Network timeouts (seconds) ────────────────────────────────────────────
    http_timeout_seconds: int = 20
    subprocess_scrape_timeout_seconds: int = 90

    # ── Watcher housekeeping ──────────────────────────────────────────────────
    watcher_dedupe_seconds: int = 120
    discord_history_check_limit: int = 5

    # ── Scrape defaults ───────────────────────────────────────────────────────
    scrape_max_items: int = 20
    scrape_timeout_seconds: int = 20
    scrape_use_ai_cleanup: bool = True

    # ── Job-watcher defaults ──────────────────────────────────────────────────
    job_default_keywords: str = "python developer"
    job_default_location: str = "Canada"
    job_default_radius_miles: int = 25
    job_default_hours_old: int = 72
    job_default_results_wanted: int = 10
    job_default_refresh_seconds: int = 300
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
        main_user_profile_key = str(main_user_profile_key).strip() or None
    if not main_user_profile_key:
        main_user_profile_key = infer_owner_profile_key(resumes_cache_root)
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

    # Behavioral settings — from settings.toml
    s = _load_settings(root / "settings.toml")
    sem = s.get("semantic", {})
    jb = s.get("jobbank", {})
    dd = s.get("dedup", {})
    net = s.get("network", {})
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
        resume_profiles_dir=root / "resumes",
        resume_cache_dir=root / ".resume_cache",
        # semantic
        semantic_enabled=bool(sem.get("enabled", True)),
        semantic_model=str(sem.get("model", "sentence-transformers/all-MiniLM-L6-v2")),
        semantic_threshold=float(sem.get("threshold", 0.30)),
        semantic_match_target=str(sem.get("match_target", "description")),
        semantic_description_char_limit=int(sem.get("description_char_limit", 2200)),
        # jobbank
        jobbank_max_search_pages=int(jb.get("max_search_pages", 5)),
        jobbank_user_agent=str(jb.get("user_agent", "Mozilla/5.0 (compatible; RebuiltJobWatcher/1.0)")),
        # dedup
        dedup_months_threshold=int(dd.get("months_threshold", 2)),
        dedup_max_fifo_files=int(dd.get("max_fifo_files", 6)),
        dedup_max_entries_per_file=int(dd.get("max_entries_per_file", 500)),
        # network
        http_timeout_seconds=int(net.get("http_timeout_seconds", 20)),
        subprocess_scrape_timeout_seconds=int(net.get("subprocess_scrape_timeout_seconds", 90)),
        # watcher housekeeping
        watcher_dedupe_seconds=int(wat.get("dedupe_seconds", 120)),
        discord_history_check_limit=int(wat.get("discord_history_check_limit", 5)),
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
        job_default_refresh_seconds=int(jd.get("refresh_seconds", 300)),
        job_default_country_indeed=str(jd.get("country_indeed", "AUTO")),
        job_default_allow_north_america=bool(jd.get("allow_north_america", False)),
        # reddit defaults
        reddit_default_subreddit=str(rd.get("subreddit", "wallpapers")),
        reddit_default_sort=str(rd.get("sort", "new")),
        reddit_default_time_filter=str(rd.get("time_filter", "day")),
        reddit_default_limit=int(rd.get("limit", 10)),
        reddit_default_refresh_seconds=int(rd.get("refresh_seconds", 300)),
    )
