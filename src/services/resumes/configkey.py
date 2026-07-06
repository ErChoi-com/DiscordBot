from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_RESUME_CACHE_TTL_SECONDS = 86400
DEFAULT_OPENROUTER_RESUME_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_RESUME_MODEL = "openai/gpt-oss-120b"


def _read_dotenv_value(*keys: str) -> str | None:
    """Read the first matching key from the project .env file.

    Self-contained so configkey stays independent of the config module,
    which itself imports from services.resumes.resume.
    Returns None on any I/O error or if none of the keys are present.
    """
    from pathlib import Path
    try:
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if not env_path.exists():
            return None
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in keys:
                cleaned = v.strip().strip('"').strip("'")
                return cleaned or None
    except OSError:
        return None
    return None


@dataclass(slots=True)
class GeminiSettings:
    api_key: str | None
    model: str
    resume_cache_ttl_seconds: int = DEFAULT_RESUME_CACHE_TTL_SECONDS
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    openrouter_model: str = DEFAULT_OPENROUTER_RESUME_MODEL
    groq_model: str = DEFAULT_GROQ_RESUME_MODEL


def _configured_api_key(config: Any | None) -> str | None:
    if config is None:
        return None
    return getattr(config, "gemini_api_key", None)


def _configured_model(config: Any | None) -> str:
    if config is None:
        return DEFAULT_GEMINI_MODEL
    return str(getattr(config, "gemini_model", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL)


def _fallback_api_key() -> str | None:
    return (
        os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("geminiAPI")
        or _read_dotenv_value("GOOGLE_AI_STUDIO_API_KEY", "GEMINI_API_KEY", "geminiAPI")
    )


def _fallback_model() -> str | None:
    return os.getenv("GEMINI_MODEL") or None


def _configured_resume_cache_ttl(config: Any | None) -> int:
    if config is None:
        return DEFAULT_RESUME_CACHE_TTL_SECONDS
    raw_value = getattr(config, "gemini_resume_cache_ttl_seconds", DEFAULT_RESUME_CACHE_TTL_SECONDS)
    try:
        return int(raw_value or DEFAULT_RESUME_CACHE_TTL_SECONDS)
    except (TypeError, ValueError):
        return DEFAULT_RESUME_CACHE_TTL_SECONDS


def _fallback_resume_cache_ttl() -> int | None:
    raw_value = os.getenv("GEMINI_RESUME_CACHE_TTL_SECONDS")
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def _configured_openrouter_api_key(config: Any | None) -> str | None:
    if config is None:
        return None
    return getattr(config, "openrouter_key", None)


def _configured_groq_api_key(config: Any | None) -> str | None:
    if config is None:
        return None
    return getattr(config, "groq_api_key", None)


def _fallback_openrouter_api_key() -> str | None:
    return (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("openRouter")
        or _read_dotenv_value("OPENROUTER_API_KEY", "openRouter")
    )


def _fallback_groq_api_key() -> str | None:
    return (
        os.getenv("GROQ_API_KEY")
        or os.getenv("groqAPI")
        or _read_dotenv_value("GROQ_API_KEY", "groqAPI")
    )


def _resolve_openrouter_model() -> str:
    return (
        os.getenv("OPENROUTER_RESUME_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or _read_dotenv_value("OPENROUTER_RESUME_MODEL", "OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_RESUME_MODEL
    )


def _resolve_groq_model() -> str:
    return (
        os.getenv("GROQ_RESUME_MODEL")
        or os.getenv("GROQ_MODEL")
        or _read_dotenv_value("GROQ_RESUME_MODEL", "GROQ_MODEL")
        or DEFAULT_GROQ_RESUME_MODEL
    )


def load_gemini_settings(config: Any | None = None) -> GeminiSettings:
    configured_api_key = _configured_api_key(config)
    api_key = configured_api_key if configured_api_key else _fallback_api_key()

    model = _configured_model(config)
    if model == DEFAULT_GEMINI_MODEL:
        model = _fallback_model() or model

    resume_cache_ttl_seconds = _configured_resume_cache_ttl(config)
    if resume_cache_ttl_seconds == DEFAULT_RESUME_CACHE_TTL_SECONDS:
        resume_cache_ttl_seconds = _fallback_resume_cache_ttl() or resume_cache_ttl_seconds

    openrouter_api_key = _configured_openrouter_api_key(config) or _fallback_openrouter_api_key()
    groq_api_key = _configured_groq_api_key(config) or _fallback_groq_api_key()

    return GeminiSettings(
        api_key=api_key,
        model=model,
        resume_cache_ttl_seconds=max(300, int(resume_cache_ttl_seconds)),
        openrouter_api_key=openrouter_api_key or None,
        groq_api_key=groq_api_key or None,
        openrouter_model=_resolve_openrouter_model(),
        groq_model=_resolve_groq_model(),
    )