from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from services.resumes.configkey import GeminiSettings

RESUME_CACHE_RECORD_NAME = "resume_explicit_cache.json"
RESUME_SOURCE_SUFFIXES = {".md", ".txt"}
RESUME_CACHE_SYSTEM_INSTRUCTION = (
    "You are a resume tailoring assistant. Treat the cached resume source files as the canonical "
    "candidate background for future resume generation requests."
)


@dataclass(slots=True)
class ResumeExplicitCacheStatus:
    status: str
    message: str
    cache_name: str | None = None
    source_files: list[Path] = field(default_factory=list)
    remote_expire_time: str | None = None


def discover_resume_source_files(profiles_dir: Path) -> list[Path]:
    if not profiles_dir.exists():
        return []
    return [
        path
        for path in sorted(profiles_dir.iterdir())
        if path.is_file() and path.suffix.lower() in RESUME_SOURCE_SUFFIXES
    ]


def load_resume_source_bundle(profiles_dir: Path) -> tuple[str, str, list[Path]] | None:
    source_files = discover_resume_source_files(profiles_dir)
    if not source_files:
        return None

    sections: list[str] = []
    digest = hashlib.sha256()
    for path in source_files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        sections.append(f"Source file: {path.name}\n{text}")

    if not sections:
        return None

    return "\n\n".join(sections), digest.hexdigest(), source_files


def normalize_gemini_model_name(model: str) -> str:
    model_name = model.strip()
    if model_name.startswith("models/"):
        return model_name
    return f"models/{model_name}"


def build_resume_cache_payload_digest(source_digest: str, model: str) -> str:
    digest = hashlib.sha256()
    digest.update(normalize_gemini_model_name(model).encode("utf-8"))
    digest.update(b"\0")
    digest.update(RESUME_CACHE_SYSTEM_INSTRUCTION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source_digest.encode("utf-8"))
    return digest.hexdigest()


def _serialize_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


class ResumeExplicitCacheManager:
    def __init__(
        self,
        profiles_dir: Path,
        cache_dir: Path,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.profiles_dir = profiles_dir
        self.cache_dir = cache_dir
        self.client_factory = client_factory or self._default_client_factory

    @property
    def record_path(self) -> Path:
        return self.cache_dir / RESUME_CACHE_RECORD_NAME

    def ensure_local_dirs(self) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_cache(self, settings: GeminiSettings) -> ResumeExplicitCacheStatus:
        self.ensure_local_dirs()

        loaded = load_resume_source_bundle(self.profiles_dir)
        if loaded is None:
            return ResumeExplicitCacheStatus(
                status="missing-profile",
                message="No resume source files found in `resumes/`. Add one or more .md or .txt files.",
            )
        combined_text, source_digest, source_files = loaded

        if not settings.api_key:
            return ResumeExplicitCacheStatus(
                status="missing-api-key",
                message="Gemini API key is missing, so the remote explicit cache cannot be created.",
                source_files=source_files,
            )

        payload_digest = build_resume_cache_payload_digest(source_digest, settings.model)
        record = self._load_record()

        try:
            client = self.client_factory(settings.api_key)
        except Exception as exc:
            return ResumeExplicitCacheStatus(
                status="error",
                message=f"Could not initialize Gemini client: {exc}",
                source_files=source_files,
            )

        model_name = normalize_gemini_model_name(settings.model)
        if record and record.get("model") == model_name and record.get("payload_digest") == payload_digest:
            cache_name = str(record.get("cache_name") or "")
            remote_cache = self._get_cache(client, cache_name)
            if remote_cache is not None:
                expire_time = _serialize_datetime(getattr(remote_cache, "expire_time", None))
                if int(record.get("ttl_seconds") or 0) != settings.resume_cache_ttl_seconds:
                    remote_cache = self._update_cache_ttl(client, cache_name, settings.resume_cache_ttl_seconds)
                    expire_time = _serialize_datetime(getattr(remote_cache, "expire_time", None))
                    record["ttl_seconds"] = settings.resume_cache_ttl_seconds
                    record["expire_time"] = expire_time
                    record["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._save_record(record)
                    return ResumeExplicitCacheStatus(
                        status="updated",
                        message=f"Updated Gemini explicit cache `{cache_name}` TTL.",
                        cache_name=cache_name,
                        source_files=source_files,
                        remote_expire_time=expire_time,
                    )

                record["expire_time"] = expire_time
                record["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save_record(record)
                return ResumeExplicitCacheStatus(
                    status="reused",
                    message=f"Reused Gemini explicit cache `{cache_name}`.",
                    cache_name=cache_name,
                    source_files=source_files,
                    remote_expire_time=expire_time,
                )

        previous_cache_name = str(record.get("cache_name") or "") if record else ""
        try:
            created_cache = self._create_cache(
                client,
                model_name,
                settings.resume_cache_ttl_seconds,
                combined_text,
                source_digest,
            )
        except Exception as exc:
            return ResumeExplicitCacheStatus(
                status="error",
                message=(
                    "Gemini explicit cache creation failed. "
                    f"The local folder is not the same thing as the API cache: {exc}"
                ),
                source_files=source_files,
            )

        cache_name = str(getattr(created_cache, "name", "") or "")
        expire_time = _serialize_datetime(getattr(created_cache, "expire_time", None))
        self._save_record(
            {
                "cache_name": cache_name,
                "model": model_name,
                "payload_digest": payload_digest,
                "ttl_seconds": settings.resume_cache_ttl_seconds,
                "source_files": [path.name for path in source_files],
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "expire_time": expire_time,
            }
        )

        if previous_cache_name and previous_cache_name != cache_name:
            self._delete_cache(client, previous_cache_name)

        return ResumeExplicitCacheStatus(
            status="created",
            message=f"Created Gemini explicit cache `{cache_name}` from local resume files.",
            cache_name=cache_name,
            source_files=source_files,
            remote_expire_time=expire_time,
        )

    def _load_record(self) -> dict[str, Any] | None:
        if not self.record_path.exists():
            return None
        try:
            payload = json.loads(self.record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if not payload.get("cache_name") or not payload.get("model") or not payload.get("payload_digest"):
            return None
        return payload

    def _save_record(self, record: dict[str, Any]) -> None:
        self.record_path.write_text(json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8")

    def _default_client_factory(self, api_key: str) -> Any:
        from google import genai

        return genai.Client(api_key=api_key)

    def _create_cache(
        self,
        client: Any,
        model_name: str,
        ttl_seconds: int,
        combined_text: str,
        source_digest: str,
    ) -> Any:
        from google.genai import types

        return client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=f"resume-profile-{source_digest[:12]}",
                system_instruction=RESUME_CACHE_SYSTEM_INSTRUCTION,
                contents=[combined_text],
                ttl=f"{ttl_seconds}s",
            ),
        )

    def _get_cache(self, client: Any, cache_name: str) -> Any | None:
        try:
            return client.caches.get(name=cache_name)
        except Exception:
            return None

    def _update_cache_ttl(self, client: Any, cache_name: str, ttl_seconds: int) -> Any:
        from google.genai import types

        return client.caches.update(
            name=cache_name,
            config=types.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s"),
        )

    def _delete_cache(self, client: Any, cache_name: str) -> None:
        try:
            client.caches.delete(cache_name)
        except Exception:
            pass