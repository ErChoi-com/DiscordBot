from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODE_DESCRIPTIONS = {
    "normal": "Default behavior for general use.",
    "build": "Build-oriented workflow.",
    "analyze": "Analysis workflow.",
    "export": "Export/report workflow.",
}
DEFAULT_MODE = "normal"

SCRAPE_DEFAULTS: dict[str, Any] = {
    "max_items": 20,
    "timeout_seconds": 20,
    "use_ai_cleanup": True,
}

JOB_DEFAULTS: dict[str, Any] = {
    "sites": ["all"],
    "keywords": "python developer",
    "location": "Canada",
    "radius_miles": 25,
    "role_filters": [],
    "exclusion_terms": [],
    "hours_old": 72,
    "results_wanted": 50,
    "refresh_seconds": 300,
    "country_indeed": "AUTO",
    "allow_north_america": False,
    "jobbank_native_query": "",
    "semantic_threshold": 0.30,
    "ats_semantic_threshold": 0.20,
    "enabled": False,
}

REDDIT_DEFAULTS: dict[str, Any] = {
    "subreddits": ["wallpapers"],
    "sort": "new",
    "time_filter": "day",
    "flair_tags": "",
    "include_nsfw": False,
    "include_spoiler": False,
    "media_mode": "image",
    "limit": 10,
    "refresh_seconds": 300,
    "enabled": False,
}

DEFAULT_CONFIG_OVERRIDES: tuple[tuple[dict[str, Any], tuple[tuple[str, str, type], ...]], ...] = (
    (
        SCRAPE_DEFAULTS,
        (
            ("max_items", "scrape_max_items", int),
            ("timeout_seconds", "scrape_timeout_seconds", int),
            ("use_ai_cleanup", "scrape_use_ai_cleanup", bool),
        ),
    ),
    (
        JOB_DEFAULTS,
        (
            ("keywords", "job_default_keywords", str),
            ("location", "job_default_location", str),
            ("radius_miles", "job_default_radius_miles", int),
            ("hours_old", "job_default_hours_old", int),
            ("results_wanted", "job_default_results_wanted", int),
            ("refresh_seconds", "job_default_refresh_seconds", int),
            ("country_indeed", "job_default_country_indeed", str),
            ("allow_north_america", "job_default_allow_north_america", bool),
        ),
    ),
    (
        REDDIT_DEFAULTS,
        (
            ("sort", "reddit_default_sort", str),
            ("time_filter", "reddit_default_time_filter", str),
            ("limit", "reddit_default_limit", int),
            ("refresh_seconds", "reddit_default_refresh_seconds", int),
        ),
    ),
)


def init_store_defaults(config: Any) -> None:
    """Overwrite the module-level default dicts from settings.toml values (via AppConfig).
    Call once from app.py after load_config().
    """
    for defaults, overrides in DEFAULT_CONFIG_OVERRIDES:
        for key, config_attr, value_type in overrides:
            defaults[key] = value_type(getattr(config, config_attr, defaults[key]))
    REDDIT_DEFAULTS["subreddits"] = [str(getattr(config, "reddit_default_subreddit", "wallpapers"))]


class RuntimeStore:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.channel_modes: dict[int, str] = {}
        self.channel_scrape_settings: dict[int, dict[str, Any]] = {}
        self.channel_job_settings: dict[int, dict[str, Any]] = {}
        self.channel_reddit_settings: dict[int, dict[str, Any]] = {}
        self.channel_job_seen: dict[int, dict[str, None]] = {}
        self.channel_cheatsheet_ids: dict[str, int] = {}

    def get_cheatsheet_message_id(self, channel_id: int, sheet_kind: str) -> int | None:
        return self.channel_cheatsheet_ids.get(f"{channel_id}:{sheet_kind}")

    def set_cheatsheet_message_id(self, channel_id: int, sheet_kind: str, message_id: int) -> None:
        self.channel_cheatsheet_ids[f"{channel_id}:{sheet_kind}"] = message_id
        self.save()

    def clear_cheatsheet_message_id(self, channel_id: int, sheet_kind: str) -> None:
        self.channel_cheatsheet_ids.pop(f"{channel_id}:{sheet_kind}", None)
        self.save()

    def get_mode(self, channel_id: int) -> str:
        return self.channel_modes.get(channel_id, DEFAULT_MODE)

    def set_mode(self, channel_id: int, mode_name: str) -> None:
        self.channel_modes[channel_id] = mode_name
        self.save()

    def get_scrape_settings(self, channel_id: int) -> dict[str, Any]:
        merged = dict(SCRAPE_DEFAULTS)
        merged.update(self.channel_scrape_settings.get(channel_id, {}))
        return merged

    def update_scrape_setting(self, channel_id: int, key: str, value: Any) -> None:
        current = self.channel_scrape_settings.setdefault(channel_id, {})
        current[key] = value
        self.save()

    def get_job_settings(self, channel_id: int) -> dict[str, Any]:
        merged = dict(JOB_DEFAULTS)
        current = self.channel_job_settings.get(channel_id, {})
        merged.update(current)
        merged["sites"] = list(current.get("sites", JOB_DEFAULTS["sites"]))
        merged["role_filters"] = list(current.get("role_filters", JOB_DEFAULTS["role_filters"]))
        merged["exclusion_terms"] = list(current.get("exclusion_terms", JOB_DEFAULTS["exclusion_terms"]))
        merged["jobbank_native_query"] = str(current.get("jobbank_native_query", JOB_DEFAULTS["jobbank_native_query"]))
        return merged

    def update_job_setting(self, channel_id: int, key: str, value: Any) -> None:
        current = self.channel_job_settings.setdefault(channel_id, {})
        current[key] = value
        self.save()

    def remove_job_channel(self, channel_id: int) -> None:
        """Remove all persisted job-watcher state for a channel."""
        self.channel_job_settings.pop(channel_id, None)
        self.channel_job_seen.pop(channel_id, None)
        self.save()

    def get_reddit_settings(self, channel_id: int) -> dict[str, Any]:
        merged = dict(REDDIT_DEFAULTS)
        merged.update(self.channel_reddit_settings.get(channel_id, {}))
        if "subreddit" in merged and "subreddits" not in merged:
            raw = str(merged.pop("subreddit")).strip()
            merged["subreddits"] = [raw[2:] if raw.lower().startswith("r/") else raw]
        else:
            merged.pop("subreddit", None)
        return merged

    def update_reddit_setting(self, channel_id: int, key: str, value: Any) -> None:
        current = self.channel_reddit_settings.setdefault(channel_id, {})
        current[key] = value
        self.save()

    def save(self) -> None:
        payload = {
            "channel_modes": {str(k): v for k, v in self.channel_modes.items()},
            "channel_scrape_settings": {str(k): v for k, v in self.channel_scrape_settings.items()},
            "channel_job_settings": {str(k): v for k, v in self.channel_job_settings.items()},
            "channel_reddit_settings": {str(k): v for k, v in self.channel_reddit_settings.items()},
            "channel_job_seen": {str(k): list(v) for k, v in self.channel_job_seen.items()},
            "channel_cheatsheet_ids": dict(self.channel_cheatsheet_ids),
        }
        try:
            self.state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"Failed to save runtime state: {exc}")

    def load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return

        self.channel_modes = {int(k): str(v) for k, v in payload.get("channel_modes", {}).items()}
        self.channel_scrape_settings = {int(k): dict(v) for k, v in payload.get("channel_scrape_settings", {}).items()}
        self.channel_job_settings = {int(k): dict(v) for k, v in payload.get("channel_job_settings", {}).items()}
        reddit_raw = payload.get("channel_reddit_settings", {})
        migrated_reddit: dict[int, dict[str, Any]] = {}
        for k, v in reddit_raw.items():
            d = dict(v)
            if "subreddit" in d:
                if "subreddits" not in d:
                    raw = str(d["subreddit"]).strip()
                    d["subreddits"] = [raw[2:] if raw.lower().startswith("r/") else raw]
                del d["subreddit"]
            migrated_reddit[int(k)] = d
        self.channel_reddit_settings = migrated_reddit
        self.channel_job_seen = {
            int(k): {str(item) for item in values}
            for k, values in payload.get("channel_job_seen", {}).items()
        }
        self.channel_cheatsheet_ids = {
            str(k): int(v)
            for k, v in payload.get("channel_cheatsheet_ids", {}).items()
        }
