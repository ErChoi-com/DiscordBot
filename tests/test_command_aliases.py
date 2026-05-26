from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import commands.handlers as handlers_module
from commands.handlers import (
    CMD_COMMANDS,
    CMD_JOB_SETTINGS,
    CMD_JOB_SETTINGS_ALIAS,
    CommandRouter,
    CMD_RESUME,
    CMD_RESUME_CHECK,
    CMD_SCRAPE,
    _command_matches,
    _expand_command_aliases,
    _extract_command_payload,
)


def test_expand_command_aliases_includes_slash_variants() -> None:
    aliases = _expand_command_aliases((CMD_JOB_SETTINGS, CMD_JOB_SETTINGS_ALIAS))

    assert CMD_JOB_SETTINGS in aliases
    assert CMD_JOB_SETTINGS_ALIAS in aliases
    assert "/jobsettings" in aliases
    assert "/jobsinit" in aliases


def test_command_matches_accepts_documented_slash_commands() -> None:
    assert _command_matches("/commands", "/commands", normalize=True)
    assert _command_matches("/jobsettings", "/jobsettings", normalize=True)
    assert _command_matches("/resume", "/resume", normalize=True)
    assert _command_matches("/resumebuild", "/resumebuild", normalize=True)
    assert _command_matches("/resumecheck", "/resumecheck", normalize=True)
    assert _command_matches("/scrape https://example.com", "/scrape", normalize=True)


def test_extract_command_payload_handles_slash_and_legacy_commands() -> None:
    assert _extract_command_payload("/scrape https://example.com", CMD_SCRAPE) == "https://example.com"
    assert _extract_command_payload(".scrape https://example.com", CMD_SCRAPE) == "https://example.com"


def test_extract_command_payload_supports_jobsettings_aliases() -> None:
    assert _extract_command_payload("/jobsettings", CMD_JOB_SETTINGS) == ""
    assert _extract_command_payload("/jobsinit", CMD_JOB_SETTINGS_ALIAS) == ""


def test_extract_command_payload_supports_commands_aliases() -> None:
    assert _extract_command_payload("/commands", CMD_COMMANDS) == ""


def test_extract_command_payload_supports_resumecheck_alias() -> None:
    assert _extract_command_payload("/resumecheck", CMD_RESUME_CHECK) == ""
    assert _extract_command_payload(".resumecheck", CMD_RESUME_CHECK) == ""


def test_extract_command_payload_supports_resumebuild_aliases() -> None:
    assert _extract_command_payload("/resumebuild", CMD_RESUME) == ""
    assert _extract_command_payload("/ernestresume", CMD_RESUME) == ""


def test_every_command_constant_has_slash_alias() -> None:
    command_constants = {
        name: value
        for name, value in handlers_module.__dict__.items()
        if name.startswith("CMD_") and isinstance(value, str)
    }

    missing_aliases: list[str] = []
    for name, command in command_constants.items():
        aliases = handlers_module._COMMAND_ALIASES.get(command, ())
        if not any(alias.startswith("/") for alias in aliases):
            missing_aliases.append(name)

    assert not missing_aliases, f"Commands without slash aliases: {missing_aliases}"


def test_command_router_init_does_not_seed_profile_cache(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[object, object]] = []

    def _fake_ensure_profile_cache(*args, **kwargs):
        calls.append((args, kwargs))
        return tmp_path / "unused"

    monkeypatch.setattr(handlers_module, "ensure_profile_cache", _fake_ensure_profile_cache)

    class _Config:
        resume_profiles_dir = tmp_path / "resumes"
        resume_cache_dir = tmp_path / ".resume_cache"
        main_user_profile_key = "owner-profile"

    _Config.resume_profiles_dir.mkdir(parents=True, exist_ok=True)
    _Config.resume_cache_dir.mkdir(parents=True, exist_ok=True)

    class _Store:
        pass

    class _WatcherManager:
        pass

    CommandRouter(client=object(), config=_Config(), store=_Store(), watcher_manager=_WatcherManager())

    assert calls == []
