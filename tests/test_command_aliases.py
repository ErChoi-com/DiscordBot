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
    _ContentOverrideMessage,
    _expand_command_aliases,
    _extract_aggressiveness_flags,
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


def test_extract_aggressive_flag_detects_and_strips_trailing_flag() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild --aggressive")
    assert aggressive is True
    assert strong is False
    assert content == ".resumebuild"


def test_extract_aggressive_flag_preserves_target_before_flag() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild ricky disappoints --aggressive")
    assert aggressive is True
    assert strong is False
    assert content == ".resumebuild ricky disappoints"


def test_extract_aggressive_flag_preserves_target_after_flag() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild --aggressive ricky disappoints")
    assert aggressive is True
    assert strong is False
    assert content == ".resumebuild ricky disappoints"


def test_extract_aggressive_flag_case_insensitive() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild --AGGRESSIVE")
    assert aggressive is True
    assert strong is False
    assert content == ".resumebuild"


def test_extract_aggressive_flag_absent_leaves_content_untouched() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild ricky disappoints")
    assert aggressive is False
    assert strong is False
    assert content == ".resumebuild ricky disappoints"


def test_extract_aggressive_flag_does_not_match_as_substring_of_a_name() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild --aggressiveperson")
    assert aggressive is False
    assert strong is False
    assert content == ".resumebuild --aggressiveperson"


def test_aggressive_flag_strip_composes_with_target_payload_extraction() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild ricky disappoints --aggressive")
    assert aggressive is True
    assert strong is False
    assert _extract_command_payload(content, CMD_RESUME) == "ricky disappoints"


def test_strong_aggressive_flag_detected_and_stripped() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild --strongaggressive")
    assert aggressive is True
    assert strong is True
    assert content == ".resumebuild"


def test_strong_aggressive_does_not_collide_with_aggressive() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild --strongaggressive")
    assert strong is True
    content2, aggressive2, strong2 = _extract_aggressiveness_flags(".resumebuild --aggressive")
    assert strong2 is False
    assert aggressive2 is True


def test_strong_aggressive_preserves_target() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(".resumebuild ricky --strongaggressive")
    assert strong is True
    assert aggressive is True
    assert content == ".resumebuild ricky"


def test_content_override_message_reports_new_content_and_delegates_rest() -> None:
    class _FakeMessage:
        def __init__(self) -> None:
            self.content = ".resumebuild ricky --aggressive"
            self.channel = "channel-sentinel"
            self.author = "author-sentinel"

        def to_reference(self, fail_if_not_exists: bool = False):
            return ("ref", fail_if_not_exists)

    real = _FakeMessage()
    wrapped = _ContentOverrideMessage(real, ".resumebuild ricky")

    assert wrapped.content == ".resumebuild ricky"
    assert real.content == ".resumebuild ricky --aggressive"  # original untouched
    assert wrapped.channel == "channel-sentinel"
    assert wrapped.author == "author-sentinel"
    assert wrapped.to_reference(fail_if_not_exists=True) == ("ref", True)


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

    from services.health import WatcherHealthTracker
    CommandRouter(client=object(), config=_Config(), store=_Store(), watcher_manager=_WatcherManager(), health=WatcherHealthTracker())

    assert calls == []


def test_extract_command_payload_supports_resumecoverbuild_aliases() -> None:
    from commands.handlers import CMD_RESUME_COVER

    assert _extract_command_payload("/resumecoverbuild", CMD_RESUME_COVER) == ""
    assert _extract_command_payload(".resumecoverbuild ricky", CMD_RESUME_COVER) == "ricky"
    assert _command_matches("/cover", "/cover", normalize=True)
    assert _command_matches("/coverbuild", "/coverbuild", normalize=True)


def test_resumecoverbuild_does_not_shadow_resumebuild() -> None:
    from commands.handlers import CMD_RESUME_COVER

    # Token-boundary matching: the longer command must never match the
    # shorter command's prefix and vice versa.
    assert not _command_matches(".resumecoverbuild", CMD_RESUME, normalize=True)
    assert not _command_matches(".resumebuild", CMD_RESUME_COVER, normalize=True)
    assert _extract_command_payload(".resumecoverbuild", CMD_RESUME) == ""
