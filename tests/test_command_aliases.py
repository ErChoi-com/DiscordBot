from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import commands.handlers as handlers_module
from commands.handlers import (
    CMD_BEST_JOBS,
    CMD_BEST_JOBS_ALIAS,
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
    _extract_llm_directive,
    MAX_LLM_DIRECTIVE_CHARS,
    parse_best_jobs_payload,
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


def test_extract_llm_directive_captures_and_strips_parenthesised_span() -> None:
    content, directive = _extract_llm_directive(
        ".resumebuild (lead with the embedded work)"
    )
    assert directive == "lead with the embedded work"
    assert content.strip() == ".resumebuild"


def test_extract_llm_directive_absent_leaves_content_untouched() -> None:
    content, directive = _extract_llm_directive(".resumebuild ricky disappoints")
    assert directive == ""
    assert content == ".resumebuild ricky disappoints"


def test_extract_llm_directive_preserves_target_on_either_side() -> None:
    """The remaining text is read as a username, so it must survive intact."""
    for raw in (
        ".resumebuild ricky (emphasise the tutoring)",
        ".resumebuild (emphasise the tutoring) ricky",
    ):
        content, directive = _extract_llm_directive(raw)
        assert directive == "emphasise the tutoring"
        assert _extract_command_payload(content, ".resumebuild") == "ricky"


def test_extract_llm_directive_composes_with_aggressiveness_flags() -> None:
    content, aggressive, strong = _extract_aggressiveness_flags(
        ".resumebuild ricky --strongaggressive (keep it to one page)"
    )
    content, directive = _extract_llm_directive(content)
    assert (aggressive, strong) == (True, True)
    assert directive == "keep it to one page"
    assert _extract_command_payload(content, ".resumebuild") == "ricky"


def test_extract_llm_directive_collapses_whitespace_across_newlines() -> None:
    content, directive = _extract_llm_directive(".resumebuild (first  steer\nhere)")
    assert directive == "first steer here"
    assert content.strip() == ".resumebuild"


def test_extract_llm_directive_consumes_nested_parens_whole() -> None:
    """A parenthesised URL must not leave residue that becomes a username.

    `\\(([^()]*)\\)` would capture just "bar" here and leave the rest of the
    span behind, which then fails target resolution.
    """
    raw = ".resumebuild (see https://en.wikipedia.org/wiki/Foo_(bar) for tone)"
    content, directive = _extract_llm_directive(raw)
    assert directive == "see https://en.wikipedia.org/wiki/Foo_(bar) for tone"
    assert _extract_command_payload(content, ".resumebuild") == ""


def test_extract_llm_directive_takes_every_group_and_keeps_the_target() -> None:
    content, directive = _extract_llm_directive(".resumebuild (emphasis) ricky (extra)")
    assert directive == "emphasis; extra"
    assert _extract_command_payload(content, ".resumebuild") == "ricky"


def test_extract_llm_directive_treats_unclosed_paren_as_directive() -> None:
    """Silently ignoring an unclosed "(" left the whole span as a username."""
    content, directive = _extract_llm_directive(".resumebuild (lead with embedded")
    assert directive == "lead with embedded"
    assert _extract_command_payload(content, ".resumebuild") == ""


def test_directive_extraction_runs_before_flag_stripping() -> None:
    """A flag written INSIDE the steer must not switch the mode on.

    The flag parser is a whole-message regex, so extraction order is the whole
    guard here.
    """
    raw = ".resumebuild (avoid sounding --aggressive in the summary)"
    content, directive = _extract_llm_directive(raw)
    content, aggressive, strong = _extract_aggressiveness_flags(content)
    assert directive == "avoid sounding --aggressive in the summary"
    assert (aggressive, strong) == (False, False)


def test_extract_llm_directive_is_length_bounded() -> None:
    content, directive = _extract_llm_directive(f".resumebuild ({'x' * 5000})")
    assert len(directive) == MAX_LLM_DIRECTIVE_CHARS


def test_extract_llm_directive_handles_empty_group() -> None:
    content, directive = _extract_llm_directive(".resumebuild ()")
    assert directive == ""
    assert content.strip() == ".resumebuild"


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


# ---------------------------------------------------------------------------
# .bestjobs argument parsing
# ---------------------------------------------------------------------------

def test_best_jobs_aliases_are_registered() -> None:
    aliases = _expand_command_aliases((CMD_BEST_JOBS, CMD_BEST_JOBS_ALIAS))

    assert ".bestjobs" in aliases
    assert ".best" in aliases
    assert "/bestjobs" in aliases
    assert "/best" in aliases


def test_best_jobs_payload_defaults_to_a_day_of_the_owner_profile() -> None:
    assert parse_best_jobs_payload("") == ("day", 10, None, True, None)


def test_best_jobs_payload_reads_window_count_and_profile_in_any_order() -> None:
    """Tokens are identified by shape, so the user need not remember an order."""
    assert parse_best_jobs_payload("week 15") == ("week", 15, None, True, None)
    assert parse_best_jobs_payload("15 week") == ("week", 15, None, True, None)
    assert parse_best_jobs_payload("week 15 ricky") == ("week", 15, "ricky", True, None)
    assert parse_best_jobs_payload("ricky 15 week") == ("week", 15, "ricky", True, None)


def test_best_jobs_payload_accepts_window_synonyms() -> None:
    for token in ("day", "today", "d", "24h"):
        assert parse_best_jobs_payload(token)[0] == "day"
    for token in ("week", "weekly", "w", "7d"):
        assert parse_best_jobs_payload(token)[0] == "week"


def test_best_jobs_payload_rejects_an_out_of_range_count() -> None:
    *_, error = parse_best_jobs_payload("999")
    assert error is not None and "between 1 and" in error

    *_, zero_error = parse_best_jobs_payload("0")
    assert zero_error is not None


def test_best_jobs_payload_rejects_multiple_unknown_tokens() -> None:
    """Two leftover words is a typo, not a profile name -- guessing which one
    was meant would silently rank the wrong profile."""
    _, _, profile, _, error = parse_best_jobs_payload("week ricky shane")
    assert profile is None
    assert error is not None and "Unrecognized options" in error


def test_best_jobs_fast_flag_disables_description_fetching() -> None:
    """Enrichment is the slow stage, so it needs an explicit escape hatch."""
    assert parse_best_jobs_payload("week --fast") == ("week", 10, None, False, None)
    # The flag must not be mistaken for a profile name.
    assert parse_best_jobs_payload("--fast ricky") == ("day", 10, "ricky", False, None)
    assert parse_best_jobs_payload("week")[3] is True
