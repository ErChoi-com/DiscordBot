from __future__ import annotations

import asyncio
import io
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable

import discord

from config import AppConfig
from services import job_match, job_service, scrape_service
from services.health import WatcherHealthTracker, build_channel_health_embed, build_all_health_embed
from services.priority_scheduler import INTERACTIVE, PriorityWorkScheduler
from services import scheduler_labels
from services.resumes.configkey import GeminiSettings, load_gemini_settings
from services.resumes.cache import ResumeExplicitCacheManager, ResumeExplicitCacheStatus
from services.resumes.cover import generate_cover_letter
from services.resumes.listing import (
    JobContext,
    condense_latex_if_overflowing,
    extract_job_context_from_message,
    generate_resume_rewrite,
    repair_latex_until_compiles,
)
from services.resumes.structured import load_structured_profile
from services.resumes.resume import (
    EXAMPLE_PROFILE_KEY,
    RESUMES_CACHE_ROOT,
    compile_latex_to_pdf,
    discord_profile_key,
    ensure_profile_cache,
    profile_cache_dir,
    resolve_discord_profile_key,
)
from state.store import RuntimeStore
from ui.views import (
    JobSettingsView,
    RedditSettingsView,
    ScrapeSettingsView,
    format_job_settings_summary,
    format_reddit_settings_summary,
    format_scrape_settings_summary,
)
from watchers.manager import WatcherManager

MessageHandler = Callable[[discord.Message], Awaitable[bool]]


class _MessageReferenceAdapter:
    def __init__(self, message_id: int, resolved: discord.Message | None = None) -> None:
        self.message_id = message_id
        self.resolved = resolved


class _InteractionMessageAdapter:
    def __init__(
        self,
        interaction: discord.Interaction,
        content: str,
        reference_message_id: int | None = None,
        resolved_reference: discord.Message | None = None,
    ) -> None:
        self.content = content
        self.channel = interaction.channel
        self.author = interaction.user
        self.guild = interaction.guild
        self.id = 0
        self.reference = (
            _MessageReferenceAdapter(reference_message_id, resolved=resolved_reference)
            if reference_message_id is not None
            else None
        )

    def to_reference(self, fail_if_not_exists: bool = False) -> None:
        _ = fail_if_not_exists
        return None


class _ContentOverrideMessage:
    """Wraps a real discord.Message but reports a different `.content`.

    Used to strip a parsed "--aggressive" flag before the message reaches
    target-user resolution, so the flag token is never mistaken for part of a
    mention/username/profile name. Every other attribute (channel, author,
    guild, reference, to_reference(), ...) delegates straight through to the
    wrapped message.
    """

    def __init__(self, message: discord.Message, content: str) -> None:
        self._message = message
        self.content = content

    def __getattr__(self, name: str) -> Any:
        return getattr(self._message, name)


@dataclass(slots=True)
class _ResumeRequest:
    """Everything the resume-family commands need after shared validation."""

    settings: GeminiSettings
    job: JobContext
    profile_key: str
    profile_dir: Path
    template_path: Path
    cache_scope: str
    reply_send_kwargs: dict[str, Any]


CHEATSHEET_KIND_GENERAL = "general"
CHEATSHEET_KIND_JOB = "job"
CHEATSHEET_KIND_REDDIT = "reddit"

CHEATSHEET_METADATA = {
    CHEATSHEET_KIND_GENERAL: ("General Commands", "Quick reference for general channel usage."),
    CHEATSHEET_KIND_JOB: ("Job Watcher Commands", "Quick reference for job watcher channels."),
    CHEATSHEET_KIND_REDDIT: ("Reddit Watcher Commands", "Quick reference for reddit watcher channels."),
}

CMD_COMMANDS = ".cmd"
CMD_STATUS = ".st"
CMD_STATUS_ALIAS = ".watch"
CMD_RESUME = ".resumebuild"
CMD_RESUME_COVER = ".resumecoverbuild"
CMD_RESUME_CHECK = ".resumecheck"
CMD_CONTINUE = ".more"
CMD_HELLO = ".hi"
CMD_JOB_SETTINGS = ".job"
CMD_JOB_SETTINGS_ALIAS = ".jobs"
CMD_JOB_PIPELINE_TEST = ".jobtest"
CMD_REDDIT_SETTINGS = ".reddit"
CMD_REDDIT_SETTINGS_ALIAS = ".rset"
CMD_CLEAR_REDDIT_SEEN = ".rclear"
CMD_RESET_REDDIT_SEEN = ".rreset"
CMD_SCRAPE_SETTINGS = ".scrapecfg"
CMD_SCRAPE_SETTINGS_ALIAS = ".cfg"
CMD_SCRAPE = ".scrape"
CMD_HEALTH = ".health"
CMD_BEST_JOBS = ".bestjobs"
CMD_BEST_JOBS_ALIAS = ".best"
CMD_QUOTA = ".quota"
CMD_RESET = ".reset"
PRIMARY_RESUME_SLASH_COMMAND = "/resumebuild"


_COMMAND_ALIASES: dict[str, tuple[str, ...]] = {
    CMD_COMMANDS: ("/cmd", "/commands"),
    CMD_STATUS: ("/st", "/status"),
    CMD_STATUS_ALIAS: ("/watch", "/watcherstatus"),
    CMD_RESUME: (PRIMARY_RESUME_SLASH_COMMAND, "/resume", "/ernestresume", "/res"),
    CMD_RESUME_COVER: ("/resumecoverbuild", "/coverbuild", "/cover"),
    CMD_RESUME_CHECK: ("/resumecheck",),
    CMD_CONTINUE: ("/more", "/continue"),
    CMD_HELLO: ("/hi", "/hello", "/hello2"),
    CMD_JOB_SETTINGS: ("/job", "/jobsettings"),
    CMD_JOB_SETTINGS_ALIAS: ("/jobs", "/jobsinit"),
    CMD_JOB_PIPELINE_TEST: ("/jobtest", "/jobpipelinetest"),
    CMD_REDDIT_SETTINGS: ("/reddit", "/redditsettings"),
    CMD_REDDIT_SETTINGS_ALIAS: ("/rset",),
    CMD_CLEAR_REDDIT_SEEN: ("/rclear", "/clearredditseen"),
    CMD_RESET_REDDIT_SEEN: ("/rreset", "/resetredditseen"),
    CMD_SCRAPE_SETTINGS: ("/scrapecfg", "/settings"),
    CMD_SCRAPE_SETTINGS_ALIAS: ("/cfg",),
    CMD_SCRAPE: ("/scrape",),
    CMD_HEALTH: ("/health", "/whealth"),
    CMD_BEST_JOBS: ("/bestjobs", "/bestmatches"),
    CMD_BEST_JOBS_ALIAS: ("/best",),
    CMD_QUOTA: ("/quota", "/quotas"),
    CMD_RESET: ("/reset",),
}


def quiet_reply_kwargs(message: Any) -> dict[str, Any]:
    """channel.send kwargs for a quiet threaded reply: references the invoking
    message without pinging its author or resolving any mentions in the body.
    One home for the literal previously copy-pasted at a dozen send sites."""
    return {
        "reference": message.to_reference(fail_if_not_exists=False),
        "mention_author": False,
        "allowed_mentions": discord.AllowedMentions.none(),
    }


def build_commands_cheatsheet_embed(sheet_kind: str = CHEATSHEET_KIND_GENERAL) -> discord.Embed:
    title, description = CHEATSHEET_METADATA.get(sheet_kind, CHEATSHEET_METADATA[CHEATSHEET_KIND_GENERAL])
    embed = discord.Embed(
        title=title,
        description=description,
        color=discord.Color.blurple(),
    )

    if sheet_kind == CHEATSHEET_KIND_JOB:
        embed.add_field(
            name="Job Watcher",
            value=(
                f"`{CMD_JOB_SETTINGS}` / `{CMD_JOB_SETTINGS_ALIAS}` - Open job settings\n"
                f"`{CMD_JOB_PIPELINE_TEST}` - Full pipeline test with metrics\n"
                f"`{CMD_BEST_JOBS}` / `{CMD_BEST_JOBS_ALIAS}` - Top archive matches for your profile - `.best week 15`\n"
                f"`{CMD_STATUS}` - Watcher status\n"
                f"`{CMD_HEALTH}` - Scrape health dashboard · `.health all` for all channels\n"
                f"`{CMD_QUOTA}` - (owner) Member shares of the AI commands - `.quota user @them 0.5`"
            ),
            inline=False,
        )
        embed.add_field(
            name="Utility",
            value=(
                f"`{CMD_COMMANDS}` - Show general command sheet\n"
                f"`{CMD_RESUME}` - Reply to a job post for tailored resume (add `--aggressive` or `--strongaggressive`)\n"
                f"`{CMD_RESUME} (...)` - text in parentheses goes straight to the model, e.g. `(lead with the embedded work)`\n"
                f"`{CMD_RESUME_COVER}` - Reply to a job post for a cover letter covering what the resume left out\n"
                f"`{CMD_RESUME_CHECK}` - Compile your current template cache\n"
                f"`{CMD_CONTINUE}` - Continue paginated output"
            ),
            inline=False,
        )
        return embed

    if sheet_kind == CHEATSHEET_KIND_REDDIT:
        embed.add_field(
            name="Reddit Watcher",
            value=(
                f"`{CMD_REDDIT_SETTINGS}` / `{CMD_REDDIT_SETTINGS_ALIAS}` - Open Reddit settings\n"
                f"`{CMD_CLEAR_REDDIT_SEEN}` / `{CMD_RESET_REDDIT_SEEN}` - Clear seen IDs\n"
                f"`{CMD_STATUS}` - Watcher status"
            ),
            inline=False,
        )
        embed.add_field(
            name="Utility",
            value=(
                f"`{CMD_COMMANDS}` - Show general command sheet\n"
                f"`{CMD_CONTINUE}` - Continue paginated output"
            ),
            inline=False,
        )
        return embed

    embed.add_field(
        name="General",
        value=(
            f"`{CMD_COMMANDS}` - Show this list\n"
            f"`{CMD_STATUS}` / `{CMD_STATUS_ALIAS}` - Watcher status\n"
            f"`{CMD_RESUME}` - Reply to a job post and generate a tailored resume draft (add `--aggressive` or `--strongaggressive`)\n"
            f"`{CMD_RESUME} (...)` - text in parentheses goes straight to the model, e.g. `(lead with the embedded work)`\n"
            f"`{CMD_RESUME_COVER}` - Reply to a job post and generate a complementary cover letter\n"
            f"`{CMD_RESUME_CHECK}` - Compile your current template cache\n"
            f"`{CMD_CONTINUE}` - Continue paginated output\n"
            f"`{CMD_HELLO}` - Quick hello test\n"
            f"`{CMD_RESET}` - (owner) Restart the bot process"
        ),
        inline=False,
    )
    embed.add_field(
        name="Job Watcher",
        value=(
            f"`{CMD_JOB_SETTINGS}` / `{CMD_JOB_SETTINGS_ALIAS}` - Open job settings\n"
            f"`{CMD_JOB_PIPELINE_TEST}` - Full pipeline test with metrics\n"
            f"`{CMD_BEST_JOBS}` / `{CMD_BEST_JOBS_ALIAS}` - Top archive matches for your profile - `.best week 15`"
        ),
        inline=False,
    )
    embed.add_field(
        name="Reddit Watcher",
        value=(
            f"`{CMD_REDDIT_SETTINGS}` / `{CMD_REDDIT_SETTINGS_ALIAS}` - Open Reddit settings\n"
            f"`{CMD_CLEAR_REDDIT_SEEN}` / `{CMD_RESET_REDDIT_SEEN}` - Clear seen IDs"
        ),
        inline=False,
    )
    embed.add_field(
        name="Scraping",
        value=(
            f"`{CMD_SCRAPE_SETTINGS}` / `{CMD_SCRAPE_SETTINGS_ALIAS}` - Open scrape settings\n"
            f"`{CMD_SCRAPE} <url> [| selector]` - Run one-time scrape"
        ),
        inline=False,
    )
    return embed

def _alternate_command_prefix(command: str) -> str:
    aliases = _COMMAND_ALIASES.get(command)
    if aliases:
        return aliases[0]
    return command


def _expand_command_aliases(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for prefix in prefixes:
        if prefix not in expanded:
            expanded.append(prefix)
        for alias in _COMMAND_ALIASES.get(prefix, ()):
            if alias not in expanded:
                expanded.append(alias)
        alternate = _alternate_command_prefix(prefix)
        if alternate not in expanded:
            expanded.append(alternate)
    return tuple(expanded)


_MENTION_ID_PATTERN = re.compile(r"<@[!&]?(\d+)>")


def parse_quota_payload(payload: str) -> tuple[str, str, str]:
    """Split a `.quota` payload into (action, target, value).

    Accepted forms, all owner-only:
        .quota                          show the current policy
        .quota default 0.5              everyone without a role or override
        .quota role @Role 0.7           per-role share
        .quota user @member 0.3         manual per-user share
        .quota clear user @member       drop an override back to inherited
    """
    parts = str(payload or "").split()
    if not parts:
        return "show", "", ""
    action = parts[0].lower()
    if action == "clear":
        scope = parts[1].lower() if len(parts) > 1 else ""
        return "clear", scope, " ".join(parts[2:])
    if action in ("default", "role", "user"):
        if action == "default":
            return "default", "", " ".join(parts[1:])
        return action, " ".join(parts[1:-1]), parts[-1] if len(parts) > 1 else ""
    return "unknown", "", ""


def _extract_command_payload(content: str, command: str) -> str:
    stripped = content.strip()
    for alias in _expand_command_aliases((command,)):
        if _command_matches(stripped, alias, normalize=True):
            return stripped[len(alias) :].strip()
    return ""


# --strongaggressive checked first (contains the substring --aggressive).
_STRONG_AGGRESSIVE_FLAG_PATTERN = re.compile(r"\s*--strongaggressive\b", re.IGNORECASE)
_AGGRESSIVE_FLAG_PATTERN = re.compile(r"\s*--aggressive\b", re.IGNORECASE)


def _extract_aggressiveness_flags(content: str) -> tuple[str, bool, bool]:
    """Strip ``--strongaggressive`` or ``--aggressive`` from command text.

    Returns ``(cleaned_content, aggressive, strong_aggressive)``.
    strong_aggressive implies aggressive.
    """
    if _STRONG_AGGRESSIVE_FLAG_PATTERN.search(content):
        return _STRONG_AGGRESSIVE_FLAG_PATTERN.sub("", content), True, True
    if _AGGRESSIVE_FLAG_PATTERN.search(content):
        return _AGGRESSIVE_FLAG_PATTERN.sub("", content), True, False
    return content, False, False


# Discord's hard message cap is 2000; a directive far past this is a paste
# accident, and an unbounded string would displace the real prompt content.
MAX_LLM_DIRECTIVE_CHARS = 600


def _extract_llm_directive(content: str) -> tuple[str, str]:
    """Strip ``(...)`` free-form model instructions from command text.

    Returns ``(cleaned_content, directive)``; the directive is ``""`` when the
    command carries no parenthesised span. Stripping matters as much as
    capturing: the resume commands treat whatever remains as a target
    username, so an unstripped ``(...)`` becomes "Could not resolve ... to a
    server member."

    Depth-tracked rather than regex-matched, because the obvious
    ``\\(([^()]*)\\)`` gets the common cases wrong: a wiki URL ending in
    ``_(bar))`` yields the directive "bar" and leaves the rest of the span
    behind as a bogus username. Nested parens are consumed whole, every
    top-level group is taken (so ``(a) ricky (b)`` still resolves the target),
    and an unclosed ``(`` runs to end-of-line instead of being ignored.
    """
    parts: list[str] = []
    kept: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(content):
        if char == "(":
            if depth == 0:
                kept.append(content[start:index])
                start = index + 1
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                parts.append(content[start:index])
                start = index + 1
    if depth > 0:  # unclosed "(" — treat the remainder as the directive
        parts.append(content[start:])
    else:
        kept.append(content[start:])

    if not parts:
        return content, ""
    directive = " ".join("; ".join(parts).split())[:MAX_LLM_DIRECTIVE_CHARS]
    return "".join(kept), directive


_BEST_JOBS_WINDOW_ALIASES: dict[str, str] = {
    "day": "day", "today": "day", "d": "day", "24h": "day", "1d": "day",
    "week": "week", "weekly": "week", "w": "week", "7d": "week", "7": "week",
}


def parse_best_jobs_payload(payload: str) -> tuple[str, int, str | None, bool, str | None]:
    """Parse ``[day|week] [N] [profile] [--fast]`` for the best-jobs command.

    Tokens are identified by shape rather than position, so `.best 20 week`
    works as well as `.best week 20`. ``--fast`` skips the description-fetch
    stage, trading ranking quality for a much faster answer. Returns
    ``(window, limit, profile_token, enrich, error)``; ``error`` is non-None
    only for input that cannot be honoured at all, so a bare `.best` is valid
    and takes every default.
    """
    window = "day"
    limit = job_match.DEFAULT_LIMIT
    enrich = True
    profile_parts: list[str] = []

    for token in str(payload).split():
        lowered = token.lower()
        if lowered in ("--fast", "-fast", "fast"):
            enrich = False
            continue
        if lowered in _BEST_JOBS_WINDOW_ALIASES:
            window = _BEST_JOBS_WINDOW_ALIASES[lowered]
            continue
        if lowered.isdigit():
            value = int(lowered)
            if not 1 <= value <= job_match.MAX_LIMIT:
                return window, limit, None, enrich, (
                    f"Result count must be between 1 and {job_match.MAX_LIMIT} (got {value})."
                )
            limit = value
            continue
        profile_parts.append(token)

    if len(profile_parts) > 1:
        return window, limit, None, enrich, (
            f"Unrecognized options: {' '.join(profile_parts)}. "
            f"Usage: `{CMD_BEST_JOBS} [day|week] [count] [profile] [--fast]`."
        )
    return window, limit, (profile_parts[0] if profile_parts else None), enrich, None


def _command_matches(content: str, prefix: str, normalize: bool) -> bool:
    if normalize:
        content_cmp = content.lower()
        prefix_cmp = prefix.lower()
    else:
        content_cmp = content
        prefix_cmp = prefix

    if not content_cmp.startswith(prefix_cmp):
        return False

    # Require token boundary so `.st123` does not match `.st`.
    if len(content_cmp) == len(prefix_cmp):
        return True
    return content_cmp[len(prefix_cmp)].isspace()


def log_panel_send_event(context: str, **fields: Any) -> None:
    try:
        log_path = Path(__file__).resolve().parents[1] / ".panel_send_events.log"
        parts = [f"ts={datetime.now(timezone.utc).isoformat()}", f"context={context}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(" | ".join(parts) + "\n")
    except Exception:
        pass


def command_handler(*prefixes: str, normalize: bool = False) -> Callable[[MessageHandler], MessageHandler]:
    expanded_prefixes = _expand_command_aliases(prefixes)
    match_prefixes = tuple(prefix.lower() for prefix in expanded_prefixes) if normalize else expanded_prefixes

    def decorator(func: MessageHandler) -> MessageHandler:
        @wraps(func)
        async def wrapper(self: CommandRouter, message: discord.Message) -> bool:
            raw_content = message.content
            match_content = raw_content.strip()
            if not any(_command_matches(match_content, prefix, normalize=normalize) for prefix in match_prefixes):
                return False
            return await func(self, message)

        return wrapper

    return decorator


class CommandRouter:
    # Set once .reset has launched a forcerun; a second one would kill the
    # replacement mid-warmup. No lock needed -- handle_reset checks and sets it
    # with no await in between. On the class so a router built without
    # __init__ still reads a sane default.
    _restart_spawned = False

    def __init__(
        self,
        client: discord.Client,
        config: AppConfig,
        store: RuntimeStore,
        watcher_manager: WatcherManager,
        health: WatcherHealthTracker,
        scheduler: PriorityWorkScheduler | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.store = store
        self.watcher_manager = watcher_manager
        self.health = health
        # Default to the watcher manager's scheduler so interactive commands
        # and background watcher work share one queue unless a caller
        # explicitly wants them isolated (e.g. tests).
        self.scheduler = scheduler or getattr(watcher_manager, "scheduler", None) or PriorityWorkScheduler()
        self.resume_cache_manager = ResumeExplicitCacheManager(
            profiles_dir=config.resume_profiles_dir,
            cache_dir=config.resume_cache_dir,
        )
        self.channel_continuations: dict[int, list[str]] = {}
        self.channel_panel_messages: dict[tuple[int, str], discord.Message] = {}

        self.handlers: tuple[MessageHandler, ...] = (
            self.handle_commands,
            self.handle_reset,
            self.handle_resume,
            self.handle_resume_cover,
            self.handle_resume_check,
            self.handle_status,
            self.handle_health,
            self.handle_job_pipeline_test,
            self.handle_best_jobs,
            self.handle_clear_reddit_seen,
            self.handle_reddit_settings,
            self.handle_hello,
            self.handle_scrape_settings,
            self.handle_job_settings,
            self.handle_continue,
            self.handle_scrape,
            self.handle_quota,
        )

    async def _run_interactive(
        self,
        fn: Callable[..., Any],
        *args: Any,
        cost: float | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run blocking work on the shared priority scheduler at interactive
        tier -- these are Discord commands the user is actively waiting on,
        so they preempt queued background watcher/scrape work of the same or
        lower cost. Pass `label` so cost is derived from that label's real
        measured median duration (rolling two-day window) instead of a guess.

        Held under work_guard() so graceful shutdown's drain_active_work()
        waits for in-flight interactive commands (e.g. .resumebuild) instead
        of abandoning them mid-execution."""
        async with self.watcher_manager.work_guard():
            return await self.scheduler.run(fn, *args, tier=INTERACTIVE, cost=cost, label=label, **kwargs)

    @command_handler(CMD_COMMANDS, normalize=True)
    async def handle_commands(self, message: discord.Message) -> bool:
        await message.channel.send(embed=build_commands_cheatsheet_embed())
        return True

    async def ensure_commands_cheatsheet_pinned(self, channel_id: int, sheet_kind: str = CHEATSHEET_KIND_GENERAL) -> None:
        channel = self.client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.client.fetch_channel(channel_id)
            except Exception as exc:
                print(f"Could not resolve channel {channel_id} for cheat sheet check: {exc}")
                return
        if channel is None:
            return
        if not hasattr(channel, "send"):
            return

        expected_title, expected_description = CHEATSHEET_METADATA.get(sheet_kind, CHEATSHEET_METADATA[CHEATSHEET_KIND_GENERAL])
        new_embed = build_commands_cheatsheet_embed(sheet_kind=sheet_kind)

        permissions = None
        if hasattr(channel, "permissions_for"):
            me = getattr(getattr(channel, "guild", None), "me", None) or getattr(self.client, "user", None)
            if me is not None:
                try:
                    permissions = channel.permissions_for(me)
                except Exception:
                    permissions = None

        if permissions is not None and not bool(getattr(permissions, "send_messages", True)):
            print(f"Missing send_messages in channel {channel_id}; cannot enforce cheat sheet pin")
            return

        can_pin = True if permissions is None else bool(getattr(permissions, "manage_messages", True))

        def _is_bot_cheatsheet(message: Any) -> bool:
            embed = message.embeds[0] if getattr(message, "embeds", None) else None
            if embed is None:
                return False
            current_user = getattr(self.client, "user", None)
            if current_user is not None and getattr(getattr(message, "author", None), "id", None) != getattr(current_user, "id", None):
                return False
            return embed.title == expected_title

        stored_id = self.store.get_cheatsheet_message_id(channel_id, sheet_kind)
        if stored_id is not None:
            try:
                existing = await channel.fetch_message(stored_id)
                if _is_bot_cheatsheet(existing):
                    try:
                        await existing.edit(embed=new_embed)
                    except Exception as exc:
                        print(f"Could not edit stored cheat sheet in channel {channel_id}: {exc}")
                    return
            except discord.NotFound:
                self.store.clear_cheatsheet_message_id(channel_id, sheet_kind)
            except Exception as exc:
                print(f"Could not fetch stored cheat sheet {stored_id} in channel {channel_id}: {exc}")

        pinned_messages: list[Any] = []
        if hasattr(channel, "pins"):
            try:
                pinned_messages = await channel.pins()
            except Exception as exc:
                print(f"Could not read pinned messages in channel {channel_id}: {exc}")

        for message in pinned_messages:
            if _is_bot_cheatsheet(message):
                try:
                    await message.edit(embed=new_embed)
                except Exception as exc:
                    print(f"Could not edit pinned cheat sheet in channel {channel_id}: {exc}")
                self.store.set_cheatsheet_message_id(channel_id, sheet_kind, message.id)
                return

        if hasattr(channel, "history"):
            try:
                async for recent in channel.history(limit=50):
                    if _is_bot_cheatsheet(recent):
                        try:
                            await recent.edit(embed=new_embed)
                        except Exception as exc:
                            print(f"Could not edit unpinned cheat sheet in channel {channel_id}: {exc}")
                        self.store.set_cheatsheet_message_id(channel_id, sheet_kind, recent.id)
                        return
            except Exception as exc:
                print(f"Could not read history in channel {channel_id}: {exc}")

        try:
            sent_message = await channel.send(embed=new_embed)
            self.store.set_cheatsheet_message_id(channel_id, sheet_kind, sent_message.id)
            if can_pin:
                try:
                    await sent_message.pin(reason="Ensure command cheat sheet is pinned for watcher channel")
                except Exception as pin_exc:
                    print(f"Could not pin cheat sheet in channel {channel_id}: {pin_exc}")
            else:
                print(f"Missing manage_messages in channel {channel_id}; cheat sheet sent without pin")
        except Exception as exc:
            print(f"Could not send cheat sheet in channel {channel_id}: {exc}")

    @command_handler(CMD_STATUS, CMD_STATUS_ALIAS, normalize=True)
    async def handle_status(self, message: discord.Message) -> bool:

        channel_id = message.channel.id

        job_enabled = bool(self.store.get_job_settings(channel_id).get("enabled"))
        reddit_enabled = bool(self.store.get_reddit_settings(channel_id).get("enabled"))

        job_task = self.watcher_manager.channel_job_tasks.get(channel_id)
        reddit_task = self.watcher_manager.channel_reddit_tasks.get(channel_id)
        job_running = bool(job_task and not job_task.done())
        reddit_running = bool(reddit_task and not reddit_task.done())
        active_process = self.watcher_manager.channel_active_process.get(channel_id, "none")

        lines = [
            f"Status for <#{channel_id}>",
            f"job watcher: enabled={job_enabled}, running={job_running}",
            f"reddit watcher: enabled={reddit_enabled}, running={reddit_running}",
            f"active process lock: `{active_process}`",
        ]
        await message.channel.send("\n".join(lines))
        return True

    @command_handler(CMD_HEALTH, normalize=True)
    async def handle_health(self, message: discord.Message) -> bool:
        channel_id = message.channel.id
        payload = _extract_command_payload(message.content, CMD_HEALTH).strip().lower()

        active_job = sum(
            1 for t in self.watcher_manager.channel_job_tasks.values() if not t.done()
        )
        active_reddit = sum(
            1 for t in self.watcher_manager.channel_reddit_tasks.values() if not t.done()
        )

        if payload == "all":
            channel_names: dict[int, str] = {}
            for cid in self.health.all_job_health():
                ch = self.client.get_channel(cid)
                channel_names[cid] = getattr(ch, "name", str(cid))
            embed = build_all_health_embed(
                self.health,
                self.watcher_manager.channel_job_tasks,
                channel_names,
                active_job,
                active_reddit,
                scheduler_stats=self.scheduler.stats(),
            )
            await message.channel.send(embed=embed)
            return True

        job_task = self.watcher_manager.channel_job_tasks.get(channel_id)
        reddit_task = self.watcher_manager.channel_reddit_tasks.get(channel_id)
        channel_name = getattr(message.channel, "name", str(channel_id))

        # None while the browser is down: that is not a session problem, and
        # session_valid() would read False and report it as one. A plain flag
        # read, so it costs nothing to ask on every .health.
        from services import browser_service

        reddit_session = (
            browser_service.session_valid() if browser_service.is_ready() else None
        )

        embeds = build_channel_health_embed(
            self.health,
            channel_id,
            channel_name,
            job_task_alive=bool(job_task and not job_task.done()),
            reddit_task_alive=bool(reddit_task and not reddit_task.done()),
            active_job_watchers=active_job,
            active_reddit_watchers=active_reddit,
            scheduler_stats=self.scheduler.stats(),
            reddit_session_valid=reddit_session,
        )
        await message.channel.send(embeds=embeds)
        return True

    async def resolve_referenced_message(self, message: discord.Message) -> discord.Message | None:
        reference = getattr(message, "reference", None)
        if reference is None or getattr(reference, "message_id", None) is None:
            return None

        resolved = getattr(reference, "resolved", None)
        if isinstance(resolved, discord.Message):
            return resolved

        try:
            return await message.channel.fetch_message(reference.message_id)
        except (AttributeError, discord.NotFound, discord.HTTPException):
            return None

    async def _read_text_attachment(self, message: discord.Message) -> str:
        """Fall back to text-file attachment(s) when a message has no body
        content: e.g. a job post pasted as one or more .txt uploads instead
        of typed. Multiple attachments are concatenated in upload order."""
        attachments = getattr(message, "attachments", None) or []
        chunks: list[str] = []
        for attachment in attachments:
            filename = str(getattr(attachment, "filename", "") or "")
            if not filename.lower().endswith((".txt", ".md")):
                continue
            try:
                raw = await attachment.read()
            except (AttributeError, discord.NotFound, discord.HTTPException):
                continue
            try:
                text = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if text:
                chunks.append(text)
        return "\n".join(chunks)

    def build_resume_missing_key_message(self) -> str:
        return (
            "No LLM API key found. Add at least one of these to `.env`:\n"
            "\u2022 Gemini (primary): `geminiAPI=...` or `GEMINI_API_KEY=...`\n"
            "\u2022 OpenRouter (fallback): `openRouter=...`\n"
            "\u2022 Groq (fallback): `groqAPI=...`"
        )

    def build_resume_guild_only_message(self) -> str:
        return f"`{CMD_RESUME}` / `{PRIMARY_RESUME_SLASH_COMMAND}` is only available in a server channel."

    def build_resume_unauthorized_message(self) -> str:
        return (
            "You can only use resume commands for your own profile cache. "
            f"The server owner can target others with `{CMD_RESUME} @user` or `{CMD_RESUME_CHECK} @user`."
        )

    @staticmethod
    def _normalized_discord_id(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _is_guild_owner(self, message: discord.Message) -> bool:
        guild_owner_id = self._normalized_discord_id(getattr(getattr(message, "guild", None), "owner_id", None))
        author_id = self._normalized_discord_id(getattr(getattr(message, "author", None), "id", None))
        return guild_owner_id is not None and author_id is not None and author_id == guild_owner_id

    def _can_access_resume_target(self, message: discord.Message, target_user_id: int | None) -> bool:
        author_id = self._normalized_discord_id(getattr(getattr(message, "author", None), "id", None))
        target_id = self._normalized_discord_id(target_user_id)

        if author_id is None or target_id is None:
            return False
        if author_id == target_id:
            return True
        return self._is_guild_owner(message)

    @staticmethod
    def _parse_user_mention_id(token: str) -> int | None:
        match = re.fullmatch(r"<@!?(\d+)>", token.strip())
        if not match:
            return None
        return int(match.group(1))

    async def _resolve_resume_target_member(
        self,
        message: discord.Message,
        command: str,
    ) -> tuple[discord.abc.User | discord.Member | None, str | None]:
        guild = getattr(message, "guild", None)
        author = getattr(message, "author", None)
        if guild is None or author is None:
            return author, None

        payload = _extract_command_payload(getattr(message, "content", ""), command)
        if not payload:
            return author, None

        raw_content = str(getattr(message, "content", "")).lstrip()
        is_slash_invocation = raw_content.startswith("/")
        tokens = payload.split()
        if is_slash_invocation:
            if len(tokens) != 1:
                return None, f"Usage: `{command}` or `{command} @user` (single username, mention, or user id)."
            target_token = tokens[0].strip()
        else:
            # Dot commands can target multi-word names (for example: `.resumecheck ricky disappoints`).
            target_token = payload.strip()

        if target_token.lower() in {"me", "self"}:
            return author, None

        target_id = self._parse_user_mention_id(target_token)
        if target_id is None and target_token.isdigit():
            target_id = int(target_token)

        if target_id is not None:
            member = guild.get_member(target_id)
            if member is None:
                try:
                    member = await guild.fetch_member(target_id)
                except (discord.NotFound, discord.Forbidden, discord.HTTPException, AttributeError):
                    member = None
            if member is None:
                return None, f"Could not find member `{target_token}` in this server."
            return member, None

        named_member = None
        if hasattr(guild, "get_member_named"):
            named_member = guild.get_member_named(target_token)
        if named_member is not None:
            return named_member, None

        members = list(getattr(guild, "members", []) or [])
        lowered = target_token.casefold()
        matches = [
            member for member in members
            if str(getattr(member, "name", "")).casefold() == lowered
            or str(getattr(member, "display_name", "")).casefold() == lowered
            or str(getattr(member, "global_name", "")).casefold() == lowered
        ]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"Multiple members match `{target_token}`. Use a mention or user id."

        return None, f"Could not resolve `{target_token}` to a server member."

    @staticmethod
    def _profile_has_content(cache_root: Path, profile_key: str | int | None) -> bool:
        """True when this profile has resume text a ranker can actually match on.

        ``profile_cache_seed_ready()`` only checks that the three seed files
        exist, and ``ensure_profile_cache()`` creates them empty -- so every
        member who has ever touched a resume command has a "ready" directory
        holding a 0-byte baseinfo.txt. Ranking against one of those produces the
        "no resume content" dead end, so the default picker skips them.
        """
        if profile_key is None:
            return False
        try:
            text = (profile_cache_dir(profile_key, cache_root) / "baseinfo.txt").read_text(
                encoding="utf-8"
            )
        except OSError:
            return False
        return bool(text.strip())

    async def _member_profile_key(
        self,
        guild: Any,
        user_id: int | None,
        cache_root: Path,
    ) -> str | None:
        """The cache key for ``user_id``'s profile, resolved through their username.

        Profile directories are named after the Discord username
        (``discord_profile_key``), never the numeric id, so a raw owner_id can
        never address one. The member may not be in the cache -- the bot runs on
        ``Intents.default()``, which has no members intent -- so fall back to an
        HTTP fetch, which does not need it.
        """
        if guild is None or user_id is None:
            return None

        member = getattr(guild, "owner", None)
        if self._normalized_discord_id(getattr(member, "id", None)) != user_id:
            member = None
        if member is None:
            getter = getattr(guild, "get_member", None)
            member = getter(user_id) if callable(getter) else None
        if member is None:
            fetch = getattr(guild, "fetch_member", None)
            if callable(fetch):
                try:
                    member = await fetch(user_id)
                except (discord.HTTPException, discord.NotFound, AttributeError, TypeError):
                    member = None

        username = self._author_profile_name(member)
        if not username:
            return None
        key = discord_profile_key(user_id, username)
        return key if (cache_root / key).is_dir() else None

    async def resolve_default_profile_key(
        self,
        message: discord.Message,
        cache_root: Path = RESUMES_CACHE_ROOT,
    ) -> str | int | None:
        """Which profile a no-argument ranking command should rank for.

        The channel's owner is the intended default. An explicitly configured
        MAIN_USER_PROFILE still wins (that is what setting it means), but the
        bare "example" fallback config.py substitutes when it is unset does not
        -- that seed profile is empty by design.
        """
        configured = self.config.main_user_profile_key
        if configured and str(configured).lower() != EXAMPLE_PROFILE_KEY:
            return configured

        guild = getattr(message, "guild", None)
        owner_id = self._normalized_discord_id(getattr(guild, "owner_id", None))
        owner_key = await self._member_profile_key(guild, owner_id, cache_root)
        if owner_key and self._profile_has_content(cache_root, owner_key):
            return owner_key

        # A DM has no owner; rank for whoever is talking to the bot.
        if guild is None:
            author = getattr(message, "author", None)
            author_key = discord_profile_key(
                self._normalized_discord_id(getattr(author, "id", None)) or 0,
                self._author_profile_name(author),
            )
            if self._profile_has_content(cache_root, author_key):
                return author_key

        # Only when the owner could not be identified at all -- no guild, or a
        # member lookup the bot could not complete. If the owner IS known and
        # their profile is simply empty, fall through to the caller's "seed one
        # first" message instead: silently ranking some other member's resume
        # would answer a question nobody asked, and would show one member's
        # matches to the whole channel.
        if owner_key is None:
            with_content = [
                path.name
                for path in sorted(cache_root.iterdir())
                if path.is_dir()
                and path.name != EXAMPLE_PROFILE_KEY
                and self._profile_has_content(cache_root, path.name)
            ] if cache_root.is_dir() else []
            if len(with_content) == 1:
                return with_content[0]

        # Nothing usable. Returning an empty profile directory here would only
        # buy the ranker's "seed it first" message one archive scan later; the
        # caller says it straight away instead.
        return None

    async def _resolve_member_profile_token(
        self,
        message: discord.Message,
        token: str,
        cache_root: Path,
    ) -> str:
        """Turn an @mention into the username the cache is keyed by.

        Anything else is passed through untouched for the folder-name resolver.
        """
        mention_id = self._parse_user_mention_id(token)
        if mention_id is None:
            return token
        # Permission is checked by the caller, but check it here too before
        # spending a member fetch on a request that is about to be refused.
        if not self._is_guild_owner(message):
            return token
        key = await self._member_profile_key(
            getattr(message, "guild", None), mention_id, cache_root
        )
        return key or token

    def _resolve_owner_profile_key_from_payload(
        self,
        message: discord.Message,
        command: str,
        cache_root: Path,
    ) -> tuple[str | None, str | None]:
        if not self._is_guild_owner(message):
            return None, "Only the server owner can target a resume cache profile directly."

        payload = _extract_command_payload(getattr(message, "content", ""), command).strip()
        if not payload:
            return None, "No target profile provided."

        # Support both exact folder names and username-like input that normalizes to a folder key.
        candidate_keys: list[str] = []
        normalized_key = discord_profile_key(0, payload)
        compact_key = re.sub(r"[-._\s]+", "", normalized_key)
        for candidate in (payload, normalized_key, compact_key):
            normalized = str(candidate).strip()
            if normalized and normalized not in candidate_keys:
                candidate_keys.append(normalized)

        for profile_key in candidate_keys:
            if (cache_root / profile_key).is_dir():
                return profile_key, None

        return None, f"Could not resolve `{payload}` to a resume cache profile folder."

    @staticmethod
    def _author_profile_name(author: discord.abc.User | discord.Member | None) -> str | None:
        if author is None:
            return None
        value = getattr(author, "name", None)
        if isinstance(value, str) and value.strip():
            return value
        return None

    def build_resume_summary_lines(
        self,
        settings: GeminiSettings,
        job: JobContext,
        cache_status: ResumeExplicitCacheStatus,
        profile_dir: Path,
        used_provider: str | None = None,
    ) -> list[str]:
        source_names = ", ".join(path.name for path in cache_status.source_files) or "none"
        if used_provider == "openrouter":
            model_label = settings.openrouter_model
        elif used_provider == "groq":
            model_label = settings.groq_model
        else:
            model_label = settings.model
        lines = [
            f"LLM model: `{model_label}`",
            f"Resume source folder: `{self.config.resume_profiles_dir.name}`",
            f"Local cache metadata: `{self.config.resume_cache_dir.name}`",
            f"Profile scaffold: `{profile_dir.name}`",
            f"Resume sources: {source_names}",
            f"Explicit cache: {cache_status.message}",
            f"Job title: {job.title}",
            f"Posting URL: {job.posting_url}",
        ]
        if cache_status.remote_expire_time:
            lines.append(f"Cache expires: {cache_status.remote_expire_time}")
        if job.apply_url:
            lines.append(f"Apply URL: {job.apply_url}")
        return lines

    async def resolve_resume_context(
        self,
        message: discord.Message,
        command: str = CMD_RESUME,
        slash_command: str = PRIMARY_RESUME_SLASH_COMMAND,
    ) -> JobContext | None:
        referenced = await self.resolve_referenced_message(message)
        if referenced is None:
            await message.channel.send(f"Reply to a job post message with `{command}` or `{slash_command}`.")
            return None

        referenced_author = getattr(referenced, "author", None)
        if not bool(getattr(referenced_author, "bot", False)):
            await message.channel.send(
                f"Reply to an ErnestBot job listing message with `{command}` or `{slash_command}`."
            )
            return None

        content = str(getattr(referenced, "content", "")).strip()
        if not content:
            content = await self._read_text_attachment(referenced)
        if not content:
            await message.channel.send(
                "The replied message does not contain a text job post to parse, "
                "and no readable text file attachment was found."
            )
            return None

        listing_lines = [line.strip() for line in content.splitlines() if line.strip()]
        first_line = listing_lines[0] if listing_lines else ""
        has_listing_header = first_line.startswith("[") and "]" in first_line
        has_url = "http://" in content or "https://" in content
        if not (has_listing_header and has_url):
            await message.channel.send(
                f"Reply to an ErnestBot job listing message with `{command}` or `{slash_command}`."
            )
            return None

        job = extract_job_context_from_message(content)
        if job is None:
            await message.channel.send("Could not find a job title and URL in the replied message.")
            return None

        return job

    async def _prepare_resume_request(
        self,
        message: discord.Message,
        command: str,
        slash_command: str,
    ) -> _ResumeRequest | None:
        """Shared validation scaffold for the resume-family commands: guild
        check, target/profile resolution, provider-key check, job context, and
        required-file checks. Sends the user-facing error and returns None on
        any failure."""
        guild_owner_id = getattr(getattr(message, "guild", None), "owner_id", None)
        if guild_owner_id is None:
            await message.channel.send(self.build_resume_guild_only_message())
            return None

        cache_root = Path(__file__).resolve().parents[1] / "services" / "resumes" / "resumes_cache"
        target_user, target_error = await self._resolve_resume_target_member(message, command)
        owner_profile_key_override: str | None = None
        if target_error:
            owner_profile_key_override, owner_override_error = self._resolve_owner_profile_key_from_payload(
                message,
                command,
                cache_root,
            )
            if owner_override_error is not None:
                await message.channel.send(target_error)
                return None
            target_user_id = self._normalized_discord_id(getattr(getattr(message, "author", None), "id", None))
            target_profile_name = None
        else:
            target_user_id = self._normalized_discord_id(getattr(target_user, "id", None))
            if not self._can_access_resume_target(message, target_user_id):
                await message.channel.send(self.build_resume_unauthorized_message())
                return None

            target_profile_name = self._author_profile_name(target_user)
            if target_user_id is None:
                await message.channel.send("Could not resolve a valid target profile for this command.")
                return None

        settings = load_gemini_settings(self.config)
        if not (settings.api_key or settings.openrouter_api_key or settings.groq_api_key):
            await message.channel.send(self.build_resume_missing_key_message())
            return None

        job = await self.resolve_resume_context(message, command, slash_command)
        if job is None:
            return None

        profile_key = owner_profile_key_override or discord_profile_key(target_user_id, target_profile_name)

        # Explicitly check for user resume cache folder and required files
        if owner_profile_key_override is None:
            try:
                profile_dir = await self._run_interactive(
                    ensure_profile_cache,
                    profile_key,
                    cache_root,
                    label=scheduler_labels.RESUME_ENSURE_PROFILE_CACHE,
                )
            except FileNotFoundError as exc:
                await message.channel.send(f"Resume cache setup failed: {exc}",
                                           **quiet_reply_kwargs(message),)
                return None

            try:
                profile_key = resolve_discord_profile_key(
                    target_user_id,
                    target_profile_name,
                    cache_root,
                )
            except FileNotFoundError as exc:
                await message.channel.send(
                    f"Resume cache error: {exc}",
                    **quiet_reply_kwargs(message),
                )
                return None
            profile_dir = cache_root / profile_key
        else:
            profile_dir = cache_root / profile_key

        # Defensive: check all required files exist
        required_files = ["baseinfo.txt", "instructions.txt", "template.tex"]
        missing = [f for f in required_files if not (profile_dir / f).exists()]
        if missing:
            await message.channel.send(
                f"Your resume cache folder `{profile_dir.name}` is missing required file(s): {', '.join(missing)}. "
                "Please add them and try again.",
                **quiet_reply_kwargs(message),
            )
            return None

        reply_send_kwargs = quiet_reply_kwargs(message)

        if owner_profile_key_override is None:
            cache_scope = str(target_user_id)
        else:
            cache_scope = f"profile_{re.sub(r'[^A-Za-z0-9._-]+', '-', profile_key)}"

        return _ResumeRequest(
            settings=settings,
            job=job,
            profile_key=profile_key,
            profile_dir=profile_dir,
            template_path=profile_dir / "template.tex",
            cache_scope=cache_scope,
            reply_send_kwargs=reply_send_kwargs,
        )

    @command_handler(CMD_RESUME, normalize=True)
    async def handle_resume(self, message: discord.Message) -> bool:
        raw_content = str(getattr(message, "content", ""))
        # Directive first: flag stripping is a whole-message regex, so running
        # it first eats a "--aggressive" written INSIDE the steer and silently
        # turns the mode on — the opposite of what "(avoid --aggressive
        # phrasing)" asked for.
        stripped_content, user_directive = _extract_llm_directive(raw_content)
        stripped_content, aggressive, strong_aggressive = _extract_aggressiveness_flags(stripped_content)
        # Keyed on the content actually changing, not on the directive being
        # non-empty: ".resumebuild ()" strips to nothing but must still be
        # re-wrapped, or the bare "()" is read as a target username.
        if stripped_content != raw_content:
            message = _ContentOverrideMessage(message, stripped_content)

        request = await self._prepare_resume_request(message, CMD_RESUME, PRIMARY_RESUME_SLASH_COMMAND)
        if request is None:
            return True

        settings = request.settings
        job = request.job
        profile_dir = request.profile_dir
        template_path = request.template_path
        reply_send_kwargs = request.reply_send_kwargs
        compile_log_dir = self.config.resume_cache_dir / "compile_logs"
        template_log_path = compile_log_dir / f"{profile_dir.name}.log"

        # Use a user-specific ResumeExplicitCacheManager for this request
        user_resume_cache_manager = ResumeExplicitCacheManager(
            profiles_dir=profile_dir,
            cache_dir=self.config.resume_cache_dir / f".cache_{request.cache_scope}",
        )

        # Structured-mode profiles never use the Gemini context cache (the
        # cached legacy instructions conflict with the JSON-only prompt), so
        # skip the cache round-trip entirely for them.
        is_structured_profile = await self._run_interactive(
            lambda: load_structured_profile(template_path, profile_dir / "baseinfo.txt") is not None,
            label=scheduler_labels.RESUME_LOAD_STRUCTURED_PROFILE,
        )
        if is_structured_profile:
            cache_status = ResumeExplicitCacheStatus(
                status="skipped",
                message="Gemini cache skipped: structured profile mode.",
            )
        elif settings.api_key:
            cache_status = await self._run_interactive(
                user_resume_cache_manager.ensure_cache, settings, label=scheduler_labels.RESUME_ENSURE_GEMINI_CACHE
            )
        else:
            cache_status = ResumeExplicitCacheStatus(
                status="skipped",
                message="Gemini cache skipped: no Gemini API key configured.",
            )

        allowance = self._member_allowance(message)
        if allowance.blocked:
            await message.channel.send(
                f"Resume generation is not available on your quota. "
                f"Ask the server owner for a share: `{CMD_QUOTA} user @you 0.5`.",
                **reply_send_kwargs,
            )
            return True

        rewrite_result = await self._run_interactive(
            generate_resume_rewrite,
            settings,
            job,
            cache_status.cache_name,
            [profile_dir / "baseinfo.txt"],
            [profile_dir / "instructions.txt"],
            template_path,
            label=scheduler_labels.RESUME_REWRITE,
            aggressive=aggressive,
            strong_aggressive=strong_aggressive,
            user_directive=user_directive,
            # A reduced share keeps the generation pass and drops the optional
            # audits in order of cost: a cheaper resume, never a broken one.
            allowance=allowance,
        )

        if rewrite_result.status != "ok" or not rewrite_result.rewritten_resume:
            await message.channel.send(f"Resume generation failed: {rewrite_result.message}", **reply_send_kwargs)
            return True

        if not rewrite_result.latex_document:
            await message.channel.send("Resume generation did not return a LaTeX document.", **reply_send_kwargs)
            return True

        compile_result = await self._run_interactive(
            compile_latex_to_pdf,
            rewrite_result.latex_document,
            job.title,
            template_path,
            template_log_path,
            self.config.resume_normalize_json_latex,
            label=scheduler_labels.RESUME_COMPILE_LATEX,
        )

        def _recompile(latex_document: str):
            return compile_latex_to_pdf(
                latex_document,
                job.title,
                template_path,
                template_log_path,
                self.config.resume_normalize_json_latex,
            )

        # Deterministic auto-fixes in compile_latex_to_pdf only catch known
        # failure patterns. For anything novel, iteratively ask an LLM to fix
        # the specific compile error (not re-tailor the resume), feeding each
        # fresh error back, bounded to two rounds to cap added latency/cost.
        llm_repair_provider: str | None = None
        llm_repair_attempted = False
        if compile_result.status == "error" and compile_result.log_excerpt:
            llm_repair_attempted = True
            repaired_latex, compile_result, llm_repair_provider, _repair_rounds = await self._run_interactive(
                repair_latex_until_compiles,
                rewrite_result.latex_document,
                compile_result,
                settings,
                _recompile,
                label=scheduler_labels.RESUME_REPAIR_LATEX,
            )
            if llm_repair_provider:
                rewrite_result.latex_document = repaired_latex

        # A compiling resume can still violate the expected format by
        # overflowing the page budget. Best-effort: ask an LLM to comment out
        # the least-relevant bullets; the original PDF is kept unless the
        # condensed one compiles with fewer pages.
        llm_condense_provider: str | None = None
        if compile_result.status == "ok":
            condensed_latex, compile_result, llm_condense_provider = await self._run_interactive(
                condense_latex_if_overflowing,
                rewrite_result.latex_document,
                compile_result,
                self.config.resume_max_pages,
                settings,
                _recompile,
                label=scheduler_labels.RESUME_CONDENSE_LATEX,
            )
            if llm_condense_provider:
                rewrite_result.latex_document = condensed_latex

        if compile_result.status == "ok" and compile_result.pdf_bytes and compile_result.pdf_name:
            structured_summary: dict[str, Any] | None = None
            try:
                parsed_summary = json.loads(rewrite_result.rewritten_resume or "")
                if isinstance(parsed_summary, dict) and parsed_summary.get("mode") == "structured":
                    structured_summary = parsed_summary
            except (json.JSONDecodeError, TypeError):
                structured_summary = None

            if structured_summary is not None and rewrite_result.used_provider is None:
                content = (
                    "Compiled PDF from canonical content (all LLM providers failed; "
                    "rendered deterministically with extracted listing keywords)."
                )
            else:
                used_provider = rewrite_result.used_provider or "gemini"
                if used_provider == "openrouter":
                    used_model = settings.openrouter_model
                elif used_provider == "groq":
                    used_model = settings.groq_model
                else:
                    used_model = settings.model
                content = f"Compiled PDF using `{used_provider}` (`{used_model}`)."
                if structured_summary is not None:
                    tailored = structured_summary.get("tailored_bullets_used", 0)
                    total_bullets = structured_summary.get("visible_bullet_count", 0)
                    content = (
                        f"{content} Tailored {tailored}/{total_bullets} bullets for this listing."
                    )
                    if structured_summary.get("strong_aggressive"):
                        content = f"{content} (strong-aggressive mode)"
                    elif structured_summary.get("aggressive"):
                        content = f"{content} (aggressive mode)"
            if compile_result.repairs_applied:
                shown_repairs = compile_result.repairs_applied[:3]
                extra_repairs = len(compile_result.repairs_applied) - len(shown_repairs)
                repair_note = ", ".join(shown_repairs)
                if extra_repairs > 0:
                    repair_note = f"{repair_note}, +{extra_repairs} more"
                content = f"{content} Auto-fixed LaTeX: {repair_note}."
            if llm_repair_provider:
                content = f"{content} LLM-repaired a compile error via `{llm_repair_provider}`."
            if llm_condense_provider:
                content = (
                    f"{content} LLM-condensed to fit {self.config.resume_max_pages} page(s) "
                    f"via `{llm_condense_provider}`."
                )
            elif compile_result.page_count and compile_result.page_count > self.config.resume_max_pages:
                content = (
                    f"{content} Note: PDF is {compile_result.page_count} pages "
                    f"(target {self.config.resume_max_pages}); auto-condense could not shrink it."
                )
            await message.channel.send(
                content=content,
                file=discord.File(io.BytesIO(compile_result.pdf_bytes), filename=compile_result.pdf_name),
                **reply_send_kwargs,
            )
        else:
            failure_dump_name = None
            try:
                failed_latex_dir = self.config.resume_cache_dir / "failed_latex"
                failed_latex_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                title_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", job.title).strip("-") or "resume"
                failure_dump_name = f"{timestamp}-{profile_dir.name}-{title_slug}.tex"
                (failed_latex_dir / failure_dump_name).write_text(
                    rewrite_result.latex_document,
                    encoding="utf-8",
                )
            except OSError:
                failure_dump_name = None

            error_msg = f"PDF compile failed: {compile_result.message}"
            if llm_repair_attempted:
                error_msg += " (LLM auto-repair was also attempted and did not produce a compiling document.)"
            if failure_dump_name:
                error_msg += f"\nSaved failed LaTeX: `{failure_dump_name}`"
            if compile_result.log_excerpt:
                error_msg += f"\n```\n{compile_result.log_excerpt[-500:]}\n```"
            await message.channel.send(error_msg, **reply_send_kwargs)
        return True

    @command_handler(CMD_RESUME_COVER, normalize=True)
    async def handle_resume_cover(self, message: discord.Message) -> bool:
        # The aggressiveness flags do not change cover-letter generation, but
        # they must still be stripped: anything left in the message is read as
        # a target username, so `.resumecoverbuild --aggressive` used to fail
        # with "Could not resolve `--aggressive` to a server member."
        raw_content = str(getattr(message, "content", ""))
        stripped_content, user_directive = _extract_llm_directive(raw_content)
        stripped_content, _aggressive, _strong = _extract_aggressiveness_flags(stripped_content)
        if stripped_content != raw_content:
            message = _ContentOverrideMessage(message, stripped_content)

        request = await self._prepare_resume_request(message, CMD_RESUME_COVER, "/resumecoverbuild")
        if request is None:
            return True

        job = request.job
        profile_dir = request.profile_dir
        reply_send_kwargs = request.reply_send_kwargs
        compile_log_dir = self.config.resume_cache_dir / "compile_logs"
        cover_log_path = compile_log_dir / f"{profile_dir.name}.coverbuild.log"

        allowance = self._member_allowance(message)
        if allowance.blocked:
            await message.channel.send(
                f"Cover letter generation is not available on your quota. "
                f"Ask the server owner for a share: `{CMD_QUOTA} user @you 0.5`.",
                **reply_send_kwargs,
            )
            return True

        cover_result = await self._run_interactive(
            generate_cover_letter,
            request.settings,
            job,
            request.template_path,
            profile_dir / "baseinfo.txt",
            label=scheduler_labels.RESUME_COVER_REWRITE,
            user_directive=user_directive,
            allowance=allowance,
        )

        if cover_result.status != "ok" or not cover_result.latex_document:
            await message.channel.send(
                f"Cover letter generation failed: {cover_result.message}", **reply_send_kwargs
            )
            return True

        compile_result = await self._run_interactive(
            compile_latex_to_pdf,
            cover_result.latex_document,
            f"Cover Letter - {job.title}",
            request.template_path,
            cover_log_path,
            self.config.resume_normalize_json_latex,
            label=scheduler_labels.RESUME_COVER_COMPILE_LATEX,
        )

        if compile_result.status == "ok" and compile_result.pdf_bytes and compile_result.pdf_name:
            content = cover_result.message
            if compile_result.repairs_applied:
                content = f"{content} Auto-fixed LaTeX: {', '.join(compile_result.repairs_applied[:3])}."
            await message.channel.send(
                content=content,
                file=discord.File(io.BytesIO(compile_result.pdf_bytes), filename=compile_result.pdf_name),
                **reply_send_kwargs,
            )
        else:
            failure_dump_name = None
            try:
                failed_latex_dir = self.config.resume_cache_dir / "failed_latex"
                failed_latex_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                title_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", job.title).strip("-") or "cover"
                failure_dump_name = f"{timestamp}-{profile_dir.name}-cover-{title_slug}.tex"
                (failed_latex_dir / failure_dump_name).write_text(
                    cover_result.latex_document,
                    encoding="utf-8",
                )
            except OSError:
                failure_dump_name = None

            error_msg = f"Cover letter PDF compile failed: {compile_result.message}"
            if failure_dump_name:
                error_msg += f"\nSaved failed LaTeX: `{failure_dump_name}`"
            if compile_result.log_excerpt:
                error_msg += f"\n```\n{compile_result.log_excerpt[-500:]}\n```"
            await message.channel.send(error_msg, **reply_send_kwargs)
        return True

    @command_handler(CMD_RESUME_CHECK, normalize=True)
    async def handle_resume_check(self, message: discord.Message) -> bool:
        # No LLM runs here — this is a raw template compile — so a directive
        # has nothing to steer. Strip it anyway so the parse stays uniform
        # across the resume family and `(...)` is never read as a username.
        check_content, _directive = _extract_llm_directive(str(getattr(message, "content", "")))
        if check_content != str(getattr(message, "content", "")):
            message = _ContentOverrideMessage(message, check_content)

        guild_owner_id = getattr(getattr(message, "guild", None), "owner_id", None)
        if guild_owner_id is None:
            await message.channel.send(f"`{CMD_RESUME_CHECK}` is only available in a server channel.")
            return True

        cache_root = Path(__file__).resolve().parents[1] / "services" / "resumes" / "resumes_cache"
        target_user, target_error = await self._resolve_resume_target_member(message, CMD_RESUME_CHECK)
        owner_profile_key_override: str | None = None
        if target_error:
            owner_profile_key_override, owner_override_error = self._resolve_owner_profile_key_from_payload(
                message,
                CMD_RESUME_CHECK,
                cache_root,
            )
            if owner_override_error is not None:
                await message.channel.send(target_error)
                return True
            profile_key = owner_profile_key_override
            profile_dir = cache_root / profile_key
        else:
            target_user_id = self._normalized_discord_id(getattr(target_user, "id", None))
            if not self._can_access_resume_target(message, target_user_id):
                await message.channel.send(self.build_resume_unauthorized_message())
                return True

            target_profile_name = self._author_profile_name(target_user)
            if target_user_id is None:
                await message.channel.send("Could not resolve a valid target profile for this command.")
                return True

            profile_key = discord_profile_key(target_user_id, target_profile_name)
            profile_dir = await self._run_interactive(
                ensure_profile_cache,
                profile_key,
                cache_root,
                label=scheduler_labels.RESUME_ENSURE_PROFILE_CACHE,
            )
            try:
                profile_key = resolve_discord_profile_key(
                    target_user_id,
                    target_profile_name,
                    cache_root,
                )
            except FileNotFoundError as exc:
                await message.channel.send(
                    f"Resume cache error: {exc}",
                    **quiet_reply_kwargs(message),
                )
                return True
            profile_dir = cache_root / profile_key
        template_path = profile_dir / "template.tex"
        compile_log_dir = self.config.resume_cache_dir / "compile_logs"
        template_log_path = compile_log_dir / f"{profile_dir.name}.resumecheck.log"

        try:
            template_text = await self._run_interactive(
                template_path.read_text, encoding="utf-8", label=scheduler_labels.RESUME_READ_TEMPLATE
            )
        except OSError as exc:
            await message.channel.send(f"Could not read current template for `{profile_dir.name}`: {exc}")
            return True

        if not template_text.strip():
            await message.channel.send("Current template is empty. Update `template.tex` first, then run `/resumecheck`.")
            return True

        compile_result = await self._run_interactive(
            compile_latex_to_pdf,
            template_text,
            f"{profile_dir.name}-resume-check",
            template_path,
            template_log_path,
            self.config.resume_normalize_json_latex,
            label=scheduler_labels.RESUME_COMPILE_LATEX,
        )

        reply_send_kwargs = quiet_reply_kwargs(message)

        if compile_result.status == "ok" and compile_result.pdf_bytes and compile_result.pdf_name:
            content = f"Resume check compile succeeded for profile `{profile_dir.name}`."
            if compile_result.repairs_applied:
                shown_repairs = compile_result.repairs_applied[:3]
                extra_repairs = len(compile_result.repairs_applied) - len(shown_repairs)
                repair_note = ", ".join(shown_repairs)
                if extra_repairs > 0:
                    repair_note = f"{repair_note}, +{extra_repairs} more"
                content = f"{content} Auto-fixed LaTeX: {repair_note}."
            await message.channel.send(
                content=content,
                file=discord.File(io.BytesIO(compile_result.pdf_bytes), filename=compile_result.pdf_name),
                **reply_send_kwargs,
            )
            return True

        error_msg = f"Resume check compile failed: {compile_result.message}"
        if compile_result.log_excerpt:
            error_msg += f"\n```\n{compile_result.log_excerpt[-500:]}\n```"
        await message.channel.send(error_msg, **reply_send_kwargs)
        return True

    @command_handler(CMD_CLEAR_REDDIT_SEEN, CMD_RESET_REDDIT_SEEN, normalize=True)
    async def handle_clear_reddit_seen(self, message: discord.Message) -> bool:

        state_file = self.config.base_dir / f".reddit_seen_{message.channel.id}.json"
        state_file.write_text(json.dumps({"seen_ids": []}, ensure_ascii=True), encoding="utf-8")
        await message.channel.send("Cleared reddit seen list for this channel.")
        return True

    @command_handler(CMD_BEST_JOBS, CMD_BEST_JOBS_ALIAS, normalize=True)
    async def handle_best_jobs(self, message: discord.Message) -> bool:
        payload = _extract_command_payload(message.content, CMD_BEST_JOBS)
        if not payload:
            payload = _extract_command_payload(message.content, CMD_BEST_JOBS_ALIAS)
        window, limit, profile_token, enrich, error = parse_best_jobs_payload(payload)
        if error:
            await message.channel.send(error, **quiet_reply_kwargs(message))
            return True

        profile_key: str | int | None
        if profile_token:
            # Same rule as the resume commands: only the server owner may point
            # a command at somebody else's resume cache. An @mention is accepted
            # as well as a folder name, since naming a member is the obvious way
            # to ask for their matches.
            profile_token = await self._resolve_member_profile_token(
                message, profile_token, RESUMES_CACHE_ROOT
            )
            resolved, resolve_error = self._resolve_owner_profile_key_from_payload(
                _ContentOverrideMessage(message, f"{CMD_BEST_JOBS} {profile_token}"),
                CMD_BEST_JOBS,
                RESUMES_CACHE_ROOT,
            )
            if resolve_error:
                await message.channel.send(resolve_error, **quiet_reply_kwargs(message))
                return True
            profile_key = resolved
        else:
            profile_key = await self.resolve_default_profile_key(message, RESUMES_CACHE_ROOT)

        if profile_key is None:
            await message.channel.send(
                f"No seeded resume profile to rank against. Run `{CMD_RESUME}` to seed one, "
                f"or name a profile: `{CMD_BEST_JOBS} {window} {limit} <profile>`.",
                **quiet_reply_kwargs(message),
            )
            return True

        allowance = self._member_allowance(message)
        if allowance.blocked:
            await message.channel.send(
                f"`{CMD_BEST_JOBS}` is not available on your quota. "
                f"Ask the server owner for a share: `{CMD_QUOTA} user @you 0.5`.",
                **quiet_reply_kwargs(message),
            )
            return True

        status_note = (
            "fetching descriptions, then ranking" if enrich else "ranking (fast, titles only)"
        )
        if not allowance.full:
            status_note += f" · {allowance.describe()}"
        status_message = await message.channel.send(
            f"Best matches for the last {window} - {status_note}...",
            **quiet_reply_kwargs(message),
        )
        # Scope the archive to this channel's own job-watcher settings --
        # otherwise the ranking draws from every channel's search, not just
        # the one the command was actually run in.
        job_settings = self.store.get_job_settings(message.channel.id)
        try:
            report = await self._run_interactive(
                job_match.best_jobs,
                profile_key,
                window=window,
                limit=limit,
                settings=load_gemini_settings(self.config),
                cache_root=RESUMES_CACHE_ROOT,
                enrich=enrich,
                llm_candidates=allowance.scale(job_match.LLM_CANDIDATES, minimum=5),
                enrich_candidates=allowance.scale(job_match.ENRICH_CANDIDATES, minimum=2),
                role_filters=job_settings["role_filters"],
                exclusion_terms=job_settings["exclusion_terms"],
                allow_north_america=bool(job_settings.get("allow_north_america", False)),
                label=scheduler_labels.BEST_JOBS_RANK,
            )
        except Exception as exc:
            await status_message.edit(content=f"`{CMD_BEST_JOBS}` failed: {exc}")
            return True

        text = job_match.format_match_report(report, window)
        try:
            await status_message.delete()
        except (discord.NotFound, discord.HTTPException):
            pass
        await message.channel.send(
            self.set_continuation(message.channel.id, text), **quiet_reply_kwargs(message)
        )
        return True

    def _member_allowance(self, message: discord.Message):
        """This author's share of the expensive work in one command."""
        from services import quota

        guild = getattr(message, "guild", None)
        # A router without a store (tests, and any future embedding that does
        # not persist state) still has to run its commands: an unreadable quota
        # means "no quota configured", never a failed command. Same fail-open
        # rule the description cache follows -- a limiter that can take the bot
        # down is worse than no limiter.
        store = getattr(self, "store", None)
        if store is None:
            from services.quota import QuotaPolicy

            policy = QuotaPolicy()
        else:
            policy = store.get_quota_policy(
                self._normalized_discord_id(getattr(guild, "id", None))
            )
        author = getattr(message, "author", None)
        role_ids = tuple(
            rid for rid in (
                self._normalized_discord_id(getattr(role, "id", None))
                for role in (getattr(author, "roles", None) or ())
            ) if rid is not None
        )
        return quota.allowance_for(
            policy,
            user_id=self._normalized_discord_id(getattr(author, "id", None)),
            role_ids=role_ids,
            is_owner=self._is_guild_owner(message),
        )

    @staticmethod
    def _quota_target_id(text: str) -> int | None:
        """A user or role id from a mention, or a raw id typed directly."""
        match = _MENTION_ID_PATTERN.search(str(text or ""))
        if match:
            return int(match.group(1))
        try:
            return int(str(text).strip())
        except (TypeError, ValueError):
            return None

    def _format_quota_policy(self, message: discord.Message, policy) -> str:
        from services import quota

        guild = getattr(message, "guild", None)
        lines = [
            f"Quota shares (0-1) for `{getattr(guild, 'name', 'this server')}`",
            f"  default: {policy.default_share:.2f} "
            f"({quota.Allowance(policy.default_share).describe()})",
        ]
        if policy.role_shares:
            lines.append("  roles:")
            for role_id, share in sorted(policy.role_shares.items()):
                role = discord.utils.get(getattr(guild, "roles", None) or [], id=role_id)
                lines.append(f"    {getattr(role, 'name', role_id)}: {share:.2f}")
        if policy.user_shares:
            lines.append("  members (manual overrides, these win over roles):")
            for user_id, share in sorted(policy.user_shares.items()):
                member = discord.utils.get(getattr(guild, "members", None) or [], id=user_id)
                lines.append(f"    {getattr(member, 'display_name', user_id)}: {share:.2f}")
        lines.append("")
        lines.append(
            f"The server owner is always {quota.FULL_SHARE:.1f} and cannot be limited. "
            f"A share below {quota.BLOCKED_BELOW} refuses the command outright."
        )
        return "\n".join(lines)

    @command_handler(CMD_RESET, normalize=True)
    async def handle_reset(self, message: discord.Message) -> bool:
        """Restart the bot: spawn `run.py --forcerun` detached (not a child of
        this process, so it outlives it) in run.bat's directory. That spawned
        process kills this one and starts a genuinely new interpreter, which
        imports every module fresh off disk -- not a fork/clone of this
        process's already-loaded state, so any code changes since this
        process started are what the restarted bot runs.
        """
        if not self._is_guild_owner(message):
            await message.channel.send(
                "Only the server owner can reset the bot.",
                **quiet_reply_kwargs(message),
            )
            return True

        from services import platform_support

        if self._restart_spawned:
            await message.channel.send(
                "A restart is already in progress.", **quiet_reply_kwargs(message)
            )
            return True
        # Claim it before the first await, or a .reset arriving mid-send races in.
        self._restart_spawned = True

        # Confirm intent BEFORE spawning: once the child's forcerun reaches its
        # kill step, this process can be gone within milliseconds (Windows'
        # taskkill /F is immediate), too fast to rely on a reply sent after.
        await message.channel.send("Restarting...", **quiet_reply_kwargs(message))

        run_py = self.config.base_dir / "run.py"
        # The child's output used to go to DEVNULL, so the one process that
        # knew why a restart failed wrote its reason nowhere -- and this process
        # is about to be killed by it, taking the question with it.
        ok, detail = platform_support.spawn_detached(
            [sys.executable, str(run_py), "--forcerun"],
            cwd=self.config.base_dir,
            log_path=self.config.base_dir / ".restart.log",
        )
        if not ok:
            # Either nothing spawned or it died on startup, so this process
            # stays -- release the claim, or one failed launch refuses every
            # later .reset for this process's life. That was the trap: a child
            # that exited immediately still made Popen succeed, so the bot said
            # "Restarting...", never restarted, and would not try again.
            self._restart_spawned = False
            await message.channel.send(
                f"Restart failed to launch: {detail}", **quiet_reply_kwargs(message)
            )
        return True

    @command_handler(CMD_QUOTA, normalize=True)
    async def handle_quota(self, message: discord.Message) -> bool:
        from services import quota

        if not self._is_guild_owner(message):
            await message.channel.send(
                "Only the server owner can view or change quotas.",
                **quiet_reply_kwargs(message),
            )
            return True

        guild_id = self._normalized_discord_id(getattr(getattr(message, "guild", None), "id", None))
        if guild_id is None:
            await message.channel.send(
                "Quotas are per-server, so this only works inside a server.",
                **quiet_reply_kwargs(message),
            )
            return True

        policy = self.store.get_quota_policy(guild_id)
        action, target, value = parse_quota_payload(
            _extract_command_payload(message.content, CMD_QUOTA)
        )

        if action == "show":
            await message.channel.send(
                self._format_quota_policy(message, policy), **quiet_reply_kwargs(message)
            )
            return True

        if action == "unknown":
            await message.channel.send(
                f"Usage: `{CMD_QUOTA}` · `{CMD_QUOTA} default 0.5` · "
                f"`{CMD_QUOTA} role @Role 0.7` · `{CMD_QUOTA} user @member 0.3` · "
                f"`{CMD_QUOTA} clear user @member`",
                **quiet_reply_kwargs(message),
            )
            return True

        if action == "clear":
            target_id = self._quota_target_id(value)
            table = policy.user_shares if target == "user" else policy.role_shares
            if target not in ("user", "role") or target_id is None:
                await message.channel.send(
                    f"Usage: `{CMD_QUOTA} clear user @member` or "
                    f"`{CMD_QUOTA} clear role @Role`.",
                    **quiet_reply_kwargs(message),
                )
                return True
            if table.pop(target_id, None) is None:
                await message.channel.send(
                    f"No {target} override was set for that.", **quiet_reply_kwargs(message)
                )
                return True
            self.store.set_quota_policy(guild_id, policy)
            await message.channel.send(
                f"Cleared the {target} override; it now inherits "
                f"{'its role or ' if target == 'user' else ''}the default.",
                **quiet_reply_kwargs(message),
            )
            return True

        share = quota.clamp_share(value)
        if share is None:
            await message.channel.send(
                f"The share must be a number from 0 to 1 -- `0.5` is half the allowance, "
                f"`0` refuses the command. Got `{value}`.",
                **quiet_reply_kwargs(message),
            )
            return True

        if action == "default":
            policy.default_share = share
            label = "default"
        else:
            target_id = self._quota_target_id(target)
            if target_id is None:
                await message.channel.send(
                    f"Name the {action} by mention or id: `{CMD_QUOTA} {action} @"
                    f"{'member' if action == 'user' else 'Role'} {share}`.",
                    **quiet_reply_kwargs(message),
                )
                return True
            if action == "user":
                policy.user_shares[target_id] = share
            else:
                policy.role_shares[target_id] = share
            label = f"{action} {target.strip()}"

        self.store.set_quota_policy(guild_id, policy)
        await message.channel.send(
            f"Set {label} to {share:.2f} -- {quota.Allowance(share).describe()}.",
            **quiet_reply_kwargs(message),
        )
        return True

    @command_handler(CMD_JOB_PIPELINE_TEST, normalize=True)
    async def handle_job_pipeline_test(self, message: discord.Message) -> bool:
        import time as _time

        try:
            return await self._job_pipeline_test_inner(message, _time)
        except Exception as exc:
            try:
                await message.channel.send(f"`.jobtest` failed: {exc}")
            except discord.HTTPException:
                print(f".jobtest unrecoverable: {exc}")
            return True

    async def _job_pipeline_test_inner(self, message: discord.Message, _time: Any) -> bool:
        settings = self.store.get_job_settings(message.channel.id)
        keywords = str(settings.get("keywords") or "").strip()
        location = str(settings.get("location") or "").strip()
        if not keywords:
            await message.channel.send("No job watcher configured for this channel. Use `.job` first.")
            return True

        sites = list(settings.get("sites") or [])
        if not sites:
            await message.channel.send("No sites configured for this channel's watcher.")
            return True

        role_filters = list(settings.get("role_filters") or [])
        exclusion_terms = list(settings.get("exclusion_terms") or [])
        results_wanted = max(1, int(settings.get("results_wanted") or 10))
        hours_old = max(1, int(settings.get("hours_old") or 72))
        radius_miles = max(0, int(settings.get("radius_miles") or 25))
        country_indeed = str(settings.get("country_indeed") or "AUTO")
        allow_na = bool(settings.get("allow_north_america", False))
        try:
            sem_threshold = float(settings.get("semantic_threshold") or 0.30)
        except (TypeError, ValueError):
            sem_threshold = 0.30

        status_msg = await message.channel.send(
            f"Running full pipeline test: `{keywords}` in `{location}` "
            f"(sites={len(sites)}, results_wanted={results_wanted})..."
        )

        t_start = _time.perf_counter()
        # _run_interactive already holds work_guard() for the duration.
        raw_items = await self._run_interactive(
            job_service.scrape_job_postings,
            sites,
            keywords,
            location,
            self.config.jobspy_python_exe,
            hours_old,
            results_wanted,
            radius_miles,
            country_indeed,
            "command:jobtest",
            allow_na,
            label=scheduler_labels.JOB_PIPELINE_TEST_SCRAPE,
        )
        t_scrape = _time.perf_counter()

        count_raw = len(raw_items)

        after_role = [i for i in raw_items if job_service.matches_role_filters(str(i.get("title", "")), role_filters)]
        count_role_removed = count_raw - len(after_role)

        after_excl = [i for i in after_role if not job_service.matches_exclusion_terms(i, exclusion_terms)]
        count_excl_removed = len(after_role) - len(after_excl)

        def _semantic_filter(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                i for i in candidates
                if job_service.matches_search_parameters_semantic(
                    i, keywords, location, role_filters, threshold=sem_threshold,
                )
            ]

        after_semantic = await self._run_interactive(
            _semantic_filter, after_excl, label=scheduler_labels.JOB_PIPELINE_TEST_SEMANTIC_FILTER
        )
        t_filter = _time.perf_counter()
        count_sem_removed = len(after_excl) - len(after_semantic)

        # Layer 1: URL dedup — read-only snapshot of channel_job_seen
        seen_snapshot: set[str] = set()
        try:
            seen_raw = self.store.channel_job_seen.get(message.channel.id) or set()
            seen_snapshot = {
                job_service.canonicalize_job_link(str(link)) or str(link)
                for link in seen_raw
                if str(link).strip()
            }
        except Exception:
            pass

        count_url_dedup = 0
        after_url_dedup: list[dict[str, Any]] = []
        local_seen: set[str] = set(seen_snapshot)
        for item in after_semantic:
            raw_link = str(item.get("link") or "").strip()
            canonical = job_service.canonicalize_job_link(raw_link) or raw_link
            if not canonical or canonical in local_seen:
                count_url_dedup += 1
                continue
            local_seen.add(canonical)
            after_url_dedup.append(item)

        # Layer 2: FIFO title-signature dedup — read-only check against listing files
        listing_file = self.watcher_manager._attached_listing_file(message.channel.id, "job")
        months_threshold = self.watcher_manager.dedup_months_threshold()
        count_fifo_dedup = 0
        after_fifo_dedup: list[dict[str, Any]] = []
        for item in after_url_dedup:
            formatted = self.watcher_manager.format_job_watcher_message(item)
            if job_service.is_message_duplicate(formatted, listing_file, months_threshold):
                count_fifo_dedup += 1
                continue
            after_fifo_dedup.append(item)

        # Layer 3: Discord history dedup — check recent messages in channel
        count_history_dedup = 0
        fresh: list[dict[str, Any]] = []
        for item in after_fifo_dedup:
            formatted = self.watcher_manager.format_job_watcher_message(item)
            if await self.watcher_manager.should_skip_duplicate_message(
                message.channel.id, message.channel, formatted, watcher_type="job",
            ):
                count_history_dedup += 1
                continue
            fresh.append(item)

        scrape_sec = t_scrape - t_start
        filter_sec = t_filter - t_scrape
        total_sec = _time.perf_counter() - t_start

        site_counts: dict[str, int] = {}
        for item in raw_items:
            label = str(item.get("site_label") or item.get("site") or "unknown")
            site_counts[label] = site_counts.get(label, 0) + 1

        lines = [
            "**Pipeline Test Results**",
            f"Query: `{keywords}` | Location: `{location}`",
            f"Sites: {', '.join(sites)} | results_wanted: {results_wanted}",
            "",
            f"**Total retrieved: {count_raw}**",
        ]
        for site_name, cnt in sorted(site_counts.items()):
            lines.append(f"  {site_name}: {cnt}")

        lines += [
            "",
            "**Filtering**",
            f"  Role filters ({', '.join(role_filters) or 'none'}): -{count_role_removed}",
            f"  Exclusion terms ({len(exclusion_terms)} term(s)): -{count_excl_removed}",
            f"  Semantic match (threshold {sem_threshold:.2f}): -{count_sem_removed}",
            "",
            "**Dedup (all read-only)**",
            f"  URL seen set ({len(seen_snapshot)} tracked): -{count_url_dedup}",
            f"  FIFO title signature: -{count_fifo_dedup}",
            f"  Discord history: -{count_history_dedup}",
            "",
            f"**Would send: {len(fresh)}** of {count_raw} retrieved",
            "",
            f"Timing — scrape: {scrape_sec:.1f}s | filters+dedup: {total_sec:.1f}s",
        ]

        if fresh:
            lines.append(f"\n**Sample results** (first 3 of {len(fresh)}):")
            for item in fresh[:3]:
                lines.append(self.watcher_manager.format_job_watcher_message(item))

        try:
            await status_msg.edit(content=self.set_continuation(message.channel.id, "\n".join(lines)))
        except discord.HTTPException:
            await message.channel.send(self.set_continuation(message.channel.id, "\n".join(lines)))
        return True

    async def dispatch(self, message: discord.Message) -> bool:
        for handler in self.handlers:
            if await handler(message):
                return True
        return False

    async def dispatch_interaction_command(
        self,
        interaction: discord.Interaction,
        content: str,
        *,
        reference_message_id: int | None = None,
        resolved_reference: discord.Message | None = None,
    ) -> bool:
        if interaction.channel is None or not hasattr(interaction.channel, "send"):
            if not interaction.response.is_done():
                await interaction.response.send_message("This command can only run in a text channel.", ephemeral=True)
            else:
                await interaction.followup.send("This command can only run in a text channel.", ephemeral=True)
            return False

        deferred_by_router = False
        if not interaction.response.is_done():
            await interaction.response.defer()
            deferred_by_router = True

        try:
            adapted_message = _InteractionMessageAdapter(
                interaction,
                content,
                reference_message_id=reference_message_id,
                resolved_reference=resolved_reference,
            )
            for handler in self.handlers:
                if await handler(adapted_message):
                    return True

            await interaction.followup.send("Command was not recognized by the router.", ephemeral=True)
            return False
        finally:
            # Remove the public "thinking..." placeholder once real output has been sent.
            if deferred_by_router:
                try:
                    await interaction.delete_original_response()
                except (discord.NotFound, discord.HTTPException, discord.ClientException):
                    pass

    def set_continuation(self, channel_id: int, text: str) -> str:
        chunks = scrape_service.chunk_text_for_discord(text)
        if len(chunks) <= 1:
            self.channel_continuations.pop(channel_id, None)
            return chunks[0]
        self.channel_continuations[channel_id] = chunks[1:]
        return f"{chunks[0]}\n\nUse `{CMD_CONTINUE}` for more ({len(chunks) - 1} part(s) left)."

    async def send_or_update_panel_message(
        self,
        message: discord.Message,
        content: str,
        view: discord.ui.View,
        panel_key: str,
        reuse_cached_message: bool = True,
    ) -> discord.Message:
        fingerprint = (message.channel.id, panel_key)
        existing = self.channel_panel_messages.get(fingerprint)
        if reuse_cached_message:
            if existing is not None:
                try:
                    await existing.edit(content=content, view=view)
                    log_panel_send_event(
                        "panel_edit",
                        channel_id=message.channel.id,
                        panel_key=panel_key,
                        message_id=getattr(existing, "id", None),
                        source_message_id=getattr(message, "id", None),
                        reuse_cached_message=int(reuse_cached_message),
                    )
                    return existing
                except (discord.NotFound, discord.HTTPException):
                    self.channel_panel_messages.pop(fingerprint, None)
        elif existing is not None:
            try:
                await existing.delete()
                log_panel_send_event(
                    "panel_delete",
                    channel_id=message.channel.id,
                    panel_key=panel_key,
                    message_id=getattr(existing, "id", None),
                    source_message_id=getattr(message, "id", None),
                )
            except (discord.NotFound, discord.HTTPException):
                pass
            self.channel_panel_messages.pop(fingerprint, None)

        sent_message = await message.channel.send(content, view=view)
        self.channel_panel_messages[fingerprint] = sent_message
        log_panel_send_event(
            "panel_send",
            channel_id=message.channel.id,
            panel_key=panel_key,
            sent_message_id=getattr(sent_message, "id", None),
            source_message_id=getattr(message, "id", None),
            reuse_cached_message=int(reuse_cached_message),
        )
        return sent_message

    @command_handler(CMD_HELLO)
    async def handle_hello(self, message: discord.Message) -> bool:
        await message.channel.send("Hello!")
        return True

    @command_handler(CMD_SCRAPE)
    async def handle_scrape(self, message: discord.Message) -> bool:

        try:
            url, selector = scrape_service.parse_scrape_command(message.content)
        except ValueError:
            await message.channel.send(f"Usage: {CMD_SCRAPE} <url> or {CMD_SCRAPE} <url> | <css selector>")
            return True

        if not self.watcher_manager.acquire_channel_process(message.channel.id, "scrape_once"):
            await message.channel.send("A scraper process is already running in this channel.")
            return True

        await message.channel.send("Scraping...")
        try:
            settings = self.store.get_scrape_settings(message.channel.id)
            if job_service.job_site_from_url(url):
                items = await self._run_interactive(
                    job_service.scrape_jobs_from_board_url,
                    url,
                    self.config.jobspy_python_exe,
                    int(settings["max_items"]),
                    label=scheduler_labels.SCRAPE_COMMAND_JOBSITE,
                )
            else:
                items = await self._run_interactive(
                    scrape_service.scrape_url,
                    url,
                    selector,
                    "Mozilla/5.0 (compatible; RebuiltScraper/1.0)",
                    int(settings["timeout_seconds"]),
                    3,
                    int(settings["max_items"]),
                    False,
                    label=scheduler_labels.SCRAPE_COMMAND_GENERIC,
                )

            if settings["use_ai_cleanup"] and self.config.openrouter_key:
                try:
                    result = await self._run_interactive(
                        scrape_service.clean_with_openrouter,
                        url,
                        items,
                        self.config.openrouter_key,
                        label=scheduler_labels.SCRAPE_COMMAND_AI_CLEANUP,
                    )
                    output = f"[{result['source_url']} via {result['model']}]\n{result['text']}"
                except Exception:
                    output = scrape_service.format_items(url, items)
            else:
                output = scrape_service.format_items(url, items)

            await message.channel.send(self.set_continuation(message.channel.id, output))
        except Exception as exc:
            await message.channel.send(f"Scrape failed: {exc}")
        finally:
            self.watcher_manager.release_channel_process(message.channel.id, "scrape_once")
        return True

    @command_handler(CMD_CONTINUE)
    async def handle_continue(self, message: discord.Message) -> bool:

        pending = self.channel_continuations.get(message.channel.id, [])
        if not pending:
            await message.channel.send("No continuation is pending for this channel.")
            return True

        chunk = pending.pop(0)
        if pending:
            chunk = f"{chunk}\n\nUse `{CMD_CONTINUE}` for more ({len(pending)} part(s) left)."
        else:
            self.channel_continuations.pop(message.channel.id, None)
        await message.channel.send(chunk)
        return True

    @command_handler(CMD_SCRAPE_SETTINGS, CMD_SCRAPE_SETTINGS_ALIAS)
    async def handle_scrape_settings(self, message: discord.Message) -> bool:

        await self.send_or_update_panel_message(
            message,
            format_scrape_settings_summary(self.store, message.channel.id),
            ScrapeSettingsView(store=self.store, channel_id=message.channel.id, owner_id=message.author.id),
            panel_key="scrape_settings",
        )
        return True

    @command_handler(CMD_JOB_SETTINGS, CMD_JOB_SETTINGS_ALIAS)
    async def handle_job_settings(self, message: discord.Message) -> bool:

        try:
            guild_owner_id = getattr(getattr(message, "guild", None), "owner_id", None)
            cache_root = Path(__file__).resolve().parents[1] / "services" / "resumes" / "resumes_cache"
            profile_key = discord_profile_key(message.author.id, self._author_profile_name(message.author))
            resume_profile_dir = await self._run_interactive(
                ensure_profile_cache,
                profile_key,
                cache_root,
                label=scheduler_labels.RESUME_ENSURE_PROFILE_CACHE,
            )
            profile_key = resolve_discord_profile_key(
                message.author.id,
                self._author_profile_name(message.author),
                cache_root,
            )
            resume_profile_dir = cache_root / profile_key

            content = format_job_settings_summary(self.store, message.channel.id)
            view = JobSettingsView(
                store=self.store,
                manager=self.watcher_manager,
                channel_id=message.channel.id,
                owner_id=message.author.id,
                resume_profile_dir=resume_profile_dir,
            )
            await self.send_or_update_panel_message(
                message,
                content,
                view,
                panel_key="job_settings",
                reuse_cached_message=False,
            )
        except Exception as exc:
            import traceback
            await message.channel.send(f"Error loading job settings: {exc}\n```\n{traceback.format_exc()}\n```")
        return True

    @command_handler(CMD_REDDIT_SETTINGS, CMD_REDDIT_SETTINGS_ALIAS, normalize=True)
    async def handle_reddit_settings(self, message: discord.Message) -> bool:
        try:
            await self.send_or_update_panel_message(
                message,
                format_reddit_settings_summary(self.store, message.channel.id),
                RedditSettingsView(
                    store=self.store,
                    manager=self.watcher_manager,
                    channel_id=message.channel.id,
                    owner_id=message.author.id,
                ),
                panel_key="reddit_settings",
                reuse_cached_message=False,
            )
        except discord.HTTPException as exc:
            print(f"Failed to send reddit settings panel in channel {message.channel.id}: {exc}")
            await message.channel.send("Could not open the Reddit settings panel. Check channel permissions for messages, embeds, and components.")
        return True
