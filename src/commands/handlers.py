from __future__ import annotations

import asyncio
import io
import json
import re
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable

import discord

from config import AppConfig
from services import job_service, scrape_service
from services.resumes.configkey import GeminiSettings, load_gemini_settings
from services.resumes.cache import ResumeExplicitCacheManager, ResumeExplicitCacheStatus
from services.resumes.listing import JobContext, extract_job_context_from_message, generate_resume_rewrite
from services.resumes.resume import (
    compile_latex_to_pdf,
    discord_profile_key,
    ensure_profile_cache,
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
CMD_RESUME_CHECK = ".resumecheck"
CMD_CONTINUE = ".more"
CMD_HELLO = ".hi"
CMD_JOB_SETTINGS = ".job"
CMD_JOB_SETTINGS_ALIAS = ".jobs"
CMD_JOBBANK_TEST = ".jtest"
CMD_JOBBANK_FILTERS = ".jfilters"
CMD_REDDIT_SETTINGS = ".reddit"
CMD_REDDIT_SETTINGS_ALIAS = ".rset"
CMD_CLEAR_REDDIT_SEEN = ".rclear"
CMD_RESET_REDDIT_SEEN = ".rreset"
CMD_SCRAPE_SETTINGS = ".scrapecfg"
CMD_SCRAPE_SETTINGS_ALIAS = ".cfg"
CMD_SCRAPE = ".scrape"
PRIMARY_RESUME_SLASH_COMMAND = "/resumebuild"


_COMMAND_ALIASES: dict[str, tuple[str, ...]] = {
    CMD_COMMANDS: ("/cmd", "/commands"),
    CMD_STATUS: ("/st", "/status"),
    CMD_STATUS_ALIAS: ("/watch", "/watcherstatus"),
    CMD_RESUME: (PRIMARY_RESUME_SLASH_COMMAND, "/resume", "/ernestresume", "/res"),
    CMD_RESUME_CHECK: ("/resumecheck",),
    CMD_CONTINUE: ("/more", "/continue"),
    CMD_HELLO: ("/hi", "/hello", "/hello2"),
    CMD_JOB_SETTINGS: ("/job", "/jobsettings"),
    CMD_JOB_SETTINGS_ALIAS: ("/jobs", "/jobsinit"),
    CMD_JOBBANK_TEST: ("/jtest", "/jobbanktest"),
    CMD_JOBBANK_FILTERS: ("/jfilters", "/jobbankfilters"),
    CMD_REDDIT_SETTINGS: ("/reddit", "/redditsettings"),
    CMD_REDDIT_SETTINGS_ALIAS: ("/rset",),
    CMD_CLEAR_REDDIT_SEEN: ("/rclear", "/clearredditseen"),
    CMD_RESET_REDDIT_SEEN: ("/rreset", "/resetredditseen"),
    CMD_SCRAPE_SETTINGS: ("/scrapecfg", "/settings"),
    CMD_SCRAPE_SETTINGS_ALIAS: ("/cfg",),
    CMD_SCRAPE: ("/scrape",),
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
                f"`{CMD_JOBBANK_TEST}` - Run Job Bank listing test\n"
                f"`{CMD_JOBBANK_FILTERS} [query|clear]` - Additional native filters\n"
                f"`{CMD_STATUS}` - Watcher status"
            ),
            inline=False,
        )
        embed.add_field(
            name="Utility",
            value=(
                f"`{CMD_COMMANDS}` - Show general command sheet\n"
                f"`{CMD_RESUME}` - Reply to a job post for tailored resume\n"
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
            f"`{CMD_RESUME}` - Reply to a job post and generate a tailored resume draft\n"
            f"`{CMD_RESUME_CHECK}` - Compile your current template cache\n"
            f"`{CMD_CONTINUE}` - Continue paginated output\n"
            f"`{CMD_HELLO}` - Quick hello test"
        ),
        inline=False,
    )
    embed.add_field(
        name="Job Watcher",
        value=(
            f"`{CMD_JOB_SETTINGS}` / `{CMD_JOB_SETTINGS_ALIAS}` - Open job settings\n"
            f"`{CMD_JOBBANK_TEST}` - Run Job Bank listing test\n"
            f"`{CMD_JOBBANK_FILTERS} [query|clear]` - Additional native filters"
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


def _extract_command_payload(content: str, command: str) -> str:
    stripped = content.strip()
    for alias in _expand_command_aliases((command,)):
        if _command_matches(stripped, alias, normalize=True):
            return stripped[len(alias) :].strip()
    return ""


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
    def __init__(self, client: discord.Client, config: AppConfig, store: RuntimeStore, watcher_manager: WatcherManager) -> None:
        self.client = client
        self.config = config
        self.store = store
        self.watcher_manager = watcher_manager
        self.resume_cache_manager = ResumeExplicitCacheManager(
            profiles_dir=config.resume_profiles_dir,
            cache_dir=config.resume_cache_dir,
        )
        self.channel_continuations: dict[int, list[str]] = {}
        self.channel_panel_messages: dict[tuple[int, str], discord.Message] = {}

        self.handlers: tuple[MessageHandler, ...] = (
            self.handle_commands,
            self.handle_resume,
            self.handle_resume_check,
            self.handle_status,
            self.handle_jobbank_test,
            self.handle_jobbank_filters,
            self.handle_clear_reddit_seen,
            self.handle_reddit_settings,
            self.handle_hello,
            self.handle_scrape_settings,
            self.handle_job_settings,
            self.handle_continue,
            self.handle_scrape,
        )

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

        def _is_cheatsheet(message: Any) -> bool:
            embed = message.embeds[0] if getattr(message, "embeds", None) else None
            if embed is None:
                return False
            if embed.title != expected_title or embed.description != expected_description:
                return False
            current_user = getattr(self.client, "user", None)
            if current_user is None:
                return True
            return getattr(getattr(message, "author", None), "id", None) == getattr(current_user, "id", None)

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

        pinned_messages: list[Any] = []
        if hasattr(channel, "pins"):
            try:
                pinned_messages = await channel.pins()
            except Exception as exc:
                print(f"Could not read pinned messages in channel {channel_id}: {exc}")

        for message in pinned_messages:
            if _is_cheatsheet(message):
                return

        # If pinning is unavailable, avoid repost spam by accepting an existing unpinned cheat sheet.
        if not can_pin and hasattr(channel, "history"):
            try:
                async for recent in channel.history(limit=50):
                    if _is_cheatsheet(recent):
                        return
            except Exception as exc:
                print(f"Could not read history in channel {channel_id}: {exc}")

        try:
            sent_message = await channel.send(embed=build_commands_cheatsheet_embed(sheet_kind=sheet_kind))
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
        return f"Only the server owner can use `{CMD_RESUME}` / `{PRIMARY_RESUME_SLASH_COMMAND}`."

    def owner_profile_key(self, guild_owner_id: int | None) -> int | str | None:
        return self.config.main_user_profile_key or guild_owner_id

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

    async def resolve_resume_context(self, message: discord.Message) -> JobContext | None:
        referenced = await self.resolve_referenced_message(message)
        if referenced is None:
            await message.channel.send(f"Reply to a job post message with `{CMD_RESUME}` or `{PRIMARY_RESUME_SLASH_COMMAND}`.")
            return None

        referenced_author = getattr(referenced, "author", None)
        if not bool(getattr(referenced_author, "bot", False)):
            await message.channel.send(
                f"Reply to an ErnestBot job listing message with `{CMD_RESUME}` or `{PRIMARY_RESUME_SLASH_COMMAND}`."
            )
            return None

        content = str(getattr(referenced, "content", "")).strip()
        if not content:
            await message.channel.send("The replied message does not contain a text job post to parse.")
            return None

        listing_lines = [line.strip() for line in content.splitlines() if line.strip()]
        first_line = listing_lines[0] if listing_lines else ""
        has_listing_header = first_line.startswith("[") and "]" in first_line
        has_url = "http://" in content or "https://" in content
        if not (has_listing_header and has_url):
            await message.channel.send(
                f"Reply to an ErnestBot job listing message with `{CMD_RESUME}` or `{PRIMARY_RESUME_SLASH_COMMAND}`."
            )
            return None

        job = extract_job_context_from_message(content)
        if job is None:
            await message.channel.send("Could not find a job title and URL in the replied message.")
            return None

        return job

    @command_handler(CMD_RESUME, normalize=True)
    async def handle_resume(self, message: discord.Message) -> bool:
        guild_owner_id = getattr(getattr(message, "guild", None), "owner_id", None)
        if guild_owner_id is None:
            await message.channel.send(self.build_resume_guild_only_message())
            return True


        # Allow both the watcher owner and the server owner to use /resumebuild
        channel_id = message.channel.id
        job_settings = self.store.get_job_settings(channel_id)
        watcher_owner_id = job_settings.get("owner_id")
        if watcher_owner_id is not None:
            allowed_ids = {guild_owner_id, watcher_owner_id}
        else:
            allowed_ids = {guild_owner_id}
        if message.author.id not in allowed_ids:
            await message.channel.send(self.build_resume_unauthorized_message())
            return True

        settings = load_gemini_settings(self.config)
        if not (settings.api_key or settings.openrouter_api_key or settings.groq_api_key):
            await message.channel.send(self.build_resume_missing_key_message())
            return True

        job = await self.resolve_resume_context(message)
        if job is None:
            return True

        cache_root = Path(__file__).resolve().parents[1] / "services" / "resumes" / "resumes_cache"
        profile_key = discord_profile_key(message.author.id, self._author_profile_name(message.author))

        # Explicitly check for user resume cache folder and required files
        try:
            profile_dir = await asyncio.to_thread(
                ensure_profile_cache,
                profile_key,
                cache_root,
                self.owner_profile_key(guild_owner_id),
            )
        except FileNotFoundError as exc:
            await message.channel.send(f"Resume cache setup failed: {exc}",
                                       reference=message.to_reference(fail_if_not_exists=False),
                                       mention_author=False,
                                       allowed_mentions=discord.AllowedMentions.none())
            return True

        try:
            profile_key = resolve_discord_profile_key(
                message.author.id,
                self._author_profile_name(message.author),
                cache_root,
            )
        except FileNotFoundError as exc:
            await message.channel.send(
                f"Resume cache error: {exc}",
                reference=message.to_reference(fail_if_not_exists=False),
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return True

        # Defensive: check all required files exist
        required_files = ["baseinfo.txt", "instructions.txt", "template.tex"]
        missing = [f for f in required_files if not (profile_dir / f).exists()]
        if missing:
            await message.channel.send(
                f"Your resume cache folder `{profile_dir.name}` is missing required file(s): {', '.join(missing)}. "
                "Please add them and try again.",
                reference=message.to_reference(fail_if_not_exists=False),
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return True

        template_path = profile_dir / "template.tex"
        compile_log_dir = self.config.resume_cache_dir / "compile_logs"
        template_log_path = compile_log_dir / f"{profile_dir.name}.log"
        reply_reference = message.to_reference(fail_if_not_exists=False)
        reply_send_kwargs = {
            "reference": reply_reference,
            "mention_author": False,
            "allowed_mentions": discord.AllowedMentions.none(),
        }


        # Use a user-specific ResumeExplicitCacheManager for this request
        user_resume_cache_manager = ResumeExplicitCacheManager(
            profiles_dir=profile_dir,
            cache_dir=self.config.resume_cache_dir / f".cache_{message.author.id}",
        )

        if settings.api_key:
            cache_status = await asyncio.to_thread(user_resume_cache_manager.ensure_cache, settings)
        else:
            cache_status = ResumeExplicitCacheStatus(
                status="skipped",
                message="Gemini cache skipped: no Gemini API key configured.",
            )

        rewrite_result = await asyncio.to_thread(
            generate_resume_rewrite,
            settings,
            job,
            cache_status.cache_name,
            [profile_dir / "baseinfo.txt"],
            [profile_dir / "instructions.txt"],
            template_path,
        )

        if rewrite_result.status != "ok" or not rewrite_result.rewritten_resume:
            await message.channel.send(f"Resume generation failed: {rewrite_result.message}", **reply_send_kwargs)
            return True

        if not rewrite_result.latex_document:
            await message.channel.send("Resume generation did not return a LaTeX document.", **reply_send_kwargs)
            return True

        compile_result = await asyncio.to_thread(
            compile_latex_to_pdf,
            rewrite_result.latex_document,
            job.title,
            template_path,
            template_log_path,
            self.config.resume_normalize_json_latex,
        )
        if compile_result.status == "ok" and compile_result.pdf_bytes and compile_result.pdf_name:
            used_provider = rewrite_result.used_provider or "gemini"
            if used_provider == "openrouter":
                used_model = settings.openrouter_model
            elif used_provider == "groq":
                used_model = settings.groq_model
            else:
                used_model = settings.model
            await message.channel.send(
                content=f"Compiled PDF using `{used_provider}` (`{used_model}`).",
                file=discord.File(io.BytesIO(compile_result.pdf_bytes), filename=compile_result.pdf_name),
                **reply_send_kwargs,
            )
        else:
            error_msg = f"PDF compile failed: {compile_result.message}"
            if compile_result.log_excerpt:
                error_msg += f"\n```\n{compile_result.log_excerpt[-500:]}\n```"
            await message.channel.send(error_msg, **reply_send_kwargs)
        return True

    @command_handler(CMD_RESUME_CHECK, normalize=True)
    async def handle_resume_check(self, message: discord.Message) -> bool:
        guild_owner_id = getattr(getattr(message, "guild", None), "owner_id", None)
        if guild_owner_id is None:
            await message.channel.send(f"`{CMD_RESUME_CHECK}` is only available in a server channel.")
            return True

        if message.author.id != guild_owner_id:
            await message.channel.send(f"Only the server owner can use `{CMD_RESUME_CHECK}`.")
            return True

        cache_root = Path(__file__).resolve().parents[1] / "services" / "resumes" / "resumes_cache"
        profile_key = discord_profile_key(message.author.id, self._author_profile_name(message.author))
        profile_dir = await asyncio.to_thread(
            ensure_profile_cache,
            profile_key,
            cache_root,
            self.owner_profile_key(guild_owner_id),
        )
        try:
            profile_key = resolve_discord_profile_key(
                message.author.id,
                self._author_profile_name(message.author),
                cache_root,
            )
        except FileNotFoundError as exc:
            await message.channel.send(
                f"Resume cache error: {exc}",
                reference=message.to_reference(fail_if_not_exists=False),
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return True
        profile_dir = cache_root / profile_key
        template_path = profile_dir / "template.tex"
        compile_log_dir = self.config.resume_cache_dir / "compile_logs"
        template_log_path = compile_log_dir / f"{profile_dir.name}.resumecheck.log"

        try:
            template_text = await asyncio.to_thread(template_path.read_text, encoding="utf-8")
        except OSError as exc:
            await message.channel.send(f"Could not read current template for `{profile_dir.name}`: {exc}")
            return True

        if not template_text.strip():
            await message.channel.send("Current template is empty. Update `template.tex` first, then run `/resumecheck`.")
            return True

        compile_result = await asyncio.to_thread(
            compile_latex_to_pdf,
            template_text,
            f"{profile_dir.name}-resume-check",
            template_path,
            template_log_path,
            self.config.resume_normalize_json_latex,
        )

        reply_reference = message.to_reference(fail_if_not_exists=False)
        reply_send_kwargs = {
            "reference": reply_reference,
            "mention_author": False,
            "allowed_mentions": discord.AllowedMentions.none(),
        }

        if compile_result.status == "ok" and compile_result.pdf_bytes and compile_result.pdf_name:
            await message.channel.send(
                content=f"Resume check compile succeeded for profile `{profile_dir.name}`.",
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

    @command_handler(CMD_JOBBANK_TEST, normalize=True)
    async def handle_jobbank_test(self, message: discord.Message) -> bool:

        settings = self.store.get_job_settings(message.channel.id)
        keywords = str(settings.get("keywords") or "").strip()
        location = str(settings.get("location") or "").strip()

        used_sample_fallback = False
        if not keywords:
            keywords = "jobs"
            used_sample_fallback = True
        if not location:
            location = "Canada"
            used_sample_fallback = True

        native_query = job_service.effective_jobbank_native_query(str(settings.get("jobbank_native_query") or ""))

        requested = max(1, min(int(settings.get("results_wanted") or 10), 10))
        items = await asyncio.to_thread(
            job_service.scrape_job_postings,
            [job_service.JOBBANK_CANADA_SITE],
            keywords,
            location,
            self.config.jobspy_python_exe,
            int(settings.get("hours_old") or 72),
            requested,
            int(settings.get("radius_miles") or 25),
            str(settings.get("country_indeed") or "AUTO"),
            "command:jobbanktest",
            bool(settings.get("allow_north_america", False)),
            native_query,
        )

        source_label = "sample params" if used_sample_fallback else "watcher params"
        native_count = len(native_query)
        if not items:
            await message.channel.send(
                "Job Bank test returned no listings "
                f"using {source_label}: '{keywords}' in '{location}' with {native_count} native filter key(s)."
            )
            return True

        lines = [
            f"Job Bank test results ({len(items)} listing(s), {source_label}, native filters={native_count}):"
        ]
        for item in items:
            lines.append(self.watcher_manager.format_job_watcher_message(item))

        await message.channel.send(self.set_continuation(message.channel.id, "\n\n".join(lines)))
        return True

    @command_handler(CMD_JOBBANK_FILTERS, normalize=True)
    async def handle_jobbank_filters(self, message: discord.Message) -> bool:

        payload = _extract_command_payload(message.content, CMD_JOBBANK_FILTERS)
        current = str(self.store.get_job_settings(message.channel.id).get("jobbank_native_query") or "")
        if not payload:
            if current:
                await message.channel.send(
                    "Current additional Job Bank native filters: "
                    f"`{current}`\nDefault Job Bank native filters are always applied."
                )
            else:
                await message.channel.send(
                    "No additional Job Bank native filters set. Default Job Bank native filters are still applied. "
                    f"Use `{CMD_JOBBANK_FILTERS} <query>` (without URL), for example: "
                    f"`{CMD_JOBBANK_FILTERS} fgeo=27234&fn21=21231`."
                )
            return True

        if payload.lower() in {"clear", "reset", "none"}:
            self.store.update_job_setting(message.channel.id, "jobbank_native_query", "")
            await message.channel.send("Cleared additional Job Bank native filters for this channel (defaults remain active).")
            return True

        query_text = payload[1:] if payload.startswith("?") else payload
        self.store.update_job_setting(message.channel.id, "jobbank_native_query", query_text)
        await message.channel.send(
            "Saved additional Job Bank native filters for this channel: "
            f"`{query_text}`\nDefaults remain active and will be merged with these."
        )
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
                items = await asyncio.to_thread(
                    job_service.scrape_jobs_from_board_url,
                    url,
                    self.config.jobspy_python_exe,
                    int(settings["max_items"]),
                )
            else:
                items = await asyncio.to_thread(
                    scrape_service.scrape_url,
                    url,
                    selector,
                    "Mozilla/5.0 (compatible; RebuiltScraper/1.0)",
                    int(settings["timeout_seconds"]),
                    3,
                    int(settings["max_items"]),
                    False,
                )

            if settings["use_ai_cleanup"] and self.config.openrouter_key:
                try:
                    result = await asyncio.to_thread(
                        scrape_service.clean_with_openrouter,
                        url,
                        items,
                        self.config.openrouter_key,
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
            resume_profile_dir = await asyncio.to_thread(
                ensure_profile_cache,
                profile_key,
                cache_root,
                self.owner_profile_key(guild_owner_id),
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
