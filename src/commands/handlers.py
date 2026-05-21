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
from services.resumes.resume import compile_latex_to_pdf, ensure_profile_cache
from state.store import MODE_DESCRIPTIONS, RuntimeStore
from ui.views import (
    JobSettingsView,
    ModeDropdownView,
    RedditSettingsView,
    ScrapeSettingsView,
    format_job_settings_summary,
    format_mode_summary,
    format_reddit_settings_summary,
    format_scrape_settings_summary,
)
from watchers.manager import WatcherManager

MessageHandler = Callable[[discord.Message], Awaitable[bool]]


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
    match_prefixes = tuple(prefix.lower() for prefix in prefixes) if normalize else prefixes

    def decorator(func: MessageHandler) -> MessageHandler:
        @wraps(func)
        async def wrapper(self: CommandRouter, message: discord.Message) -> bool:
            raw_content = message.content
            match_content = raw_content.strip().lower() if normalize else raw_content
            if not any(match_content.startswith(prefix) for prefix in match_prefixes):
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
        if config.main_user_profile_key is not None:
            ensure_profile_cache(config.main_user_profile_key, seed_profile_key=config.main_user_profile_key)
        self.channel_continuations: dict[int, list[str]] = {}
        self.channel_panel_messages: dict[tuple[int, str], discord.Message] = {}

        self.handlers: tuple[MessageHandler, ...] = (
            self.handle_commands,
            self.handle_resume,
            self.handle_status,
            self.handle_jobbank_test,
            self.handle_jobbank_filters,
            self.handle_clear_reddit_seen,
            self.handle_reddit_settings,
            self.handle_hello,
            self.handle_modes,
            self.handle_mode,
            self.handle_scrape_settings,
            self.handle_job_settings,
            self.handle_continue,
            self.handle_scrape,
        )

    @command_handler("$commands", normalize=True)
    async def handle_commands(self, message: discord.Message) -> bool:

        embed = discord.Embed(
            title="Bot Commands",
            description="Quick reference for this channel.",
            color=discord.Color.blurple(),
        )
        embed.add_field(
            name="General",
            value=(
                "`$commands` - Show this list\n"
                "`$status` / `$watcherstatus` - Watcher status\n"
                "`$modes` - Open mode picker\n"
                "`$mode [name]` - Show or set mode\n"
                "`$resume` - Reply to a job post and generate a tailored resume draft\n"
                "`$continue` - Continue paginated output\n"
                "`$hello2` - Quick hello test"
            ),
            inline=False,
        )
        embed.add_field(
            name="Job Watcher",
            value=(
                "`$jobsettings` / `$jobsinit` - Open job settings\n"
                "`$jobbanktest` - Run Job Bank listing test\n"
                "`$jobbankfilters [query|clear]` - Additional native filters"
            ),
            inline=False,
        )
        embed.add_field(
            name="Reddit Watcher",
            value=(
                "`$redditsettings` - Open Reddit settings\n"
                "`$redditinit` - Deprecated alias for `$redditsettings`\n"
                "`$clearredditseen` / `$resetredditseen` - Clear seen IDs"
            ),
            inline=False,
        )
        embed.add_field(
            name="Scraping",
            value=(
                "`$scrapesettings` / `$settings` - Open scrape settings\n"
                "`$scrape <url> [| selector]` - Run one-time scrape"
            ),
            inline=False,
        )
        await message.channel.send(embed=embed)
        return True

    @command_handler("$status", "$watcherstatus", normalize=True)
    async def handle_status(self, message: discord.Message) -> bool:

        channel_id = message.channel.id
        mode = self.store.get_mode(channel_id)

        job_enabled = bool(self.store.get_job_settings(channel_id).get("enabled"))
        reddit_enabled = bool(self.store.get_reddit_settings(channel_id).get("enabled"))

        job_task = self.watcher_manager.channel_job_tasks.get(channel_id)
        reddit_task = self.watcher_manager.channel_reddit_tasks.get(channel_id)
        job_running = bool(job_task and not job_task.done())
        reddit_running = bool(reddit_task and not reddit_task.done())
        active_process = self.watcher_manager.channel_active_process.get(channel_id, "none")

        lines = [
            f"Status for <#{channel_id}>",
            f"mode: `{mode}`",
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
        return "`$resume` is only available in a server channel."

    def build_resume_unauthorized_message(self) -> str:
        return "Only the server owner can use `$resume`."

    def owner_profile_key(self, guild_owner_id: int | None) -> int | str | None:
        return self.config.main_user_profile_key or guild_owner_id

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
            await message.channel.send("Reply to a job post message with `$resume`.")
            return None

        referenced_author = getattr(referenced, "author", None)
        if not bool(getattr(referenced_author, "bot", False)):
            await message.channel.send("Reply to an ErnestBot job listing message with `$resume`.")
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
            await message.channel.send("Reply to an ErnestBot job listing message with `$resume`.")
            return None

        job = extract_job_context_from_message(content)
        if job is None:
            await message.channel.send("Could not find a job title and URL in the replied message.")
            return None

        return job

    @command_handler("$resume", normalize=True)
    async def handle_resume(self, message: discord.Message) -> bool:
        guild_owner_id = getattr(getattr(message, "guild", None), "owner_id", None)
        if guild_owner_id is None:
            await message.channel.send(self.build_resume_guild_only_message())
            return True

        if message.author.id != guild_owner_id:
            await message.channel.send(self.build_resume_unauthorized_message())
            return True

        settings = load_gemini_settings(self.config)
        if not (settings.api_key or settings.openrouter_api_key or settings.groq_api_key):
            await message.channel.send(self.build_resume_missing_key_message())
            return True

        job = await self.resolve_resume_context(message)
        if job is None:
            return True

        profile_key: int | str = message.author.id
        if message.author.id == guild_owner_id:
            profile_key = self.owner_profile_key(guild_owner_id) or message.author.id

        profile_dir = await asyncio.to_thread(
            ensure_profile_cache,
            profile_key,
            Path(__file__).resolve().parents[1] / "services" / "resumes" / "resumes_cache",
            self.owner_profile_key(guild_owner_id),
        )
        template_path = profile_dir / "template.tex"
        compile_log_dir = self.config.resume_cache_dir / "compile_logs"
        template_log_path = compile_log_dir / f"{profile_dir.name}.log"
        reply_reference = message.to_reference(fail_if_not_exists=False)
        reply_send_kwargs = {
            "reference": reply_reference,
            "mention_author": False,
            "allowed_mentions": discord.AllowedMentions.none(),
        }

        if settings.api_key:
            cache_status = await asyncio.to_thread(self.resume_cache_manager.ensure_cache, settings)
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

    @command_handler("$clearredditseen", "$resetredditseen", normalize=True)
    async def handle_clear_reddit_seen(self, message: discord.Message) -> bool:

        state_file = self.config.base_dir / f".reddit_seen_{message.channel.id}.json"
        state_file.write_text(json.dumps({"seen_ids": []}, ensure_ascii=True), encoding="utf-8")
        await message.channel.send("Cleared reddit seen list for this channel.")
        return True

    @command_handler("$jobbanktest", normalize=True)
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

    @command_handler("$jobbankfilters", normalize=True)
    async def handle_jobbank_filters(self, message: discord.Message) -> bool:

        payload = message.content[len("$jobbankfilters") :].strip()
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
                    "Use `$jobbankfilters <query>` (without URL), for example: `$jobbankfilters fgeo=27234&fn21=21231`."
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

    def set_continuation(self, channel_id: int, text: str) -> str:
        chunks = scrape_service.chunk_text_for_discord(text)
        if len(chunks) <= 1:
            self.channel_continuations.pop(channel_id, None)
            return chunks[0]
        self.channel_continuations[channel_id] = chunks[1:]
        return f"{chunks[0]}\n\nUse `$continue` for more ({len(chunks) - 1} part(s) left)."

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

    @command_handler("$hello2")
    async def handle_hello(self, message: discord.Message) -> bool:
        await message.channel.send("Hello!")
        return True

    @command_handler("$modes")
    async def handle_modes(self, message: discord.Message) -> bool:

        await message.channel.send(
            "Select a mode from the dropdown:",
            view=ModeDropdownView(store=self.store, channel_id=message.channel.id, owner_id=message.author.id),
        )
        return True

    @command_handler("$mode")
    async def handle_mode(self, message: discord.Message) -> bool:

        payload = message.content[len("$mode") :].strip().lower()
        if not payload:
            await message.channel.send(format_mode_summary(self.store, message.channel.id))
            return True

        if payload not in MODE_DESCRIPTIONS:
            await message.channel.send("Unknown mode. Use `$mode` or `$modes`.")
            return True

        self.store.set_mode(message.channel.id, payload)
        await message.channel.send(f"Mode set to `{payload}` for this channel.")
        return True

    @command_handler("$scrape")
    async def handle_scrape(self, message: discord.Message) -> bool:

        try:
            url, selector = scrape_service.parse_scrape_command(message.content)
        except ValueError:
            await message.channel.send("Usage: $scrape <url> or $scrape <url> | <css selector>")
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

    @command_handler("$continue")
    async def handle_continue(self, message: discord.Message) -> bool:

        pending = self.channel_continuations.get(message.channel.id, [])
        if not pending:
            await message.channel.send("No continuation is pending for this channel.")
            return True

        chunk = pending.pop(0)
        if pending:
            chunk = f"{chunk}\n\nUse `$continue` for more ({len(pending)} part(s) left)."
        else:
            self.channel_continuations.pop(message.channel.id, None)
        await message.channel.send(chunk)
        return True

    @command_handler("$scrapesettings", "$settings")
    async def handle_scrape_settings(self, message: discord.Message) -> bool:

        await self.send_or_update_panel_message(
            message,
            format_scrape_settings_summary(self.store, message.channel.id),
            ScrapeSettingsView(store=self.store, channel_id=message.channel.id, owner_id=message.author.id),
            panel_key="scrape_settings",
        )
        return True

    @command_handler("$jobsettings", "$jobsinit")
    async def handle_job_settings(self, message: discord.Message) -> bool:

        try:
            guild_owner_id = getattr(getattr(message, "guild", None), "owner_id", None)
            profile_key: int | str = message.author.id
            if guild_owner_id is not None and message.author.id == guild_owner_id:
                profile_key = self.owner_profile_key(guild_owner_id) or message.author.id
            resume_profile_dir = await asyncio.to_thread(
                ensure_profile_cache,
                profile_key,
                Path(__file__).resolve().parents[1] / "services" / "resumes" / "resumes_cache",
                self.owner_profile_key(guild_owner_id),
            )

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

    @command_handler("$redditsettings", "$redditinit", normalize=True)
    async def handle_reddit_settings(self, message: discord.Message) -> bool:
        normalized_content = message.content.strip().lower()

        if normalized_content.startswith("$redditinit"):
            await message.channel.send("`$redditinit` is deprecated. Use `$redditsettings`.")
            return True

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
