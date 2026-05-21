from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable

import discord

from config import AppConfig
from services import job_service, reddit_service
from state.store import RuntimeStore


class WatcherManager:
    def __init__(self, client: discord.Client, config: AppConfig, store: RuntimeStore) -> None:
        self.client = client
        self.config = config
        self.store = store

        self.channel_job_tasks: dict[int, asyncio.Task[Any]] = {}
        self.channel_reddit_tasks: dict[int, asyncio.Task[Any]] = {}
        self.channel_active_process: dict[int, str] = {}
        self.channel_recent_messages: dict[int, str] = {}

    def dedup_months_threshold(self) -> int:
        return int(getattr(self.config, "dedup_months_threshold", job_service.DEDUP_MONTHS_THRESHOLD))

    def remember_message(self, channel_id: int, content: str) -> None:
        self.channel_recent_messages[channel_id] = content.strip()

    async def should_skip_duplicate_message(self, channel_id: int, channel: Any, content: str) -> bool:
        normalized = content.strip()
        remembered = self.channel_recent_messages.get(channel_id)
        if remembered == normalized:
            return True

        if not hasattr(channel, "history"):
            return False

        try:
            async for recent in channel.history(limit=self.config.discord_history_check_limit):
                if recent is None:
                    continue

                recent_author = getattr(recent, "author", None)
                current_user = getattr(self.client, "user", None)
                if recent_author is None:
                    continue
                if current_user is not None:
                    if getattr(recent_author, "id", None) != getattr(current_user, "id", None):
                        continue
                elif not bool(getattr(recent_author, "bot", False)):
                    continue

                # Only suppress if the most recent bot-authored message is identical.
                if str(getattr(recent, "content", "")).strip() == normalized:
                    self.remember_message(channel_id, normalized)
                    return True
                return False
        except Exception as exc:
            print(f"Watcher dedupe check failed: {exc}")

        return False

    async def _is_duplicate_message(self, channel_id: int, channel: Any, content: str) -> bool:
        listing_file = self.config.base_dir / f".message_listing_{channel_id}.json"
        if job_service.check_and_record_message(
            content,
            listing_file,
            months_threshold=self.dedup_months_threshold(),
        ):
            return True

        return await self.should_skip_duplicate_message(channel_id, channel, content)

    async def send_watcher_message(self, channel_id: int, content: str, check_duplicate: bool = True) -> bool:
        channel = self.client.get_channel(channel_id)
        if channel is None:
            return False

        text = content[:1900]

        if check_duplicate and await self._is_duplicate_message(channel_id, channel, text):
            return False

        await channel.send(text)
        self.remember_message(channel_id, text)
        return True

    async def send_reddit_gallery_message(self, channel_id: int, item: dict[str, Any]) -> bool:
        channel = self.client.get_channel(channel_id)
        if channel is None:
            return False

        title = str(item.get("title") or "(untitled)")
        permalink = str(item.get("permalink") or item.get("link") or "")
        gallery_links = [
            str(url).strip()
            for url in item.get("gallery_links", [])
            if str(url).strip().startswith("http")
        ]
        if len(gallery_links) <= 1:
            return await self.send_watcher_message(channel_id, f"{title}\n{str(item.get('link') or '')}".strip())

        dedupe_key = "\n".join([title, permalink, *gallery_links])

        if await self._is_duplicate_message(channel_id, channel, dedupe_key):
            return False

        embeds: list[discord.Embed] = []
        for index, url in enumerate(gallery_links[:10]):
            if index == 0:
                embed = discord.Embed(title=title, url=permalink or None)
            else:
                embed = discord.Embed(url=permalink or None)
            embed.set_image(url=url)
            embeds.append(embed)

        text = f"{title}\n{permalink}".strip()
        await channel.send(content=(text or None), embeds=embeds)
        self.remember_message(channel_id, dedupe_key)
        return True

    @staticmethod
    def format_job_watcher_message(item: dict[str, Any]) -> str:
        site_label = str(item.get("site_label") or "Unknown site")
        title = str(item.get("title") or "(job)").strip()
        if len(title) > 220:
            title = f"{title[:217].rstrip()}..."

        link = str(item.get("link") or "").strip()

        lines = [f"[{site_label}] {title}"]
        if link:
            lines.append(link)

        apply_link = str(item.get("apply_link") or "").strip()
        if apply_link:
            lines.append(f"Apply: {apply_link}")

        contact_emails = [str(value).strip() for value in item.get("contact_emails", []) if str(value).strip()]
        contact_emails = list(dict.fromkeys(contact_emails))
        if contact_emails:
            lines.append(f"Email: {', '.join(contact_emails[:2])}")

        contact_phones = [str(value).strip() for value in item.get("contact_phones", []) if str(value).strip()]
        contact_phones = list(dict.fromkeys(contact_phones))
        if contact_phones:
            lines.append(f"Phone: {', '.join(contact_phones[:2])}")

        return "\n".join(lines).strip()

    @staticmethod
    def format_reddit_watcher_message(item: dict[str, Any], max_len: int = 1850) -> str:
        title = str(item.get("title") or "(untitled)")
        link = str(item.get("link") or "")
        permalink = str(item.get("permalink") or link)
        gallery_links = [
            str(url).strip()
            for url in item.get("gallery_links", [])
            if str(url).strip().startswith("http")
        ]

        if len(gallery_links) <= 1:
            return f"{title}\n{link}".strip()

        lines = [title, f"Gallery post: {permalink}"]
        for url in gallery_links:
            candidate = "\n".join(lines + [url])
            if len(candidate) > max_len:
                break
            lines.append(url)
        return "\n".join(lines)

    def acquire_channel_process(self, channel_id: int, process_name: str) -> bool:
        current = self.channel_active_process.get(channel_id)
        if current and current != process_name:
            return False
        self.channel_active_process[channel_id] = process_name
        return True

    def release_channel_process(self, channel_id: int, process_name: str) -> None:
        if self.channel_active_process.get(channel_id) == process_name:
            self.channel_active_process.pop(channel_id, None)

    def _finalize_channel_task(
        self,
        task_map: dict[int, asyncio.Task[Any]],
        channel_id: int,
        process_name: str,
        task: asyncio.Task[Any],
    ) -> None:
        current = task_map.get(channel_id)
        if current is task:
            task_map.pop(channel_id, None)
        self.release_channel_process(channel_id, process_name)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"{process_name} stopped for channel {channel_id}: {exc}")

    def create_channel_task(
        self,
        task_map: dict[int, asyncio.Task[Any]],
        channel_id: int,
        process_name: str,
        coroutine: Awaitable[Any],
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine)
        task_map[channel_id] = task
        task.add_done_callback(lambda completed: self._finalize_channel_task(task_map, channel_id, process_name, completed))
        return task

    async def _run_job_watcher(self, channel_id: int) -> None:
        while self.store.get_job_settings(channel_id)["enabled"]:
            settings = self.store.get_job_settings(channel_id)
            try:
                items = await asyncio.to_thread(
                    job_service.scrape_job_postings,
                    settings["sites"],
                    settings["keywords"],
                    settings["location"],
                    self.config.jobspy_python_exe,
                    int(settings["hours_old"]),
                    int(settings["results_wanted"]),
                    int(settings.get("radius_miles", 25)),
                    str(settings.get("country_indeed") or "AUTO"),
                    f"channel:{channel_id}",
                    bool(settings.get("allow_north_america", False)),
                    job_service.effective_jobbank_native_query(str(settings.get("jobbank_native_query") or "")),
                )
                role_filters = list(settings.get("role_filters", []))
                exclusion_terms = list(settings.get("exclusion_terms", []))
                items = [item for item in items if job_service.matches_role_filters(str(item.get("title", "")), role_filters)]
                items = [item for item in items if not job_service.matches_exclusion_terms(item, exclusion_terms)]
                items = [
                    item
                    for item in items
                    if job_service.matches_search_parameters_semantic(
                        item,
                        str(settings.get("keywords") or ""),
                        str(settings.get("location") or ""),
                        role_filters,
                    )
                ]
                seen = self.store.channel_job_seen.setdefault(channel_id, set())
                fresh = [item for item in reversed(items) if item.get("link") and item["link"] not in seen]
                
                # Check final formatted messages for duplicates
                listing_file = self.config.base_dir / f".message_listing_{channel_id}.json"
                for item in fresh:
                    formatted_msg = self.format_job_watcher_message(item)
                    # Check if this exact formatted message is a duplicate
                    if not job_service.check_and_record_message(
                        formatted_msg,
                        listing_file,
                        months_threshold=self.dedup_months_threshold(),
                    ):
                        # Not a duplicate, send it
                        await self.send_watcher_message(channel_id, formatted_msg, check_duplicate=False)
                    # Always track the link to avoid re-scraping it
                    seen.add(item["link"])
                self.store.save()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"Job watcher error for channel {channel_id}: {exc}")

            await asyncio.sleep(max(60, int(settings.get("refresh_seconds", 300))))

    async def _run_reddit_watcher(self, channel_id: int) -> None:
        settings = self.store.get_reddit_settings(channel_id)
        flair_tags = [part.strip() for part in str(settings["flair_tags"]).split(",") if part.strip()]
        state_file = self.config.base_dir / f".reddit_seen_{channel_id}.json"

        async def post_reddit_item(item: dict[str, Any]) -> bool:
            gallery_links = [
                str(url).strip()
                for url in item.get("gallery_links", [])
                if str(url).strip().startswith("http")
            ]
            if len(gallery_links) > 1:
                return await self.send_reddit_gallery_message(channel_id, item)
            return await self.send_watcher_message(channel_id, self.format_reddit_watcher_message(item))

        await reddit_service.poll_subreddit_media(
            subreddit=str(settings["subreddit"]),
            post_callback=post_reddit_item,
            sort=str(settings["sort"]),
            time_filter=str(settings["time_filter"]),
            flair_tags=flair_tags,
            include_nsfw=bool(settings["include_nsfw"]),
            include_spoiler=bool(settings["include_spoiler"]),
            media_mode=str(settings["media_mode"]),
            limit=int(settings["limit"]),
            interval_seconds=max(60, int(settings.get("refresh_seconds", 300))),
            state_file=Path(state_file),
        )

    def start_job_watcher(self, channel_id: int) -> bool:
        existing = self.channel_job_tasks.get(channel_id)
        if existing and not existing.done():
            return True
        if not self.acquire_channel_process(channel_id, "job_watcher"):
            return False

        self.store.update_job_setting(channel_id, "enabled", True)
        self.create_channel_task(self.channel_job_tasks, channel_id, "job_watcher", self._run_job_watcher(channel_id))
        return True

    def stop_job_watcher(self, channel_id: int) -> None:
        self.store.update_job_setting(channel_id, "enabled", False)
        task = self.channel_job_tasks.get(channel_id)
        if task and not task.done():
            task.cancel()
        else:
            self.release_channel_process(channel_id, "job_watcher")

    def start_reddit_watcher(self, channel_id: int) -> bool:
        existing = self.channel_reddit_tasks.get(channel_id)
        if existing and not existing.done():
            return True
        if not self.acquire_channel_process(channel_id, "reddit_watcher"):
            return False

        self.store.update_reddit_setting(channel_id, "enabled", True)
        self.create_channel_task(self.channel_reddit_tasks, channel_id, "reddit_watcher", self._run_reddit_watcher(channel_id))
        return True

    def stop_reddit_watcher(self, channel_id: int) -> None:
        self.store.update_reddit_setting(channel_id, "enabled", False)
        task = self.channel_reddit_tasks.get(channel_id)
        if task and not task.done():
            task.cancel()
        else:
            self.release_channel_process(channel_id, "reddit_watcher")

    def restore_enabled_watchers(self) -> dict[str, int]:
        restored = {"job": 0, "reddit": 0}
        for channel_id, settings in self.store.channel_job_settings.items():
            if settings.get("enabled"):
                if self.start_job_watcher(channel_id):
                    restored["job"] += 1
                else:
                    self.store.update_job_setting(channel_id, "enabled", False)
        for channel_id, settings in self.store.channel_reddit_settings.items():
            if settings.get("enabled"):
                if self.start_reddit_watcher(channel_id):
                    restored["reddit"] += 1
                else:
                    self.store.update_reddit_setting(channel_id, "enabled", False)
        return restored
