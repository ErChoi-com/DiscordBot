from __future__ import annotations

import asyncio
import contextlib
import math
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

_T = TypeVar("_T")

import discord

from config import AppConfig
from services import job_service, reddit_service
from services.ats_service import ATS_PLATFORMS as _ATS_PLATFORMS, BAMBOOHR as _BAMBOOHR, scrape_ats_platform as _scrape_ats_platform
from services.health import WatcherHealthTracker
from services.priority_scheduler import BACKGROUND, PriorityWorkScheduler
from services import scheduler_labels
from state.store import RuntimeStore


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _sync_geonames() -> bool:
    """Refresh data/geonames_raw and rebuild geo.db when new data is published.

    A no-op on the vast majority of runs: the script compares the published
    manifest against the local one and exits immediately when they match.

    Off when GEONAMES_AUTO_SYNC is set to 0/false -- a host on a metered link,
    or one that manages data/ itself, should be able to opt out. Never raises:
    stale geo data degrades location matching, it does not break the scrape,
    and this must not be able to kill the loop that calls it.
    """
    if os.environ.get("GEONAMES_AUTO_SYNC", "1").strip().lower() in {"0", "false", "no"}:
        return False
    script = Path(__file__).resolve().parents[2] / "sync_geonames.py"
    if not script.exists():
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=900,
        )
    except Exception as exc:
        print(f"[geonames-sync] skipped: {exc}")
        return False
    # The script prefixes its own lines, so echo the last one verbatim.
    tail = (result.stdout or result.stderr or "").strip().splitlines()
    if tail:
        print(tail[-1])
    return result.returncode == 0


# Where the ATS fleet rotation left off, per platform. Matches the existing
# `.bot_state*.json` ignore rule because that is what it is -- runtime state,
# not data: losing it costs one repeated rotation, never a missed company.
_ATS_ROTATION_STATE = ".bot_state.ats_rotation.json"

# Platforms are scraped concurrently and share one state file, so the
# read-modify-write has to be serialised or two platforms landing together lose
# one of the two cursors.
_ATS_ROTATION_LOCK = threading.Lock()


class WatcherManager:
    def __init__(
        self,
        client: discord.Client,
        config: AppConfig,
        store: RuntimeStore,
        health: WatcherHealthTracker,
        scheduler: PriorityWorkScheduler | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.store = store
        self.health = health
        # Callers (app.py) should pass a shared scheduler so interactive
        # commands and background watcher work actually arbitrate against
        # each other; a private one here is only a safety net for callers
        # (tests) that don't care about cross-domain prioritization.
        self.scheduler = scheduler or PriorityWorkScheduler()

        self.channel_job_tasks: dict[int, asyncio.Task[Any]] = {}
        self.channel_reddit_tasks: dict[int, asyncio.Task[Any]] = {}
        self.channel_active_process: dict[int, str] = {}
        self.channel_recent_messages: dict[int, str] = {}
        self.channel_listing_files: dict[tuple[int, str], Path] = {}
        self.channel_send_locks: dict[tuple[int, str], asyncio.Lock] = {}
        self.cheatsheet_ensurer: Callable[[int, str], Awaitable[None]] | None = None
        self._ats_scrape_task: asyncio.Task[Any] | None = None
        self._watchdog_task: asyncio.Task[Any] | None = None

        self._active_send_cycles: int = 0
        self._send_idle_event: asyncio.Event = asyncio.Event()
        self._send_idle_event.set()

        self._active_work_cycles: int = 0
        self._work_idle_event: asyncio.Event = asyncio.Event()
        self._work_idle_event.set()

    @contextlib.asynccontextmanager
    async def work_guard(self):
        """Acquire while any significant work is running. Shutdown waits for all guards to release."""
        self._active_work_cycles += 1
        self._work_idle_event.clear()
        try:
            yield
        finally:
            self._active_work_cycles = max(0, self._active_work_cycles - 1)
            if self._active_work_cycles == 0:
                self._work_idle_event.set()

    async def _tracked_to_thread(
        self,
        fn: Callable[..., _T],
        *args: Any,
        cost: float | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> _T:
        """Run fn on the shared priority scheduler (background tier) and hold
        a work_guard for the duration. Pass `label` so the scheduler can
        derive `cost` from that label's actual measured median duration
        (rolling two-day window) instead of a static guess; pass an explicit
        `cost` only to override that."""
        async with self.work_guard():
            return await self.scheduler.run(fn, *args, tier=BACKGROUND, cost=cost, label=label, **kwargs)

    async def drain_active_work(self, timeout: float = 120.0) -> bool:
        """Block until all work_guard holders have released. Returns False on timeout or cancellation."""
        if self._work_idle_event.is_set():
            return True
        print(f"[shutdown] Waiting up to {timeout:.0f}s for {self._active_work_cycles} active operation(s) to finish...")
        try:
            await asyncio.wait_for(self._work_idle_event.wait(), timeout=timeout)
            return True
        except BaseException:
            return False

    def _enter_send_cycle(self) -> None:
        self._active_send_cycles += 1
        self._send_idle_event.clear()

    def _exit_send_cycle(self) -> None:
        self._active_send_cycles = max(0, self._active_send_cycles - 1)
        if self._active_send_cycles == 0:
            self._send_idle_event.set()

    async def drain_active_sends(self, timeout: float = 30.0) -> bool:
        """Block until no watcher is mid-send. Returns False on timeout."""
        if self._send_idle_event.is_set():
            return True
        print(f"[shutdown] Waiting up to {timeout:.0f}s for {self._active_send_cycles} active send cycle(s)...")
        try:
            await asyncio.wait_for(self._send_idle_event.wait(), timeout=timeout)
            return True
        except BaseException:
            return False

    def set_cheatsheet_ensurer(self, ensurer: Callable[[int, str], Awaitable[None]] | None) -> None:
        self.cheatsheet_ensurer = ensurer

    def _schedule_cheatsheet_check(self, channel_id: int, sheet_kind: str) -> None:
        if self.cheatsheet_ensurer is None:
            return
        asyncio.create_task(self.cheatsheet_ensurer(channel_id, sheet_kind))

    def dedup_months_threshold(self) -> int:
        return int(getattr(self.config, "dedup_months_threshold", job_service.DEDUP_MONTHS_THRESHOLD))

    def _dedup_listing_file(self, channel_id: int, watcher_type: str) -> Path:
        """Canonical per-watcher listing key used to derive dedup storage namespace."""
        return self.config.base_dir / f".message_listing_{channel_id}_{watcher_type}.json"

    def _attached_listing_file(self, channel_id: int, watcher_type: str) -> Path:
        """Return channel/watcher dedup listing file attached at watcher startup."""
        key = (channel_id, watcher_type)
        attached = self.channel_listing_files.get(key)
        if attached is not None:
            return attached
        fallback = self._dedup_listing_file(channel_id, watcher_type)
        self.channel_listing_files[key] = fallback
        return fallback

    def _ensure_watcher_dedup_directory(self, channel_id: int, watcher_type: str) -> None:
        listing_file = self._attached_listing_file(channel_id, watcher_type)
        try:
            job_service.ensure_dedup_directory_for_listing_file(listing_file)
            changed = job_service.normalize_dedup_storage_for_listing_file(listing_file)
            if changed:
                print(f"Watcher dedup normalized for {channel_id}/{watcher_type}")
        except Exception as exc:
            print(f"Watcher dedup directory init failed for {channel_id}/{watcher_type}: {exc}")

    async def _channel_exists(self, channel_id: int) -> bool:
        cached = self.client.get_channel(channel_id)
        if cached is not None:
            return True
        try:
            fetched = await self.client.fetch_channel(channel_id)
        except discord.NotFound:
            return False
        except discord.Forbidden:
            return False
        except discord.HTTPException as exc:
            # Fail open on transient API/network issues to avoid purging valid watchers.
            print(f"Channel existence probe failed for {channel_id}: {exc}")
            return True
        except Exception as exc:
            # Unexpected failures should not trigger destructive state cleanup.
            print(f"Unexpected channel probe failure for {channel_id}: {exc}")
            return True
        if fetched is None:
            return False
        return True

    def _purge_deleted_job_channel(self, channel_id: int) -> None:
        self.store.remove_job_channel(channel_id)
        self.channel_job_tasks.pop(channel_id, None)
        self.channel_active_process.pop(channel_id, None)
        self.channel_recent_messages.pop((channel_id, "job"), None)
        listing_file = self._attached_listing_file(channel_id, "job")
        self.channel_listing_files.pop((channel_id, "job"), None)
        dedup_dir = job_service.dedup_directory_for_listing_file(listing_file)
        if dedup_dir.exists():
            try:
                for child in dedup_dir.glob("*"):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                dedup_dir.rmdir()
            except Exception as exc:
                print(f"Could not remove dedup directory for deleted channel {channel_id}: {exc}")


    def remember_message(self, channel_id: int, content: str, watcher_type: str = "job") -> None:
        key = (channel_id, watcher_type)
        self.channel_recent_messages[key] = content.strip()

    async def should_skip_duplicate_message(self, channel_id: int, channel: Any, content: str, watcher_type: str = "job") -> bool:
        normalized = content.strip()
        key = (channel_id, watcher_type)
        remembered = self.channel_recent_messages.get(key)
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
                    self.remember_message(channel_id, normalized, watcher_type=watcher_type)
                    return True
                return False
        except Exception as exc:
            print(f"Watcher dedupe check failed: {exc}")


        return False

    async def _is_duplicate_message(self, channel_id: int, channel: Any, content: str, watcher_type: str = "job") -> bool:
        listing_file = self._attached_listing_file(channel_id, watcher_type)
        if job_service.is_message_duplicate(
            content,
            listing_file,
            months_threshold=self.dedup_months_threshold(),
        ):
            return True

        return await self.should_skip_duplicate_message(channel_id, channel, content, watcher_type=watcher_type)


    def _record_dedup_after_send(self, channel_id: int, watcher_type: str, content: str) -> bool:
        """Persist watcher dedup record only after a successful message send."""
        listing_file = self._attached_listing_file(channel_id, watcher_type)
        recorded = job_service.record_message_for_dedup(content, listing_file)
        if not recorded:
            print(f"Watcher dedup record failed for {channel_id}/{watcher_type}")
        return recorded

    def _channel_send_lock(self, channel_id: int, watcher_type: str) -> asyncio.Lock:
        key = (channel_id, watcher_type)
        lock = self.channel_send_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.channel_send_locks[key] = lock
        return lock


    async def send_watcher_message(self, channel_id: int, content: str, check_duplicate: bool = True, watcher_type: str = "job", dedup_key: str | None = None) -> bool:
        channel = self.client.get_channel(channel_id)
        if channel is None:
            return False

        text = content[:1900]
        key = (dedup_key or text)[:1900]

        async with self._channel_send_lock(channel_id, watcher_type):
            if check_duplicate and await self._is_duplicate_message(channel_id, channel, key, watcher_type=watcher_type):
                return False

            await channel.send(text)
            self._record_dedup_after_send(channel_id, watcher_type, key)
            self.remember_message(channel_id, key, watcher_type=watcher_type)
            return True

    async def send_reddit_gallery_message(self, channel_id: int, item: dict[str, Any]) -> bool:
        channel = self.client.get_channel(channel_id)
        if channel is None:
            return False

        title = str(item.get("title") or "(untitled)")
        permalink = str(item.get("permalink") or item.get("link") or "")
        print(f"[watcher][diag] gallery msg ch={channel_id} id={item.get('id')} title={title[:60]} gallery={len(item.get('gallery_links',[]))}")
        gallery_links = [
            str(url).strip()
            for url in item.get("gallery_links", [])
            if str(url).strip().startswith("http")
        ]
        stable_key = f"{title}\n{permalink}".strip()
        if len(gallery_links) <= 1:
            return await self.send_watcher_message(
                channel_id,
                f"{title}\n{str(item.get('link') or '')}".strip(),
                watcher_type="reddit",
                dedup_key=stable_key,
            )

        dedupe_key = stable_key

        async with self._channel_send_lock(channel_id, "reddit"):
            if await self._is_duplicate_message(channel_id, channel, dedupe_key, watcher_type="reddit"):
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
            self._record_dedup_after_send(channel_id, "reddit", dedupe_key)
            self.remember_message(channel_id, dedupe_key, watcher_type="reddit")
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
            if not await self._channel_exists(channel_id):
                print(f"Purging job watcher for deleted or inaccessible channel {channel_id}.")
                self._purge_deleted_job_channel(channel_id)
                return
            settings = self.store.get_job_settings(channel_id)
            _scrape_start = self.health.begin_scrape()
            try:
                items = await self._tracked_to_thread(
                    job_service.scrape_job_postings,
                    settings["sites"],
                    settings["keywords"],
                    settings["location"],
                    self.config.jobspy_python_exe,
                    int(settings.get("hours_old") or 72),
                    int(settings["results_wanted"]),
                    int(settings.get("radius_miles", 25)),
                    str(settings.get("country_indeed") or "AUTO"),
                    f"channel:{channel_id}",
                    bool(settings.get("allow_north_america", False)),
                    label=scheduler_labels.job_scrape_label(channel_id),
                )
                raw_count = len(items)
                role_filters = list(settings.get("role_filters", []))
                exclusion_terms = list(settings.get("exclusion_terms", []))
                items = [item for item in items if job_service.matches_role_filters(str(item.get("title", "")), role_filters)]
                items = [item for item in items if not job_service.matches_exclusion_terms(item, exclusion_terms)]
                _ats_set = set(_ATS_PLATFORMS)
                _sem_th = _safe_float(settings.get("semantic_threshold"), 0.30)
                _ats_th = _safe_float(settings.get("ats_semantic_threshold"), _sem_th)

                def _semantic_filter(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
                    # SentenceTransformer inference is CPU-bound and, on first call, loads
                    # a model from disk -- run off the event loop so it can't freeze Discord
                    # gateway/interaction handling (this previously ran inline in the
                    # coroutine and stalled the whole bot for tens of seconds per scrape).
                    return [
                        item
                        for item in candidates
                        if job_service.matches_search_parameters_semantic(
                            item,
                            str(settings.get("keywords") or ""),
                            str(settings.get("location") or ""),
                            role_filters,
                            threshold=_ats_th if _ats_set.intersection(item.get("sites") or []) else _sem_th,
                        )
                    ]

                items = await self._tracked_to_thread(
                    _semantic_filter, items, label=scheduler_labels.semantic_filter_label(channel_id)
                )
                filtered_count = len(items)
                seen = self.store.channel_job_seen.setdefault(channel_id, set())
                batch_seen_links = {
                    job_service.canonicalize_job_link(str(link)) or str(link)
                    for link in seen
                    if str(link).strip()
                }
                fresh: list[dict[str, Any]] = []
                for item in reversed(items):
                    raw_link = str(item.get("link") or "").strip()
                    canonical_link = job_service.canonicalize_job_link(raw_link) or raw_link
                    if not canonical_link or canonical_link in batch_seen_links:
                        continue
                    batch_seen_links.add(canonical_link)
                    item["link"] = canonical_link
                    fresh.append(item)

                sent_count = 0
                sent_items: list[dict[str, Any]] = []
                self._enter_send_cycle()
                try:
                    for item in fresh:
                        formatted_msg = self.format_job_watcher_message(item)
                        sent = await self.send_watcher_message(channel_id, formatted_msg, check_duplicate=True)
                        if sent:
                            sent_count += 1
                            seen.add(str(item.get("link") or "").strip())
                            self.store.save()
                            sent_items.append(item)
                    cap = job_service.DEDUP_SEEN_LINKS_CAP
                    if len(seen) > cap:
                        excess = len(seen) - cap
                        for _ in range(excess):
                            seen.pop()
                        self.store.save()
                finally:
                    self._exit_send_cycle()

                if sent_items:
                    try:
                        from datetime import date as _date
                        from services.jba.merge_data import log_jobs
                        today = _date.today().isoformat()
                        for _item in sent_items:
                            if not _item.get("date_posted"):
                                _item["date_posted"] = today
                        log_jobs(sent_items)
                    except Exception:
                        pass

                self.health.record_scrape_success(
                    channel_id, _scrape_start, raw_count, filtered_count, sent_count
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"Job watcher error for channel {channel_id}: {exc}")
                self.health.record_scrape_error(channel_id, _scrape_start, str(exc))

            await asyncio.sleep(max(60, int(settings.get("refresh_seconds", 300))))

    async def _run_reddit_watcher(self, channel_id: int) -> None:
        settings = self.store.get_reddit_settings(channel_id)
        subreddits = [str(s).strip() for s in settings.get("subreddits", ["wallpapers"]) if str(s).strip()]
        if not subreddits:
            subreddits = ["wallpapers"]
        flair_tags = [part.strip() for part in str(settings["flair_tags"]).split(",") if part.strip()]

        async def post_reddit_item(item: dict[str, Any]) -> bool:
            gallery_links = [
                str(url).strip()
                for url in item.get("gallery_links", [])
                if str(url).strip().startswith("http")
            ]
            print(f"[watcher][diag] post_reddit_item ch={channel_id} id={item.get('id')} link={str(item.get('link',''))[:80]} gallery={len(gallery_links)}")
            if len(gallery_links) > 1:
                return await self.send_reddit_gallery_message(channel_id, item)
            stable_key = f"{str(item.get('title') or '')}\n{str(item.get('permalink') or '')}".strip()
            return await self.send_watcher_message(
                channel_id,
                self.format_reddit_watcher_message(item),
                watcher_type="reddit",
                dedup_key=stable_key,
            )

        async def poll_one(subreddit: str) -> None:
            # Use the legacy single-subreddit filename when only one is configured
            # to preserve existing dedup state across upgrades.
            if len(subreddits) == 1:
                state_file = self.config.base_dir / f".reddit_seen_{channel_id}.json"
            else:
                state_file = self.config.base_dir / f".reddit_seen_{channel_id}_{subreddit}.json"
            await reddit_service.poll_subreddit_media(
                subreddit=subreddit,
                post_callback=post_reddit_item,
                sort=str(settings["sort"]),
                time_filter=str(settings["time_filter"]),
                flair_tags=flair_tags,
                include_nsfw=bool(settings["include_nsfw"]),
                include_spoiler=bool(settings["include_spoiler"]),
                media_mode=str(settings["media_mode"]),
                limit=int(settings["limit"]),
                interval_seconds=max(60, int(settings.get("refresh_seconds", 300))),
                state_file=state_file,
                scheduler=self.scheduler,
            )

        if len(subreddits) == 1:
            await poll_one(subreddits[0])
        else:
            await asyncio.gather(*[poll_one(sub) for sub in subreddits])

    def _priority_countries(self) -> list[str]:
        """The countries the enabled channels actually want, for scrape ordering.

        The ATS loop is process-wide and channel-agnostic, so there is no single
        channel to ask -- the union of what every enabled channel wants is the
        honest answer, and it keeps a second channel scoped elsewhere from being
        starved by the first.

        Derived from the settings that already exist rather than a new one:
        `location` names the region and `allow_north_america` widens it to the
        US, which are the same two inputs the region gate itself reads.
        """
        from services.jba import geo_priority

        wanted: list[str] = []
        for settings in self.store.channel_job_settings.values():
            if not settings.get("enabled"):
                continue
            code = geo_priority.country_of(str(settings.get("location") or ""))
            if code and code not in wanted:
                wanted.append(code)
            if settings.get("allow_north_america") and "US" not in wanted:
                wanted.append("US")
        return wanted

    def _ordered_slugs(self, platform: str) -> list[str] | None:
        """This platform's fleet, boards likely to yield a wanted country first.

        Ordering only: every slug in the fleet is returned, so the scrape submits
        exactly the work it would have anyway and makes no extra request. It
        matters when a cycle is cut off at the budget, because the tail is
        cancelled and file order decides who was in it. Measured across the
        fleet, CA-yielding boards inside the first 250 per platform go from 43
        to 1,031; for US, 400 to 1,843.

        Two orderings compose here. Boards known to post in a wanted country go
        to the front, and everything behind them is rotated to resume where the
        last cycle stopped, so the companies a cut-off cycle cancels are not the
        same ones every time. With no archive and no enabled channel the first
        does nothing and the second still applies -- a fleet nobody has an
        opinion about is exactly the case where always asking it in file order
        costs the most.

        None means "no opinion at all", and `scrape_ats_platform` then loads the
        fleet itself exactly as before. Only an empty fleet or a failure gets
        there.
        """
        try:
            from services import ats_service
            from services.jba import geo_priority

            fleet = list(ats_service.load_company_lists().get(platform) or [])
            if not fleet:
                return None
            preferred: set[str] = set()
            for country in self._priority_countries():
                preferred |= geo_priority.slugs_for(country)
            head, tail = geo_priority.partition(fleet, preferred)
            return head + self._rotate_tail(platform, tail, len(head))
        except Exception as exc:
            # Ordering is an optimisation; losing it must not cost the scrape.
            print(f"[ats-scrape] {platform}: could not order the fleet ({exc})")
            return None

    def _ats_rotation_state_path(self) -> Path:
        base = getattr(getattr(self, "config", None), "base_dir", None)
        return Path(base or ".") / _ATS_ROTATION_STATE

    def _rotate_tail(self, platform: str, tail: list[str], head_size: int) -> list[str]:
        """`tail`, resumed from the point the previous cycle actually reached.

        The head is the geo-preferred group and does not move: it is at the
        front precisely because it should be asked first on every cycle. What
        follows it is what a cut-off cycle cancels, and in fleet-file order that
        is the same companies every cycle, indefinitely. Rotating the tail by
        how far the last cycle got means the cancelled remainder moves instead,
        so a platform that reaches 43% of its fleet each pass still offers the
        whole fleet a turn rather than asking the same 43% forever.

        Nothing is dropped -- the result is a permutation of `tail`, so the
        cycle submits exactly the work it would have submitted anyway.

        Two approximations, both deliberate. `completed` counts the whole
        fan-out, so the head size is subtracted to get how far into the tail the
        cycle reached; and workers finish roughly, not exactly, in submission
        order. Being a little off costs asking a handful of companies twice,
        which is what every cycle did before this.
        """
        from services import ats_traversal

        if not tail:
            return []
        path = self._ats_rotation_state_path()
        with _ATS_ROTATION_LOCK:
            state = ats_traversal.load_state(path)
            entry = state.get("platforms", {}).get(platform) or {}
            # The rotation the previous cycle was handed, reconstructed so
            # "it reached N" can be turned back into "it stopped here".
            previous = ats_traversal.rotate(tail, entry.get("last_digest"))
            reached = max(0, self._ats_last_fanout(platform)["completed"] - head_size)
            cursor = ats_traversal.cursor_after(previous, reached)
            if cursor is None:
                # No cycle has reported yet, or it reached nothing past the
                # head. Advancing on that would step over companies that were
                # never asked, which is the failure this exists to prevent.
                return previous
            ats_traversal.record_progress(
                state, platform, cursor,
                wrapped=reached >= len(previous),
                now=datetime.now(timezone.utc).timestamp(),
            )
            ats_traversal.save_state(path, state)
            return ats_traversal.rotate(tail, cursor)

    @staticmethod
    def _ats_last_fanout(platform: str) -> dict[str, int]:
        """Submitted/completed company counts from that platform's last fan-out.

        Zeroes when the scraper never got as far as fanning out (an unknown
        platform, or an empty company list). Zero submitted reads as "no cycle
        recorded" downstream rather than "reached nothing", which are different
        and were previously indistinguishable.
        """
        try:
            from services import ats_service

            fan = ats_service.LAST_FANOUT.get(platform) or {}
            return {
                "submitted": int(fan.get("submitted", 0)),
                "completed": int(fan.get("completed", 0)),
            }
        except Exception:
            return {"submitted": 0, "completed": 0}

    @staticmethod
    def _is_new_utc_day(today: str, current_date: str | None) -> bool:
        """True when the UTC day rolled over on a loop that was already running.

        Split out of the loop so it can be tested at all: the branch it guards
        sits inside a `while True` that no test can drive, and asserting on the
        loop's source text cannot tell `if rolled_over` from `if False`.

        `current_date is None` is the loop's first pass, and `run.py` has just
        synced -- refreshing there would repeat that fetch seconds later for
        nothing.
        """
        return current_date is not None and today != current_date

    async def _refresh_company_lists(self) -> None:
        """Re-pull the published harvest and drop the cached fleet.

        Three things had to line up for a company discovered upstream to reach
        a scrape, and the last two were missing. The workflow republishes the
        harvest weekly; `run.py` runs the sync once, at boot; and
        `load_company_lists` caches for the life of the process. So a bot up
        for a fortnight scraped the fleet it started with and could not see a
        company found since, however many times the harvest was republished.
        `reload_company_lists` existed for this and nothing had ever called it.

        Daily rather than per-cycle: the harvest changes weekly at most, and a
        depth=1 fetch of an unchanged branch is nearly free, so this is cheap
        enough to be unconditional and rare enough not to matter if it is not.

        Failure is not an outage. The sync is additive over what is already on
        disk, so a refresh that cannot reach GitHub leaves the fleet exactly as
        it was -- which is the behaviour there was before this existed.
        """
        script = Path(self.config.base_dir) / "sync_ats_companies.py"
        if not script.exists():
            return
        try:
            def _sync() -> int:
                return subprocess.run(
                    [sys.executable, str(script)],
                    cwd=str(self.config.base_dir),
                    capture_output=True, text=True,
                ).returncode

            rc = await self._tracked_to_thread(
                _sync, label=scheduler_labels.ATS_COMPANY_SYNC
            )
            if rc != 0:
                print(f"[ats-scrape] company list refresh exited {rc}; "
                      "keeping the lists already on disk")
            from services import ats_service
            before = sum(len(v) for v in ats_service.load_company_lists().values())
            ats_service.reload_company_lists()
            after = sum(len(v) for v in ats_service.load_company_lists().values())
            print(f"[ats-scrape] company lists refreshed: {before:,} -> {after:,} slugs")
        except Exception as exc:
            # An optimisation, like the ordering: losing it must not stop the
            # scrape that was about to run.
            print(f"[ats-scrape] could not refresh company lists ({exc})")

    async def _run_ats_scrape_loop(self) -> None:
        ATS_PLATFORMS, BAMBOOHR, scrape_ats_platform = _ATS_PLATFORMS, _BAMBOOHR, _scrape_ats_platform
        from services.jba.merge_data import commit_archives_daily, log_jobs

        ATS_SCRAPES_PER_DAY = 4
        ATS_SCRAPE_INTERVAL = 86400 // ATS_SCRAPES_PER_DAY  # 6 hours
        scrapes_today: int = 0
        current_date: str | None = None
        last_scrape_ts: float = 0.0
        self.health.set_ats_task_alive(True)

        while True:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != current_date:
                rolled_over = self._is_new_utc_day(today, current_date)
                current_date = today
                scrapes_today = 0
                if rolled_over:
                    await self._refresh_company_lists()

            now_ts = datetime.now(timezone.utc).timestamp()
            wait = ATS_SCRAPE_INTERVAL - (now_ts - last_scrape_ts)
            if scrapes_today > 0 and wait > 0:
                await asyncio.sleep(min(wait, 3600.0))
                continue
            if scrapes_today >= ATS_SCRAPES_PER_DAY:
                await asyncio.sleep(3600)
                continue

            active_settings = [
                s for s in self.store.channel_job_settings.values()
                if s.get("enabled")
            ]
            if not active_settings:
                await asyncio.sleep(3600)
                continue

            platforms = tuple(
                p for p in ATS_PLATFORMS
                if p != BAMBOOHR or self.config.ats_bamboohr_enabled
            )

            health_tracker = self.health
            # Declare the full roster every cycle, so `.health` lists every
            # platform that exists rather than only those that have reported --
            # and names the ones switched off instead of omitting them.
            health_tracker.set_ats_roster(
                ATS_PLATFORMS,
                disabled=[p for p in ATS_PLATFORMS if p not in platforms],
            )

            def _scrape_one(platform: str) -> list[dict[str, Any]]:
                try:
                    result = scrape_ats_platform(
                        platform=platform,
                        keywords="",
                        location="",
                        results_wanted=0,
                        # First caller of a parameter that has existed unused
                        # since it was written. Passing the fleet in the order
                        # we want it asked is the whole feature; the contents
                        # are identical to what the default would have loaded.
                        company_slugs=self._ordered_slugs(platform),
                    )
                    new_count = log_jobs(result) if result else 0
                    fan = self._ats_last_fanout(platform)
                    if result:
                        print(
                            f"[ats-scrape] {platform}: {len(result):,} scraped, "
                            f"{new_count:,} new to DB, "
                            f"{fan['completed']:,}/{fan['submitted']:,} companies reached"
                        )
                    health_tracker.record_ats_platform_result(
                        platform, len(result), new_count=new_count,
                        submitted=fan["submitted"], completed=fan["completed"],
                    )
                    return result
                except Exception as exc:
                    # Coverage is recorded on the error path too: a cycle that
                    # raised after reaching 9,000 of 10,000 companies is a
                    # different failure from one that reached 12.
                    fan = self._ats_last_fanout(platform)
                    print(f"[ats-scrape] {platform} error: {exc}")
                    health_tracker.record_ats_platform_result(
                        platform, 0, error=str(exc),
                        submitted=fan["submitted"], completed=fan["completed"],
                    )
                    return []

            try:
                async def _scrape_one_bounded(platform: str) -> list[dict[str, Any]]:
                    try:
                        return await asyncio.wait_for(
                            self._tracked_to_thread(
                                _scrape_one, platform, label=scheduler_labels.ats_scrape_label(platform)
                            ),
                            timeout=600,
                        )
                    except Exception as exc:
                        # A platform cut off at the 600s bound is exactly when
                        # coverage matters most -- without it the log says the
                        # cycle failed but not how far it got.
                        fan = self._ats_last_fanout(platform)
                        print(
                            f"[ats-scrape] {platform} timed out or errored: {exc} "
                            f"({fan['completed']:,}/{fan['submitted']:,} companies reached)"
                        )
                        health_tracker.record_ats_platform_result(
                            platform, 0, error=str(exc),
                            submitted=fan["submitted"], completed=fan["completed"],
                        )
                        return []

                # Each platform is its own scheduler submission (BACKGROUND tier,
                # labeled per-platform) rather than a nested ThreadPoolExecutor --
                # a nested pool's threads would do real concurrent CPU work fully
                # invisible to (and uncounted by) the scheduler's own worker
                # accounting, undermining the priority arbitration interactive
                # commands rely on during this up-to-30-minute cycle. This also
                # gives each platform its own measured cost instead of one
                # blended "ats_scrape" bucket.
                per_platform_results = await asyncio.wait_for(
                    asyncio.gather(*[_scrape_one_bounded(p) for p in platforms]),
                    timeout=1800,
                )
                all_results = [item for chunk in per_platform_results for item in chunk]

                if all_results:
                    print(f"[ats-scrape] Logged {len(all_results)} jobs across {len(platforms)} platforms")
                scrapes_today += 1
                last_scrape_ts = datetime.now(timezone.utc).timestamp()
                self.health.record_ats_scrape_complete(len(all_results), scrapes_today, ATS_SCRAPES_PER_DAY)
                print(f"[ats-scrape] Scrape {scrapes_today}/{ATS_SCRAPES_PER_DAY} complete for {today}")

                # Publish whatever archive zips have landed since yesterday.
                # Self-limiting to once a UTC day, and a no-op on the days when
                # no weekly rollover or monthly consolidation produced a zip.
                # Off unless JBA_ARCHIVE_GIT_COMMIT is set; push additionally
                # needs JBA_ARCHIVE_GIT_PUSH. git shells out and can block on a
                # network push, so it goes to a thread rather than the loop.
                await self._tracked_to_thread(
                    commit_archives_daily, label=scheduler_labels.ATS_SCRAPE
                )

                # Pick up republished GeoNames data. Checked every scrape
                # cycle rather than daily because the check itself is a few
                # hundred bytes of manifest -- the 12MB archive is only
                # fetched when a checksum actually moves, which the publishing
                # workflow does about weekly. Subprocess rather than an import
                # because the sync lives in scripts/, which is not on the
                # package path, and a geo.db rebuild in-process would hold the
                # GIL for several seconds.
                await self._tracked_to_thread(
                    _sync_geonames, label=scheduler_labels.GEONAMES_SYNC
                )
            except Exception as exc:
                print(f"[ats-scrape] Scrape cycle error: {exc}")

    def _ensure_ats_scrape_loop(self) -> None:
        if self._ats_scrape_task is not None and not self._ats_scrape_task.done():
            return
        self._ats_scrape_task = asyncio.create_task(self._run_ats_scrape_loop())
        print("[ats-scrape] Background ATS scrape loop started")

    async def _run_watchdog(self, interval_seconds: int = 30) -> None:
        """Periodically detect watchers/loops that are enabled in store but whose
        task died (e.g. an exception escaping a loop's own try/except, such as a
        corrupt settings read outside the guarded block) and restart them. Without
        this, a single unexpected failure would silently kill that watcher until
        the whole bot process restarts."""
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                for channel_id, settings in list(self.store.channel_job_settings.items()):
                    if not settings.get("enabled"):
                        continue
                    task = self.channel_job_tasks.get(channel_id)
                    if task is None or task.done():
                        print(f"[watchdog] Job watcher for channel {channel_id} is dead but enabled; restarting.")
                        self.health.record_watchdog_restart(f"job watcher (channel {channel_id})")
                        self.start_job_watcher(channel_id)

                for channel_id, settings in list(self.store.channel_reddit_settings.items()):
                    if not settings.get("enabled"):
                        continue
                    task = self.channel_reddit_tasks.get(channel_id)
                    if task is None or task.done():
                        print(f"[watchdog] Reddit watcher for channel {channel_id} is dead but enabled; restarting.")
                        self.health.record_watchdog_restart(f"reddit watcher (channel {channel_id})")
                        self.start_reddit_watcher(channel_id)

                any_job_enabled = any(
                    s.get("enabled") for s in self.store.channel_job_settings.values()
                )
                if any_job_enabled and (self._ats_scrape_task is None or self._ats_scrape_task.done()):
                    print("[watchdog] ATS scrape loop is dead but job watchers are enabled; restarting.")
                    self.health.record_watchdog_restart("ATS scrape loop")
                    self._ensure_ats_scrape_loop()

                stats = self.scheduler.stats()
                print(
                    "[watchdog] scheduler: "
                    f"workers={stats['workers']} active={stats['active']} "
                    f"queued={stats['queued']} (interactive={stats['queued_interactive']}, "
                    f"background={stats['queued_background']}) "
                    f"completed={stats['completed']} promoted={stats['promoted']}"
                )
                if stats["label_costs"]:
                    label_summary = ", ".join(
                        f"{label}={info['median_seconds']}s(n={info['samples']})"
                        for label, info in sorted(stats["label_costs"].items())
                    )
                    print(f"[watchdog] scheduler learned costs: {label_summary}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[watchdog] cycle error: {exc}")

    def ensure_watchdog(self) -> None:
        if self._watchdog_task is not None and not self._watchdog_task.done():
            return
        self._watchdog_task = asyncio.create_task(self._run_watchdog())
        print("[watchdog] Watcher supervisor started")

    def start_job_watcher(self, channel_id: int) -> bool:
        existing = self.channel_job_tasks.get(channel_id)
        if existing and not existing.done():
            return True
        if not self.acquire_channel_process(channel_id, "job_watcher"):
            return False

        self.channel_listing_files[(channel_id, "job")] = self._dedup_listing_file(channel_id, "job")
        self._ensure_watcher_dedup_directory(channel_id, "job")
        self.store.update_job_setting(channel_id, "enabled", True)
        self.health.record_job_watcher_started(channel_id)
        self.create_channel_task(self.channel_job_tasks, channel_id, "job_watcher", self._run_job_watcher(channel_id))
        self._ensure_ats_scrape_loop()
        self._schedule_cheatsheet_check(channel_id, "job")
        return True

    def stop_job_watcher(self, channel_id: int) -> None:
        self.store.update_job_setting(channel_id, "enabled", False)
        task = self.channel_job_tasks.get(channel_id)
        if task and not task.done():
            task.cancel()
        else:
            self.release_channel_process(channel_id, "job_watcher")
        self.channel_listing_files.pop((channel_id, "job"), None)

    def start_reddit_watcher(self, channel_id: int) -> bool:
        existing = self.channel_reddit_tasks.get(channel_id)
        if existing and not existing.done():
            return True
        if not self.acquire_channel_process(channel_id, "reddit_watcher"):
            return False

        self.channel_listing_files[(channel_id, "reddit")] = self._dedup_listing_file(channel_id, "reddit")
        self._ensure_watcher_dedup_directory(channel_id, "reddit")
        self.store.update_reddit_setting(channel_id, "enabled", True)
        self.create_channel_task(self.channel_reddit_tasks, channel_id, "reddit_watcher", self._run_reddit_watcher(channel_id))
        self._schedule_cheatsheet_check(channel_id, "reddit")
        return True

    def stop_reddit_watcher(self, channel_id: int) -> None:
        self.store.update_reddit_setting(channel_id, "enabled", False)
        task = self.channel_reddit_tasks.get(channel_id)
        if task and not task.done():
            task.cancel()
        else:
            self.release_channel_process(channel_id, "reddit_watcher")
        self.channel_listing_files.pop((channel_id, "reddit"), None)

    async def restore_enabled_watchers(self) -> dict[str, int]:
        restored = {"job": 0, "reddit": 0}
        for channel_id, settings in list(self.store.channel_job_settings.items()):
            if settings.get("enabled"):
                if not await self._channel_exists(channel_id):
                    print(f"Purging job watcher for deleted or inaccessible channel {channel_id} during restore.")
                    self._purge_deleted_job_channel(channel_id)
                    continue
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
