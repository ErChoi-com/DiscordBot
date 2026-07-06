from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands

from commands.handlers import CommandRouter
from config import load_config
from services import job_service
from services.health import WatcherHealthTracker
from services.resumes.resume import migrate_legacy_profile_keys_with_usernames
from services import browser_service
from state.store import RuntimeStore, init_store_defaults
from watchers.manager import WatcherManager


def create_client() -> discord.Client:
    intents = discord.Intents.default()
    intents.message_content = True
    return commands.Bot(command_prefix=commands.when_mentioned_or("."), intents=intents)


def read_tracked_process_id(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def register_current_process(pid_path: Path) -> None:
    pid_path.write_text(str(os.getpid()), encoding="utf-8")


def unregister_current_process(pid_path: Path) -> None:
    tracked_process_id = read_tracked_process_id(pid_path)
    if tracked_process_id != os.getpid():
        return
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_runtime_lock_owner(lock_path: Path) -> int | None:
    if not lock_path.exists():
        return None
    try:
        raw_text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw_text:
        return None
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            return int(raw_text)
        except ValueError:
            return None
    try:
        return int(payload.get("pid"))
    except (TypeError, ValueError):
        return None


def acquire_runtime_lock(lock_path: Path) -> bool:
    current_pid = os.getpid()

    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner_pid = read_runtime_lock_owner(lock_path)
            if owner_pid == current_pid:
                return True
            if owner_pid is not None and is_process_running(owner_pid):
                return False
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                return False
            continue

        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": current_pid}, ensure_ascii=True))
        except Exception:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return True


def release_runtime_lock(lock_path: Path) -> None:
    owner_pid = read_runtime_lock_owner(lock_path)
    if owner_pid != os.getpid():
        return
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def bind_client_events(
    client: discord.Client,
    watcher_manager: WatcherManager,
    router: CommandRouter,
    sync_guild_id: int | None = None,
    force_sync_once: bool = False,
    force_sync_marker_path: Path | None = None,
) -> None:
    slash_synced = False

    async def _learn_usernames_from_history(max_messages_per_channel: int = 200) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for guild in client.guilds:
            guild_me = getattr(guild, "me", None)
            for channel in getattr(guild, "text_channels", []):
                try:
                    if guild_me is not None:
                        perms = channel.permissions_for(guild_me)
                        if not bool(getattr(perms, "read_message_history", False)):
                            continue
                    async for message in channel.history(limit=max_messages_per_channel):
                        author = getattr(message, "author", None)
                        user_id = getattr(author, "id", None)
                        username = getattr(author, "name", None)
                        if isinstance(user_id, int) and isinstance(username, str) and username.strip():
                            mapping[user_id] = username.strip()
                except (discord.Forbidden, discord.HTTPException):
                    continue
                except Exception:
                    continue
        return mapping

    def _append_interaction_trace(entry: str) -> None:
        try:
            trace_path = Path(__file__).resolve().parents[1] / ".interaction_trace.log"
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(entry + "\n")
        except OSError:
            pass

    async def _run_slash_command(
        interaction: discord.Interaction,
        content: str,
        *,
        reference_message_id: int | None = None,
    ) -> None:
        await router.dispatch_interaction_command(
            interaction,
            content,
            reference_message_id=reference_message_id,
        )

    @client.tree.command(name="commands", description="Show command cheat sheet")
    async def slash_commands(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/commands")

    @client.tree.command(name="status", description="Show watcher status for this channel")
    async def slash_status(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/status")

    @client.tree.command(name="resumebuild", description="Generate a tailored resume from a job listing message")
    @discord.app_commands.describe(message_id="Job listing message ID from ErnestBot")
    async def slash_resumebuild(interaction: discord.Interaction, message_id: int | None = None) -> None:
        await _run_slash_command(interaction, "/resumebuild", reference_message_id=message_id)

    @client.tree.command(name="resumecheck", description="Compile your current cached template into a PDF")
    async def slash_resumecheck(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/resumecheck")

    @client.tree.command(name="continue", description="Send next chunk of pending long output")
    async def slash_continue(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/continue")

    @client.tree.command(name="hello2", description="Quick hello test")
    async def slash_hello(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/hello2")

    @client.tree.command(name="jobsettings", description="Open job watcher settings panel")
    async def slash_jobsettings(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/jobsettings")

    @client.tree.command(name="jobsinit", description="Alias for jobsettings")
    async def slash_jobsinit(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/jobsinit")

    @client.tree.command(name="jobbanktest", description="Run a Job Bank test scrape with current settings")
    async def slash_jobbanktest(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/jobbanktest")

    @client.tree.command(name="jobbankfilters", description="Show, set, or clear additional Job Bank native filters")
    @discord.app_commands.describe(query="Filter query text, or 'clear' to reset")
    async def slash_jobbankfilters(interaction: discord.Interaction, query: str | None = None) -> None:
        content = "/jobbankfilters" if not query else f"/jobbankfilters {query}"
        await _run_slash_command(interaction, content)

    @client.tree.command(name="redditsettings", description="Open Reddit watcher settings panel")
    async def slash_redditsettings(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/redditsettings")

    @client.tree.command(name="clearredditseen", description="Clear seen Reddit IDs for this channel")
    async def slash_clearredditseen(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/clearredditseen")

    @client.tree.command(name="resetredditseen", description="Alias for clearredditseen")
    async def slash_resetredditseen(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/resetredditseen")

    @client.tree.command(name="settings", description="Open scrape settings panel")
    async def slash_settings(interaction: discord.Interaction) -> None:
        await _run_slash_command(interaction, "/settings")

    @client.tree.command(name="scrape", description="Run a one-time scrape")
    @discord.app_commands.describe(url="URL to scrape", selector="Optional CSS selector")
    async def slash_scrape(
        interaction: discord.Interaction,
        url: str,
        selector: str | None = None,
    ) -> None:
        content = f"/scrape {url}"
        if selector:
            content = f"{content} | {selector}"
        await _run_slash_command(interaction, content)

    @client.event
    async def on_ready() -> None:
        nonlocal slash_synced
        restored = await watcher_manager.restore_enabled_watchers()
        print("Restored scraper processes: " f"jobs={restored['job']}, reddit={restored['reddit']}")
        if not slash_synced:
            try:
                if force_sync_once and sync_guild_id is not None:
                    target_guild = discord.Object(id=sync_guild_id)
                    # One-time forced refresh: clear guild-scoped commands and republish globals.
                    client.tree.clear_commands(guild=target_guild)
                    guild_synced = await client.tree.sync(guild=target_guild)
                    print(
                        "One-time force sync: cleared guild slash commands "
                        f"for guild={sync_guild_id}; remaining guild commands={len(guild_synced)}."
                    )

                synced = await client.tree.sync()
                print(f"Synced {len(synced)} global slash command(s).")
                if force_sync_once and force_sync_marker_path is not None and force_sync_marker_path.exists():
                    try:
                        force_sync_marker_path.unlink(missing_ok=True)
                        print(f"One-time force sync completed. Removed marker: {force_sync_marker_path.name}")
                    except OSError as marker_exc:
                        print(f"Could not remove force-sync marker: {marker_exc}")
                slash_synced = True
            except Exception as exc:
                print(f"Failed to sync slash commands: {exc}")

        try:
            username_by_user_id = await _learn_usernames_from_history()
            cache_root = Path(__file__).resolve().parent / "services" / "resumes" / "resumes_cache"
            migrations = migrate_legacy_profile_keys_with_usernames(username_by_user_id, cache_root)
            if migrations:
                print(f"Migrated {len(migrations)} legacy resume profile folder(s) to username keys.")
                for old_name, new_name in migrations:
                    print(f"  {old_name} -> {new_name}")
        except Exception as exc:
            print(f"History-based username migration skipped: {exc}")

        print("Logged in as {0.user}".format(client))

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author == client.user:
            return

        content = (message.content or "").strip()
        if not content:
            print(
                "Received message with empty content "
                f"in channel {message.channel.id} from user {getattr(message.author, 'id', 'unknown')}. "
                "If commands are being typed, Discord message content intent is likely unavailable."
            )
            return

        if content.startswith("."):
            print(f"Dispatching command from channel {message.channel.id}: {content}")
        await router.dispatch(message)

    @client.event
    async def on_interaction(interaction: discord.Interaction) -> None:
        # Trace every interaction so we can prove whether this process receives button/modal events.
        interaction_type = getattr(interaction, "type", None)
        interaction_data: dict[str, Any] = getattr(interaction, "data", {}) or {}
        custom_id = str(interaction_data.get("custom_id") or "")
        component_type = str(interaction_data.get("component_type") or "")
        channel_id = getattr(getattr(interaction, "channel", None), "id", None)
        user_id = getattr(getattr(interaction, "user", None), "id", None)
        _append_interaction_trace(
            " | ".join(
                [
                    f"type={interaction_type}",
                    f"custom_id={custom_id}",
                    f"component_type={component_type}",
                    f"channel_id={channel_id}",
                    f"user_id={user_id}",
                ]
            )
        )


def run_bot() -> None:
    config = load_config()
    if not config.discord_token:
        print("Missing token. Add 'discordtoken=YOUR_TOKEN' to .env and try again.")
        raise SystemExit(1)

    if not acquire_runtime_lock(config.lock_path):
        owner_pid = read_runtime_lock_owner(config.lock_path)
        if owner_pid is None:
            print("Another runtime owner is active.")
        else:
            print(f"Another runtime owner is active (pid={owner_pid}).")
        raise SystemExit(0)

    browser_service.start()

    # Push settings.toml values into service/store module globals
    job_service.configure_job_service(config)
    init_store_defaults(config)

    store = RuntimeStore(config.state_path)
    store.load()

    health = WatcherHealthTracker()
    client = create_client()
    watcher_manager = WatcherManager(client=client, config=config, store=store, health=health)
    router = CommandRouter(client=client, config=config, store=store, watcher_manager=watcher_manager, health=health)
    watcher_manager.set_cheatsheet_ensurer(router.ensure_commands_cheatsheet_pinned)
    sync_guild_id_raw = (
        os.getenv("DISCORD_SYNC_GUILD_ID")
        or os.getenv("DISCORD_GUILD_ID")
        or "1116052711406841866"
    )
    try:
        sync_guild_id = int(sync_guild_id_raw)
    except (TypeError, ValueError):
        sync_guild_id = None

    force_sync_marker_path = config.base_dir / ".force_slash_sync_once"
    force_sync_once = (
        os.getenv("DISCORD_FORCE_SYNC_ONCE", "").strip() == "1"
        or force_sync_marker_path.exists()
    )
    if force_sync_once:
        print("One-time slash force sync is enabled for this startup.")

    bind_client_events(
        client,
        watcher_manager,
        router,
        sync_guild_id=sync_guild_id,
        force_sync_once=force_sync_once,
        force_sync_marker_path=force_sync_marker_path,
    )

    _original_close = client.close

    async def _graceful_close() -> None:
        try:
            try:
                work_drained = await watcher_manager.drain_active_work(timeout=120)
                print("[shutdown] All scrapes complete." if work_drained else "[shutdown] Timed out waiting for scrapes. Closing anyway.")
            except Exception as exc:
                print(f"[shutdown] drain_active_work raised unexpectedly: {exc}. Closing anyway.")
            try:
                sends_drained = await watcher_manager.drain_active_sends(timeout=30)
                print("[shutdown] All watcher sends complete. Closing." if sends_drained else "[shutdown] Timed out waiting for watcher sends. Closing anyway.")
            except Exception as exc:
                print(f"[shutdown] drain_active_sends raised unexpectedly: {exc}. Closing anyway.")
        finally:
            await _original_close()

    client.close = _graceful_close  # type: ignore[method-assign]

    register_current_process(config.pid_path)
    try:
        client.run(config.discord_token)
    finally:
        browser_service.stop()
        unregister_current_process(config.pid_path)
        release_runtime_lock(config.lock_path)


if __name__ == "__main__":
    run_bot()
