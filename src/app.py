from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import discord

from commands.handlers import CommandRouter
from config import load_config
from services import job_service
from state.store import RuntimeStore, init_store_defaults
from watchers.manager import WatcherManager


def create_client() -> discord.Client:
    intents = discord.Intents.default()
    intents.message_content = True
    return discord.Client(intents=intents)


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


def bind_client_events(client: discord.Client, watcher_manager: WatcherManager, router: CommandRouter) -> None:
    def _append_interaction_trace(entry: str) -> None:
        try:
            trace_path = Path(__file__).resolve().parents[1] / ".interaction_trace.log"
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(entry + "\n")
        except OSError:
            pass

    @client.event
    async def on_ready() -> None:
        restored = watcher_manager.restore_enabled_watchers()
        print("Restored scraper processes: " f"jobs={restored['job']}, reddit={restored['reddit']}")
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

        if content.startswith("$"):
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

    # Push settings.toml values into service/store module globals
    job_service.configure_job_service(config)
    init_store_defaults(config)

    store = RuntimeStore(config.state_path)
    store.load()

    client = create_client()
    watcher_manager = WatcherManager(client=client, config=config, store=store)
    router = CommandRouter(client=client, config=config, store=store, watcher_manager=watcher_manager)
    bind_client_events(client, watcher_manager, router)

    register_current_process(config.pid_path)
    try:
        client.run(config.discord_token)
    finally:
        unregister_current_process(config.pid_path)
        release_runtime_lock(config.lock_path)


if __name__ == "__main__":
    run_bot()
