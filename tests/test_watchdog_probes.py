"""ensure_watchdog starts the two probes that watch the halves it cannot.

The supervisor task runs *on* the loop, so it cannot observe the loop
stopping, and it never touches the websocket, so it cannot observe the gateway
going quiet. Those are the two failures that produced this run's symptoms --
commands answering slowly, and "Can't keep up, websocket is 41.5s behind" --
and each has its own watcher started here. Wiring is the whole feature: a
probe that is written and never started reports nothing at all.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import gateway_watch, loop_watch  # noqa: E402
from services.health import WatcherHealthTracker  # noqa: E402
from state.store import RuntimeStore  # noqa: E402
from watchers.manager import WatcherManager  # noqa: E402


class _Config:
    def __init__(self, root: Path):
        self.data_dir = root
        self.state_file = root / ".bot_state.json"


@pytest.fixture
def manager(tmp_path):
    client = object()
    return WatcherManager(
        client=client,
        config=_Config(tmp_path),
        store=RuntimeStore(tmp_path / ".bot_state.json"),
        health=WatcherHealthTracker(),
    )


@pytest.fixture
def started(monkeypatch):
    """Record what each probe was started with, without starting threads."""
    calls: dict[str, object] = {}
    monkeypatch.setattr(loop_watch, "start", lambda *a, **k: calls.setdefault("loop", True))
    monkeypatch.setattr(gateway_watch, "start", lambda client, *a, **k: calls.setdefault("gateway", client))
    return calls


def test_the_watchdog_starts_both_probes_and_hands_the_gateway_its_client(manager, started):
    async def go():
        manager.ensure_watchdog()

    asyncio.run(go())
    assert started.get("loop") is True
    # The gateway probe reads the websocket off the client, so being handed
    # the real client rather than a placeholder is the whole of its access.
    assert started.get("gateway") is manager.client


def test_a_probe_that_cannot_start_costs_a_line_not_the_watchdog(manager, monkeypatch, capsys):
    """These are diagnostics. A bot that refuses to supervise its watchers
    because a diagnostic raised would be strictly worse off than one with no
    diagnostic at all."""
    def _boom(*args, **kwargs):
        raise RuntimeError("no loop here")

    monkeypatch.setattr(loop_watch, "start", _boom)
    monkeypatch.setattr(gateway_watch, "start", _boom)

    async def go():
        manager.ensure_watchdog()
        assert manager._watchdog_task is not None and not manager._watchdog_task.done()
        manager._watchdog_task.cancel()

    asyncio.run(go())
    out = capsys.readouterr().out
    assert "loop watch unavailable" in out
    assert "gateway watch unavailable" in out


def test_a_second_call_does_not_start_a_second_supervisor(manager, started):
    """The probes are idempotent per process on their own, but the supervisor
    task is not, and ensure_watchdog is called from every watcher start."""
    async def go():
        manager.ensure_watchdog()
        first = manager._watchdog_task
        manager.ensure_watchdog()
        assert manager._watchdog_task is first
        first.cancel()

    asyncio.run(go())
