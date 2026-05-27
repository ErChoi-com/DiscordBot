"""Tests for command cheatsheet pinning behavior.

pyright may not resolve runtime-only test path injection for `commands.*`.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commands.handlers import (
    CHEATSHEET_KIND_GENERAL,
    CHEATSHEET_KIND_JOB,
    CHEATSHEET_KIND_REDDIT,
    CommandRouter,
    build_commands_cheatsheet_embed,
)


@dataclass
class _FakePerms:
    send_messages: bool = True
    manage_messages: bool = True


class _FakeSentMessage:
    def __init__(self, embed):
        self.embeds = [embed]
        self.author = SimpleNamespace(id=999)
        self.pin_calls = 0

    async def pin(self, reason: str | None = None):
        self.pin_calls += 1


class _FakeChannel:
    def __init__(self, pinned=None, history_items=None, perms: _FakePerms | None = None):
        self._pinned = list(pinned or [])
        self._history_items = list(history_items or [])
        self._perms = perms or _FakePerms()
        self.sent_messages: list[_FakeSentMessage] = []
        self.guild = SimpleNamespace(me=SimpleNamespace(id=999))

    async def pins(self):
        return list(self._pinned)

    def permissions_for(self, _member):
        return self._perms

    async def send(self, embed=None, **_kwargs):
        sent = _FakeSentMessage(embed)
        self.sent_messages.append(sent)
        return sent

    async def history(self, limit=50):
        count = 0
        for item in self._history_items:
            if count >= limit:
                break
            count += 1
            yield item


class _FakeClient:
    def __init__(self, channel):
        self._channel = channel
        self.user = SimpleNamespace(id=999)

    def get_channel(self, _channel_id):
        return self._channel

    async def fetch_channel(self, _channel_id):
        return self._channel


def _make_router(channel):
    router = object.__new__(CommandRouter)
    router.client = _FakeClient(channel)
    return router


def test_build_commands_cheatsheet_embed_by_kind() -> None:
    general = build_commands_cheatsheet_embed(CHEATSHEET_KIND_GENERAL)
    job = build_commands_cheatsheet_embed(CHEATSHEET_KIND_JOB)
    reddit = build_commands_cheatsheet_embed(CHEATSHEET_KIND_REDDIT)

    assert general.title == "General Commands"
    assert job.title == "Job Watcher Commands"
    assert reddit.title == "Reddit Watcher Commands"


def test_ensure_commands_cheatsheet_pinned_reuses_existing_pinned() -> None:
    existing_embed = build_commands_cheatsheet_embed(CHEATSHEET_KIND_JOB)
    existing = SimpleNamespace(embeds=[existing_embed], author=SimpleNamespace(id=999))
    channel = _FakeChannel(pinned=[existing])
    router = _make_router(channel)

    asyncio.run(router.ensure_commands_cheatsheet_pinned(123, CHEATSHEET_KIND_JOB))

    assert len(channel.sent_messages) == 0


def test_ensure_commands_cheatsheet_pinned_sends_and_pins_when_missing() -> None:
    channel = _FakeChannel(pinned=[])
    router = _make_router(channel)

    asyncio.run(router.ensure_commands_cheatsheet_pinned(123, CHEATSHEET_KIND_REDDIT))

    assert len(channel.sent_messages) == 1
    assert channel.sent_messages[0].embeds[0].title == "Reddit Watcher Commands"
    assert channel.sent_messages[0].pin_calls == 1
