"""`.reset` must gate on guild ownership and launch a truly detached restart.

Drives CommandRouter.handle_reset directly (same pattern as
test_resume_directive_wiring.py) rather than mocking at the dispatch layer,
because the two things that matter -- who is allowed to call it, and what
gets spawned -- both live inside the handler body.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commands.handlers import CommandRouter


class _Channel:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, content=None, **kwargs):
        self.sent.append(str(content))
        return SimpleNamespace(id=1)


def _message(*, author_id: int, owner_id: int):
    message = SimpleNamespace(
        content=".reset",
        guild=SimpleNamespace(owner_id=owner_id),
        author=SimpleNamespace(id=author_id, name="author", display_name="author"),
        channel=_Channel(),
        id=999,
    )
    message.to_reference = lambda fail_if_not_exists=False: None
    return message


def _router(base_dir: Path) -> CommandRouter:
    router = object.__new__(CommandRouter)
    router.config = SimpleNamespace(base_dir=base_dir)
    return router


def test_reset_rejects_non_owner_and_never_spawns(monkeypatch, tmp_path: Path) -> None:
    from services import platform_support

    spawned: list[object] = []
    monkeypatch.setattr(
        platform_support,
        "spawn_detached",
        lambda *a, **k: spawned.append((a, k)) or (True, ""),
    )

    router = _router(tmp_path)
    message = _message(author_id=1, owner_id=50)

    handled = asyncio.run(router.handle_reset(message))

    assert handled is True
    assert spawned == []
    assert any("Only the server owner" in text for text in message.channel.sent)


def test_reset_owner_spawns_detached_forcerun(monkeypatch, tmp_path: Path) -> None:
    from services import platform_support

    spawned: list[tuple[list[str], Path]] = []

    def _fake_spawn_detached(args, cwd=None, system=None, **kwargs):
        spawned.append((list(args), Path(cwd) if cwd is not None else None))
        return True, ""

    monkeypatch.setattr(platform_support, "spawn_detached", _fake_spawn_detached)

    router = _router(tmp_path)
    message = _message(author_id=50, owner_id=50)

    handled = asyncio.run(router.handle_reset(message))

    assert handled is True
    assert len(spawned) == 1
    args, cwd = spawned[0]
    assert args[0] == sys.executable
    assert args[1] == str(tmp_path / "run.py")
    assert args[2] == "--forcerun"
    assert cwd == tmp_path
    assert any("Restarting" in text for text in message.channel.sent)


def test_reset_reports_spawn_failure(monkeypatch, tmp_path: Path) -> None:
    from services import platform_support

    monkeypatch.setattr(platform_support, "spawn_detached", lambda *a, **k: (False, "boom"))

    router = _router(tmp_path)
    message = _message(author_id=50, owner_id=50)

    handled = asyncio.run(router.handle_reset(message))

    assert handled is True
    assert any("Restart failed" in text and "boom" in text for text in message.channel.sent)


def test_a_second_reset_does_not_spawn_a_second_forcerun(monkeypatch, tmp_path: Path) -> None:
    """Two `.reset`s before the first restart lands must spawn exactly one
    forcerun. Two overlapping forceruns were observed in production: the second
    kills the replacement the first just started, mid-warmup.
    """
    from services import platform_support

    spawned: list[list[str]] = []
    monkeypatch.setattr(
        platform_support,
        "spawn_detached",
        lambda args, cwd=None, system=None, **kwargs: (spawned.append(list(args)), (True, ""))[1],
    )

    router = _router(tmp_path)
    first = _message(author_id=50, owner_id=50)
    second = _message(author_id=50, owner_id=50)

    assert asyncio.run(router.handle_reset(first)) is True
    assert asyncio.run(router.handle_reset(second)) is True

    assert len(spawned) == 1, "the second .reset spawned another forcerun"
    assert any("Restarting" in text for text in first.channel.sent)
    assert any("already in progress" in text for text in second.channel.sent)


def test_a_failed_launch_does_not_block_retrying_the_reset(monkeypatch, tmp_path: Path) -> None:
    """If nothing was spawned, this process is not going away -- refusing every
    later .reset would strand the bot with no way to restart itself.
    """
    from services import platform_support

    attempts: list[list[str]] = []
    outcomes = iter([(False, "boom"), (True, "")])

    def _spawn(args, cwd=None, system=None, **kwargs):
        attempts.append(list(args))
        return next(outcomes)

    monkeypatch.setattr(platform_support, "spawn_detached", _spawn)

    router = _router(tmp_path)
    failed = _message(author_id=50, owner_id=50)
    retried = _message(author_id=50, owner_id=50)

    asyncio.run(router.handle_reset(failed))
    asyncio.run(router.handle_reset(retried))

    assert len(attempts) == 2, "the retry after a failed launch was refused"
    assert any("Restart failed" in text for text in failed.channel.sent)
    assert any("Restarting" in text for text in retried.channel.sent)
