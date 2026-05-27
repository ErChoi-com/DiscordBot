from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commands.handlers import CMD_RESUME_CHECK, CommandRouter


class _FakeGuild:
    def __init__(self, owner_id: int, members: list[SimpleNamespace]) -> None:
        self.owner_id = owner_id
        self.members = members

    def get_member(self, user_id: int):
        for member in self.members:
            if member.id == user_id:
                return member
        return None

    async def fetch_member(self, user_id: int):
        return self.get_member(user_id)

    def get_member_named(self, name: str):
        lowered = name.casefold()
        for member in self.members:
            if str(getattr(member, "name", "")).casefold() == lowered:
                return member
        return None


def _make_message(
    *,
    author_id: int,
    guild_owner_id: int,
    content: str = ".resumecheck",
    author_name: str = "author",
    guild_members: list[SimpleNamespace] | None = None,
):
    members = guild_members or [SimpleNamespace(id=author_id, name=author_name)]
    author = next((member for member in members if member.id == author_id), None)
    if author is None:
        author = SimpleNamespace(id=author_id, name=author_name)
        members = [*members, author]

    return SimpleNamespace(
        content=content,
        guild=SimpleNamespace(owner_id=guild_owner_id),
        author=author,
    )


def test_can_access_resume_target_allows_regular_member_self() -> None:
    router = object.__new__(CommandRouter)
    message = _make_message(author_id=10, guild_owner_id=10)

    assert router._can_access_resume_target(message, 10)


def test_can_access_resume_target_rejects_regular_member_other_user() -> None:
    router = object.__new__(CommandRouter)
    message = _make_message(author_id=20, guild_owner_id=10)

    assert not router._can_access_resume_target(message, 30)


def test_can_access_resume_target_allows_guild_owner_targeting_other() -> None:
    router = object.__new__(CommandRouter)
    message = _make_message(author_id=10, guild_owner_id=10)

    assert router._can_access_resume_target(message, 30)


def test_resolve_resume_target_member_defaults_to_author() -> None:
    router = object.__new__(CommandRouter)
    author = SimpleNamespace(id=50, name="author")
    message = _make_message(
        author_id=50,
        guild_owner_id=10,
        content=".resumecheck",
        author_name="author",
        guild_members=[author],
    )
    message.guild = _FakeGuild(owner_id=10, members=[author])

    target, error = asyncio.run(router._resolve_resume_target_member(message, CMD_RESUME_CHECK))

    assert error is None
    assert target is author


def test_resolve_resume_target_member_by_username() -> None:
    router = object.__new__(CommandRouter)
    author = SimpleNamespace(id=50, name="author")
    target_member = SimpleNamespace(id=60, name="otherusername")
    members = [author, target_member]
    message = _make_message(
        author_id=50,
        guild_owner_id=10,
        content=".resumecheck otherusername",
        author_name="author",
        guild_members=members,
    )
    message.guild = _FakeGuild(owner_id=10, members=members)

    target, error = asyncio.run(router._resolve_resume_target_member(message, CMD_RESUME_CHECK))

    assert error is None
    assert target is target_member


def test_resolve_resume_target_member_by_mention() -> None:
    router = object.__new__(CommandRouter)
    author = SimpleNamespace(id=50, name="author")
    target_member = SimpleNamespace(id=70, name="mentiontarget")
    members = [author, target_member]
    message = _make_message(
        author_id=50,
        guild_owner_id=10,
        content=".resumecheck <@70>",
        author_name="author",
        guild_members=members,
    )
    message.guild = _FakeGuild(owner_id=10, members=members)

    target, error = asyncio.run(router._resolve_resume_target_member(message, CMD_RESUME_CHECK))

    assert error is None
    assert target is target_member


def test_resolve_resume_target_member_by_multiword_display_name_for_dot_command() -> None:
    router = object.__new__(CommandRouter)
    author = SimpleNamespace(id=50, name="author")
    target_member = SimpleNamespace(id=80, name="rickydisappoints", display_name="ricky disappoints")
    members = [author, target_member]
    message = _make_message(
        author_id=50,
        guild_owner_id=10,
        content=".resumecheck ricky disappoints",
        author_name="author",
        guild_members=members,
    )
    message.guild = _FakeGuild(owner_id=10, members=members)

    target, error = asyncio.run(router._resolve_resume_target_member(message, CMD_RESUME_CHECK))

    assert error is None
    assert target is target_member


def test_resolve_resume_target_member_rejects_multiple_tokens() -> None:
    router = object.__new__(CommandRouter)
    author = SimpleNamespace(id=50, name="author")
    message = _make_message(
        author_id=50,
        guild_owner_id=10,
        content=".resumecheck user extra",
        author_name="author",
        guild_members=[author],
    )
    message.guild = _FakeGuild(owner_id=10, members=[author])

    target, error = asyncio.run(router._resolve_resume_target_member(message, CMD_RESUME_CHECK))

    assert target is None
    assert error is not None


def test_owner_profile_key_resolver_accepts_exact_cache_folder_name(tmp_path: Path) -> None:
    router = object.__new__(CommandRouter)
    profile_dir = tmp_path / "rickydisappoints"
    profile_dir.mkdir(parents=True)

    message = _make_message(
        author_id=10,
        guild_owner_id=10,
        content=".resumecheck rickydisappoints",
        author_name="owner",
    )

    profile_key, error = router._resolve_owner_profile_key_from_payload(message, CMD_RESUME_CHECK, tmp_path)

    assert error is None
    assert profile_key == "rickydisappoints"


def test_owner_profile_key_resolver_accepts_normalized_name(tmp_path: Path) -> None:
    router = object.__new__(CommandRouter)
    profile_dir = tmp_path / "rickydisappoints"
    profile_dir.mkdir(parents=True)

    message = _make_message(
        author_id=10,
        guild_owner_id=10,
        content=".resumecheck Ricky Disappoints",
        author_name="owner",
    )

    profile_key, error = router._resolve_owner_profile_key_from_payload(message, CMD_RESUME_CHECK, tmp_path)

    assert error is None
    assert profile_key == "rickydisappoints"


def test_owner_profile_key_resolver_rejects_non_owner(tmp_path: Path) -> None:
    router = object.__new__(CommandRouter)
    profile_dir = tmp_path / "rickydisappoints"
    profile_dir.mkdir(parents=True)

    message = _make_message(
        author_id=11,
        guild_owner_id=10,
        content=".resumecheck rickydisappoints",
        author_name="member",
    )

    profile_key, error = router._resolve_owner_profile_key_from_payload(message, CMD_RESUME_CHECK, tmp_path)

    assert profile_key is None
    assert error is not None
