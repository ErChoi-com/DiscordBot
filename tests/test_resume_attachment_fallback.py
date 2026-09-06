from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import discord

from commands.handlers import CommandRouter


class _FakeAttachment:
    def __init__(self, filename: str, data: bytes, *, fails: bool = False) -> None:
        self.filename = filename
        self._data = data
        self._fails = fails

    async def read(self) -> bytes:
        if self._fails:
            raise discord.HTTPException(SimpleNamespace(status=500, reason="boom"), "boom")
        return self._data


class _SentMessages(list):
    async def send(self, *args, **kwargs) -> None:
        self.append((args, kwargs))


JOB_TEXT = (
    "[Software Engineer @ Acme]\n"
    "Apply: https://acme.example.com/careers/123\n"
    "Great role building things."
)


def _make_referenced(*, content: str, attachments: list, is_bot: bool = True):
    return SimpleNamespace(
        content=content,
        author=SimpleNamespace(bot=is_bot),
        attachments=attachments,
    )


def _make_message(referenced) -> SimpleNamespace:
    async def fetch_message(message_id):
        return referenced

    channel = SimpleNamespace(send=_SentMessages().send, fetch_message=fetch_message)
    message = SimpleNamespace(
        reference=SimpleNamespace(message_id=1, resolved=None),
        channel=channel,
    )
    return message


def test_resolve_resume_context_falls_back_to_txt_attachment() -> None:
    router = object.__new__(CommandRouter)
    referenced = _make_referenced(
        content="",
        attachments=[_FakeAttachment("job.txt", JOB_TEXT.encode("utf-8"))],
    )
    message = _make_message(referenced)

    job = asyncio.run(router.resolve_resume_context(message))

    assert job is not None
    assert job.title == "[Software Engineer @ Acme]"
    assert job.posting_url == "https://acme.example.com/careers/123"


def test_resolve_resume_context_ignores_non_text_attachments() -> None:
    router = object.__new__(CommandRouter)
    referenced = _make_referenced(
        content="",
        attachments=[_FakeAttachment("resume.pdf", b"%PDF-1.4 binary junk")],
    )
    message = _make_message(referenced)

    job = asyncio.run(router.resolve_resume_context(message))

    assert job is None


def test_resolve_resume_context_skips_attachment_read_errors() -> None:
    router = object.__new__(CommandRouter)
    referenced = _make_referenced(
        content="",
        attachments=[
            _FakeAttachment("broken.txt", b"", fails=True),
            _FakeAttachment("job.md", JOB_TEXT.encode("utf-8")),
        ],
    )
    message = _make_message(referenced)

    job = asyncio.run(router.resolve_resume_context(message))

    assert job is not None
    assert job.posting_url == "https://acme.example.com/careers/123"


def test_resolve_resume_context_concatenates_multiple_text_attachments() -> None:
    router = object.__new__(CommandRouter)
    header = "[Software Engineer @ Acme]"
    body = "Apply: https://acme.example.com/careers/123\nGreat role building things."
    referenced = _make_referenced(
        content="",
        attachments=[
            _FakeAttachment("title.txt", header.encode("utf-8")),
            _FakeAttachment("body.md", body.encode("utf-8")),
        ],
    )
    message = _make_message(referenced)

    job = asyncio.run(router.resolve_resume_context(message))

    assert job is not None
    assert job.title == header
    assert job.posting_url == "https://acme.example.com/careers/123"


def test_resolve_resume_context_prefers_message_content_over_attachment() -> None:
    router = object.__new__(CommandRouter)
    other_text = (
        "[Other Job @ Other Co]\n"
        "Apply: https://other.example.com/careers/999\n"
    )
    referenced = _make_referenced(
        content=JOB_TEXT,
        attachments=[_FakeAttachment("job.txt", other_text.encode("utf-8"))],
    )
    message = _make_message(referenced)

    job = asyncio.run(router.resolve_resume_context(message))

    assert job is not None
    assert job.posting_url == "https://acme.example.com/careers/123"
