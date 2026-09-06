"""The ``(...)`` directive must survive the real command path.

Every other test covers one hop: parsing (test_command_aliases), the prompt
block (test_structured_resume). This one drives `CommandRouter.handle_resume`
itself, because that is where the argument is actually read off the Discord
message and handed to the service — the hop where a typo would make `(...)`
parse perfectly and still do nothing.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commands import handlers as handlers_module
from commands.handlers import CMD_RESUME, CMD_RESUME_COVER, CommandRouter, _ResumeRequest
from services.resumes.listing import JobContext

PROFILE_DIR = ROOT / "src" / "services" / "resumes" / "resumes_cache" / "xboxsignout._"

pytestmark = pytest.mark.skipif(
    not (PROFILE_DIR / "template.tex").exists(),
    reason="needs the xboxsignout._ profile cache",
)


class _Channel:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, content=None, **kwargs):
        self.sent.append(str(content))
        return SimpleNamespace(id=1)


def _message(content: str):
    author = SimpleNamespace(id=50, name="author", display_name="author")
    return SimpleNamespace(
        content=content,
        guild=SimpleNamespace(owner_id=50),
        author=author,
        channel=_Channel(),
        id=999,
    )


def _router(monkeypatch, captured: dict, *, cover: bool = False):
    """A router whose only stubs are the pieces that would hit the network."""
    router = object.__new__(CommandRouter)
    router.config = SimpleNamespace(
        resume_cache_dir=ROOT / ".resume_cache",
        resume_normalize_json_latex=False,
    )

    async def _run_interactive(fn, *args, cost=None, label=None, **kwargs):
        # Same contract as the real one: forward everything but its own knobs.
        return fn(*args, **kwargs)

    router._run_interactive = _run_interactive

    async def _prepare(message, command, slash_command):
        # Record what the handler left on the message, so we also prove the
        # directive was STRIPPED before target/username resolution sees it.
        captured["content_seen_by_prepare"] = str(getattr(message, "content", ""))
        return _ResumeRequest(
            settings=SimpleNamespace(
                api_key="", openrouter_api_key="k", groq_api_key="", model="m"
            ),
            job=JobContext(
                title="Software Engineer Co-op",
                posting_url="http://example.test/job",
                apply_url="",
                source_message="",
            ),
            profile_key="xboxsignout._",
            profile_dir=PROFILE_DIR,
            template_path=PROFILE_DIR / "template.tex",
            cache_scope="test",
            reply_send_kwargs={},
        )

    router._prepare_resume_request = _prepare

    def _fake_rewrite(*args, **kwargs):
        captured["user_directive"] = kwargs.get("user_directive")
        captured["aggressive"] = kwargs.get("aggressive")
        captured["strong_aggressive"] = kwargs.get("strong_aggressive")
        return SimpleNamespace(
            status="error", message="stopped for test", rewritten_resume=None,
            latex_document=None, used_provider=None, summary=None,
        )

    def _fake_cover(*args, **kwargs):
        captured["user_directive"] = kwargs.get("user_directive")
        return SimpleNamespace(
            status="error", message="stopped for test", latex_document=None,
            used_provider=None, summary=None,
        )

    monkeypatch.setattr(handlers_module, "generate_resume_rewrite", _fake_rewrite)
    monkeypatch.setattr(handlers_module, "generate_cover_letter", _fake_cover)
    return router


@pytest.mark.parametrize(
    "content, expected_directive, expected_flags",
    [
        (f"{CMD_RESUME} (lead with the embedded work)", "lead with the embedded work", (False, False)),
        (f"{CMD_RESUME} --aggressive (keep it short)", "keep it short", (True, False)),
        (f"{CMD_RESUME} --strongaggressive (one page only)", "one page only", (True, True)),
        (f"{CMD_RESUME} (avoid sounding --aggressive)", "avoid sounding --aggressive", (False, False)),
        (f"{CMD_RESUME}", None, (False, False)),
    ],
)
def test_directive_reaches_the_service_from_handle_resume(
    monkeypatch, content, expected_directive, expected_flags
) -> None:
    captured: dict = {}
    router = _router(monkeypatch, captured)
    assert asyncio.run(router.handle_resume(_message(content))) is True

    assert captured["user_directive"] == (expected_directive or "")
    assert (captured["aggressive"], captured["strong_aggressive"]) == expected_flags
    # The parenthesised span must be gone before username resolution runs.
    assert "(" not in captured["content_seen_by_prepare"]


def test_directive_reaches_the_service_from_handle_resume_cover(monkeypatch) -> None:
    captured: dict = {}
    router = _router(monkeypatch, captured, cover=True)
    message = _message(f"{CMD_RESUME_COVER} (emphasise the tutoring work)")
    assert asyncio.run(router.handle_resume_cover(message)) is True

    assert captured["user_directive"] == "emphasise the tutoring work"
    assert "(" not in captured["content_seen_by_prepare"]
