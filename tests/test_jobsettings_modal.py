# pyright: reportMissingImports=false

import asyncio
import sys
from pathlib import Path

import discord

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.views import JobTextModal, TemplateEditModal, _sanitize_modal_text_input_labels


class _DummyStore:
    def get_job_settings(self, channel_id: int) -> dict[str, object]:
        return {
            "keywords": "python developer",
            "location": "Canada",
            "radius_miles": 25,
            "results_wanted": 10,
        }


def test_job_text_modal_component_shape_is_stable() -> None:
    # Repeated instantiation should not accumulate extra modal components.
    for _ in range(20):
        modal = JobTextModal(store=_DummyStore(), channel_id=123)
        assert len(modal.children) == 5


def test_job_text_modal_labels_respect_discord_limit() -> None:
    modal = JobTextModal(store=_DummyStore(), channel_id=123)
    for child in modal.children:
        label = getattr(child, "label", "") or ""
        assert 1 <= len(label) <= 45


def test_modal_label_sanitizer_trims_overlong_labels() -> None:
    class _LongLabelModal(discord.ui.Modal, title="Long label test"):
        def __init__(self) -> None:
            super().__init__()
            self.first = discord.ui.TextInput(label=("x" * 70), required=False)
            self.second = discord.ui.TextInput(label="ok", required=False)
            self.add_item(self.first)
            self.add_item(self.second)

    modal = _LongLabelModal()
    lengths = _sanitize_modal_text_input_labels(modal)

    assert lengths == [45, 2]
    assert len(modal.first.label or "") == 45
    assert modal.second.label == "ok"


class _FakeResponse:
    def __init__(self) -> None:
        self.defer_calls: list[dict[str, object]] = []

    async def defer(self, **kwargs) -> None:
        self.defer_calls.append(dict(kwargs))


class _FakeFollowup:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    async def send(self, content: str, ephemeral: bool = False) -> None:
        self.messages.append((content, ephemeral))


class _FakeInteraction:
    def __init__(self) -> None:
        self.response = _FakeResponse()
        self.followup = _FakeFollowup()


def test_template_edit_modal_submit_uses_thinking_defer_and_followup(tmp_path: Path) -> None:
    profile_dir = tmp_path / "123456"
    profile_dir.mkdir(parents=True)
    (profile_dir / "template.tex").write_text("\\\n", encoding="utf-8")

    modal = TemplateEditModal(profile_dir=profile_dir)

    async def _fake_preview_message(_template_path: Path) -> str:
        return "preview ok"

    modal._build_preview_message = _fake_preview_message  # type: ignore[method-assign]
    interaction = _FakeInteraction()

    asyncio.run(modal.on_submit(interaction))

    assert interaction.response.defer_calls == [{"thinking": True, "ephemeral": True}]
    assert interaction.followup.messages == [("preview ok", True)]
