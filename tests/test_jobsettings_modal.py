# pyright: reportMissingImports=false

import asyncio
import sys
from pathlib import Path

import discord

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.views import (
    JobTextModal,
    ResumeFileEditModal,
    TemplateEditModal,
    _sanitize_modal_text_input_labels,
)


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


def test_template_edit_modal_routes_compile_through_scheduler_when_provided(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression guard: this compile call used to be a raw asyncio.to_thread,
    bypassing the priority scheduler entirely for a live, user-waited-on
    interactive flow (clicking 'Edit template' in the job settings panel)."""
    from types import SimpleNamespace

    from services.priority_scheduler import PriorityWorkScheduler
    from services import scheduler_labels
    import ui.views as views_module

    def _fake_compile(*args, **kwargs):
        return SimpleNamespace(status="ok", pdf_bytes=b"%PDF-fake", message="")

    monkeypatch.setattr(views_module, "compile_latex_to_pdf", _fake_compile)

    profile_dir = tmp_path / "999"
    profile_dir.mkdir(parents=True)
    template_path = profile_dir / "template.tex"
    template_path.write_text("\\documentclass{article}", encoding="utf-8")

    scheduler = PriorityWorkScheduler(max_workers=1)
    try:
        modal = TemplateEditModal(profile_dir=profile_dir, scheduler=scheduler)
        result = asyncio.run(modal._build_preview_message(template_path))
        assert isinstance(result, tuple)

        stats = scheduler.stats()
        entry = stats["label_costs"].get(scheduler_labels.RESUME_TEMPLATE_PREVIEW_COMPILE)
        assert entry is not None, "compile must have gone through the scheduler, not asyncio.to_thread"
        assert entry["samples"] >= 1
    finally:
        scheduler.shutdown()


def test_template_edit_modal_falls_back_to_asyncio_to_thread_without_scheduler(
    tmp_path: Path, monkeypatch
) -> None:
    """No scheduler passed (e.g. TemplateEditModal constructed directly, as in
    the other test above) must still work via the plain asyncio.to_thread path."""
    from types import SimpleNamespace

    import ui.views as views_module

    def _fake_compile(*args, **kwargs):
        return SimpleNamespace(status="ok", pdf_bytes=b"%PDF-fake", message="")

    monkeypatch.setattr(views_module, "compile_latex_to_pdf", _fake_compile)

    profile_dir = tmp_path / "998"
    profile_dir.mkdir(parents=True)
    template_path = profile_dir / "template.tex"
    template_path.write_text("\\documentclass{article}", encoding="utf-8")

    modal = TemplateEditModal(profile_dir=profile_dir)
    assert modal.scheduler is None
    result = asyncio.run(modal._build_preview_message(template_path))
    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Resume-cache file editing
#
# One button/modal per file, each split across parts like the template editor.
# These write and read the real files under tmp_path -- only the Discord
# interaction transport is faked.
# ---------------------------------------------------------------------------

from ui.views import (  # noqa: E402
    DISCORD_MODAL_TEXT_LIMIT,
    RESUME_FILE_MODAL_PARTS,
    ResumeFileEditModal,
)


def _profile(tmp_path: Path, **files: str) -> Path:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        # baseinfo_txt -> baseinfo.txt, template_tex -> template.tex
        stem, _, suffix = name.rpartition("_")
        (profile_dir / f"{stem}.{suffix}").write_text(text, encoding="utf-8")
    return profile_dir


def test_resume_file_modal_edits_one_file_per_modal(tmp_path: Path) -> None:
    profile_dir = _profile(tmp_path, baseinfo_txt="base text", instructions_txt="rules text")

    baseinfo_modal = ResumeFileEditModal(profile_dir=profile_dir, file_name="baseinfo.txt")
    instructions_modal = ResumeFileEditModal(profile_dir=profile_dir, file_name="instructions.txt")

    assert baseinfo_modal.parts[0].default == "base text"
    assert instructions_modal.parts[0].default == "rules text"
    # The other file is not in this modal at all, so submitting cannot clobber it.
    assert all("instructions" not in (c.label or "") for c in baseinfo_modal.children)


def test_resume_file_modal_labels_respect_discord_limits(tmp_path: Path) -> None:
    profile_dir = _profile(tmp_path, baseinfo_txt="x")
    for file_name in ("baseinfo.txt", "instructions.txt"):
        modal = ResumeFileEditModal(profile_dir=profile_dir, file_name=file_name)
        assert len(modal.children) == RESUME_FILE_MODAL_PARTS
        assert len(modal.title) <= 45
        for child in modal.children:
            assert 1 <= len(getattr(child, "label", "") or "") <= 45


def test_resume_file_modal_round_trips_a_file_longer_than_one_field(tmp_path: Path) -> None:
    # The exact case the old combined modal refused to save: a real baseinfo.txt
    # bigger than a single 4000-char text input.
    long_text = ("A" * DISCORD_MODAL_TEXT_LIMIT) + ("B" * 1500)
    profile_dir = _profile(tmp_path, baseinfo_txt=long_text, instructions_txt="untouched")

    modal = ResumeFileEditModal(profile_dir=profile_dir, file_name="baseinfo.txt")
    assert modal._truncated is False
    assert modal.parts[0].default == "A" * DISCORD_MODAL_TEXT_LIMIT
    assert modal.parts[1].default == "B" * 1500
    assert modal.parts[2].default == ""

    # Submit the loaded values back unchanged, as an untouched modal would.
    for field in modal.parts:
        field._value = field.default
    interaction = _FakeInteraction()
    asyncio.run(modal.on_submit(interaction))

    assert (profile_dir / "baseinfo.txt").read_text(encoding="utf-8") == long_text
    assert (profile_dir / "instructions.txt").read_text(encoding="utf-8") == "untouched"
    assert "Saved" in interaction.followup.messages[0][0]


def test_resume_file_modal_saves_an_edit_to_a_split_file(tmp_path: Path) -> None:
    profile_dir = _profile(tmp_path, baseinfo_txt=("A" * DISCORD_MODAL_TEXT_LIMIT) + "tail")

    modal = ResumeFileEditModal(profile_dir=profile_dir, file_name="baseinfo.txt")
    modal.parts[0]._value = "rewritten head"
    modal.parts[1]._value = "rewritten tail"
    modal.parts[2]._value = ""

    asyncio.run(modal.on_submit(_FakeInteraction()))

    assert (profile_dir / "baseinfo.txt").read_text(encoding="utf-8") == "rewritten headrewritten tail"


def test_resume_file_modal_refuses_to_save_a_file_it_could_not_fully_load(tmp_path: Path) -> None:
    capacity = DISCORD_MODAL_TEXT_LIMIT * RESUME_FILE_MODAL_PARTS
    original = "C" * (capacity + 10)
    profile_dir = _profile(tmp_path, baseinfo_txt=original)

    modal = ResumeFileEditModal(profile_dir=profile_dir, file_name="baseinfo.txt")
    assert modal._truncated is True

    for field in modal.parts:
        field._value = field.default
    interaction = _FakeInteraction()
    asyncio.run(modal.on_submit(interaction))

    # The tail that never made it into a field must not be written away.
    assert (profile_dir / "baseinfo.txt").read_text(encoding="utf-8") == original
    assert "truncated" in interaction.followup.messages[0][0]


def test_resume_file_modal_creates_a_missing_file(tmp_path: Path) -> None:
    profile_dir = tmp_path / "fresh"

    modal = ResumeFileEditModal(profile_dir=profile_dir, file_name="baseinfo.txt")
    modal.parts[0]._value = "brand new content"
    modal.parts[1]._value = ""
    modal.parts[2]._value = ""

    asyncio.run(modal.on_submit(_FakeInteraction()))

    assert (profile_dir / "baseinfo.txt").read_text(encoding="utf-8") == "brand new content"


# ---------------------------------------------------------------------------
# Three editors, one format
#
# base info / instructions / template must be the same modal shape and the same
# three adjacent buttons -- the template one only adds a compile preview.
# ---------------------------------------------------------------------------

from ui.views import JobSettingsView  # noqa: E402

RESUME_EDITOR_FILES = ("baseinfo.txt", "instructions.txt", "template.tex")


def _bare_view() -> JobSettingsView:
    """A JobSettingsView with only its declared buttons -- no store/manager."""
    view = JobSettingsView.__new__(JobSettingsView)
    discord.ui.View.__init__(view, timeout=None)
    return view


def test_template_editor_is_the_shared_resume_file_editor(tmp_path: Path) -> None:
    profile_dir = _profile(tmp_path, template_tex="\\documentclass{article}")
    modal = TemplateEditModal(profile_dir=profile_dir)

    assert isinstance(modal, ResumeFileEditModal)
    assert modal.file_name == "template.tex"
    assert modal.parts[0].default == "\\documentclass{article}"


def test_all_three_editors_have_the_same_modal_shape(tmp_path: Path) -> None:
    profile_dir = _profile(
        tmp_path, baseinfo_txt="a", instructions_txt="b", template_tex="c"
    )
    modals = [
        ResumeFileEditModal(profile_dir=profile_dir, file_name="baseinfo.txt"),
        ResumeFileEditModal(profile_dir=profile_dir, file_name="instructions.txt"),
        TemplateEditModal(profile_dir=profile_dir),
    ]

    shapes = {
        (
            len(m.children),
            tuple(c.style for c in m.children),
            tuple(c.max_length for c in m.children),
            tuple(c.required for c in m.children),
        )
        for m in modals
    }
    assert len(shapes) == 1, "the three editors must present identical fields"
    assert [m.title for m in modals] == [
        "Edit baseinfo.txt",
        "Edit instructions.txt",
        "Edit template.tex",
    ]


def test_the_three_editor_buttons_are_adjacent_and_share_a_style() -> None:
    buttons = [c for c in _bare_view().children if isinstance(c, discord.ui.Button)]
    labels = [b.label for b in buttons]

    editors = ["Edit base info", "Edit instructions", "Edit template"]
    first = labels.index(editors[0])
    assert labels[first:first + 3] == editors, "the three editors must sit next to each other"

    editor_buttons = buttons[first:first + 3]
    assert len({b.style for b in editor_buttons}) == 1
    assert len({b.row for b in editor_buttons}) == 1
    assert all(b.custom_id for b in editor_buttons), "panel buttons must be persistent"

    # Discord rejects a view with more than five components in one row.
    rows: dict[int, int] = {}
    for child in _bare_view().children:
        rows[child.row] = rows.get(child.row, 0) + 1
    assert all(count <= 5 for count in rows.values()), rows


def test_template_editor_still_previews_while_the_others_just_confirm(tmp_path: Path) -> None:
    profile_dir = _profile(tmp_path, template_tex="old", baseinfo_txt="old")

    template_modal = TemplateEditModal(profile_dir=profile_dir)

    async def _fake_preview(_path: Path) -> str:
        return "preview ok"

    template_modal._build_preview_message = _fake_preview  # type: ignore[method-assign]
    for field in template_modal.parts:
        field._value = ""
    template_modal.parts[0]._value = "new template"

    template_interaction = _FakeInteraction()
    asyncio.run(template_modal.on_submit(template_interaction))

    assert (profile_dir / "template.tex").read_text(encoding="utf-8") == "new template"
    assert template_interaction.followup.messages == [("preview ok", True)]
    # The preview is slow, so the template editor alone claims the thinking slot.
    assert template_interaction.response.defer_calls == [{"thinking": True, "ephemeral": True}]

    plain_modal = ResumeFileEditModal(profile_dir=profile_dir, file_name="baseinfo.txt")
    for field in plain_modal.parts:
        field._value = ""
    plain_modal.parts[0]._value = "new base info"
    plain_interaction = _FakeInteraction()
    asyncio.run(plain_modal.on_submit(plain_interaction))

    assert (profile_dir / "baseinfo.txt").read_text(encoding="utf-8") == "new base info"
    assert plain_interaction.response.defer_calls == [{"ephemeral": True}]
    assert "Saved `baseinfo.txt`" in plain_interaction.followup.messages[0][0]


def test_every_editor_refuses_to_save_a_file_it_could_not_fully_load(tmp_path: Path) -> None:
    capacity = DISCORD_MODAL_TEXT_LIMIT * RESUME_FILE_MODAL_PARTS
    for file_name in RESUME_EDITOR_FILES:
        profile_dir = tmp_path / file_name.replace(".", "_")
        profile_dir.mkdir(parents=True, exist_ok=True)
        original = "Z" * (capacity + 1)
        (profile_dir / file_name).write_text(original, encoding="utf-8")

        if file_name == "template.tex":
            modal = TemplateEditModal(profile_dir=profile_dir)
        else:
            modal = ResumeFileEditModal(profile_dir=profile_dir, file_name=file_name)

        for field in modal.parts:
            field._value = field.default
        interaction = _FakeInteraction()
        asyncio.run(modal.on_submit(interaction))

        assert (profile_dir / file_name).read_text(encoding="utf-8") == original
        assert "truncated" in interaction.followup.messages[0][0]


# ---------------------------------------------------------------------------
# One watcher button
#
# Start and Stop are a single toggle. These drive the real callback against a
# store/manager pair that records calls, so the assertions are about what the
# button actually did, not what it is labelled.
# ---------------------------------------------------------------------------


class _ToggleStore:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def get_job_settings(self, channel_id: int) -> dict:
        return {"enabled": self.enabled}


class _ToggleManager:
    """Mirrors WatcherManager's real contract: start returns False when another
    scraper already holds the channel, and both write the enabled flag."""

    def __init__(self, store: _ToggleStore, can_start: bool = True) -> None:
        self.store = store
        self.can_start = can_start
        self.calls: list[str] = []

    def start_job_watcher(self, channel_id: int) -> bool:
        self.calls.append("start")
        if not self.can_start:
            return False
        self.store.enabled = True
        return True

    def stop_job_watcher(self, channel_id: int) -> None:
        self.calls.append("stop")
        self.store.enabled = False


class _EditResponse:
    def __init__(self) -> None:
        self.edits: list[dict] = []
        self.messages: list[tuple[str, bool]] = []

    async def edit_message(self, **kwargs) -> None:
        self.edits.append(kwargs)

    async def send_message(self, content: str, ephemeral: bool = False) -> None:
        self.messages.append((content, ephemeral))


class _EditInteraction:
    def __init__(self) -> None:
        self.response = _EditResponse()


def _toggle_view(enabled: bool = False, can_start: bool = True):
    store = _ToggleStore(enabled)
    manager = _ToggleManager(store, can_start=can_start)
    view = JobSettingsView.__new__(JobSettingsView)
    discord.ui.View.__init__(view, timeout=None)
    view.store = store
    view.manager = manager
    view.channel_id = 42
    view.sync_watcher_toggle()
    return view, store, manager


def _toggle_button(view: JobSettingsView) -> discord.ui.Button:
    button = view._watcher_toggle_button()
    assert button is not None
    return button


def test_the_panel_has_exactly_one_watcher_button() -> None:
    labels = [
        c.label for c in _bare_view().children if isinstance(c, discord.ui.Button)
    ]
    watcher_labels = [label for label in labels if "watcher" in (label or "").lower()]
    assert watcher_labels == ["Start watcher"]


def test_watcher_toggle_starts_then_stops_on_successive_clicks(monkeypatch) -> None:
    import ui.views as views_module

    monkeypatch.setattr(views_module, "format_job_settings_summary", lambda *a: "summary")
    view, store, manager = _toggle_view(enabled=False)
    button = _toggle_button(view)

    assert button.label == "Start watcher"
    assert button.style is discord.ButtonStyle.success

    first = _EditInteraction()
    asyncio.run(button.callback(first))

    assert manager.calls == ["start"]
    assert store.enabled is True
    assert button.label == "Stop watcher"
    assert button.style is discord.ButtonStyle.danger
    # The panel is redrawn with this same view, so the flipped label is what ships.
    assert first.response.edits[0]["view"] is view

    second = _EditInteraction()
    asyncio.run(button.callback(second))

    assert manager.calls == ["start", "stop"]
    assert store.enabled is False
    assert button.label == "Start watcher"
    assert button.style is discord.ButtonStyle.success


def test_watcher_toggle_opens_as_stop_when_the_watcher_is_already_running() -> None:
    # A panel posted for a channel whose watcher survived a restart must not
    # offer "Start" and silently re-enable an already-running watcher.
    view, _store, manager = _toggle_view(enabled=True)

    assert _toggle_button(view).label == "Stop watcher"
    assert manager.calls == []


def test_watcher_toggle_reports_a_busy_channel_without_flipping(monkeypatch) -> None:
    import ui.views as views_module

    monkeypatch.setattr(views_module, "format_job_settings_summary", lambda *a: "summary")
    view, store, manager = _toggle_view(enabled=False, can_start=False)
    button = _toggle_button(view)

    interaction = _EditInteraction()
    asyncio.run(button.callback(interaction))

    assert manager.calls == ["start"]
    assert store.enabled is False
    assert button.label == "Start watcher", "a refused start must not look like it worked"
    assert interaction.response.edits == []
    assert "Another scraper process" in interaction.response.messages[0][0]
