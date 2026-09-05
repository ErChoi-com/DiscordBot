from __future__ import annotations

import asyncio
import io
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discord

from services import job_service, scheduler_labels
from services.priority_scheduler import INTERACTIVE, PriorityWorkScheduler
from services.resumes.resume import compile_latex_to_pdf
from state.store import MODE_DESCRIPTIONS, RuntimeStore
from watchers.manager import WatcherManager


DISCORD_MODAL_TEXT_LIMIT = 4000
# Discord allows 5 text inputs per modal; 3 parts x 4000 chars is the same
# budget the template editor uses, and leaves room to grow.
RESUME_FILE_MODAL_PARTS = 3
DISCORD_TEXT_INPUT_LABEL_LIMIT = 45


LAST_PANEL_OUTPUTS: dict[tuple[str, int], str] = {}


def _sanitize_modal_text_input_labels(modal: discord.ui.Modal) -> list[int]:
    """Normalize modal text-input labels to Discord's 1..45 char constraint."""
    lengths: list[int] = []
    for child in modal.children:
        if not isinstance(child, discord.ui.TextInput):
            continue
        underlying = getattr(child, "_underlying", None)
        if underlying is None:
            continue
        label = getattr(underlying, "label", None)
        if not isinstance(label, str):
            continue
        normalized = label.strip() or "Field"
        if len(normalized) > DISCORD_TEXT_INPUT_LABEL_LIMIT:
            normalized = normalized[:DISCORD_TEXT_INPUT_LABEL_LIMIT]
        if normalized != label:
            try:
                underlying.label = normalized
            except Exception:
                # If discord.py rejects runtime relabeling, keep the original and let send_modal fail loudly.
                pass
        lengths.append(len(getattr(underlying, "label", "") or ""))
    return lengths


def format_mode_summary(store: RuntimeStore, channel_id: int) -> str:
    current = store.get_mode(channel_id)
    mode_lines = [f"Current mode: `{current}`", "Available modes:"]
    for mode_name, description in MODE_DESCRIPTIONS.items():
        mode_lines.append(f"- `{mode_name}`: {description}")
    return "\n".join(mode_lines)


def format_scrape_settings_summary(store: RuntimeStore, channel_id: int) -> str:
    settings = store.get_scrape_settings(channel_id)
    ai_state = "On" if settings["use_ai_cleanup"] else "Off"
    return (
        "Scrape settings for this channel:\n"
        f"- Max items: `{settings['max_items']}`\n"
        f"- Request timeout: `{settings['timeout_seconds']}s`\n"
        f"- AI cleanup: `{ai_state}`"
    )


def format_job_settings_summary(store: RuntimeStore, channel_id: int) -> str:
    settings = store.get_job_settings(channel_id)
    status = "Running" if settings["enabled"] else "Stopped"
    raw_sites = list(settings["sites"])
    sites = ", ".join(job_service.JOBSPY_SITE_LABELS.get(site, site.replace("_", " ").title()) for site in raw_sites)
    role_filters = ", ".join(settings["role_filters"]) if settings["role_filters"] else "Any"
    exclusion_terms = ", ".join(settings["exclusion_terms"]) if settings["exclusion_terms"] else "None"
    region_mode = "North America (Canada + US)" if settings.get("allow_north_america") else "Canada only"
    summary = (
        "Job watcher for this channel:\n"
        f"- Status: `{status}`\n"
        f"- Sites: `{sites}`\n"
        f"- Keywords: `{settings['keywords']}`\n"
        f"- Location: `{settings['location']}`\n"
        f"- Region filter: `{region_mode}`\n"
        f"- Radius: `{settings['radius_miles']}` miles\n"
        f"- Role filter: `{role_filters}`\n"
        f"- Exclusion terms: `{exclusion_terms}`\n"
        f"- Date window: last `{settings['hours_old']}` hours\n"
        f"- Results per check: `{settings['results_wanted']}`\n"
        f"- Refresh rate: every `{settings['refresh_seconds']}` seconds\n"
        f"- Match threshold: `{settings.get('semantic_threshold', 0.30)}`\n"
        f"- ATS match threshold: `{settings.get('ats_semantic_threshold', settings.get('semantic_threshold', 0.30))}`"
    )
    log_panel_output("job_settings", channel_id, summary)
    return summary


def format_reddit_content_mode(include_nsfw: bool, include_spoiler: bool) -> str:
    if include_nsfw and include_spoiler:
        return "both"
    if include_nsfw:
        return "nsfw"
    if include_spoiler:
        return "spoiler"
    return "safe"


def parse_reddit_content_mode(raw_value: str) -> tuple[bool, bool]:
    normalized = str(raw_value).strip().lower()
    aliases = {
        "safe": (False, False),
        "safe only": (False, False),
        "spoiler": (False, True),
        "spoilers": (False, True),
        "nsfw": (True, False),
        "both": (True, True),
        "all": (True, True),
    }
    if normalized not in aliases:
        raise ValueError("Content filter must be one of: safe, spoiler, nsfw, both.")
    return aliases[normalized]


def parse_reddit_media_mode(raw_value: str) -> str:
    normalized = str(raw_value).strip().lower()
    aliases = {
        "image": "image",
        "images": "image",
        "video": "video",
        "videos": "video",
        "link": "link",
        "links": "link",
        "post link": "link",
        "post links": "link",
        "all": "all",
        "all media": "all",
    }
    if normalized not in aliases:
        raise ValueError("Media mode must be one of: image, video, link, all.")
    return aliases[normalized]


def parse_bounded_int(raw_value: str, field_name: str, minimum: int, maximum: int) -> int:
    try:
        value = int(str(raw_value).strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a whole number.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return value


def log_interaction_failure(context: str, exc: Exception) -> None:
    try:
        log_path = Path(__file__).resolve().parents[2] / ".interaction_errors.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stack = traceback.format_exc()
        payload = (
            f"[{context}] {type(exc).__name__}: {exc}\n"
            f"{stack}\n"
            f"{'-' * 80}\n"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
    except Exception:
        pass


async def _send_view_error(
    interaction: discord.Interaction,
    error: Exception,
    log_context: str,
    retry_command: str | None = None,
) -> None:
    """Shared on_error body for settings views/modals: log, then tell the
    user ephemerally (response if still open, followup otherwise). Was
    copy-pasted per view class with only the log context and retry hint
    varying."""
    log_interaction_failure(log_context, error)
    hint = f" Please run `{retry_command}` and try again." if retry_command else " Please try again."
    message = "Interaction failed." + hint
    if not interaction.response.is_done():
        await interaction.response.send_message(message, ephemeral=True)
    else:
        await interaction.followup.send(message, ephemeral=True)


# Module-level so tests can redirect it. Resolved inside the function this had
# no seam at all, and the modal tests appended straight into the file the live
# bot writes: 7,442 of 8,016 lines were synthetic channel_id=123 records before
# tests/conftest.py started isolating it.
INTERACTION_EVENTS_PATH = Path(__file__).resolve().parents[2] / ".interaction_events.log"


def log_interaction_event(context: str, **fields: Any) -> None:
    try:
        log_path = INTERACTION_EVENTS_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)
        parts = [f"ts={datetime.now(timezone.utc).isoformat()}", f"context={context}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(" | ".join(parts) + "\n")
    except Exception:
        pass


def log_panel_output(panel_name: str, channel_id: int, content: str) -> None:
    key = (panel_name, channel_id)
    previous = LAST_PANEL_OUTPUTS.get(key)
    LAST_PANEL_OUTPUTS[key] = content
    preview = content.replace("\n", " ")[:140]
    log_interaction_event(
        "panel_output",
        panel=panel_name,
        channel_id=channel_id,
        content_length=len(content),
        duplicate=int(previous == content),
        preview=preview,
    )


def format_reddit_settings_summary(store: RuntimeStore, channel_id: int) -> str:
    settings = store.get_reddit_settings(channel_id)
    flair = settings["flair_tags"] or "Any"
    flags = []
    if settings["include_nsfw"]:
        flags.append("NSFW")
    if settings["include_spoiler"]:
        flags.append("Spoilers")
    flag_text = ", ".join(flags) if flags else "Safe only"
    status = "Running" if settings["enabled"] else "Stopped"
    return (
        "Reddit media watcher for this channel:\n"
        f"- Status: `{status}`\n"
        f"- Subreddits: `r/{', r/'.join(str(s) for s in settings['subreddits'])}`\n"
        f"- Media mode: `{settings['media_mode']}`\n"
        f"- Sort: `{settings['sort']}`\n"
        f"- Time filter: `{settings['time_filter']}`\n"
        f"- Flair tags: `{flair}`\n"
        f"- Content filter: `{flag_text}`\n"
        f"- Items per check: `{settings['limit']}`\n"
        f"- Refresh rate: every `{settings['refresh_seconds']}` seconds"
    )


class ModeDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label=name, value=name, description=desc[:100])
            for name, desc in MODE_DESCRIPTIONS.items()
        ]
        super().__init__(placeholder="Choose a mode...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change the mode.", ephemeral=True)
            return
        selected_mode = self.values[0]
        self.store.set_mode(self.channel_id, selected_mode)
        await interaction.response.send_message(f"Mode set to `{selected_mode}` for this channel.")


class ModeDropdownView(discord.ui.View):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        super().__init__(timeout=None)
        self.add_item(ModeDropdown(store=store, channel_id=channel_id, owner_id=owner_id))

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item[Any]) -> None:
        await _send_view_error(interaction, error, "mode_dropdown_view.on_error", ".cmd")


class MaxItemsDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="5 items", value="5"),
            discord.SelectOption(label="10 items", value="10"),
            discord.SelectOption(label="20 items", value="20", description="Recommended"),
            discord.SelectOption(label="30 items", value="30"),
            discord.SelectOption(label="50 items", value="50"),
        ]
        super().__init__(placeholder="Max items", min_values=1, max_values=1, options=options, row=0)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return
        self.store.update_scrape_setting(self.channel_id, "max_items", int(self.values[0]))
        await interaction.response.edit_message(content=format_scrape_settings_summary(self.store, self.channel_id), view=self.view)


class TimeoutDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="10 seconds", value="10"),
            discord.SelectOption(label="20 seconds", value="20", description="Recommended"),
            discord.SelectOption(label="30 seconds", value="30"),
            discord.SelectOption(label="45 seconds", value="45"),
        ]
        super().__init__(placeholder="Timeout", min_values=1, max_values=1, options=options, row=1)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return
        self.store.update_scrape_setting(self.channel_id, "timeout_seconds", int(self.values[0]))
        await interaction.response.edit_message(content=format_scrape_settings_summary(self.store, self.channel_id), view=self.view)


class AiCleanupDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="AI cleanup: On", value="on"),
            discord.SelectOption(label="AI cleanup: Off", value="off"),
        ]
        super().__init__(placeholder="AI cleanup", min_values=1, max_values=1, options=options, row=2)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return
        self.store.update_scrape_setting(self.channel_id, "use_ai_cleanup", self.values[0] == "on")
        await interaction.response.edit_message(content=format_scrape_settings_summary(self.store, self.channel_id), view=self.view)


class ScrapeSettingsView(discord.ui.View):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        super().__init__(timeout=None)
        self.add_item(MaxItemsDropdown(store=store, channel_id=channel_id, owner_id=owner_id))
        self.add_item(TimeoutDropdown(store=store, channel_id=channel_id, owner_id=owner_id))
        self.add_item(AiCleanupDropdown(store=store, channel_id=channel_id, owner_id=owner_id))

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item[Any]) -> None:
        await _send_view_error(interaction, error, "scrape_settings_view.on_error", ".scrapecfg")


class JobSourceDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label=label, value=value)
            for label, value in job_service.jobspy_site_options()
        ]
        super().__init__(placeholder="Job sources", min_values=1, max_values=len(options), options=options, row=0)

    async def callback(self, interaction: discord.Interaction) -> None:
        self.store.update_job_setting(self.channel_id, "sites", list(self.values))
        await interaction.response.edit_message(content=format_job_settings_summary(self.store, self.channel_id), view=self.view)


class JobDateDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="Last 24 hours", value="24"),
            discord.SelectOption(label="Last 72 hours", value="72", description="Recommended"),
            discord.SelectOption(label="Last 7 days", value="168"),
            discord.SelectOption(label="Last 30 days", value="720"),
        ]
        super().__init__(placeholder="Date window", min_values=1, max_values=1, options=options, row=1)

    async def callback(self, interaction: discord.Interaction) -> None:
        self.store.update_job_setting(self.channel_id, "hours_old", int(self.values[0]))
        await interaction.response.edit_message(content=format_job_settings_summary(self.store, self.channel_id), view=self.view)


class JobRoleFilterDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="Internship", value="internship"),
            discord.SelectOption(label="Entry level", value="entry"),
            discord.SelectOption(label="Junior", value="junior"),
            discord.SelectOption(label="Mid level", value="mid"),
            discord.SelectOption(label="Senior", value="senior"),
        ]
        super().__init__(placeholder="Role filters (optional)", min_values=0, max_values=5, options=options, row=3)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return
        self.store.update_job_setting(self.channel_id, "role_filters", list(self.values))
        await interaction.response.edit_message(content=format_job_settings_summary(self.store, self.channel_id), view=self.view)


class JobRefreshDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="15 minutes", value="900", description="Recommended"),
            discord.SelectOption(label="30 minutes", value="1800"),
            discord.SelectOption(label="45 minutes", value="2700"),
            discord.SelectOption(label="60 minutes", value="3600"),
            discord.SelectOption(label="75 minutes", value="4500"),
            discord.SelectOption(label="90 minutes", value="5400"),
            discord.SelectOption(label="105 minutes", value="6300"),
            discord.SelectOption(label="120 minutes", value="7200"),
        ]
        super().__init__(placeholder="Refresh rate", min_values=1, max_values=1, options=options, row=2)

    async def callback(self, interaction: discord.Interaction) -> None:
        self.store.update_job_setting(self.channel_id, "refresh_seconds", int(self.values[0]))
        await interaction.response.edit_message(content=format_job_settings_summary(self.store, self.channel_id), view=self.view)


class JobTextModal(discord.ui.Modal, title="Edit Job Watcher"):
    def __init__(self, store: RuntimeStore, channel_id: int, panel_message: discord.Message | None = None, parent_view: discord.ui.View | None = None):
        super().__init__()
        self.store = store
        self.channel_id = channel_id
        self.panel_message = panel_message
        self.parent_view = parent_view

        # Build inputs per instance to avoid stale modal component metadata across reloads.
        self.keywords = discord.ui.TextInput(label="Keywords", placeholder="python developer", required=True, max_length=120)
        self.location = discord.ui.TextInput(label="Location", placeholder="United States", required=True, max_length=120)
        self.radius_miles = discord.ui.TextInput(label="Radius (miles)", placeholder="25", required=True, max_length=4)
        self.results_wanted = discord.ui.TextInput(label="Results per check", placeholder="10", required=True, max_length=3)
        self.role_filters = discord.ui.TextInput(
            label="Role filters (comma-separated)",
            placeholder="internship, entry, junior, mid, senior",
            required=False,
            max_length=100,
        )
        self.add_item(self.keywords)
        self.add_item(self.location)
        self.add_item(self.radius_miles)
        self.add_item(self.results_wanted)
        self.add_item(self.role_filters)

        current = self.store.get_job_settings(channel_id)
        self.keywords.default = str(current["keywords"])
        self.location.default = str(current["location"])
        self.radius_miles.default = str(current.get("radius_miles", 25))
        self.results_wanted.default = str(current["results_wanted"])
        role_filter_list = current.get("role_filters", [])
        self.role_filters.default = ", ".join(role_filter_list) if role_filter_list else ""
        log_interaction_event(
            "job_text_modal.init",
            channel_id=channel_id,
            panel_message_id=getattr(panel_message, "id", None),
            keywords=current.get("keywords", ""),
        )

    async def on_submit(self, interaction: discord.Interaction) -> None:
        log_interaction_event(
            "job_text_modal.submit.start",
            channel_id=self.channel_id,
            interaction_message_id=getattr(interaction.message, "id", None),
            user_id=getattr(getattr(interaction, "user", None), "id", None),
        )
        try:
            # Acknowledge immediately — must happen within 3 s or Discord shows "Interaction Failed"
            await interaction.response.defer(ephemeral=True)

            radius_miles = parse_bounded_int(str(self.radius_miles.value), "Radius", 1, 500)
            results_wanted = parse_bounded_int(str(self.results_wanted.value), "Results per check", 1, 400)
            raw_role_filters = str(self.role_filters.value).strip()
            _valid_roles = {"internship", "entry", "junior", "mid", "senior"}
            role_filter_list = [t.strip().lower() for t in raw_role_filters.split(",") if t.strip()] if raw_role_filters else []
            invalid = [r for r in role_filter_list if r not in _valid_roles]
            if invalid:
                raise ValueError(f"Invalid role filters: {', '.join(invalid)}. Use: internship, entry, junior, mid, senior")
        except ValueError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except Exception as exc:
            log_interaction_failure("job_text_modal.defer_or_parse", exc)
            if not interaction.response.is_done():
                await interaction.response.send_message("Interaction failed while validating input.", ephemeral=True)
            else:
                await interaction.followup.send("Interaction failed while validating input.", ephemeral=True)
            return

        try:
            self.store.update_job_setting(self.channel_id, "keywords", str(self.keywords.value).strip())
            self.store.update_job_setting(self.channel_id, "location", str(self.location.value).strip())
            self.store.update_job_setting(self.channel_id, "radius_miles", radius_miles)
            self.store.update_job_setting(self.channel_id, "results_wanted", results_wanted)
            self.store.update_job_setting(self.channel_id, "role_filters", role_filter_list)
            log_interaction_event(
                "job_text_modal.submit.saved",
                channel_id=self.channel_id,
                panel_message_id=getattr(self.panel_message, "id", None),
                user_id=getattr(getattr(interaction, "user", None), "id", None),
                keywords=str(self.keywords.value).strip(),
                location=str(self.location.value).strip(),
            )

            if self.panel_message is not None:
                try:
                    await self.panel_message.edit(
                        content=format_job_settings_summary(self.store, self.channel_id),
                        view=self.parent_view,
                    )
                except (discord.NotFound, discord.HTTPException):
                    pass
            await interaction.followup.send("​", ephemeral=True)
        except Exception as exc:
            log_interaction_failure("job_text_modal.persist_or_refresh", exc)
            await interaction.followup.send("Interaction failed while saving settings.", ephemeral=True)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await _send_view_error(interaction, error, "job_text_modal.on_error")


class JobExclusionTermsModal(discord.ui.Modal, title="Edit Filters"):
    def __init__(self, store: RuntimeStore, channel_id: int, panel_message: discord.Message | None = None, parent_view: discord.ui.View | None = None):
        super().__init__()
        self.store = store
        self.channel_id = channel_id
        self.panel_message = panel_message
        self.parent_view = parent_view

        self.exclusion_terms = discord.ui.TextInput(
            label="Exclusion terms (comma-separated)",
            placeholder="contract, remote, part.?time",
            required=False,
            max_length=500,
            style=discord.TextStyle.paragraph,
        )
        self.add_item(self.exclusion_terms)

        current = self.store.get_job_settings(channel_id)
        exclusion_list = current.get("exclusion_terms", [])
        self.exclusion_terms.default = ", ".join(exclusion_list) if exclusion_list else ""

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        raw_exclusions = str(self.exclusion_terms.value).strip()
        exclusion_list = [term.strip() for term in raw_exclusions.split(",") if term.strip()] if raw_exclusions else []

        try:
            self.store.update_job_setting(self.channel_id, "exclusion_terms", exclusion_list)
            if self.panel_message is not None:
                try:
                    await self.panel_message.edit(
                        content=format_job_settings_summary(self.store, self.channel_id),
                        view=self.parent_view,
                    )
                except (discord.NotFound, discord.HTTPException):
                    pass
            await interaction.followup.send("​", ephemeral=True)
        except Exception as exc:
            log_interaction_failure("job_exclusion_terms_modal.persist_or_refresh", exc)
            await interaction.followup.send("Interaction failed while saving filters.", ephemeral=True)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await _send_view_error(interaction, error, "job_exclusion_terms_modal.on_error")


class JobThresholdModal(discord.ui.Modal, title="Edit Match Threshold"):
    def __init__(self, store: RuntimeStore, channel_id: int, panel_message: discord.Message | None = None, parent_view: discord.ui.View | None = None):
        super().__init__()
        self.store = store
        self.channel_id = channel_id
        self.panel_message = panel_message
        self.parent_view = parent_view

        self.semantic_threshold = discord.ui.TextInput(
            label="Match threshold (0.0 – 1.0)",
            placeholder="0.30  — lower = more results, higher = stricter",
            required=True,
            max_length=5,
        )
        self.add_item(self.semantic_threshold)

        self.ats_semantic_threshold = discord.ui.TextInput(
            label="ATS match threshold (0.0 – 1.0)",
            placeholder="0.30  — separate threshold for ATS sources",
            required=False,
            max_length=5,
        )
        self.add_item(self.ats_semantic_threshold)

        current = self.store.get_job_settings(channel_id)
        self.semantic_threshold.default = str(current.get("semantic_threshold", 0.30))
        self.ats_semantic_threshold.default = str(current.get("ats_semantic_threshold", current.get("semantic_threshold", 0.30)))

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        raw = str(self.semantic_threshold.value).strip()
        try:
            value = float(raw)
            if not (0.0 <= value <= 1.0):
                raise ValueError()
        except ValueError:
            await interaction.followup.send("Threshold must be a number between 0.0 and 1.0.", ephemeral=True)
            return

        ats_value = value
        ats_raw = str(self.ats_semantic_threshold.value).strip()
        if ats_raw:
            try:
                ats_value = float(ats_raw)
                if not (0.0 <= ats_value <= 1.0):
                    raise ValueError()
            except ValueError:
                await interaction.followup.send("ATS threshold must be a number between 0.0 and 1.0.", ephemeral=True)
                return

        try:
            self.store.update_job_setting(self.channel_id, "semantic_threshold", value)
            self.store.update_job_setting(self.channel_id, "ats_semantic_threshold", ats_value)
            if self.panel_message is not None:
                try:
                    await self.panel_message.edit(
                        content=format_job_settings_summary(self.store, self.channel_id),
                        view=self.parent_view,
                    )
                except (discord.NotFound, discord.HTTPException):
                    pass
            await interaction.followup.send("​", ephemeral=True)
        except Exception as exc:
            log_interaction_failure("job_threshold_modal.persist_or_refresh", exc)
            await interaction.followup.send("Interaction failed while saving threshold.", ephemeral=True)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await _send_view_error(interaction, error, "job_threshold_modal.on_error")


class ResumeFileEditModal(discord.ui.Modal):
    """Edit one resume-cache text file, split across several modal fields.

    One 4000-char text input cannot hold a real baseinfo.txt or template.tex,
    so each file is chunked across ``RESUME_FILE_MODAL_PARTS`` fields on load
    and rejoined on submit. Every resume-cache editor -- base info,
    instructions, template -- is this class, so the three panel buttons behave
    identically; TemplateEditModal only adds a compile preview after saving.
    """

    # Overridden by subclasses that do slow work after saving and therefore
    # need Discord's "thinking" placeholder.
    defer_kwargs: dict[str, Any] = {"ephemeral": True}
    error_source = "resume_file_modal.on_error"

    def __init__(
        self,
        profile_dir: Path,
        file_name: str,
        panel_message: discord.Message | None = None,
        parent_view: discord.ui.View | None = None,
    ):
        super().__init__(title=f"Edit {file_name}")
        self.profile_dir = profile_dir
        self.file_name = file_name
        self.panel_message = panel_message
        self.parent_view = parent_view

        self.parts: list[discord.ui.TextInput] = []
        for index in range(RESUME_FILE_MODAL_PARTS):
            field = discord.ui.TextInput(
                label=f"{file_name} (Part {index + 1} of {RESUME_FILE_MODAL_PARTS})",
                required=False,
                max_length=DISCORD_MODAL_TEXT_LIMIT,
                style=discord.TextStyle.paragraph,
            )
            self.add_item(field)
            self.parts.append(field)

        self._truncated = False
        self._load_file()

    @property
    def file_path(self) -> Path:
        return self.profile_dir / self.file_name

    def _load_file(self) -> None:
        path = self.file_path
        if not path.exists():
            return
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return

        remaining = text
        for field in self.parts:
            field.default = remaining[:DISCORD_MODAL_TEXT_LIMIT]
            remaining = remaining[DISCORD_MODAL_TEXT_LIMIT:]

        if remaining:
            self._truncated = True

    async def on_submit(self, interaction: discord.Interaction) -> None:
        # Modal submit interactions need a deferred channel message response.
        await interaction.response.defer(**self.defer_kwargs)

        if self._truncated:
            await interaction.followup.send(
                (
                    f"`{self.file_name}` is longer than "
                    f"{DISCORD_MODAL_TEXT_LIMIT * RESUME_FILE_MODAL_PARTS} chars and was "
                    "truncated before editing. Not saving, to avoid losing the tail."
                ),
                ephemeral=True,
            )
            return

        combined = "".join(str(field.value) for field in self.parts)
        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(combined, encoding="utf-8")
        except OSError as exc:
            await interaction.followup.send(
                f"Failed to save `{self.file_name}`: {exc}", ephemeral=True
            )
            return

        await self._after_save(interaction, combined)

    async def _after_save(self, interaction: discord.Interaction, text: str) -> None:
        """What to report once the file is on disk. Subclasses may do more."""
        await interaction.followup.send(
            f"Saved `{self.file_name}` ({len(text):,} chars) in `{self.profile_dir.name}`.",
            ephemeral=True,
        )

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await _send_view_error(interaction, error, self.error_source)


class TemplateEditModal(ResumeFileEditModal):
    """The template.tex editor: the shared editor plus a compiled PDF preview."""

    defer_kwargs = {"thinking": True, "ephemeral": True}
    error_source = "template_edit_modal.on_error"

    def __init__(
        self,
        profile_dir: Path,
        panel_message: discord.Message | None = None,
        parent_view: discord.ui.View | None = None,
        scheduler: PriorityWorkScheduler | None = None,
    ):
        self.scheduler = scheduler
        super().__init__(
            profile_dir=profile_dir,
            file_name="template.tex",
            panel_message=panel_message,
            parent_view=parent_view,
        )

    async def _after_save(self, interaction: discord.Interaction, text: str) -> None:
        preview_result = await self._build_preview_message(self.file_path)
        if isinstance(preview_result, tuple):
            preview_message, preview_file = preview_result
        else:
            preview_message, preview_file = str(preview_result), None

        if preview_file is not None:
            try:
                await interaction.followup.send(preview_message, file=preview_file, ephemeral=True)
                return
            except TypeError:
                # Test doubles may not support file uploads; fall back to text-only response.
                pass

        await interaction.followup.send(preview_message, ephemeral=True)

    async def _build_preview_message(self, template_path: Path) -> str | tuple[str, discord.File]:
        preview_log_path = self.profile_dir / "template.preview.log"

        try:
            template_text = template_path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Template updated successfully, but preview could not be read: {exc}"

        compile_args = (
            compile_latex_to_pdf,
            template_text,
            f"{self.profile_dir.name}-template-preview",
            template_path,
            preview_log_path,
            True,
        )
        if self.scheduler is not None:
            compile_result = await self.scheduler.run(
                *compile_args, tier=INTERACTIVE, label=scheduler_labels.RESUME_TEMPLATE_PREVIEW_COMPILE
            )
        else:
            compile_result = await asyncio.to_thread(*compile_args)
        if compile_result.status != "ok" or not compile_result.pdf_bytes:
            return (
                "Template updated successfully, but preview generation failed: "
                f"{compile_result.message}"
            )

        preview_name = f"{self.profile_dir.name}-template-preview.pdf"
        preview_file = discord.File(io.BytesIO(compile_result.pdf_bytes), filename=preview_name)
        return ("Template updated successfully. Preview PDF attached.", preview_file)


class JobSettingsView(discord.ui.View):
    WATCHER_TOGGLE_BUTTON_ID = "job_settings:watcher_toggle"
    RESUME_BASEINFO_BUTTON_ID = "job_settings:resume_baseinfo"
    RESUME_INSTRUCTIONS_BUTTON_ID = "job_settings:resume_instructions"
    TEMPLATE_BUTTON_ID = "job_settings:template"

    def __init__(
        self,
        store: RuntimeStore,
        manager: WatcherManager,
        channel_id: int,
        owner_id: int,
        resume_profile_dir: Path,
    ):
        # Keep this panel active to avoid Discord "Interaction Failed" from expired components.
        super().__init__(timeout=None)
        self.store = store
        self.manager = manager
        self.channel_id = channel_id
        self.owner_id = owner_id
        self.resume_profile_dir = resume_profile_dir
        self.resume_cache_root = resume_profile_dir.parent
        log_interaction_event(
            "job_settings_view.init",
            channel_id=channel_id,
            owner_id=owner_id,
        )
        self.add_item(JobSourceDropdown(store=store, channel_id=channel_id, owner_id=owner_id))
        self.add_item(JobDateDropdown(store=store, channel_id=channel_id, owner_id=owner_id))
        self.add_item(JobRefreshDropdown(store=store, channel_id=channel_id, owner_id=owner_id))
        # Both of these were built and then never attached to anything.
        # JobRoleFilterDropdown has had options, a callback and its own row
        # since it was written; role_filters could only be reached through the
        # free-text modal, where "internship" had to be typed exactly.
        # JobRegionDropdown is new here only in the sense that the setting it
        # writes had no control at all -- see its docstring.
        self.sync_watcher_toggle()
        self.sync_region_toggle()

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item[Any]) -> None:
        await _send_view_error(interaction, error, "job_settings_view.on_error", ".job")

    def _is_allowed(self, interaction: discord.Interaction) -> bool:
        user_id = interaction.user.id
        if user_id == self.owner_id:
            return True
        guild_owner_id = getattr(getattr(interaction, "guild", None), "owner_id", None)
        if guild_owner_id is not None and user_id == guild_owner_id:
            return True
        main_user_id = getattr(getattr(self.manager, "config", None), "main_user_id", None)
        if main_user_id is not None and user_id == main_user_id:
            return True
        return False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not self._is_allowed(interaction):
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return False
        return True

    @staticmethod
    def _interaction_profile_name(interaction: discord.Interaction) -> str | None:
        user = getattr(interaction, "user", None)
        if user is None:
            return None
        value = getattr(user, "name", None)
        if isinstance(value, str) and value.strip():
            return value
        return None

    def _resolve_profile_dir_for_clicker(self, interaction: discord.Interaction) -> Path:
        return self.resume_profile_dir

    @discord.ui.button(label="Edit exclusions", style=discord.ButtonStyle.secondary, row=3)
    async def edit_filters(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        modal = JobExclusionTermsModal(store=self.store, channel_id=self.channel_id, panel_message=interaction.message, parent_view=self)
        _sanitize_modal_text_input_labels(modal)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Advanced", style=discord.ButtonStyle.secondary, row=3)
    async def edit_threshold(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        modal = JobThresholdModal(store=self.store, channel_id=self.channel_id, panel_message=interaction.message, parent_view=self)
        _sanitize_modal_text_input_labels(modal)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Edit text/results", style=discord.ButtonStyle.primary, row=3)
    async def edit_text(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        log_interaction_event(
            "job_settings_view.edit_text",
            channel_id=self.channel_id,
            owner_id=self.owner_id,
            user_id=getattr(getattr(interaction, "user", None), "id", None),
            message_id=getattr(getattr(interaction, "message", None), "id", None),
        )
        modal = JobTextModal(store=self.store, channel_id=self.channel_id, panel_message=interaction.message, parent_view=self)
        label_lengths = _sanitize_modal_text_input_labels(modal)
        log_interaction_event(
            "job_settings_view.edit_text.modal_labels",
            channel_id=self.channel_id,
            lengths=label_lengths,
        )
        await interaction.response.send_modal(modal)

    async def _open_resume_file_modal(self, interaction: discord.Interaction, file_name: str) -> None:
        profile_dir = self._resolve_profile_dir_for_clicker(interaction)
        if file_name == "template.tex":
            modal: ResumeFileEditModal = TemplateEditModal(
                profile_dir=profile_dir,
                panel_message=interaction.message,
                parent_view=self,
                scheduler=self.manager.scheduler,
            )
        else:
            modal = ResumeFileEditModal(
                profile_dir=profile_dir,
                file_name=file_name,
                panel_message=interaction.message,
                parent_view=self,
            )
        _sanitize_modal_text_input_labels(modal)
        await interaction.response.send_modal(modal)

    # The three resume-cache editors sit together on row 4, same style, same
    # modal shape -- they are the same editor pointed at different files.
    @discord.ui.button(
        label="Edit base info",
        style=discord.ButtonStyle.green,
        row=4,
        custom_id=RESUME_BASEINFO_BUTTON_ID,
    )
    async def edit_resume_baseinfo(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._open_resume_file_modal(interaction, "baseinfo.txt")

    @discord.ui.button(
        label="Edit instructions",
        style=discord.ButtonStyle.green,
        row=4,
        custom_id=RESUME_INSTRUCTIONS_BUTTON_ID,
    )
    async def edit_resume_instructions(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._open_resume_file_modal(interaction, "instructions.txt")

    @discord.ui.button(
        label="Edit template",
        style=discord.ButtonStyle.green,
        row=4,
        custom_id=TEMPLATE_BUTTON_ID,
    )
    async def edit_template(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._open_resume_file_modal(interaction, "template.tex")

    def _watcher_is_enabled(self) -> bool:
        try:
            return bool(self.store.get_job_settings(self.channel_id).get("enabled"))
        except Exception:
            return False

    def _watcher_toggle_button(self) -> discord.ui.Button | None:
        for child in self.children:
            if getattr(child, "custom_id", None) == self.WATCHER_TOGGLE_BUTTON_ID:
                return child  # type: ignore[return-value]
        return None

    REGION_TOGGLE_BUTTON_ID = "job_settings:region_toggle"

    def _region_toggle_button(self) -> discord.ui.Button | None:
        for child in self.children:
            if getattr(child, "custom_id", None) == self.REGION_TOGGLE_BUTTON_ID:
                return child  # type: ignore[return-value]
        return None

    def _region_allows_us(self) -> bool:
        try:
            return bool(self.store.get_job_settings(self.channel_id).get("allow_north_america", False))
        except Exception:
            return False

    def sync_region_toggle(self) -> None:
        """Label the region button with the state it is in, not the action.

        The watcher button opposite it is labelled with its action ("Stop
        watcher"), because starting and stopping are verbs. A region is a state,
        and "Switch to Canada only" next to a summary line reading "Region
        filter: Canada only" reads as a contradiction, so this names the mode.
        """
        button = self._region_toggle_button()
        if button is None:
            return
        if self._region_allows_us():
            button.label = "Region: Canada + US"
            button.style = discord.ButtonStyle.primary
        else:
            button.label = "Region: Canada only"
            button.style = discord.ButtonStyle.secondary

    @discord.ui.button(
        label="Region: Canada only",
        style=discord.ButtonStyle.secondary,
        row=3,
        custom_id=REGION_TOGGLE_BUTTON_ID,
    )
    async def toggle_region(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """Widen or narrow the region gate.

        allow_north_america gates the archive (job_match._matches_channel_region),
        the live scrape (job_service.filter_rows_by_region) and the watcher, and
        until now had no control anywhere -- format_job_settings_summary printed
        which mode was active while offering no way to change it. It is not
        cosmetic: the ATS fleet is global, and over a measured 3-day window 1.9%
        of archived ATS rows were Canadian against roughly 29% US, so a channel
        left on the default discards the largest single block of what the bot
        has already collected.

        A button rather than a dropdown because the panel's five action rows are
        full: rows 0-2 are selects, and a select needs a whole row. This is also
        why JobRoleFilterDropdown has never been attached -- role_filters stays
        reachable through the text modal.

        Canada is kept under both settings, matching filter_rows_by_region, so
        this only ever widens what reaches the channel.
        """
        if not self._is_allowed(interaction):
            await interaction.response.send_message(
                "Only the channel owner can change this.", ephemeral=True
            )
            return
        self.store.update_job_setting(
            self.channel_id, "allow_north_america", not self._region_allows_us()
        )
        self.sync_region_toggle()
        await interaction.response.edit_message(
            content=format_job_settings_summary(self.store, self.channel_id), view=self
        )

    def sync_watcher_toggle(self) -> None:
        """Point the single watcher button at whichever action is available now.

        Called on construction and after every toggle, so the label always
        describes what clicking it will do rather than what state it is in.
        """
        button = self._watcher_toggle_button()
        if button is None:
            return
        if self._watcher_is_enabled():
            button.label = "Stop watcher"
            button.style = discord.ButtonStyle.danger
        else:
            button.label = "Start watcher"
            button.style = discord.ButtonStyle.success

    @discord.ui.button(
        label="Start watcher",
        style=discord.ButtonStyle.success,
        row=4,
        custom_id=WATCHER_TOGGLE_BUTTON_ID,
    )
    async def toggle_watcher(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if self._watcher_is_enabled():
            self.manager.stop_job_watcher(self.channel_id)
        elif not self.manager.start_job_watcher(self.channel_id):
            # Leave the button alone: nothing changed, so re-rendering the panel
            # would only redraw the same state.
            await interaction.response.send_message(
                "Another scraper process is running in this channel.", ephemeral=True
            )
            return

        self.sync_watcher_toggle()
        await interaction.response.edit_message(
            content=format_job_settings_summary(self.store, self.channel_id), view=self
        )



class RedditSortDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="New", value="new"),
            discord.SelectOption(label="Top", value="top"),
            discord.SelectOption(label="Trending", value="trending"),
            discord.SelectOption(label="Rising", value="rising"),
            discord.SelectOption(label="Hot", value="hot"),
        ]
        super().__init__(placeholder="Sort order", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return
        self.store.update_reddit_setting(self.channel_id, "sort", self.values[0])
        await interaction.response.edit_message(content=format_reddit_settings_summary(self.store, self.channel_id), view=self.view)


class RedditTimeDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="Hour", value="hour"),
            discord.SelectOption(label="Day", value="day", description="Recommended"),
            discord.SelectOption(label="Week", value="week"),
            discord.SelectOption(label="Month", value="month"),
            discord.SelectOption(label="Year", value="year"),
            discord.SelectOption(label="All time", value="all"),
        ]
        super().__init__(placeholder="Time filter", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return
        self.store.update_reddit_setting(self.channel_id, "time_filter", self.values[0])
        await interaction.response.edit_message(content=format_reddit_settings_summary(self.store, self.channel_id), view=self.view)


class RedditMediaDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="All media types", value="all"),
            discord.SelectOption(label="Images", value="image"),
            discord.SelectOption(label="Videos", value="video"),
            discord.SelectOption(label="Post links", value="link"),
        ]
        super().__init__(placeholder="Media mode", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return
        self.store.update_reddit_setting(self.channel_id, "media_mode", self.values[0])
        await interaction.response.edit_message(content=format_reddit_settings_summary(self.store, self.channel_id), view=self.view)


class RedditRefreshDropdown(discord.ui.Select):
    def __init__(self, store: RuntimeStore, channel_id: int, owner_id: int):
        self.store = store
        self.channel_id = channel_id
        self.owner_id = owner_id
        options = [
            discord.SelectOption(label="1 minute", value="60"),
            discord.SelectOption(label="2 minutes", value="120"),
            discord.SelectOption(label="5 minutes", value="300", description="Recommended"),
            discord.SelectOption(label="10 minutes", value="600"),
            discord.SelectOption(label="15 minutes", value="900"),
        ]
        super().__init__(placeholder="Refresh rate", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return
        self.store.update_reddit_setting(self.channel_id, "refresh_seconds", int(self.values[0]))
        await interaction.response.edit_message(content=format_reddit_settings_summary(self.store, self.channel_id), view=self.view)


class RedditTextModal(discord.ui.Modal, title="Edit Reddit Watcher"):
    subreddit = discord.ui.TextInput(label="Subreddits (comma-separated)", placeholder="wallpapers, earthporn", required=True, max_length=200)
    flair_tags = discord.ui.TextInput(label="Flair tags (comma-separated)", placeholder="Desktop, Anime", required=False, max_length=120)
    media_mode = discord.ui.TextInput(label="Media mode", placeholder="image | video | link | all", required=True, max_length=20)
    content_filter = discord.ui.TextInput(label="Content filter", placeholder="safe | spoiler | nsfw | both", required=True, max_length=12)
    limit = discord.ui.TextInput(label="Items per check", placeholder="10", required=True, max_length=3)

    def __init__(self, store: RuntimeStore, channel_id: int):
        super().__init__()
        self.store = store
        self.channel_id = channel_id
        current = self.store.get_reddit_settings(channel_id)
        self.subreddit.default = ", ".join(str(s) for s in current["subreddits"])
        self.flair_tags.default = str(current["flair_tags"])
        self.media_mode.default = str(current["media_mode"])
        self.content_filter.default = format_reddit_content_mode(bool(current["include_nsfw"]), bool(current["include_spoiler"]))
        self.limit.default = str(current["limit"])

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            media_mode = parse_reddit_media_mode(str(self.media_mode.value))
            include_nsfw, include_spoiler = parse_reddit_content_mode(str(self.content_filter.value))
            limit = parse_bounded_int(str(self.limit.value), "Items per check", 1, 100)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return

        parsed_subs = [(s[2:] if s.lower().startswith("r/") else s).lower() for s in (p.strip() for p in str(self.subreddit.value).split(",")) if s]
        if not parsed_subs:
            parsed_subs = ["wallpapers"]
        self.store.update_reddit_setting(self.channel_id, "subreddits", parsed_subs)
        self.store.update_reddit_setting(self.channel_id, "flair_tags", str(self.flair_tags.value).strip())
        self.store.update_reddit_setting(self.channel_id, "media_mode", media_mode)
        self.store.update_reddit_setting(self.channel_id, "include_nsfw", include_nsfw)
        self.store.update_reddit_setting(self.channel_id, "include_spoiler", include_spoiler)
        self.store.update_reddit_setting(self.channel_id, "limit", limit)
        await interaction.response.send_message(format_reddit_settings_summary(self.store, self.channel_id), ephemeral=True)


class RedditSettingsView(discord.ui.View):
    def __init__(self, store: RuntimeStore, manager: WatcherManager, channel_id: int, owner_id: int):
        super().__init__(timeout=None)
        self.store = store
        self.manager = manager
        self.channel_id = channel_id
        self.owner_id = owner_id
        self.add_item(RedditSortDropdown(store=store, channel_id=channel_id, owner_id=owner_id))
        self.add_item(RedditTimeDropdown(store=store, channel_id=channel_id, owner_id=owner_id))
        self.add_item(RedditMediaDropdown(store=store, channel_id=channel_id, owner_id=owner_id))
        self.add_item(RedditRefreshDropdown(store=store, channel_id=channel_id, owner_id=owner_id))

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item[Any]) -> None:
        await _send_view_error(interaction, error, "reddit_settings_view.on_error", ".reddit")

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the menu owner can change settings.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Edit subreddit/media/filter", style=discord.ButtonStyle.primary, row=4)
    async def edit_text(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(RedditTextModal(store=self.store, channel_id=self.channel_id))

    @discord.ui.button(label="Start watcher", style=discord.ButtonStyle.success, row=4)
    async def start(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.manager.start_reddit_watcher(self.channel_id):
            await interaction.response.send_message("Another scraper process is running in this channel.", ephemeral=True)
            return
        await interaction.response.edit_message(content=format_reddit_settings_summary(self.store, self.channel_id), view=self)

    @discord.ui.button(label="Stop watcher", style=discord.ButtonStyle.danger, row=4)
    async def stop(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.manager.stop_reddit_watcher(self.channel_id)
        await interaction.response.edit_message(content=format_reddit_settings_summary(self.store, self.channel_id), view=self)
