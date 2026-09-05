"""The region gate, reachable from the job settings panel.

`allow_north_america` gates the archive (job_match._matches_channel_region), the
live scrape (job_service.filter_rows_by_region) and the watcher (manager.py) --
and had no control anywhere. format_job_settings_summary printed which mode was
active, so the panel reported a setting it offered no way to change.

It is not cosmetic. The ATS fleet is global: over a measured 3-day window 1.9%
of archived ATS rows were Canadian against roughly 29% US, so a channel left on
the default discards the largest single block of what the bot already collected.

A button rather than a dropdown because the panel's five action rows are full --
rows 0-2 are selects and a select occupies a whole row. That is also why
JobRoleFilterDropdown has never been attached to anything; role_filters remains
reachable through the free-text modal.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import discord
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.views import JobSettingsView  # noqa: E402


class _Store:
    def __init__(self, allow_north_america: bool = False):
        self.settings: dict[str, object] = {
            "enabled": True,
            "allow_north_america": allow_north_america,
            "sites": ["indeed"], "keywords": "k", "location": "Canada",
            "radius_miles": 25, "role_filters": [], "exclusion_terms": [],
            "hours_old": 24, "results_wanted": 10, "refresh_seconds": 900,
            "semantic_threshold": 0.3, "ats_semantic_threshold": 0.3,
        }
        self.writes: list[tuple[str, object]] = []

    def get_job_settings(self, channel_id: int) -> dict[str, object]:
        return dict(self.settings)

    def update_job_setting(self, channel_id: int, key: str, value: object) -> None:
        self.writes.append((key, value))
        self.settings[key] = value


class _Interaction:
    def __init__(self, user_id: int = 456):
        self.edited: dict[str, object] = {}
        self.sent: list[str] = []
        self.user = type("U", (), {"id": user_id})()
        self.guild = None
        outer = self

        class _Resp:
            def is_done(self): return False
            async def edit_message(self, **kw): outer.edited = kw
            async def send_message(self, msg, **kw): outer.sent.append(msg)

        self.response = _Resp()


def _panel(store: _Store) -> JobSettingsView:
    return JobSettingsView(
        store=store, manager=None, channel_id=123, owner_id=456,
        resume_profile_dir=Path("."),
    )


def _button(view: JobSettingsView) -> discord.ui.Button:
    btn = view._region_toggle_button()
    assert btn is not None, "the region control is not on the panel"
    return btn


# ── it exists and fits ───────────────────────────────────────────────────────

def test_the_panel_carries_a_region_control():
    assert _button(_panel(_Store())) is not None


def test_the_panel_still_fits_discords_row_budget():
    """Five action rows, width 5 each; a select fills a row on its own. Adding
    a control that does not fit silently fails to render, which would put the
    setting straight back out of reach.
    """
    view = _panel(_Store())
    width: dict[int, int] = {}
    for item in view.children:
        w = 5 if isinstance(item, discord.ui.Select) else 1
        width[item.row] = width.get(item.row, 0) + w
    assert all(r <= 4 for r in width), f"a row is outside Discord's 0-4 range: {sorted(width)}"
    assert all(v <= 5 for v in width.values()), f"a row overflows: {width}"


def test_no_two_selects_share_a_row():
    view = _panel(_Store())
    rows = [i.row for i in view.children if isinstance(i, discord.ui.Select)]
    assert len(rows) == len(set(rows)), f"selects collide: {rows}"


# ── the label reports the state ──────────────────────────────────────────────

def test_the_label_reports_canada_only_by_default():
    assert "Canada only" in (_button(_panel(_Store(False))).label or "")


def test_the_label_reports_north_america_when_the_gate_is_open():
    assert "Canada + US" in (_button(_panel(_Store(True))).label or "")


def test_the_label_is_refreshed_after_a_toggle():
    store = _Store(False)
    view = _panel(store)
    asyncio.run(view.toggle_region.callback(_Interaction()))
    assert "Canada + US" in (_button(view).label or ""), "the label went stale after toggling"


# ── it writes what the filters read ──────────────────────────────────────────

def test_toggling_from_canada_opens_the_gate():
    store = _Store(False)
    asyncio.run(_panel(store).toggle_region.callback(_Interaction()))
    assert ("allow_north_america", True) in store.writes


def test_toggling_from_north_america_closes_it_again():
    store = _Store(True)
    asyncio.run(_panel(store).toggle_region.callback(_Interaction()))
    assert ("allow_north_america", False) in store.writes


def test_the_stored_value_is_a_bool():
    """Every consumer does bool(settings.get("allow_north_america", False)), so
    a stored string would be truthy and make "Canada only" silently mean both.
    """
    store = _Store(False)
    view = _panel(store)
    for _ in range(2):
        asyncio.run(view.toggle_region.callback(_Interaction()))
    assert all(isinstance(v, bool) for k, v in store.writes if k == "allow_north_america")


def test_toggling_twice_returns_to_the_original_setting():
    store = _Store(False)
    view = _panel(store)
    for _ in range(2):
        asyncio.run(view.toggle_region.callback(_Interaction()))
    assert store.settings["allow_north_america"] is False


def test_the_summary_is_refreshed_so_it_cannot_contradict_the_button():
    store = _Store(False)
    inter = _Interaction()
    asyncio.run(_panel(store).toggle_region.callback(inter))
    assert "North America" in str(inter.edited.get("content", ""))


# ── permission ───────────────────────────────────────────────────────────────

def test_a_stranger_cannot_change_the_region_and_nothing_is_written():
    """Same gate the rest of the panel uses. A write that happened before the
    refusal would leave the setting changed and the user told it was not.
    """
    store = _Store(False)
    inter = _Interaction(user_id=999)
    asyncio.run(_panel(store).toggle_region.callback(inter))

    assert store.writes == [], "a rejected click still changed the setting"
    assert inter.sent and "owner" in inter.sent[0].lower()


@pytest.mark.parametrize("start,expected", [(False, True), (True, False)])
def test_each_starting_state_flips_to_the_other(start, expected):
    store = _Store(start)
    asyncio.run(_panel(store).toggle_region.callback(_Interaction()))
    assert store.settings["allow_north_america"] is expected
