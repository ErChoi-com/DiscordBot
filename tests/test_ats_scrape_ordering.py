"""Ordering the ATS fleet from the countries the enabled channels actually want.

The ATS loop is process-wide and channel-agnostic, so there is no single channel
to ask which region matters. The union of what every enabled channel wants is
the honest answer, and it keeps a channel scoped to one country from starving a
channel scoped to another.

The ordering itself is a stable partition -- every slug still gets submitted, so
the scrape does the same work and makes no extra request. It only decides who is
inside the part that ran when a cycle is cut off at its budget.

This is also the first caller of `scrape_ats_platform(company_slugs=...)`, a
parameter that has existed unused since it was written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from watchers.manager import WatcherManager  # noqa: E402


class _Store:
    def __init__(self, channels: dict[int, dict]):
        self.channel_job_settings = channels


def _manager(channels: dict[int, dict]) -> WatcherManager:
    """A manager with only the attributes these methods touch."""
    mgr = WatcherManager.__new__(WatcherManager)
    mgr.store = _Store(channels)  # type: ignore[attr-defined]
    return mgr


# ── which countries the scrape should favour ─────────────────────────────────

def test_an_enabled_channels_location_decides_the_country():
    mgr = _manager({1: {"enabled": True, "location": "Canada"}})
    assert mgr._priority_countries() == ["CA"]


def test_a_disabled_channel_does_not_get_a_vote():
    """A stopped watcher is not scraping, so letting it steer the shared loop
    would spend the priority slots on a channel receiving nothing.
    """
    mgr = _manager({
        1: {"enabled": False, "location": "Germany"},
        2: {"enabled": True, "location": "Canada"},
    })
    assert mgr._priority_countries() == ["CA"]


def test_every_enabled_channel_is_represented():
    """Two channels scoped to different countries must both be served; taking
    only the first would starve the second permanently.
    """
    mgr = _manager({
        1: {"enabled": True, "location": "Canada"},
        2: {"enabled": True, "location": "Germany"},
    })
    assert set(mgr._priority_countries()) == {"CA", "DE"}


def test_allow_north_america_adds_the_us():
    mgr = _manager({1: {"enabled": True, "location": "Canada", "allow_north_america": True}})
    assert set(mgr._priority_countries()) == {"CA", "US"}


def test_a_country_is_not_repeated_when_several_channels_share_it():
    mgr = _manager({
        1: {"enabled": True, "location": "Canada"},
        2: {"enabled": True, "location": "Canada"},
    })
    assert mgr._priority_countries() == ["CA"]


def test_an_unrecognisable_location_contributes_nothing_rather_than_guessing():
    mgr = _manager({1: {"enabled": True, "location": "Anywhere Really"}})
    assert mgr._priority_countries() == []


def test_no_enabled_channels_means_no_opinion():
    assert _manager({})._priority_countries() == []


# ── the ordering handed to the scraper ───────────────────────────────────────

def test_preferred_boards_are_asked_first(monkeypatch):
    from services import ats_service
    from services.jba import geo_priority

    monkeypatch.setattr(
        ats_service, "load_company_lists", lambda: {"greenhouse": ["zeta", "acme", "beta"]}
    )
    monkeypatch.setattr(
        geo_priority, "slugs_for", lambda country, *a, **k: frozenset({"acme"}) if country == "CA" else frozenset()
    )

    mgr = _manager({1: {"enabled": True, "location": "Canada"}})
    assert mgr._ordered_slugs("greenhouse") == ["acme", "zeta", "beta"]


def test_ordering_never_drops_a_board(monkeypatch):
    """The property that makes this safe to put in front of the scraper: the
    fleet handed back is the same fleet, so the cycle submits the same work.
    """
    from services import ats_service
    from services.jba import geo_priority

    fleet = [f"c{i}" for i in range(300)]
    monkeypatch.setattr(ats_service, "load_company_lists", lambda: {"lever": list(fleet)})
    monkeypatch.setattr(
        geo_priority, "slugs_for", lambda country, *a, **k: frozenset({"c7", "c250"})
    )

    got = _manager({1: {"enabled": True, "location": "Canada"}})._ordered_slugs("lever")
    assert sorted(got) == sorted(fleet)


def test_no_wanted_country_leaves_the_scraper_to_load_its_own_fleet(monkeypatch):
    """None means "no opinion", which is what a checkout with no archive or no
    enabled channel gets -- identical to the behaviour before this existed.
    """
    from services import ats_service

    monkeypatch.setattr(ats_service, "load_company_lists", lambda: {"lever": ["a", "b"]})
    assert _manager({})._ordered_slugs("lever") is None


def test_an_unknown_platform_yields_no_opinion(monkeypatch):
    from services import ats_service

    monkeypatch.setattr(ats_service, "load_company_lists", lambda: {})
    mgr = _manager({1: {"enabled": True, "location": "Canada"}})
    assert mgr._ordered_slugs("nonesuch") is None


def test_a_failure_while_ordering_does_not_take_the_scrape_down(monkeypatch):
    """Ordering is an optimisation. If it raises, the scrape must still run
    with its own fleet rather than the platform being skipped.
    """
    from services import ats_service

    def _boom():
        raise RuntimeError("archive unreadable")

    monkeypatch.setattr(ats_service, "load_company_lists", _boom)
    mgr = _manager({1: {"enabled": True, "location": "Canada"}})
    assert mgr._ordered_slugs("greenhouse") is None


def test_boards_wanted_by_any_enabled_channel_are_promoted(monkeypatch):
    """The union, not the first channel: a German-scoped board must reach the
    front even when a Canadian channel is listed first.
    """
    from services import ats_service
    from services.jba import geo_priority

    monkeypatch.setattr(
        ats_service, "load_company_lists", lambda: {"ashby": ["x", "de_board", "y"]}
    )
    monkeypatch.setattr(
        geo_priority, "slugs_for",
        lambda country, *a, **k: frozenset({"de_board"}) if country == "DE" else frozenset(),
    )

    mgr = _manager({
        1: {"enabled": True, "location": "Canada"},
        2: {"enabled": True, "location": "Germany"},
    })
    assert mgr._ordered_slugs("ashby")[0] == "de_board"
