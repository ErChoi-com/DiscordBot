"""Asking Canada-yielding company boards first, learned from the archive.

The ATS fleet is collected globally -- neither harvester nor scraper has any
notion of country -- while a channel is usually scoped to one. Measured over a
3-day window, 1.9% of archived ATS rows were Canadian against roughly 29% US.

This module orders; it never filters. Every slug handed to rank() comes back,
so a cycle submits exactly the work it would have anyway. The order only
decides who is inside the part that ran when a cycle is cut off.

Hermetic: no archive, no network. The scan is exercised through `build`, which
takes rows directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.jba import geo_priority as G  # noqa: E402


# ── the CA ambiguity, which is the whole difficulty ──────────────────────────

@pytest.mark.parametrize("location", [
    "Toronto, ON, CA",
    "Montréal, Québec, ca",
    "Vancouver, British Columbia, Canada",
    "Ottawa, Ontario",
    "Calgary, AB, CA",
    "Canada",
])
def test_canadian_locations_are_recognised(location):
    assert G._is_canadian(location) is True


@pytest.mark.parametrize("location", [
    "San Francisco, CA",          # California, not Canada -- the trap
    "Los Angeles, CA, US",
    "Austin, TX, us",
    "Boston, MA, US",
    "Dubai, AE",
    "Berlin, DE",
    "Remote, CA",                 # bare tail, no province beside it
    "",
])
def test_non_canadian_locations_are_rejected(location):
    assert G._is_canadian(location) is False


def test_a_bare_ca_tail_needs_a_province_beside_it():
    """"Toronto, ON, CA" and "San Francisco, CA" end in the same two letters.
    A tail alone can never decide it; a province code is what separates them.
    """
    assert G._is_canadian("Somewhere, ON, CA") is True
    assert G._is_canadian("Somewhere, CA") is False


# ── building the set ─────────────────────────────────────────────────────────

def test_a_company_with_one_canadian_posting_qualifies():
    rows = [("acme", "Austin, TX, us"), ("acme", "Toronto, ON, CA")]
    assert G.build(rows) == {"acme"}


def test_a_company_with_no_canadian_posting_does_not():
    assert G.build([("acme", "Austin, TX, us"), ("acme", "Berlin, DE")]) == set()


def test_slugs_are_matched_case_insensitively():
    assert G.build([("ACME", "Toronto, ON, CA")]) == {"acme"}


def test_rows_without_a_company_are_skipped():
    assert G.build([("", "Toronto, ON, CA"), ("  ", "Ottawa, Ontario")]) == set()


# ── workday's slug shape ─────────────────────────────────────────────────────

def test_a_workday_slug_matches_on_its_tenant():
    """The fleet stores `tenant|host|site`; the archive records only the tenant.
    Before this, workday reported 0 Canada-yielding companies out of 14,990 --
    an exact match found nothing at all.
    """
    assert "ahri" in G.match_keys("ahri|wd3|ahri1")
    assert G.rank(["ahri|wd3|ahri1", "other|wd1|x"], {"ahri"}) == ["ahri|wd3|ahri1", "other|wd1|x"]


def test_a_plain_slug_yields_only_itself():
    assert G.match_keys("acme") == ("acme",)


def test_match_keys_keeps_the_full_slug_too():
    """A platform that does record the full triple must still match."""
    assert "ahri|wd3|ahri1" in G.match_keys("ahri|wd3|ahri1")


# ── ranking: order only, never membership ────────────────────────────────────

def test_ranking_moves_preferred_slugs_to_the_front():
    got = G.rank(["a", "b", "c", "d"], {"c", "a"})
    assert got[:2] == ["a", "c"]


def test_ranking_never_drops_or_duplicates_a_slug():
    """The property that makes this safe: a cycle submits exactly the work it
    would have submitted anyway. If this can lose a slug it is a filter, and a
    filter is what the design promises not to be.
    """
    fleet = [f"c{i}" for i in range(500)]
    preferred = {f"c{i}" for i in range(0, 500, 7)}
    got = G.rank(fleet, preferred)
    assert sorted(got) == sorted(fleet)
    assert len(got) == len(fleet)


def test_relative_order_is_preserved_within_both_groups():
    got = G.rank(["a", "b", "c", "d", "e"], {"d", "b"})
    assert got == ["b", "d", "a", "c", "e"]


def test_an_empty_preference_set_changes_nothing():
    fleet = ["a", "b", "c"]
    assert G.rank(fleet, set()) == fleet


def test_every_slug_preferred_changes_nothing_either():
    fleet = ["a", "b", "c"]
    assert G.rank(fleet, {"a", "b", "c"}) == fleet


def test_ranking_an_empty_fleet_is_not_an_error():
    assert G.rank([], {"a"}) == []


# ── cache ────────────────────────────────────────────────────────────────────

def test_a_missing_cache_reads_as_empty(tmp_path):
    assert G.load_cache(tmp_path / "nope.json")["canada_slugs"] == []


def test_a_corrupt_cache_reads_as_empty_rather_than_raising(tmp_path):
    p = tmp_path / "geo.json"
    p.write_text("{ not json", encoding="utf-8")
    assert G.load_cache(p)["canada_slugs"] == []


def test_a_cache_of_the_wrong_shape_reads_as_empty(tmp_path):
    p = tmp_path / "geo.json"
    p.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert G.load_cache(p)["canada_slugs"] == []


def test_the_cache_round_trips_and_is_written_atomically(tmp_path):
    p = tmp_path / "geo.json"
    G.save_cache({"canada_slugs": ["acme"], "sources": {"x.zip": [1, 2.0]}}, p)
    assert list(tmp_path.glob("*.tmp")) == []
    assert G.load_cache(p)["canada_slugs"] == ["acme"]


def test_an_empty_cache_leaves_the_order_untouched(tmp_path):
    """A checkout with no archive yet must not reorder anything, rather than
    treating "nothing known" as "nothing preferred" and silently shuffling.
    """
    fleet = ["a", "b", "c"]
    assert G.rank(fleet, G.canada_slugs(tmp_path / "nope.json")) == fleet
