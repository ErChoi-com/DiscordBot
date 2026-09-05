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

def test_a_company_is_indexed_under_every_country_it_posts_in():
    """Keyed by country, not by a Canada flag: a board that posts in both is
    a first-choice for either channel, and collapsing that to one boolean is
    what would tie this to a single region.
    """
    rows = [("acme", "Austin, TX, us"), ("acme", "Toronto, ON, CA")]
    got = G.build(rows)
    assert got["CA"] == {"acme"}
    assert got["US"] == {"acme"}


def test_a_country_with_no_postings_is_simply_absent():
    assert "CA" not in G.build([("acme", "Austin, TX, us")])


def test_slugs_are_matched_case_insensitively():
    assert G.build([("ACME", "Toronto, ON, CA")])["CA"] == {"acme"}


def test_rows_without_a_company_are_skipped():
    assert G.build([("", "Toronto, ON, CA"), ("  ", "Ottawa, Ontario")]) == {}


def test_an_undecidable_location_is_indexed_nowhere():
    """"San Francisco, CA" could be either country. Indexing it under a guess
    would spend priority slots on the wrong region.
    """
    assert G.build([("acme", "San Francisco, CA")]) == {}


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
    assert G.load_cache(tmp_path / "nope.json")["by_country"] == {}


def test_a_corrupt_cache_reads_as_empty_rather_than_raising(tmp_path):
    p = tmp_path / "geo.json"
    p.write_text("{ not json", encoding="utf-8")
    assert G.load_cache(p)["by_country"] == {}


def test_a_cache_of_the_wrong_shape_reads_as_empty(tmp_path):
    p = tmp_path / "geo.json"
    p.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert G.load_cache(p)["by_country"] == {}


def test_the_cache_round_trips_and_is_written_atomically(tmp_path):
    p = tmp_path / "geo.json"
    G.save_cache({"by_country": {"CA": ["acme"]}, "sources": {"x.zip": [1, 2.0]}}, p)
    assert list(tmp_path.glob("*.tmp")) == []
    assert G.slugs_for("CA", p) == frozenset({"acme"})


def test_slugs_for_an_unseen_country_is_empty_not_an_error(tmp_path):
    p = tmp_path / "geo.json"
    G.save_cache({"by_country": {"CA": ["acme"]}, "sources": {}}, p)
    assert G.slugs_for("JP", p) == frozenset()


def test_a_country_code_is_looked_up_case_insensitively(tmp_path):
    p = tmp_path / "geo.json"
    G.save_cache({"by_country": {"CA": ["acme"]}, "sources": {}}, p)
    assert G.slugs_for("ca", p) == frozenset({"acme"})


def test_countries_reports_what_the_archive_has_seen(tmp_path):
    """So a channel scoped somewhere unusual can be told whether prioritising
    its region would do anything at all before it is wired in.
    """
    p = tmp_path / "geo.json"
    G.save_cache({"by_country": {"CA": ["a"], "US": ["a", "b", "c"]}, "sources": {}}, p)
    assert G.countries(p) == {"US": 3, "CA": 1}


def test_an_empty_cache_leaves_the_order_untouched(tmp_path):
    """A checkout with no archive yet must not reorder anything, rather than
    treating "nothing known" as "nothing preferred" and silently shuffling.
    """
    fleet = ["a", "b", "c"]
    assert G.rank(fleet, G.slugs_for("CA", tmp_path / "nope.json")) == fleet


# ── country generality: the module must not be Canada-shaped ────────────────

@pytest.mark.parametrize("location,expected", [
    ("Berlin, Germany", "DE"),
    ("Stockholm, Sweden", "SE"),
    ("Amsterdam, NL", "NL"),
    ("Sydney, AU", "AU"),
    ("Dubai, AE", "AE"),
    ("London, GB", "GB"),
    ("Austin, TX", "US"),
    ("Austin, TX, us", "US"),
])
def test_countries_other_than_canada_resolve(location, expected):
    assert G.country_of(location) == expected


@pytest.mark.parametrize("location", [
    "San Francisco, CA",   # California or Canada
    "Berlin, DE",          # Germany or Delaware
    "Dover, DE",           # Delaware or Germany
    "Something, IN",       # Indiana or India
    "Somewhere, GA",       # Georgia the state or the country
])
def test_a_code_that_is_both_a_us_state_and_a_country_stays_undecided(location):
    """Reading DE as Delaware labelled every German posting American in a first
    cut. Neither reading can be had from the code alone, and guessing spends the
    priority slots on the wrong country -- worse than not prioritising.
    """
    assert G.country_of(location) == ""


@pytest.mark.parametrize("location,expected", [
    ("Dover, DE, US", "US"),
    ("Toronto, ON, CA", "CA"),
    ("Berlin, Germany", "DE"),
])
def test_an_ambiguous_code_resolves_when_something_beside_it_decides(location, expected):
    assert G.country_of(location) == expected


def test_ranking_works_for_a_country_that_is_not_the_default():
    """The point of keying by country: a channel scoped to Germany gets the
    same treatment by passing a different code, not by editing this module.
    """
    rows = [("acme", "Berlin, Germany"), ("other", "Toronto, ON, CA")]
    idx = G.build(rows)
    assert G.rank(["other", "acme"], idx["DE"]) == ["acme", "other"]
    assert G.rank(["acme", "other"], idx["CA"]) == ["other", "acme"]


# ── the partition boundary, which the rotation needs ────────────────────────

def test_partition_returns_the_two_groups_rank_concatenates():
    front, back = G.partition(["a", "b", "c", "d"], {"c", "a"})
    assert front == ["a", "c"]
    assert back == ["b", "d"]


def test_partition_agrees_with_rank_on_every_input():
    """One definition of "preferred", not two. `rank` is defined in terms of
    this, and a caller that splits the result itself would drift from it.
    """
    fleet = [f"c{i}" for i in range(200)]
    preferred = {f"c{i}" for i in range(0, 200, 3)}
    front, back = G.partition(fleet, preferred)
    assert front + back == G.rank(fleet, preferred)


def test_partition_loses_nothing():
    fleet = [f"c{i}" for i in range(100)]
    front, back = G.partition(fleet, {"c7"})
    assert sorted(front + back) == sorted(fleet)
    assert len(front) + len(back) == len(fleet)


def test_nothing_preferred_means_an_empty_front_not_an_empty_back():
    """The caller rotates `back`. Collapsing "no opinion" into an empty back
    would hand the whole fleet to the head and rotate nothing at all, which is
    exactly the checkout that needs the rotation most.
    """
    front, back = G.partition(["a", "b", "c"], set())
    assert front == []
    assert back == ["a", "b", "c"]


def test_a_workday_tenant_lands_in_the_front():
    front, back = G.partition(["ahri|wd3|ahri1", "other|wd1|x"], {"ahri"})
    assert front == ["ahri|wd3|ahri1"]
    assert back == ["other|wd1|x"]
