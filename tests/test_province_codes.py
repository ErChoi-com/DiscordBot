"""A bare Canadian province code was read as a country.

country_of resolves a two-letter tail with no country beside it -- "Austin, TX"
is the US -- and that branch only ever checked US state codes. A Canadian
province fell past it into the tail-as-country guess, so "Toronto, ON" came
back as the country "ON".

It hid because it was wrong consistently. build() keys the index with the same
function, so those boards were filed under "ON" too, slugs_for("ON") returned a
non-empty set, and the fleet ordering looked like it was working. Measured on
the live index at the time of the fix: a channel scoped to "Mississauga, ON"
prioritised a 146-board "ON" bucket while 931 boards sat under "CA". On a bot
whose whole point is Canadian postings, that is the ordering missing most of
the country it exists to surface.

The obvious fix -- resolve every province code -- is wrong, and this file pins
why. Four province codes are also ISO country codes, and a bare "City, XX"
gives nothing to choose between them with. Reading "Amsterdam, NL" as Canada
would be the same mistake in the other direction, and the module already
refuses to make it for "San Francisco, CA".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.jba import geo_priority as g  # noqa: E402


# -- the bug --------------------------------------------------------------

@pytest.mark.parametrize("location", [
    "Toronto, ON",
    "Vancouver, BC",
    "Montreal, QC",
    "Calgary, AB",
    "Halifax, NS",
    "Fredericton, NB",
    "Winnipeg, MB",
    "Yellowknife, NT",
    "Whitehorse, YT",
])
def test_a_bare_province_code_resolves_to_canada(location):
    assert g.country_of(location) == "CA"


def test_a_province_code_is_not_returned_as_a_country():
    """The actual failure: the province came back as the country name."""
    assert g.country_of("Toronto, ON") != "ON"


def test_a_city_scoped_channel_reaches_the_canadian_boards():
    """What the bug cost. _priority_countries feeds country_of a channel's own
    location setting, and free text a person types is a city, not a country.
    """
    code = g.country_of("Mississauga, ON")
    assert code == "CA"


# -- and the fix does not overreach ---------------------------------------

@pytest.mark.parametrize("location,expected", [
    ("Amsterdam, NL", "NL"),   # Netherlands, not Newfoundland and Labrador
    ("Lima, PE", "PE"),        # Peru, not Prince Edward Island
    ("Bratislava, SK", "SK"),  # Slovakia, not Saskatchewan
    ("Alofi, NU", "NU"),       # Niue, not Nunavut
])
def test_a_province_code_that_is_also_a_country_is_left_alone(location, expected):
    """Resolving these to Canada would be the same error mirrored: a bare code
    is the whole of the evidence and it points both ways."""
    assert g.country_of(location) == expected


def test_the_excluded_codes_are_exactly_the_colliding_ones():
    """Pinned so the exclusion cannot quietly grow into "provinces we gave up
    on" -- each one is out because it is an ISO country code, and that is the
    only reason any of them may be out.
    """
    excluded = set(g._CA_SUBDIVISION_CODES) - set(g._CA_ONLY_SUBDIVISION_CODES)
    assert excluded == {"nl", "pe", "sk", "nu"}


def test_no_us_state_code_is_treated_as_canadian():
    for code in g._US_SUBDIVISION_CODES:
        assert g.country_of(f"Somewhere, {code.upper()}") != "CA", code


def test_the_two_subdivision_tables_do_not_overlap():
    """The fix checks both in one branch, which is only safe while no code
    means a province in one country and a state in the other."""
    assert not (set(g._CA_ONLY_SUBDIVISION_CODES) & set(g._US_SUBDIVISION_CODES))


# -- the rules that already worked still do -------------------------------

def test_the_genuinely_ambiguous_code_is_still_refused():
    """"San Francisco, CA" is California and "Toronto, ON, CA" is Canada. The
    whole reason this function does not read a bare tail as a country."""
    assert g.country_of("San Francisco, CA") == ""


def test_a_colliding_code_still_resolves_when_the_country_is_named():
    """These are only unresolvable when the bare code is all there is."""
    assert g.country_of("St John's, NL, Canada") == "CA"
    assert g.country_of("Charlottetown, PE, Canada") == "CA"


def test_a_province_name_still_resolves():
    assert g.country_of("Saskatoon, Saskatchewan") == "CA"


def test_us_states_still_resolve():
    assert g.country_of("Austin, TX") == "US"


def test_other_countries_still_resolve():
    assert g.country_of("London, UK") == "GB"


def test_an_empty_location_resolves_to_nothing():
    assert g.country_of("") == ""


# -- the index is keyed by the same function ------------------------------

def test_the_index_files_canadian_boards_under_canada(tmp_path):
    """build() and the channel lookup both go through country_of, which is why
    the bug was self-consistent: both halves were wrong the same way, so the
    ordering returned boards and looked correct.
    """
    rows = [("acme", "Toronto, ON"), ("beta", "Vancouver, BC"),
            ("gamma", "Austin, TX")]
    by_country = g.build(rows)

    assert by_country.get("CA") == {"acme", "beta"}
    assert "ON" not in by_country
    assert "BC" not in by_country


def test_a_channel_and_the_index_agree_on_the_same_city(tmp_path):
    """The end-to-end shape of the failure: the lookup key and the index key
    must be the same string for a board to ever be prioritised."""
    by_country = g.build([("acme", "Mississauga, ON")])
    wanted = g.country_of("Toronto, ON")

    assert wanted in by_country
    assert by_country[wanted] == {"acme"}
