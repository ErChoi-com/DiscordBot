"""Ambiguous city names must resolve to the well-known city, not an arbitrary one.

`geo_cities` holds one row per (city, country) with no population column and in
GeoNames id order, so `_lookup_city` returning entries[0] resolved "Toronto" to
AU -- a 5,670-person town in New South Wales -- ahead of Toronto, Ontario.
_infer_country then reported AU for the search and CA for the job, and
_matches_location rejected every Toronto posting for a search that said
"Toronto". Silent, and in the direction of showing nothing.
"""
import sqlite3

import pytest

from services.jba.geo_db import load_geo_lookup_from_db


def _build(tmp_path, city_rows, location_rows, with_locations=True):
    path = tmp_path / "geo.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE geo_cities (city_lower TEXT NOT NULL, "
        "country_code TEXT NOT NULL, admin1_code TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE geo_admin1_by_name (name_lower TEXT NOT NULL, "
        "country_code TEXT NOT NULL, admin1_code TEXT NOT NULL)"
    )
    conn.executemany("INSERT INTO geo_cities VALUES (?,?,?)", city_rows)
    if with_locations:
        conn.execute(
            "CREATE TABLE locations (city_norm TEXT NOT NULL, admin_norm TEXT, "
            "country TEXT NOT NULL, population INTEGER NOT NULL DEFAULT 0)"
        )
        conn.executemany("INSERT INTO locations VALUES (?,?,?,?)", location_rows)
    conn.commit()
    conn.close()
    return path


# Insertion order deliberately puts the small AU town first, which is what the
# real geo.db does.
_TORONTO_CITIES = [("toronto", "AU", "02"), ("toronto", "CA", "08"), ("toronto", "US", "OH")]
_TORONTO_LOCATIONS = [
    ("toronto", "New South Wales", "AU", 5670),
    ("toronto", "Ontario", "CA", 2794356),
    ("toronto", "Ohio", "US", 4882),
]


def test_ambiguous_city_resolves_to_the_most_populous_country(tmp_path):
    cities, _, _ = load_geo_lookup_from_db(_build(tmp_path, _TORONTO_CITIES, _TORONTO_LOCATIONS))
    assert [cc for cc, _a1 in cities["toronto"]] == ["CA", "AU", "US"]


def test_unambiguous_city_is_untouched(tmp_path):
    cities, _, _ = load_geo_lookup_from_db(
        _build(tmp_path, [("guelph", "CA", "08")], [("guelph", "Ontario", "CA", 121688)])
    )
    assert cities["guelph"] == [["CA", "08"]]


def test_city_absent_from_locations_sorts_last_but_is_kept(tmp_path):
    """A geo_cities row with no population row must not be dropped -- geo_cities
    has 1.17M rows against locations' 235k, so most names have no population."""
    cities, _, _ = load_geo_lookup_from_db(
        _build(
            tmp_path,
            [("springfield", "ZZ", "01"), ("springfield", "US", "IL")],
            [("springfield", "Illinois", "US", 116565)],
        )
    )
    assert [cc for cc, _a1 in cities["springfield"]] == ["US", "ZZ"]


def test_missing_locations_table_degrades_instead_of_raising(tmp_path):
    """A partially built geo.db must fall back to the old arbitrary ordering
    rather than taking every location filter down."""
    cities, _, _ = load_geo_lookup_from_db(
        _build(tmp_path, _TORONTO_CITIES, [], with_locations=False)
    )
    assert [cc for cc, _a1 in cities["toronto"]] == ["AU", "CA", "US"]
