"""Building data/geo.db from the tracked GeoNames sources.

data/geo.db is ~78 MB and gitignored, so a fresh clone has none. sqlite3.connect
CREATES an empty file rather than failing, so the absence surfaces as
"no such table: geo_cities" on the first ATS location lookup.

These build a real database from real GeoNames-format fixtures and query it the
way the app does, rather than asserting on row counts alone.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from services import ats_service

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = REPO_ROOT / "scripts" / "build_geo_db.py"
    spec = importlib.util.spec_from_file_location("build_geo_db", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_geo_db"] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


# GeoNames cities500 rows: 19 tab-separated columns.
def _city_row(geonameid, name, ascii_name, alternates, lat, lng, country, admin1, population):
    cols = [""] * 19
    cols[0], cols[1], cols[2], cols[3] = str(geonameid), name, ascii_name, alternates
    cols[4], cols[5] = str(lat), str(lng)
    cols[8], cols[10], cols[14] = country, admin1, str(population)
    return "\t".join(cols)


@pytest.fixture
def sources(tmp_path):
    cities = tmp_path / "cities500.txt"
    cities.write_text("\n".join([
        _city_row(1, "Toronto", "Toronto", "Toronto,TOR", 43.7, -79.4, "CA", "08", 2731571),
        _city_row(2, "München", "Muenchen", "Munich,Muenchen,MUC", 48.1, 11.5, "DE", "02", 1260391),
        # Same city name in a different country: population decides the winner.
        _city_row(3, "London", "London", "London", 51.5, -0.12, "GB", "ENG", 8961989),
        _city_row(4, "London", "London", "London", 42.98, -81.24, "CA", "08", 383822),
    ]) + "\n", encoding="utf-8")

    admin1 = tmp_path / "admin1CodesASCII.txt"
    admin1.write_text("\n".join([
        "CA.08\tOntario\tOntario\t6093943",
        "DE.02\tBavaria\tBavaria\t2951839",
        "GB.ENG\tEngland\tEngland\t6269131",
        # Accented name: the shipped DB stores this class of row mojibake'd.
        "AD.06\tSant Julià de Loria\tSant Julia de Loria\t3039162",
    ]) + "\n", encoding="utf-8")
    return cities, admin1


def test_build_produces_every_table_the_lookups_need(sources, tmp_path):
    cities, admin1 = sources
    out = tmp_path / "geo.db"
    builder.build(cities, admin1, out)

    conn = sqlite3.connect(out)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()

    assert set(builder.REQUIRED_TABLES) <= tables
    # The runtime cache table must exist too, or resolve_glassdoor_location fails.
    assert "glassdoor_location_cache" in tables


def test_locations_carry_normalized_keys_matching_geolocation(sources, tmp_path):
    """locations.city_norm is compared against geolocation.normalize() output at
    query time; any divergence silently stops every city from matching."""
    from services.jba.geolocation import normalize as app_normalize

    cities, admin1 = sources
    out = tmp_path / "geo.db"
    builder.build(cities, admin1, out)

    conn = sqlite3.connect(out)
    row = conn.execute(
        "SELECT city, city_norm, admin, admin_norm, country FROM locations WHERE city = 'München'"
    ).fetchone()
    conn.close()

    assert row is not None, "accented city name was not stored"
    city, city_norm, admin, admin_norm, country = row
    assert city_norm == app_normalize(city) == "munchen"
    assert admin == "Bavaria" and admin_norm == app_normalize("Bavaria")
    assert country == "DE"


def test_accented_admin1_survives_as_utf8(sources, tmp_path):
    """The shipped database stores this name mojibake'd, and ats_service matches
    admin1 on a plain name.lower() -- so a mangled name never matches."""
    cities, admin1 = sources
    out = tmp_path / "geo.db"
    builder.build(cities, admin1, out)

    conn = sqlite3.connect(out)
    names = {r[0] for r in conn.execute("SELECT name_lower FROM geo_admin1_by_name")}
    conn.close()

    assert "sant julià de loria" in names, f"accented admin1 was mangled: {sorted(names)}"


def test_alternate_names_resolve_to_the_city(sources, tmp_path):
    """Alternates are why 'Muenchen' and 'MUC' find Munich."""
    cities, admin1 = sources
    out = tmp_path / "geo.db"
    builder.build(cities, admin1, out)

    conn = sqlite3.connect(out)
    keys = {r[0] for r in conn.execute("SELECT city_lower FROM geo_cities")}
    conn.close()

    assert {"münchen", "muenchen", "munich"} <= keys
    assert "muc" in keys, "3-character alternates should be kept"


def test_short_alternates_are_dropped_but_short_primaries_kept(tmp_path):
    """Two-letter alternates ('al', 'do') are stray tokens, and _infer_country
    tests membership in this set -- admitting them misreads ordinary words as
    cities. A genuinely short PRIMARY name must still be indexed."""
    cities = tmp_path / "cities500.txt"
    cities.write_text(
        _city_row(1, "Ai", "Ai", "al,do,Aitown", 35.0, 139.0, "JP", "01", 5000) + "\n",
        encoding="utf-8",
    )
    admin1 = tmp_path / "admin1CodesASCII.txt"
    admin1.write_text("JP.01\tHokkaido\tHokkaido\t2130037\n", encoding="utf-8")
    out = tmp_path / "geo.db"
    builder.build(cities, admin1, out)

    conn = sqlite3.connect(out)
    keys = {r[0] for r in conn.execute("SELECT city_lower FROM geo_cities")}
    conn.close()

    assert "ai" in keys, "a short PRIMARY name must still be indexed"
    assert "aitown" in keys
    assert "al" not in keys and "do" not in keys


def test_build_is_idempotent_and_replaces_an_existing_db(sources, tmp_path):
    cities, admin1 = sources
    out = tmp_path / "geo.db"
    first = builder.build(cities, admin1, out)
    second = builder.build(cities, admin1, out)

    assert first == second
    conn = sqlite3.connect(out)
    count = conn.execute("SELECT COUNT(*) FROM locations").fetchone()[0]
    conn.close()
    assert count == first["locations"], "rebuild appended instead of replacing"


def test_built_db_answers_the_real_lookup_query(sources, tmp_path):
    """Exercise geo_db's own loader against the built file, not a hand-written
    query -- that is what the app actually calls."""
    from services.jba import geo_db

    cities, admin1 = sources
    out = tmp_path / "geo.db"
    builder.build(cities, admin1, out)

    maps = geo_db.build_lookup_from_db(out)
    assert maps["city"]["toronto"] == pytest.approx([43.7, -79.4])
    # Two Londons: the more populous one wins the bare-city key.
    assert maps["city"]["london"] == pytest.approx([51.5, -0.12])
    # Disambiguated keys keep both.
    assert maps["city_country"]["london|CA"] == pytest.approx([42.98, -81.24])


def test_check_rejects_a_missing_or_empty_database(tmp_path, capsys):
    assert builder.check(tmp_path / "absent.db") is False
    assert "does not exist" in capsys.readouterr().out

    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()          # what connect() silently creates
    assert builder.check(empty) is False
    assert "missing tables" in capsys.readouterr().out


def test_check_rejects_a_database_whose_tables_exist_but_are_empty(tmp_path, capsys):
    """Distinct from the no-tables case above, which short-circuits earlier. A
    build interrupted after schema creation leaves exactly this state, and it
    would answer every lookup with 'no such city' rather than an error."""
    partial = tmp_path / "partial.db"
    conn = sqlite3.connect(partial)
    conn.executescript(builder._SCHEMA)
    conn.commit()
    conn.close()

    assert builder.check(partial) is False
    assert "is empty" in capsys.readouterr().out


def test_check_accepts_a_freshly_built_database(sources, tmp_path, capsys):
    cities, admin1 = sources
    out = tmp_path / "geo.db"
    builder.build(cities, admin1, out)

    assert builder.check(out) is True


# ---------------------------------------------------------------------------
# Degradation when the database is absent
# ---------------------------------------------------------------------------

def test_missing_geo_db_degrades_instead_of_raising(monkeypatch, capsys):
    """A fresh host has no geo.db. That must not take the ATS scrape down."""
    def _boom():
        raise sqlite3.OperationalError("no such table: geo_cities")

    monkeypatch.setattr(ats_service, "_load_geo_lookup", _boom)
    monkeypatch.setattr(ats_service, "_geo_loaded", False)
    monkeypatch.setattr(ats_service, "_geo_cities", {})
    monkeypatch.setattr(ats_service, "_geo_admin1_name", {})

    ats_service._ensure_geo_loaded()          # must not raise

    out = capsys.readouterr().out
    assert "geo.db unavailable" in out and "build_geo_db.py" in out
    # Country matching still works -- it comes from pycountry, not geo.db.
    assert ats_service._iso_country_codes, "country codes should still load"
