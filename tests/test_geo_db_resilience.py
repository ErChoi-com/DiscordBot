"""A momentary lock on geo.db must not cost the process its city matching.

Two defects, one symptom. `_open` re-asserted `PRAGMA journal_mode=WAL` on
every connect; journal mode rewrites the database header, so it needs an
exclusive lock and a single other open connection denies it. Reads share the
file fine -- that pragma was the only reason a read could fail at all.

Then `_load_geo_into_globals` set `_geo_loaded = True` on the failure path too,
and `_ensure_geo_loaded` short-circuits on that flag. So one contended open at
startup disabled city/region matching for the whole life of the process. Seen
on the real bot: it logged "geo.db unavailable (database is locked)" at 19:17
and ran country-only for the entire session, while the file opened read-write
in 0.00s minutes later.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service
from services.jba import geo_db


@pytest.fixture
def geo_file(tmp_path: Path) -> Path:
    db = tmp_path / "geo.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE geo_cities (name TEXT, country TEXT, admin1 TEXT)")
    conn.execute("INSERT INTO geo_cities VALUES ('toronto', 'CA', 'ON')")
    conn.commit()
    conn.close()
    return db


def test_open_survives_a_reader_holding_the_file(geo_file: Path):
    """The regression itself, in the shape production actually hit.

    Measured on a fresh delete-mode database, against a second connection:

        other connection      journal_mode=WAL   plain read
        idle open             OK                 OK
        holding a read txn    BLOCKED            OK
        BEGIN EXCLUSIVE       BLOCKED            BLOCKED

    A plain reader with a transaction open denies the pragma an exclusive
    lock while leaving reads perfectly available -- so the pragma, and only
    the pragma, turned a usable database into "database is locked". On this
    machine the readers were stray pytest and MCP-server processes.
    """
    holder = sqlite3.connect(geo_file)
    holder.execute("BEGIN")
    holder.execute("SELECT * FROM geo_cities").fetchall()
    try:
        conn = geo_db._open(geo_file)  # must not raise
        assert conn.execute("SELECT COUNT(*) FROM geo_cities").fetchone()[0] == 1
        conn.close()
    finally:
        holder.rollback()
        holder.close()


def test_an_exclusive_writer_is_what_the_retry_is_for(geo_file: Path):
    """The one case no open can survive -- reads really are unavailable. It is
    also transient, which is exactly why the load must not latch on it."""
    holder = sqlite3.connect(geo_file)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(sqlite3.OperationalError):
            geo_db._open(geo_file).execute("SELECT COUNT(*) FROM geo_cities").fetchone()
    finally:
        holder.rollback()
        holder.close()
    # Once it clears, the very next open works.
    conn = geo_db._open(geo_file)
    assert conn.execute("SELECT COUNT(*) FROM geo_cities").fetchone()[0] == 1
    conn.close()


def test_the_journal_pragma_is_still_applied_when_it_can_be(geo_file: Path):
    """Best effort means effort. An uncontended open should still get WAL."""
    conn = geo_db._open(geo_file)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _reset_geo_state():
    """These globals are process-wide and this module deliberately breaks them."""
    saved = (
        ats_service._geo_loaded, ats_service._countries_loaded,
        ats_service._geo_failures, ats_service._geo_next_retry,
        dict(ats_service._geo_cities), dict(ats_service._geo_admin1_name),
        set(ats_service._iso_country_codes), dict(ats_service._ca_us_subdiv_codes),
    )
    ats_service._geo_loaded = False
    ats_service._countries_loaded = False
    ats_service._geo_failures = 0
    ats_service._geo_next_retry = 0.0
    # Cleared, not just saved. These two are populated by any earlier test in
    # the session, so leaving them alone made
    # test_country_tables_are_built_even_while_geo_db_is_down pass on residue --
    # a mutant that skipped building them entirely survived because of it.
    ats_service._iso_country_codes.clear()
    ats_service._ca_us_subdiv_codes.clear()
    yield
    (ats_service._geo_loaded, ats_service._countries_loaded,
     ats_service._geo_failures, ats_service._geo_next_retry,
     ats_service._geo_cities, ats_service._geo_admin1_name,
     iso, subdiv) = saved
    ats_service._iso_country_codes.clear()
    ats_service._iso_country_codes.update(iso)
    ats_service._ca_us_subdiv_codes.clear()
    ats_service._ca_us_subdiv_codes.update(subdiv)


def test_a_transient_failure_is_retried_instead_of_latched(monkeypatch):
    """The actual bug. A lock that clears must not cost the whole process."""
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return ({"toronto": [("CA", "ON")]}, {}, {})

    monkeypatch.setattr(ats_service, "_load_geo_lookup", _flaky)
    monkeypatch.setattr(ats_service, "GEO_RETRY_INTERVAL_S", 0.0)

    ats_service._ensure_geo_loaded()
    assert ats_service._geo_cities == {}, "first attempt fails and degrades"
    assert not ats_service._geo_loaded, "and must NOT be marked loaded"

    ats_service._ensure_geo_loaded()
    assert calls["n"] == 2, "the second lookup retries"
    assert ats_service._geo_cities == {"toronto": [("CA", "ON")]}
    assert ats_service._geo_loaded, "and latches once it actually worked"


def test_a_success_is_loaded_exactly_once(monkeypatch):
    """Retrying must not mean reloading 632k rows on every lookup."""
    calls = {"n": 0}

    def _ok():
        calls["n"] += 1
        return ({"toronto": [("CA", "ON")]}, {}, {})

    monkeypatch.setattr(ats_service, "_load_geo_lookup", _ok)
    for _ in range(5):
        ats_service._ensure_geo_loaded()
    assert calls["n"] == 1


def test_a_permanent_failure_is_rate_limited_not_retried_per_lookup(monkeypatch):
    """A genuinely missing geo.db must not cost a connect on every lookup."""
    calls = {"n": 0}

    def _missing():
        calls["n"] += 1
        raise sqlite3.OperationalError("no such table: geo_cities")

    monkeypatch.setattr(ats_service, "_load_geo_lookup", _missing)
    monkeypatch.setattr(ats_service, "GEO_RETRY_INTERVAL_S", 3600.0)

    for _ in range(50):
        ats_service._ensure_geo_loaded()
    assert calls["n"] == 1, "50 lookups inside the retry window, one attempt"


def test_the_failure_is_reported_once_not_on_every_retry(monkeypatch, capsys):
    """Retrying every 2 minutes for hours must not fill the log with it."""
    monkeypatch.setattr(
        ats_service, "_load_geo_lookup",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )
    monkeypatch.setattr(ats_service, "GEO_RETRY_INTERVAL_S", 0.0)
    for _ in range(5):
        ats_service._ensure_geo_loaded()
    assert capsys.readouterr().out.count("geo.db unavailable") == 1


def test_recovery_is_announced(monkeypatch, capsys):
    """Otherwise the log's last word on geo is that it was broken."""
    calls = {"n": 0}

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return ({"toronto": [("CA", "ON")]}, {}, {})

    monkeypatch.setattr(ats_service, "_load_geo_lookup", _flaky)
    monkeypatch.setattr(ats_service, "GEO_RETRY_INTERVAL_S", 0.0)
    for _ in range(3):
        ats_service._ensure_geo_loaded()
    assert "recovered after 2 failed attempt" in capsys.readouterr().out


def test_country_tables_are_built_even_while_geo_db_is_down(monkeypatch):
    """Country-only matching is the documented degraded mode, so the pycountry
    tables must be populated on the failure path, not just the success one."""
    monkeypatch.setattr(
        ats_service, "_load_geo_lookup",
        lambda: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )
    monkeypatch.setattr(ats_service, "GEO_RETRY_INTERVAL_S", 0.0)
    ats_service._ensure_geo_loaded()
    assert "CA" in ats_service._iso_country_codes
    assert ats_service._ca_us_subdiv_codes.get("ON") == "CA"
