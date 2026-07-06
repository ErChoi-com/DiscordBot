"""
Central SQLite-backed geo database (geo.db in data/).

Provides read-only access to the pre-built geo tables and caches Glassdoor
location ID lookups so the autocomplete endpoint is only hit once per city.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR.parent.parent.parent / "data"

GEO_DB_PATH = _DATA_DIR / "geo.db"


def _open(db_path: Path = GEO_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ── Build lookup dicts (same interface as geolocation.build_lookup) ──────────

def build_lookup_from_db(db_path: Path = GEO_DB_PATH) -> dict[str, dict]:
    """Return the four-dict structure expected by geolocation.lookup_location()."""
    maps: dict[str, dict] = {
        "city_admin_country": {},
        "city_country": {},
        "city_admin": {},
        "city": {},
    }
    _pop: dict[str, dict] = {"city_country": {}, "city_admin": {}, "city": {}}

    conn = _open(db_path)
    try:
        for city_norm, admin_norm, country, lat, lng, pop in conn.execute(
            "SELECT city_norm, admin_norm, country, lat, lng, population FROM locations"
        ):
            coords = [lat, lng]
            pop = int(pop or 0)

            if admin_norm and country:
                key = f"{city_norm}|{admin_norm}|{country}"
                maps["city_admin_country"].setdefault(key, coords)

            if country:
                key = f"{city_norm}|{country}"
                if pop > _pop["city_country"].get(key, -1):
                    maps["city_country"][key] = coords
                    _pop["city_country"][key] = pop

            if admin_norm:
                key = f"{city_norm}|{admin_norm}"
                if pop > _pop["city_admin"].get(key, -1):
                    maps["city_admin"][key] = coords
                    _pop["city_admin"][key] = pop

            if pop > _pop["city"].get(city_norm, -1):
                maps["city"][city_norm] = coords
                _pop["city"][city_norm] = pop
    finally:
        conn.close()

    return maps


# ── Load geo lookup dicts (same interface as ats_service._load_geo_lookup) ───

def load_geo_lookup_from_db(db_path: Path = GEO_DB_PATH) -> tuple[dict, dict, dict]:
    """Return (cities, admin1_by_code, admin1_by_name) dicts for ats_service.

    admin1_by_code is returned as an empty dict — ats_service loads it but
    never queries _geo_admin1_code; all subdivision lookups go through
    pycountry (_ca_us_subdiv_codes) or _geo_admin1_name instead.
    """
    cities: dict[str, list] = {}
    admin1_by_name: dict[str, list] = {}

    conn = _open(db_path)
    try:
        for city_lower, cc, a1 in conn.execute(
            "SELECT city_lower, country_code, admin1_code FROM geo_cities"
        ):
            cities.setdefault(city_lower, []).append([cc, a1])

        for name_lower, cc, a1 in conn.execute(
            "SELECT name_lower, country_code, admin1_code FROM geo_admin1_by_name"
        ):
            admin1_by_name.setdefault(name_lower, []).append([cc, a1])
    finally:
        conn.close()

    return cities, {}, admin1_by_name


# ── Glassdoor location ID resolver ───────────────────────────────────────────

_gd_mem_cache: dict[str, tuple[str, str]] = {}
_gd_cache_lock = threading.Lock()
_GD_CACHE_TTL = 7 * 24 * 3600  # 1 week

def _build_country_terms() -> frozenset[str]:
    """Build the country/continent guard set from pycountry + static extras.

    pycountry covers all 249 ISO 3166-1 countries (names, official names,
    common names, alpha-3 codes).  We skip alpha-2 codes entirely because
    many collide with US-state abbreviations (CA, IN, CO, AR, ID, IL, DE…).
    The static extras cover continents, recruitment region shortcodes, and
    a handful of common aliases pycountry doesn't know about.
    """
    terms: set[str] = {
        # continents
        "africa", "asia", "europe", "oceania", "antarctica", "australasia",
        "north america", "south america", "latin america", "central america",
        "middle east", "caribbean", "scandinavia", "balkans",
        "southeast asia", "south asia", "east asia", "central asia",
        "sub-saharan africa", "north africa", "west africa", "east africa",
        "southern africa",
        # recruitment region shortcodes
        "emea", "apac", "apj", "amer", "namer", "latam", "mena", "dach",
        "global", "worldwide", "international", "anywhere",
        # common aliases pycountry omits
        "usa", "us", "uk", "gb", "uae", "america",
        "england", "scotland", "wales", "northern ireland",
        "great britain", "britain", "holland", "deutschland",
        "brasil", "turkiye", "czechia", "south korea", "korea",
        "hong kong", "ivory coast",
    }
    try:
        import pycountry
        for c in pycountry.countries:
            terms.add(c.name.lower())
            terms.add(c.alpha_3.lower())
            if hasattr(c, "official_name"):
                terms.add(c.official_name.lower())
            if hasattr(c, "common_name"):
                terms.add(c.common_name.lower())
    except ImportError:
        pass
    return frozenset(terms)


_COUNTRY_TERMS: frozenset[str] = _build_country_terms()

_CREATE_CACHE_TABLE = """
    CREATE TABLE IF NOT EXISTS glassdoor_location_cache (
        location_key TEXT PRIMARY KEY,
        loc_id       TEXT NOT NULL,
        loc_type     TEXT NOT NULL,
        resolved_at  INTEGER NOT NULL
    )
"""


def resolve_glassdoor_location(
    location_str: str,
    session: Any,
    db_path: Path = GEO_DB_PATH,
) -> tuple[str, str]:
    """Return (locId, locType) for a location string.

    Resolution order:
      1. Country-term guard — country-level strings skip autocomplete entirely
      2. In-memory cache (process-lifetime)
      3. glassdoor_location_cache table in geo.db
      4. Glassdoor findPopularLocationAjax.htm autocomplete endpoint
      5. Fallback: ("0", "C")
    """
    city = location_str.split(",")[0].strip() if location_str else ""
    if not city:
        return ("0", "C")

    cache_key = city.lower()

    # Remote / country-level location: skip autocomplete entirely
    _REMOTE_PREFIXES = ("remote", "work from home", "wfh", "anywhere", "worldwide")
    if cache_key in _COUNTRY_TERMS or any(cache_key.startswith(p) for p in _REMOTE_PREFIXES):
        return ("0", "C")

    with _gd_cache_lock:
        if cache_key in _gd_mem_cache:
            return _gd_mem_cache[cache_key]

    try:
        conn = _open(db_path)
        try:
            conn.execute(_CREATE_CACHE_TABLE)
            row = conn.execute(
                "SELECT loc_id, loc_type, resolved_at FROM glassdoor_location_cache WHERE location_key = ?",
                (cache_key,),
            ).fetchone()
        finally:
            conn.close()
        if row and (time.time() - int(row[2])) < _GD_CACHE_TTL:
            result: tuple[str, str] = (row[0], row[1])
            with _gd_cache_lock:
                _gd_mem_cache[cache_key] = result
            return result
    except Exception:
        pass

    try:
        resp = session.get(
            "https://www.glassdoor.com/findPopularLocationAjax.htm",
            params={"term": city},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list):
                first = data[0]
                loc_id = str(first.get("locationId") or first.get("realId") or "0")
                loc_type = str(first.get("locationType") or "C")
                result = (loc_id, loc_type)
                try:
                    conn = _open(db_path)
                    try:
                        conn.execute(_CREATE_CACHE_TABLE)
                        conn.execute(
                            "INSERT OR REPLACE INTO glassdoor_location_cache "
                            "(location_key, loc_id, loc_type, resolved_at) VALUES (?, ?, ?, ?)",
                            (cache_key, loc_id, loc_type, int(time.time())),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                except Exception:
                    pass
                with _gd_cache_lock:
                    _gd_mem_cache[cache_key] = result
                return result
    except Exception:
        pass

    return ("0", "C")
