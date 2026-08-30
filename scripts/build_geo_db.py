"""Build data/geo.db from the tracked GeoNames source files.

data/geo.db is ~78 MB and gitignored, so a fresh clone does not have it. Without
it every ATS location lookup raises `sqlite3.OperationalError: no such table:
geo_cities` -- sqlite3.connect() happily creates an empty database, so there is
no missing-file error to catch.

Inputs (in data/geonames_raw/, pulled by scripts/sync_geonames.py --
gitignored, because they are republished weekly and 40MB a refresh):
    cities500.txt          GeoNames dump, tab-separated, 19 columns
    admin1CodesASCII.txt   "CC.A1 <tab> name <tab> asciiname <tab> geonameid"

Usage:
    python scripts/build_geo_db.py                 # writes data/geo.db
    python scripts/build_geo_db.py --check         # verify an existing db only

Encoding note: the raw files are UTF-8 and are read as UTF-8 explicitly. An
earlier build evidently read them under a locale codec -- the shipped database
stores "Sant Julia de Loria" mojibake'd -- and since ats_service matches these
tables on a plain `name.lower()`, any mangled name simply never matches. Reading
them correctly is a fix, not just tidiness.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "geonames_raw"
CITIES_FILE = RAW_DIR / "cities500.txt"
ADMIN1_FILE = RAW_DIR / "admin1CodesASCII.txt"
GEO_DB_PATH = DATA_DIR / "geo.db"

SCHEMA_VERSION = "1"

# Alternate names shorter than this are abbreviations/stray tokens, not places.
_MIN_ALTERNATE_LEN = 3

# cities500.txt column indices (0-based) from the GeoNames export format.
_C_NAME, _C_ASCII, _C_ALT = 1, 2, 3
_C_LAT, _C_LNG = 4, 5
_C_COUNTRY, _C_ADMIN1, _C_POP = 8, 10, 14

_SCHEMA = """
CREATE TABLE _meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE locations (
    id          INTEGER PRIMARY KEY,
    city        TEXT    NOT NULL,
    city_norm   TEXT    NOT NULL,
    admin       TEXT    NOT NULL,
    admin_norm  TEXT    NOT NULL,
    country     TEXT    NOT NULL,
    lat         REAL    NOT NULL,
    lng         REAL    NOT NULL,
    population  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE geo_cities (
    city_lower   TEXT NOT NULL,
    country_code TEXT NOT NULL,
    admin1_code  TEXT NOT NULL
);
CREATE TABLE geo_admin1_by_name (
    name_lower   TEXT NOT NULL,
    country_code TEXT NOT NULL,
    admin1_code  TEXT NOT NULL
);
CREATE TABLE geo_admin1_by_code (
    code         TEXT NOT NULL,
    country_code TEXT NOT NULL,
    admin1_code  TEXT NOT NULL
);
-- Runtime cache, populated by geo_db.resolve_glassdoor_location().
CREATE TABLE glassdoor_location_cache (
    location_key TEXT PRIMARY KEY,
    loc_id       TEXT NOT NULL,
    loc_type     TEXT NOT NULL,
    resolved_at  INTEGER NOT NULL
);
"""

_INDEXES = """
CREATE INDEX idx_loc_cac  ON locations(city_norm, admin_norm, country);
CREATE INDEX idx_loc_cc   ON locations(city_norm, country);
CREATE INDEX idx_loc_ca   ON locations(city_norm, admin_norm);
CREATE INDEX idx_loc_c    ON locations(city_norm, population DESC);
CREATE INDEX idx_geo_cities ON geo_cities(city_lower);
CREATE INDEX idx_geo_a1name ON geo_admin1_by_name(name_lower);
CREATE INDEX idx_geo_a1code ON geo_admin1_by_code(code);
"""

REQUIRED_TABLES = ("locations", "geo_cities", "geo_admin1_by_name", "geo_admin1_by_code")


def normalize(value: str) -> str:
    """Match services.jba.geolocation.normalize().

    Kept in step deliberately: locations.city_norm is compared against the
    output of that function at query time, so any divergence silently stops
    every city from matching.
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFD", str(value))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    for ch in (".", "'", "`"):
        text = text.replace(ch, "")
    text = " ".join(text.lower().split())
    if text.startswith("saint "):
        text = "st " + text[6:]
    return text.replace(" saint ", " st ")


def load_admin1(path: Path) -> dict[str, tuple[str, str, str]]:
    """'CC.A1' -> (name, country_code, admin1_code)."""
    table: dict[str, tuple[str, str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or "." not in parts[0]:
                continue
            country, _, admin1 = parts[0].partition(".")
            table[parts[0]] = (parts[1], country, admin1)
    return table


def build(cities: Path, admin1: Path, out: Path) -> dict[str, int]:
    admin = load_admin1(admin1)

    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out))
    try:
        conn.executescript(_SCHEMA)

        locations: list[tuple] = []
        city_rows: list[tuple[str, str, str]] = []
        seen_city_rows: set[tuple[str, str, str]] = set()

        with cities.open(encoding="utf-8") as handle:
            for line in handle:
                cols = line.rstrip("\n").split("\t")
                if len(cols) <= _C_POP:
                    continue
                name = cols[_C_NAME].strip()
                country = cols[_C_COUNTRY].strip()
                if not name or not country:
                    continue
                admin1_code = cols[_C_ADMIN1].strip()
                admin_name = admin.get(f"{country}.{admin1_code}", ("", "", ""))[0]

                try:
                    lat, lng = float(cols[_C_LAT]), float(cols[_C_LNG])
                except ValueError:
                    continue
                try:
                    population = int(cols[_C_POP] or 0)
                except ValueError:
                    population = 0

                locations.append((
                    name, normalize(name), admin_name, normalize(admin_name),
                    country, lat, lng, population,
                ))

                # ats_service matches these on a plain name.lower(), so index the
                # display name, the ASCII form, and every alternate spelling --
                # that is what lets "Munchen"/"Muenchen" resolve to Munich.
                alternates = cols[_C_ALT].split(",") if cols[_C_ALT] else []
                candidates = [(name, True), (cols[_C_ASCII], True)]
                candidates += [(alt, False) for alt in alternates]
                for candidate, is_primary in candidates:
                    key = candidate.strip().lower()
                    if not key:
                        continue
                    # Very short ALTERNATES are abbreviations and stray tokens
                    # ("al", "do", "ea") rather than usable place names. They are
                    # the only entries that turn an otherwise unambiguous city
                    # key ambiguous, and _infer_country tests membership in this
                    # set -- so admitting them would misread ordinary words as
                    # cities. Primary names are kept whatever their length.
                    if not is_primary and len(key) < _MIN_ALTERNATE_LEN:
                        continue
                    row = (key, country, admin1_code)
                    if row not in seen_city_rows:
                        seen_city_rows.add(row)
                        city_rows.append(row)

        conn.executemany(
            "INSERT INTO locations "
            "(city, city_norm, admin, admin_norm, country, lat, lng, population) "
            "VALUES (?,?,?,?,?,?,?,?)",
            locations,
        )
        conn.executemany("INSERT INTO geo_cities VALUES (?,?,?)", city_rows)

        admin_name_rows = {
            (name.strip().lower(), country, code)
            for name, country, code in admin.values() if name.strip()
        }
        conn.executemany("INSERT INTO geo_admin1_by_name VALUES (?,?,?)", sorted(admin_name_rows))
        conn.executemany(
            "INSERT INTO geo_admin1_by_code VALUES (?,?,?)",
            sorted({(f"{c}.{a}", c, a) for _n, c, a in admin.values()}),
        )

        conn.executescript(_INDEXES)
        conn.execute("INSERT INTO _meta (key, value) VALUES ('schema_version', ?)",
                     (SCHEMA_VERSION,))
        conn.commit()

        return {
            "locations": len(locations),
            "geo_cities": len(city_rows),
            "geo_admin1_by_name": len(admin_name_rows),
        }
    finally:
        conn.close()


def check(path: Path) -> bool:
    """Verify a database has the tables the lookups need, and is non-empty."""
    if not path.exists():
        print(f"[geo-db] {path} does not exist. Run: python scripts/build_geo_db.py")
        return False
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        present = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        missing = [t for t in REQUIRED_TABLES if t not in present]
        if missing:
            print(f"[geo-db] {path} is missing tables: {', '.join(missing)}")
            return False
        for table in REQUIRED_TABLES:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count == 0:
                print(f"[geo-db] {path}: table {table} is empty")
                return False
            print(f"[geo-db]   {table}: {count:,} rows")
        return True
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data/geo.db from GeoNames sources.")
    parser.add_argument("--check", action="store_true",
                        help="verify the existing database instead of rebuilding")
    parser.add_argument("--out", type=Path, default=GEO_DB_PATH)
    args = parser.parse_args()

    if args.check:
        return 0 if check(args.out) else 1

    for path in (CITIES_FILE, ADMIN1_FILE):
        if not path.exists():
            print(f"[geo-db] Missing source file: {path}", file=sys.stderr)
            return 1

    print(f"[geo-db] Building {args.out} from {RAW_DIR}...")
    counts = build(CITIES_FILE, ADMIN1_FILE, args.out)
    for table, count in counts.items():
        print(f"[geo-db]   {table}: {count:,} rows")
    size_mb = args.out.stat().st_size / 1024 ** 2
    print(f"[geo-db] Done — {args.out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
