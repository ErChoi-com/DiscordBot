"""Pull the GeoNames source files and rebuild data/geo.db from them.

The companion to sync_ats_companies.py: that keeps the ATS company lists
current, this keeps the geo data current. data/geonames_raw/ was otherwise
frozen at whenever it was last fetched by hand.

Where the data comes from
-------------------------
The `geonames-data` branch of this repo, published weekly by
.github/workflows/geonames-sync.yml, not download.geonames.org directly. One
scheduled job fetches from GeoNames and every host pulls from the branch, so a
GeoNames outage or a rate limit costs one workflow run rather than breaking
every bot at once. `--from-source` bypasses the branch when that is what you
want.

Fetched over HTTPS rather than by cloning the branch, deliberately: a container
built from the Dockerfile has no git and no checkout, and this has to work
there too.

manifest.json on that branch carries a sha256 per file and is a few hundred
bytes, so `--if-changed` can run as often as you like -- it downloads the 12MB
archive only when a checksum actually moves.

Rebuilding data/geo.db is part of the job, not an afterthought: nothing reads
the raw files at runtime -- ats_service and job_match both go through geo.db --
so refreshing the sources without rebuilding would change nothing observable.

Usage:
    python sync_geonames.py                # pull if changed, rebuild
    python sync_geonames.py --force        # pull regardless
    python sync_geonames.py --from-source  # straight from GeoNames
    python sync_geonames.py --check        # report state, change nothing
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "data" / "geonames_raw"
STAMP = RAW_DIR / ".sync_manifest.json"

REPO = "ErChoi-com/DiscordBot"
BRANCH = "geonames-data"
BRANCH_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

GEONAMES_URL = "https://download.geonames.org/export/dump"

#: A plain urlopen sends "Python-urllib", which some CDNs refuse.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; job-bot geonames sync)"}
_TIMEOUT = 180

#: Below this the download is broken, not merely small. The dump has held ~234k
#: rows for years; a truncated file would otherwise rebuild geo.db with most of
#: the world missing, and sqlite reports no error for that.
_MIN_CITY_LINES = 200_000
_MIN_ADMIN_LINES = 3_000


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return response.read()


def _write_atomic(path: Path, payload: bytes) -> None:
    """Replace ``path`` only once the whole payload is in hand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _local_manifest() -> dict:
    try:
        return json.loads(STAMP.read_text(encoding="utf-8"))
    except Exception:
        return {}


def remote_manifest() -> dict:
    """The published manifest, or {} when it cannot be read."""
    try:
        return json.loads(_fetch(f"{BRANCH_URL}/manifest.json").decode("utf-8"))
    except Exception as exc:
        print(f"[geonames-sync] Could not read the published manifest: {exc}")
        return {}


def _checked(cities_txt: bytes, admin: bytes) -> None:
    """Refuse obviously broken downloads before they overwrite good data."""
    lines = cities_txt.count(b"\n")
    if lines < _MIN_CITY_LINES:
        raise ValueError(f"cities500.txt has only {lines:,} lines")
    lines = admin.count(b"\n")
    if lines < _MIN_ADMIN_LINES:
        raise ValueError(f"admin1CodesASCII.txt has only {lines:,} lines")


def pull(from_source: bool = False) -> dict:
    """Download both source files. Returns the manifest describing them."""
    base = GEONAMES_URL if from_source else BRANCH_URL
    print(f"[geonames-sync] Fetching cities500.zip from {base} ...")
    archive = _fetch(f"{base}/cities500.zip")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        cities = bundle.read("cities500.txt")

    print("[geonames-sync] Fetching admin1CodesASCII.txt ...")
    admin = _fetch(f"{base}/admin1CodesASCII.txt")

    _checked(cities, admin)

    _write_atomic(RAW_DIR / "cities500.txt", cities)
    _write_atomic(RAW_DIR / "admin1CodesASCII.txt", admin)
    print(f"[geonames-sync]   cities500.txt {len(cities) / 1e6:.1f} MB, "
          f"{cities.count(chr(10).encode()):,} lines")
    return {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": base,
        "files": {
            "cities500.zip": hashlib.sha256(archive).hexdigest(),
            "admin1CodesASCII.txt": hashlib.sha256(admin).hexdigest(),
        },
    }


def rebuild() -> dict:
    """Regenerate data/geo.db from the files now in data/geonames_raw/."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_geo_db import CITIES_FILE, ADMIN1_FILE, GEO_DB_PATH, build

    print("[geonames-sync] Rebuilding data/geo.db ...")
    # Build beside the live database and swap. A failed build must not leave a
    # half-written geo.db: sqlite3.connect() creates an empty file rather than
    # erroring, so a truncated db reads as "no such table" at runtime.
    staging = GEO_DB_PATH.with_suffix(".db.new")
    staging.unlink(missing_ok=True)
    counts = build(CITIES_FILE, ADMIN1_FILE, staging)
    shutil.move(str(staging), str(GEO_DB_PATH))
    print(f"[geonames-sync] Rebuilt geo.db: {counts}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="pull even when the checksums match")
    parser.add_argument("--from-source", action="store_true",
                        help="fetch from download.geonames.org, not the branch")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="skip the geo.db rebuild")
    parser.add_argument("--check", action="store_true",
                        help="report local vs published state and exit")
    args = parser.parse_args()

    local = _local_manifest()
    if args.check:
        published = remote_manifest()
        print(f"  local     : {local.get('updated', '<never synced>')}")
        print(f"  published : {published.get('updated', '<unavailable>')}")
        same = local.get("files") == published.get("files") and bool(local.get("files"))
        print(f"  up to date: {same}")
        return 0

    started = time.monotonic()
    if not args.force and not args.from_source:
        published = remote_manifest()
        if not published.get("files"):
            # No manifest means the branch has not been published yet, or the
            # network is down. Either way there is nothing to pull, and trying
            # anyway would fetch a 404 for 12MB on every scheduled run.
            print("[geonames-sync] No published data to pull; leaving sources as they are.")
            return 0
        if published["files"] == local.get("files"):
            print("[geonames-sync] Already on the published data; nothing to do.")
            return 0

    try:
        manifest = pull(from_source=args.from_source)
    except Exception as exc:
        print(f"[geonames-sync] Pull failed: {exc}", file=sys.stderr)
        print("[geonames-sync] Keeping the existing sources.", file=sys.stderr)
        return 1

    if not args.no_rebuild:
        try:
            rebuild()
        except Exception as exc:
            print(f"[geonames-sync] Rebuild failed: {exc}", file=sys.stderr)
            print("[geonames-sync] geo.db is unchanged and still usable.", file=sys.stderr)
            return 1

    # Written last: the stamp means "these sources are in place and geo.db was
    # built from them", so an interrupted run must not claim to be up to date.
    _write_atomic(STAMP, json.dumps(manifest, indent=2).encode("utf-8"))
    print(f"[geonames-sync] Done in {time.monotonic() - started:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
