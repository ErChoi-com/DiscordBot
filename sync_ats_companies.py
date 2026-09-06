"""Refresh the two company-list sources the ATS scraper reads.

There are two, and they are not interchangeable:

  data/ats_companies/  a sparse clone of the upstream job-board-aggregator
                       repo, which covers six platforms.
  data/ats_harvest/    the Common Crawl harvest this repo publishes itself
                       from .github/workflows/ats-harvest.yml, which covers
                       every platform the scraper supports.

`ats_service.load_company_lists` unions the second on top of the first, so the
harvest is strictly additive: a harvest that fails to arrive costs the
companies it would have added and nothing else.

That "costs nothing else" is exactly why this file has to be loud. A harvest
that never arrives looks identical to a harvest with nothing new in it, and
that is not hypothetical -- it is what was happening. The fetch was hardcoded
to a remote named `origin` while this checkout calls its remote `DiscordBot`,
so every fetch exited 128 and the handler printed "No ats-harvest branch yet".
The branch existed the whole time. Meanwhile the platform list here named six
of the sixteen the workflow publishes, so even a working fetch would have left
ten platforms on whatever was last harvested by hand.

Both are fixed by not hardcoding either one: the remote is discovered, and the
platform list comes from the manifest the workflow publishes beside the data.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR / "data" / "ats_companies"

FILES = [
    "ashby_companies.json",
    "bamboohr_companies.json",
    "greenhouse_companies.json",
    "icims_companies.json",
    "lever_companies.json",
    "workday_companies.json",
]


REMOTE_URL = "https://github.com/Feashliaa/job-board-aggregator"


def _bootstrap() -> bool:
    """Create data/ats_companies as a sparse clone.

    data/ats_companies is gitignored (it carries its own history), so on a fresh
    clone the directory does not exist. Skipping there left ats_service with an
    empty company list -- it would scrape nothing and report no error, which
    looks identical to "no jobs found". Clone it instead of skipping.
    """
    if (REPO_DIR / ".git").exists():
        # Never rmtree a real clone. main() already checks this, but the guard
        # belongs next to the destructive call, not one frame up.
        return True
    if REPO_DIR.exists():
        print(f"[ats-sync] data/ats_companies exists but isn't a git clone — "
              "clearing stale cache before cloning")
        shutil.rmtree(REPO_DIR)
    print(f"[ats-sync] data/ats_companies missing — cloning {REMOTE_URL}")
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none", "--no-checkout",
         REMOTE_URL, str(REPO_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[ats-sync] Clone failed: {result.stderr.strip()}", file=sys.stderr)
        print("[ats-sync] ATS scraping will find NO companies until this succeeds.",
              file=sys.stderr)
        return False
    return True


# ── the harvest this repo publishes for itself ──────────────────────────────

# This directory is owned by this script and replaced wholesale from the
# branch. It is deliberately NOT data/ats_harvest, which holds the local
# harvester's own output: writing the published files over those would have
# deleted 14,278 companies the bot already scrapes, because CI stopped sweeping
# Wayback (commit d2e68f5) while the local files still carry what it found.
# ats_service unions both directories, so neither can shrink the other.
HARVEST_DIR = BASE_DIR / "data" / "ats_harvest_ci"
LOCAL_HARVEST_DIR = BASE_DIR / "data" / "ats_harvest"
HARVEST_BRANCH = "ats-harvest"
DEAD_SLUG_DIR = BASE_DIR / "data" / "dead_slugs"

# The workflow writes this beside the data: {"files": {name: {sha256, count}}}.
# It is the list of what was published, so reading the platform set from it
# means adding a platform to the workflow needs no change here.
MANIFEST_NAME = "manifest.json"

# Published beside the company lists but not one of them. `_`-prefixed names
# are the harvester's own bookkeeping -- `_crawls.json` records which Common
# Crawl indexes it has already swept -- and are skipped by prefix.
NON_PLATFORM_FILES = frozenset({MANIFEST_NAME})


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Every git call goes through here, captured the same way."""
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True)


def _last_line(stream: bytes) -> str:
    text = stream.decode("utf-8", "replace").strip()
    return text.splitlines()[-1] if text else "no detail"


def _remote_names() -> list[str]:
    """Configured remotes, `origin` first when it exists.

    Asking git rather than assuming. The branch name is fixed by the workflow;
    the remote name is a local choice, and getting it wrong here failed closed
    and quietly for weeks.
    """
    out = _git("remote")
    if out.returncode != 0:
        return []
    names = [n.strip() for n in out.stdout.decode("utf-8", "replace").splitlines() if n.strip()]
    return sorted(names, key=lambda n: (n != "origin", n))


def _fetch_harvest() -> str | None:
    """Fetch the harvest branch from whichever remote publishes it.

    Returns the remote that answered, or None. Every remote that refused is
    named with its own error, so "the harvest did not arrive" can never again
    be reported as "there is no harvest branch".
    """
    remotes = _remote_names()
    if not remotes:
        print("[ats-sync] No git remotes configured — skipping harvest")
        return None

    problems: list[str] = []
    for remote in remotes:
        result = _git("fetch", "--depth=1", remote, HARVEST_BRANCH)
        if result.returncode == 0:
            if remote != "origin":
                print(f"[ats-sync] harvest branch found on remote '{remote}'")
            return remote
        problems.append(f"{remote}: {_last_line(result.stderr)}")

    print(f"[ats-sync] Could not fetch '{HARVEST_BRANCH}' from any remote — "
          f"the harvest will not be updated. Tried {'; '.join(problems)}",
          file=sys.stderr)
    return None


def _published(name: str) -> bytes | None:
    """One file's bytes from the fetched branch, or None if it is not there."""
    result = _git("show", f"FETCH_HEAD:{name}")
    return result.stdout if result.returncode == 0 else None


def _published_names() -> list[str]:
    """Every file at the root of the fetched branch."""
    result = _git("ls-tree", "--name-only", "FETCH_HEAD")
    if result.returncode != 0:
        return []
    return [n.strip() for n in result.stdout.decode("utf-8", "replace").splitlines() if n.strip()]


def _read_manifest() -> dict[str, dict]:
    """The manifest's `files` map, or {} when it is missing or malformed.

    Empty is not fatal: `_platform_files` falls back to whatever JSON the
    branch carries. Refusing to sync without a manifest would make a
    bookkeeping file the single point of failure for the whole fleet.
    """
    blob = _published(MANIFEST_NAME)
    if blob is None:
        return {}
    try:
        data = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[ats-sync] {MANIFEST_NAME} is unreadable ({exc}) — "
              "syncing without integrity checks")
        return {}
    files = data.get("files") if isinstance(data, dict) else None
    return files if isinstance(files, dict) else {}


def _platform_files(manifest: dict[str, dict]) -> list[str]:
    """The company-list files to pull, newest source of truth first.

    The manifest when there is one, the branch listing when there is not. A
    hardcoded list is what let ten of sixteen platforms go unsynced without a
    word, so there is deliberately no hardcoded list to fall back to.
    """
    if manifest:
        names = list(manifest)
    else:
        names = _published_names()
    return sorted(
        n for n in names
        if n.endswith(".json") and n not in NON_PLATFORM_FILES and not n.startswith("_")
    )


def _verified(name: str, blob: bytes, expected: dict | None) -> list | None:
    """Parse and check a published file, or None to keep the local copy.

    Checked before anything is written, never after. Replacing a good local
    list with a truncated one is how the fleet silently shrinks, and
    `ats_service` cannot tell a short list from a complete one -- it reads both
    as "these are the companies". sha256 catches corruption in transit; the
    count catches a file that still parses after losing entries.
    """
    try:
        payload = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[ats-sync] harvest {name} is unreadable, keeping local copy: {exc}")
        return None
    if not isinstance(payload, list):
        print(f"[ats-sync] harvest {name} is not a list, keeping local copy")
        return None
    if not expected:
        return payload

    digest = expected.get("sha256")
    if digest and hashlib.sha256(blob).hexdigest() != digest:
        print(f"[ats-sync] harvest {name} does not match its manifest checksum, "
              "keeping local copy", file=sys.stderr)
        return None
    count = expected.get("count")
    if isinstance(count, int) and len(payload) != count:
        print(f"[ats-sync] harvest {name} has {len(payload)} entries but the "
              f"manifest says {count}, keeping local copy", file=sys.stderr)
        return None
    return payload


def _write_atomic(target: Path, blob: bytes) -> None:
    """Same tmp+replace shape the rest of the bot's state writes use.

    A sync killed mid-write must not leave a truncated company list that the
    next boot reads as a smaller fleet.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".json.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(blob)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _sync_harvest() -> tuple[int, int]:
    """Pull the published harvest. Returns (written, available).

    Both numbers, because one of them alone cannot be read. "6 written" is
    healthy against 6 available and a silent regression against 16, and telling
    those apart from the log is the whole point.

    Not fatal on failure: the harvest is additive over upstream, so not having
    it means scraping the upstream list alone -- which is what happened before
    this existed.
    """
    if _git("rev-parse", "--show-toplevel").returncode != 0:
        print("[ats-sync] Not a git checkout — skipping harvest")
        return 0, 0

    remote = _fetch_harvest()
    if remote is None:
        return 0, 0

    manifest = _read_manifest()
    names = _platform_files(manifest)
    if not names:
        print(f"[ats-sync] '{HARVEST_BRANCH}' carries no company lists",
              file=sys.stderr)
        return 0, 0

    written = 0
    for name in names:
        blob = _published(name)
        if blob is None:
            print(f"[ats-sync] harvest {name} is in the manifest but not on "
                  f"the branch", file=sys.stderr)
            continue
        payload = _verified(name, blob, manifest.get(name))
        if payload is None:
            continue
        _write_atomic(HARVEST_DIR / name, blob)
        written += 1
        print(f"[ats-sync] harvest {name[:-5]}: {len(payload):,} companies")

    if written < len(names):
        print(f"[ats-sync] {len(names) - written} of {len(names)} harvest "
              f"file(s) were not updated — see above", file=sys.stderr)

    _merge_dead_marks()
    return written, len(names)


def _merge_dead_marks() -> int:
    """Fold any dead marks the branch carries into the local ones.

    Driven by what is actually published rather than a platform list, for the
    same reason as the company lists. Today the workflow publishes none: ATS
    probing was deliberately moved off the runners, so the marks are made on
    the bot's own host where the platforms answer. This stays because that is a
    decision that could be revisited, and a no-op that says so is better than
    one that looks like a successful merge of nothing.

    Merged rather than overwritten, keeping the newer date on a conflict. The
    local file is live state the running bot mutates, so replacing it would
    discard marks made since the last publish.

    Revivals need no special handling: a mark only suppresses a slug until
    DEAD_SLUG_RECHECK_DAYS elapses, so a company that came back is re-probed on
    its own within a week.
    """
    listed = _git("ls-tree", "--name-only", "FETCH_HEAD:dead")
    if listed.returncode != 0:
        return 0
    names = [n.strip() for n in listed.stdout.decode("utf-8", "replace").splitlines()
             if n.strip().endswith(".json")]
    if not names:
        return 0

    merged = 0
    for name in names:
        blob = _published(f"dead/{name}")
        if blob is None:
            continue
        try:
            published = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"[ats-sync] dead/{name} unreadable, skipping: {exc}")
            continue
        if not isinstance(published, dict):
            continue

        target = DEAD_SLUG_DIR / name
        local: dict = {}
        if target.exists():
            try:
                loaded = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    local = loaded
            except (OSError, json.JSONDecodeError):
                local = {}

        added = 0
        for slug, marked in published.items():
            slug, marked = str(slug), str(marked)
            current = local.get(slug)
            if current is None:
                added += 1
                local[slug] = marked
            elif marked > current:      # ISO dates compare lexicographically
                local[slug] = marked

        _write_atomic(target, json.dumps(local, indent=2, sort_keys=True).encode("utf-8"))
        merged += 1
        print(f"[ats-sync] dead {name[:-5]}: +{added} new marks "
              f"({len(local)} total)")
    return merged


def main() -> int:
    if not (REPO_DIR / ".git").exists() and not _bootstrap():
        return 1

    t_start = time.monotonic()

    print("[ats-sync] Fetching remote (sparse, depth=1)...")
    fetch = subprocess.run(
        ["git", "fetch", "--depth=1", "--filter=blob:none", "origin", "main"],
        cwd=REPO_DIR, capture_output=True, text=True,
    )
    if fetch.returncode != 0:
        print(f"[ats-sync] Fetch failed: {fetch.stderr.strip()}", file=sys.stderr)
        return fetch.returncode

    updated = 0
    for filename in FILES:
        result = subprocess.run(
            ["git", "show", f"origin/main:data/{filename}"],
            cwd=REPO_DIR, capture_output=True,
        )
        if result.returncode != 0:
            print(f"[ats-sync] Could not read {filename}: {result.stderr.decode().strip()}", file=sys.stderr)
            continue
        (REPO_DIR / filename).write_bytes(result.stdout)
        updated += 1

    harvested, available = _sync_harvest()

    elapsed = time.monotonic() - t_start
    print(f"[ats-sync] Done — upstream {updated}/{len(FILES)}, "
          f"harvest {harvested}/{available} in {elapsed:.1f}s.")
    return 0 if updated == len(FILES) else 1


if __name__ == "__main__":
    sys.exit(main())
