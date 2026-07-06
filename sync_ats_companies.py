import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent / "data" / "ats_companies"

FILES = [
    "ashby_companies.json",
    "bamboohr_companies.json",
    "greenhouse_companies.json",
    "icims_companies.json",
    "lever_companies.json",
    "workday_companies.json",
]


def main() -> int:
    if not (REPO_DIR / ".git").exists():
        print("[ats-sync] data/ats_companies is not a git repo — skipping.")
        return 0

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

    elapsed = time.monotonic() - t_start
    print(f"[ats-sync] Done — {updated}/{len(FILES)} files synced in {elapsed:.1f}s.")
    return 0 if updated == len(FILES) else 1


if __name__ == "__main__":
    sys.exit(main())
