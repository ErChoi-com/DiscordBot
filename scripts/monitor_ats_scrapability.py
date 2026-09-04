#!/usr/bin/env python3
"""Ask every ATS scraper whether it can still return rows, and remember the answer.

A scraper returning `[]` is indistinguishable from a platform with no openings,
so a platform can go dark and nothing says so. `scripts/check_platform_yield.py`
catches that after the fact by reading the archive; this catches it while it is
happening, by driving the real scrapers against boards already confirmed live.

Two properties matter, and both are enforced here rather than assumed:

**It writes no ATS state.** The scrape path marks slugs dead on 404/410, clears
marks on 200, and flushes on the way out. Those writes are correct when the bot
makes them and wrong when a monitor does: this samples a handful of boards, so a
transient failure here would condemn companies on evidence the run was never
designed to gather. `neutralise_state_writes` replaces all three with no-ops
before a single request goes out, and `main` refuses to run if it cannot.

**It remembers.** The in-process health tracker resets on restart and has no
baseline, so "silent three runs running" cannot survive the thing it is meant to
diagnose. State lives in .ats_validation/ (gitignored, alongside the other
validation reports) and carries the streak across runs and restarts.

Read-only, so this is safe to run on a schedule. Exit status is 2 when a
platform crosses the silence threshold, 0 otherwise, so a scheduler can alert.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CHECKED_DIR = REPO_ROOT / "data" / "ats_checked"
STATE_PATH = REPO_ROOT / ".ats_validation" / "scrapability_monitor.json"

# Three is enough to tell "this platform answers" from "this platform does not",
# and small enough that running every half hour is not a load anyone notices.
# 16 platforms x 3 boards x 48 runs/day is ~2,300 requests spread over a day.
DEFAULT_SAMPLE = 3
DEFAULT_THRESHOLD = 3


def neutralise_state_writes(ats_service: Any) -> list[str]:
    """Replace every dead-slug write in the scrape path with a no-op.

    Returns the names patched so a caller can assert the list is complete --
    a silently-missed writer is the whole risk this function exists to remove.
    """
    patched = []
    for name in ("_mark_dead", "_mark_alive", "flush_dead_slugs"):
        if not hasattr(ats_service, name):
            continue
        setattr(ats_service, name, _noop_for(name))
        patched.append(name)
    return patched


def _noop_for(name: str):
    if name == "flush_dead_slugs":
        return lambda: None
    return lambda platform, slug: None


def load_state(path: Path) -> dict[str, Any]:
    """Previous runs' results, or an empty state if there are none.

    A corrupt or unreadable file reads as empty rather than raising: losing the
    streak makes the next alarm late, while refusing to start makes it never
    come at all.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"platforms": {}}
    if not isinstance(data, dict) or not isinstance(data.get("platforms"), dict):
        return {"platforms": {}}
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def update_platform(
    state: dict[str, Any], platform: str, row_count: int, now: float, error: str = ""
) -> dict[str, Any]:
    """Fold one run's result for one platform into the state. Pure.

    An error is recorded but does NOT count as silence: a scraper that raised
    has a diagnosis already, and folding it in here would let one failure mode
    mask the other. Only a clean run that returned nothing is silence.
    """
    entry = state.setdefault("platforms", {}).setdefault(
        platform, {"consecutive_silent": 0, "last_nonempty_at": 0.0, "runs": 0}
    )
    entry["runs"] = int(entry.get("runs", 0)) + 1
    entry["last_run_at"] = now
    entry["last_row_count"] = row_count
    entry["last_error"] = error

    if error:
        entry["last_was_error"] = True
        return entry
    entry["last_was_error"] = False
    if row_count > 0:
        entry["consecutive_silent"] = 0
        entry["last_nonempty_at"] = now
    else:
        entry["consecutive_silent"] = int(entry.get("consecutive_silent", 0)) + 1
    return entry


def silent_platforms(state: dict[str, Any], threshold: int) -> list[tuple[str, int]]:
    """Platforms silent for `threshold` runs running, worst first."""
    out = [
        (name, int(entry.get("consecutive_silent", 0)))
        for name, entry in state.get("platforms", {}).items()
        if int(entry.get("consecutive_silent", 0)) >= threshold
    ]
    return sorted(out, key=lambda pair: (-pair[1], pair[0]))


def sample_slugs(platform: str, count: int, checked_dir: Path, seed: Any = None) -> list[str]:
    """`count` confirmed-live slugs for a platform, or [] if there are none.

    Deliberately re-drawn each run rather than pinned: a fixed sample would
    report a platform healthy on the strength of three boards that happen to
    still work, and would never notice the rest of the fleet rotting.
    """
    path = checked_dir / f"{platform}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    slugs = sorted(data) if isinstance(data, dict) else sorted(data or [])
    if not slugs:
        return []
    rng = random.Random(seed)
    return rng.sample(slugs, min(count, len(slugs)))


def _fmt_age(now: float, then: float) -> str:
    if not then:
        return "never"
    mins = (now - then) / 60.0
    if mins < 90:
        return f"{mins:.0f}m ago"
    hours = mins / 60.0
    return f"{hours:.0f}h ago" if hours < 48 else f"{hours / 24:.0f}d ago"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help=f"confirmed-live boards per platform (default {DEFAULT_SAMPLE})")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                    help=f"consecutive silent runs before alarming (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--state", type=Path, default=STATE_PATH)
    ap.add_argument("--platform", action="append", default=None,
                    help="limit to one platform (repeatable)")
    args = ap.parse_args(argv)

    from services import ats_service

    patched = neutralise_state_writes(ats_service)
    required = {"_mark_dead", "_mark_alive", "flush_dead_slugs"}
    missing = required - set(patched)
    if missing:
        # Refusing is the safe direction: an unpatched writer would let a
        # three-board sample condemn companies for the whole TTL.
        print(f"refusing to run: could not neutralise {sorted(missing)}", file=sys.stderr)
        return 1

    platforms = sorted(args.platform or ats_service._SCRAPERS)
    state = load_state(args.state)
    now = time.time()

    print(f"ATS scrapability -- {len(platforms)} platforms, {args.sample} board(s) each, "
          f"state {args.state.name}")
    print(f"{'platform':<16} {'rows':>5} {'silent':>7} {'last yield':>12}  note")
    print("-" * 68)

    for platform in platforms:
        slugs = sample_slugs(platform, args.sample, CHECKED_DIR, seed=f"{now}:{platform}")
        if not slugs:
            update_platform(state, platform, 0, now, error="no confirmed-live boards")
            print(f"{platform:<16} {'-':>5} {'-':>7} {'-':>12}  no confirmed-live file")
            continue
        try:
            rows = ats_service.scrape_ats_platform(
                platform=platform, keywords="", location="",
                results_wanted=5, company_slugs=list(slugs),
            )
            err = ""
        except Exception as exc:  # noqa: BLE001 - a raising scraper is a result
            rows, err = [], f"{type(exc).__name__}: {exc}"[:60]

        entry = update_platform(state, platform, len(rows), now, error=err)
        note = err or ("" if rows else "returned nothing")
        print(f"{platform:<16} {len(rows):>5} {entry['consecutive_silent']:>7} "
              f"{_fmt_age(now, entry.get('last_nonempty_at', 0.0)):>12}  {note}")

    save_state(args.state, state)

    alarms = silent_platforms(state, args.threshold)
    print("-" * 68)
    if alarms:
        print(f"\n{len(alarms)} platform(s) silent for >= {args.threshold} runs:")
        for name, streak in alarms:
            print(f"  {name}: {streak} runs with no rows")
        return 2
    print("\nNo platform has been silent long enough to alarm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
