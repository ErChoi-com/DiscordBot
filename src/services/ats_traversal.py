"""Decide which companies an ATS cycle should ask, and remember where it got to.

`scrape_ats_platform` submits its whole company list to a bounded pool and
cancels whatever has not finished when the budget runs out. The list is in file
order, so "whatever finished" is always its head -- the same few hundred slugs,
every cycle, for the life of the bot. The other ~98% of the fleet is submitted
and cancelled without ever being asked.

That stayed invisible while the head was still yielding. Once its jobs were all
archived, `_drop_already_archived` correctly began dropping them as duplicates,
and ATS output fell from 8,434 rows on 2026-08-06 to 30 on 2026-08-07 across
every platform at once. Nothing broke: the bot had exhausted the only slice it
could reach.

Two decisions carry this module.

**Order by digest, not by name.** Sorting on blake2b(slug) is uncorrelated with
the first letter, so any prefix of that order is a cross-section of the whole
fleet rather than the A's. It generalises the trick `ats_service._recheck_days_for`
already uses to stop dead-slug re-probes coming due all at once.

**Resume from a digest, not an index.** The fleet is not stable -- the harvest
adds slugs weekly and reconciliation removes them. An integer offset silently
skips or repeats entries when the list changes underneath it. A digest is
intrinsic to its slug, so inserting or removing other members never moves it,
and `bisect` gives a well-defined resume point even when the recorded digest is
itself gone.
"""
from __future__ import annotations

import json
import os
from bisect import bisect_right
from hashlib import blake2b
from pathlib import Path
from typing import Any, Iterable, Sequence

# Wide enough that a collision between two slugs in a ~30k fleet is not worth
# reasoning about, narrow enough to stay readable in the state file.
_DIGEST_BYTES = 8


def digest_for(slug: str) -> str:
    """Stable sort key for a slug, independent of every other slug."""
    return blake2b(slug.encode("utf-8"), digest_size=_DIGEST_BYTES).hexdigest()


def stable_order(slugs: Iterable[str]) -> list[tuple[str, str]]:
    """(digest, slug) pairs in digest order, deduped.

    Deterministic across processes and machines: the same fleet always yields
    the same order, which is what lets a cursor recorded by one run mean the
    same thing to the next.
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for slug in slugs:
        s = str(slug).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        pairs.append((digest_for(s), s))
    pairs.sort()
    return pairs


def next_slice(
    order: Sequence[tuple[str, str]], last_digest: str | None, count: int
) -> tuple[list[str], str | None, bool]:
    """The next `count` slugs after `last_digest`, wrapping at the end.

    Returns (slugs, new_last_digest, wrapped). `wrapped` reports that the walk
    ran off the end and resumed at the front, which is how a caller counts
    completed passes over the fleet.

    A `last_digest` that no longer exists is not an error -- bisect still places
    it among the current entries, so a fleet that shrank under the cursor
    resumes at the right neighbourhood rather than restarting.
    """
    if count <= 0 or not order:
        return [], last_digest, False

    start = 0 if not last_digest else bisect_right(order, (last_digest, chr(0x10FFFF)))
    wrapped = False
    picked: list[tuple[str, str]] = []
    n = len(order)

    if start >= n:
        start, wrapped = 0, True

    # Cap at the fleet size: asking for more than exists should hand back the
    # whole fleet once, not loop it repeatedly in a single cycle.
    want = min(count, n)
    idx = start
    for _ in range(want):
        if idx >= n:
            idx, wrapped = 0, True
        picked.append(order[idx])
        idx += 1

    return [slug for _, slug in picked], picked[-1][0], wrapped


def load_state(path: Path) -> dict[str, Any]:
    """Per-platform cursors, or an empty state.

    An unreadable or malformed file reads as empty. Losing the cursor costs one
    repeated slice; refusing to start would stop the fleet being walked at all,
    which is the condition this module exists to end.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"platforms": {}}
    if not isinstance(data, dict) or not isinstance(data.get("platforms"), dict):
        return {"platforms": {}}
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Write atomically -- same tmp+replace shape as _save_dead_slugs.

    A cycle killed mid-write must not leave a truncated file that the next run
    discards as malformed, silently restarting the walk.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def record_progress(
    state: dict[str, Any], platform: str, last_digest: str | None, wrapped: bool, now: float
) -> dict[str, Any]:
    """Fold one cycle's advance into the state. Pure."""
    entry = state.setdefault("platforms", {}).setdefault(
        platform, {"last_digest": None, "cycle_wraps": 0, "slices": 0}
    )
    entry["last_digest"] = last_digest
    entry["slices"] = int(entry.get("slices", 0)) + 1
    entry["updated_at"] = now
    if wrapped:
        entry["cycle_wraps"] = int(entry.get("cycle_wraps", 0)) + 1
    return entry


def plan_cycle(
    state: dict[str, Any], platform: str, slugs: Iterable[str], count: int, now: float
) -> list[str]:
    """The slugs this cycle should ask for, advancing the cursor as a side effect.

    The one call a scrape loop needs. `state` is mutated but not persisted --
    the caller saves it after the cycle, so a crashed cycle re-asks its slice
    rather than skipping it.
    """
    order = stable_order(slugs)
    entry = state.get("platforms", {}).get(platform) or {}
    picked, new_digest, wrapped = next_slice(order, entry.get("last_digest"), count)
    if picked:
        record_progress(state, platform, new_digest, wrapped, now)
    return picked


def coverage(state: dict[str, Any], platform: str, fleet_size: int, count: int) -> dict[str, Any]:
    """How far through `platform` the walk has got, for reporting.

    `cycles_per_pass` answers the question the old design could not even ask:
    how long until every company has been offered a turn.
    """
    entry = state.get("platforms", {}).get(platform) or {}
    per_pass = -(-max(fleet_size, 1) // max(count, 1)) if count > 0 else 0
    return {
        "platform": platform,
        "fleet": fleet_size,
        "slices_done": int(entry.get("slices", 0)),
        "full_passes": int(entry.get("cycle_wraps", 0)),
        "cycles_per_pass": per_pass,
        "started": entry.get("last_digest") is not None,
    }
