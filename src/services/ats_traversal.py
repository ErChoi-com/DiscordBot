"""Decide which companies an ATS cycle should ask, and remember where it got to.

**Half of this is wired in.** `WatcherManager._rotate_tail` calls `rotate` and
`cursor_after` on every ATS cycle. `next_slice` and `plan_cycle` are not called
by anything, and should not be wired without reading the correction below --
they hand back a bounded slice and leave the rest of the fleet unasked, which at
the count they were sized for would have cut coverage by a factor of forty.

*Correction, and it matters.* This docstring previously said the fan-out reached
a few hundred slugs and that "the other ~98% of the fleet is submitted and
cancelled without ever being asked". That is false, and it was committed as
fact. The bot's own logs record the opposite: `[ats] timed out waiting on N of M
fetches` reports what was **cancelled**, so completion is M-N, and across
platforms that is 43-100% -- typically 73-98%, with workday completing 14,989 of
14,990. The error was taking the printed number as the answer instead of its
complement. `services.health.ats_fleet_coverage` now records the reached
fraction directly so the question does not have to be inferred from a log again.

The August drop this module cited is real but has a different cause. Row count
and distinct-company count collapsed together -- 2026-08-06: 8,459 rows from
2,023 companies; 08-07: 48 rows from 16 -- which is the signature of a backlog
being absorbed, not of coverage shrinking. Companies reached for the first time
dump their whole existing job list into the archive as new; once that is
banked, later cycles find only genuinely new postings. Current volume is
roughly the steady-state rate.

So what is this module actually for? Ordering, on the cycles that do not
complete. When a pass finishes everything -- the common case -- order is
irrelevant and this changes nothing. When one does not (icims has come in at
43%), the tail is cancelled, and in file order the tail is always the same
companies. Digest order and a resume cursor make the *unfinished* remainder
rotate rather than being permanently the same set. That is a real but bounded
benefit, much smaller than the one first claimed here.

`rotate` is the form that benefit takes without any cost attached: the caller
submits the whole fleet exactly as before -- same requests, same politeness
ceiling, same load on the platform -- and only the identity of the cancelled
tail moves. `next_slice` buys sharper ordering by not asking the rest, which is
a trade, and this repo does not want one.

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
    order: Sequence[tuple[str, str]],
    last_digest: str | None,
    count: int,
    is_skipped: Any = None,
    max_scan: int | None = None,
) -> tuple[list[str], str | None, bool]:
    """The next `count` askable slugs after `last_digest`, wrapping at the end.

    Returns (slugs, new_last_digest, wrapped). `wrapped` reports that the walk
    ran off the end and resumed at the front, which is how a caller counts
    completed passes over the fleet.

    A `last_digest` that no longer exists is not an error -- bisect still places
    it among the current entries, so a fleet that shrank under the cursor
    resumes at the right neighbourhood rather than restarting.

    `is_skipped(slug)` marks a slug the scraper would refuse before making any
    request -- in practice `ats_service._is_dead`. Skipped slugs must not
    consume slice positions: 26% of the fleet carries a live dead mark, and for
    lever it is 70%, so a slice of 250 raw slugs asks only 75 companies and the
    cycle then ends early with its budget unspent. They still advance the
    cursor, or the walk would re-scan the same dead run on every cycle and
    never reach past it.

    `max_scan` bounds the work when skipped slugs are dense, so a pathological
    run of dead marks costs a bounded scan rather than a walk of the whole
    fleet. Falling short of `count` is the correct outcome there -- the next
    cycle resumes from where the scan stopped.
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
    scan_cap = min(n, max_scan if max_scan is not None else (want * 10 if is_skipped else want))

    cursor = last_digest
    idx = start
    scanned = 0
    while len(picked) < want and scanned < scan_cap:
        if idx >= n:
            idx, wrapped = 0, True
        entry = order[idx]
        cursor = entry[0]
        if is_skipped is None or not is_skipped(entry[1]):
            picked.append(entry)
        idx += 1
        scanned += 1

    return [slug for _, slug in picked], cursor, wrapped


def rotate(slugs: Iterable[str], last_digest: str | None = None) -> list[str]:
    """The whole fleet in digest order, beginning just after `last_digest`.

    The counterpart to `next_slice`, and the one a scrape loop should reach for
    first. `next_slice` hands back a bounded slice and leaves the rest unasked,
    which trades coverage for order; this drops nothing. The caller submits
    exactly the fleet it would have submitted anyway, and the rotation only
    decides which companies sit in the tail that a cut-off cycle never reaches.
    A cycle that completes its fleet -- the common case -- is unaffected.

    Over successive cycles the cut-off tail moves, so a platform that reaches
    43% each time still offers every company a turn rather than asking the same
    43% forever.
    """
    order = stable_order(slugs)
    if not order:
        return []
    start = 0 if not last_digest else bisect_right(order, (last_digest, chr(0x10FFFF)))
    if start >= len(order):
        # The recorded cursor sits past the end -- a shrunken fleet, or a pass
        # that finished on the last entry. Beginning again is the whole point
        # of a rotation, not an error.
        start = 0
    return [slug for _, slug in order[start:]] + [slug for _, slug in order[:start]]


def cursor_after(rotated: Sequence[str], reached: int) -> str | None:
    """Where a cycle that reached `reached` of `rotated` stopped, as a digest.

    None means "leave the cursor alone": a cycle that asked nobody has no new
    information, and advancing on it would step over companies that were never
    asked -- the failure this rotation exists to prevent.

    `reached` past the end is a completed pass, which lands on the final entry
    so the next rotation starts from the top again.
    """
    if reached <= 0 or not rotated:
        return None
    return digest_for(rotated[min(int(reached), len(rotated)) - 1])


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


# orphan-ok: unwired for the same reason next_slice is, and it is next_slice
# that this calls. The scrape loop walks the fleet with rotate/cursor_after
# instead (manager._rotate_tail), which asks the whole fleet in a rotated order
# rather than a fixed-size slice of it -- at the current counts a slice would
# cut coverage roughly forty-fold. Kept as the assembled form of the slice walk
# for a caller that genuinely wants a bounded slice; the scrape loop is not it.
def plan_cycle(
    state: dict[str, Any],
    platform: str,
    slugs: Iterable[str],
    count: int,
    now: float,
    is_skipped: Any = None,
    max_scan: int | None = None,
) -> list[str]:
    """The slugs this cycle should ask for, advancing the cursor as a side effect.

    The one call a scrape loop needs. `state` is mutated but not persisted --
    the caller saves it after the cycle, so a crashed cycle re-asks its slice
    rather than skipping it.

    Pass `is_skipped=ats_service._is_dead`-style predicate so dead-marked boards
    do not eat slice positions. Note the cursor advances even when nothing was
    picked, so a slice landing entirely inside a run of dead marks still moves
    the walk forward instead of retrying that run forever.
    """
    order = stable_order(slugs)
    entry = state.get("platforms", {}).get(platform) or {}
    picked, new_digest, wrapped = next_slice(
        order, entry.get("last_digest"), count, is_skipped=is_skipped, max_scan=max_scan
    )
    if new_digest is not None:
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
