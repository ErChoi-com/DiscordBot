"""Probe harvested ATS company slugs and record which boards are actually gone.

Harvesting is optimistic: an archived URL proves a board existed when it was
crawled, not that it exists now. Measured live rates for newly harvested slugs
run from 87% (BambooHR) down to 11% (iCIMS), so a list that is only ever added
to fills with companies that will never answer, and every one of them costs a
request per scrape cycle forever.

This closes that loop. It asks each platform whether a board exists and writes
the misses into ``data/dead_slugs/<platform>.json`` -- the same file, the same
``{slug: YYYY-MM-DD}`` shape, and the same meaning that ``ats_service`` already
uses, so the bot needs no new concept to benefit. Nothing is deleted: a dead
mark suppresses a slug until ``DEAD_SLUG_RECHECK_DAYS`` elapses and then it is
probed again, which is what lets a board that was down for a day come back.

Why probe here rather than let the bot discover deadness: the bot only learns a
slug is dead by scraping it during a real cycle, so a freshly harvested batch
costs thousands of wasted requests spread over weeks of user-facing runs. Doing
it once, in CI, in parallel, gets the same answer before the slugs ever reach a
scrape cycle.

Usage:
    python scripts/validate_ats_slugs.py                    # everything unchecked
    python scripts/validate_ats_slugs.py --platform lever
    python scripts/validate_ats_slugs.py --limit 2000       # bound a CI run
    python scripts/validate_ats_slugs.py --sample 200 --dry-run
    python scripts/validate_ats_slugs.py --recheck-dead     # re-probe dead marks

Exit codes:
    0  validation ran
    1  every platform failed to probe (network or endpoints changed)
    2  harness error
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
COMPANY_DIR = DATA_DIR / "ats_companies"
HARVEST_DIR = DATA_DIR / "ats_harvest"
DEAD_DIR = DATA_DIR / "dead_slugs"

PLATFORMS = ("greenhouse", "lever", "ashby", "workday", "icims", "bamboohr",
             "paylocity")

COMPANY_FILES = {p: f"{p}_companies.json" for p in PLATFORMS}
# Upstream ships this one under a different name, and as {guid, name, jobs}
# objects rather than a flat list of identifiers.
COMPANY_FILES["paylocity"] = "paylocity_companies_clean.json"

# Mirrors ats_service.DEAD_SLUG_RECHECK_DAYS. A slug marked dead more recently
# than this is not re-probed, so repeated runs cost nothing for known-dead
# companies while still letting them back in eventually.
RECHECK_DAYS = 7

REQUEST_TIMEOUT = 25

# Wall-clock ceiling for one platform's audit sample. The audit is a
# measurement, not work: a partial sample still estimates the rate, while an
# unbounded one can outlast the run it annotates.
AUDIT_BUDGET_SECONDS = 240.0

# Per-platform concurrency. These hit real ATS endpoints, so they stay at or
# below what ats_service already uses for the same host.
WORKERS = {"greenhouse": 16, "lever": 16, "ashby": 8, "workday": 12,
           "icims": 12, "bamboohr": 12,
           # Paylocity sheds load by refusing connections rather than slowing
           # down, and it does so steeply. Measured over 60-slug batches:
           #
           #   workers  unreachable  raw rate  useful rate
           #      2         0/60      3.2/s      3.2/s
           #      4        20/60      5.3/s      3.5/s
           #      8        48/60     13.2/s      2.7/s
           #     12        53/60     21.5/s      2.5/s
           #
           # Raw throughput keeps climbing while useful throughput does not:
           # past two workers most requests come back with no verdict, so the
           # extra concurrency buys refusals. A refusal is recorded unknown and
           # changes no marks, so the cost is wasted requests rather than wrong
           # answers -- but the slug still has to be probed again later.
           "paylocity": 2}

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36")


class Unreachable(Exception):
    """The probe could not reach the platform at all (as opposed to a 404)."""


def _request(url: str, *, method: str = "GET", payload: dict | None = None,
             timeout: int = REQUEST_TIMEOUT) -> tuple[int | None, bytes, str]:
    """Returns (status, first bytes, final url). None status means no answer.

    The final URL matters: BambooHR answers an unknown tenant with a 200 that
    redirects to its marketing site, so status alone cannot tell them apart.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json,*/*"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(2048), resp.geturl()
    except urllib.error.HTTPError as exc:
        return exc.code, b"", url
    except Exception:
        return None, b"", url


# --------------------------------------------------------------------------
# Per-platform liveness. Each returns True (board exists), False (it does not),
# or raises Unreachable when the answer is "we could not tell".
#
# Every one of these was calibrated against known-live and known-dead slugs;
# the obvious URL is wrong for four of the six platforms.
# --------------------------------------------------------------------------

def _decide(status: int | None) -> bool:
    if status is None:
        raise Unreachable("no response")
    if status == 200:
        return True
    if status in (301, 302, 403, 404, 410, 422):
        return False
    # 5xx and rate limits say nothing about whether the company exists.
    raise Unreachable(f"HTTP {status}")


def live_greenhouse(slug: str) -> bool:
    status, _, _ = _request(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs")
    return _decide(status)


# Lever's US and EU regions are separate deployments and a company exists in
# exactly one: amicustherapeutics is 200 on api.eu.lever.co and 404 on
# api.lever.co. Checking one host marks the entire EU population dead.
LEVER_API_HOSTS = ("api.lever.co", "api.eu.lever.co")


def live_lever(slug: str) -> bool:
    unreachable = 0
    for host in LEVER_API_HOSTS:
        status, _, _ = _request(f"https://{host}/v0/postings/{slug}?mode=json")
        try:
            if _decide(status):
                return True
        except Unreachable:
            unreachable += 1
    if unreachable == len(LEVER_API_HOSTS):
        raise Unreachable("no region answered")
    return False


def live_ashby(slug: str) -> bool:
    # jobs.ashbyhq.com is a single-page app that answers 200 for every route,
    # including nonsense, so it cannot distinguish a real board. The posting API
    # 404s properly.
    status, _, _ = _request(f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
    return _decide(status)


def live_bamboohr(slug: str) -> bool:
    # An unknown tenant is also 200: it redirects to www.bamboohr.com. The tells
    # are the final URL leaving the tenant host and a real board answering JSON
    # rather than a marketing page.
    status, body, final = _request(f"https://{slug}.bamboohr.com/careers/list")
    if status is None:
        raise Unreachable("no response")
    if status != 200:
        return _decide(status)
    if f"{slug}.bamboohr.com" not in final:
        return False
    return body.lstrip()[:1] in (b"{", b"[")


def live_icims(slug: str) -> bool:
    # iCIMS runs two mutually exclusive host conventions. Over a 120-slug
    # sample, 46 resolved only as careers-<slug>.icims.com, 34 only as
    # <slug>.icims.com, and zero resolved both ways -- so testing one form
    # misreports roughly 42% of live boards as dead.
    forms = [slug] if slug.startswith("careers-") else [f"careers-{slug}", slug]
    unreachable = 0
    for host in forms:
        status, _, _ = _request(f"https://{host}.icims.com/sitemap.xml")
        try:
            if _decide(status):
                return True
        except Unreachable:
            unreachable += 1
    if unreachable == len(forms):
        raise Unreachable("no form answered")
    return False


def live_paylocity(slug: str) -> bool:
    """Paylocity answers 200 for an unknown company too.

    A missing board renders a 4.3KB shell titled "Job Not Found" while a real
    one renders "<Company> - Job Opportunities", so status and even response
    size are not the discriminator -- the title is. Same trap as Ashby's SPA
    and BambooHR's marketing redirect.
    """
    status, body, _ = _request(
        f"https://recruiting.paylocity.com/recruiting/jobs/All/{slug}",
        timeout=REQUEST_TIMEOUT)
    if status is None:
        raise Unreachable("no response")
    if status != 200:
        return _decide(status)
    return b"Job Not Found" not in body


def live_workday(slug: str) -> bool:
    # The CXS endpoint needs a real JSON body: without one a live tenant answers
    # 500 and a nonexistent one 422, and the plain page GET is an SPA that 200s
    # for anything. Only a bodied POST separates them.
    parts = slug.split("|")
    if len(parts) != 3:
        return False
    tenant, host, site = parts
    status, _, _ = _request(
        f"https://{tenant}.{host}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs",
        method="POST",
        payload={"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""},
    )
    return _decide(status)


PROBES: dict[str, Callable[[str], bool]] = {
    "greenhouse": live_greenhouse, "lever": live_lever, "ashby": live_ashby,
    "workday": live_workday, "icims": live_icims, "bamboohr": live_bamboohr,
    "paylocity": live_paylocity,
}


# --------------------------------------------------------------------------
# Slug and dead-mark storage, matching ats_service's on-disk contract
# --------------------------------------------------------------------------

def _read_list(path: Path) -> list[str]:
    """Read an identifier list, tolerating upstream's object-shaped Paylocity
    file. Without the dict branch every entry would stringify to "{'guid': ...}"
    and the whole platform would probe as garbage."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, dict):
            item = item.get("guid") or item.get("id") or ""
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def load_candidates(platform: str) -> list[str]:
    """Every slug the bot would scrape: upstream list plus local harvest."""
    seen: set[str] = set()
    out: list[str] = []
    for path in (COMPANY_DIR / COMPANY_FILES[platform], HARVEST_DIR / f"{platform}.json"):
        for slug in _read_list(path):
            if slug not in seen:
                seen.add(slug)
                out.append(slug)
    return out


def load_dead(platform: str) -> dict[str, str]:
    path = DEAD_DIR / f"{platform}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    # Older revisions stored a bare list; treat those as marked today so they
    # still expire rather than being dropped.
    if isinstance(data, list):
        today = _dt.date.today().isoformat()
        return {str(k): today for k in data}
    return {}


def save_dead(platform: str, dead: dict[str, str]) -> None:
    """Write atomically, in ats_service's exact format.

    write_text truncates in place, so an interrupted write leaves a half-written
    file that ats_service parses as empty -- silently resurrecting every dead
    slug on the platform.
    """
    DEAD_DIR.mkdir(parents=True, exist_ok=True)
    target = DEAD_DIR / f"{platform}.json"
    tmp = target.with_suffix(f".json.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(dead, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def save_dead_merged(platform: str, result: dict, today: _dt.date,
                     ttl_days: int) -> dict[str, int]:
    """Re-read the on-disk marks, apply this run's verdicts, then write.

    The map this run loaded is minutes to tens of minutes stale by the time it
    is written, and the running bot marks the same files from its own scrape
    cycles -- on this machine it wrote ~3,500 greenhouse marks during a single
    validation pass. Writing back the map we loaded would silently discard
    every one of them, and the bot would do the same to us on its next flush.
    Both writers are individually atomic, which makes the loss invisible rather
    than corrupt.

    Re-reading immediately before the write narrows the race to the write
    itself. It cannot close it without a cross-process lock, but the remaining
    window is milliseconds instead of minutes, and the recheck cycle repairs
    anything lost in it.
    """
    dead_map = load_dead(platform)
    expired = purge_expired(dead_map, today, ttl_days)
    changes = apply_results(dead_map, result, today)
    save_dead(platform, dead_map)
    changes["expired"] = expired
    changes["dead_total"] = len(dead_map)
    return changes


def _stale(marked: str, today: _dt.date, recheck_days: int) -> bool:
    """True when a dead mark is old enough to be worth re-probing."""
    try:
        when = _dt.date.fromisoformat(marked)
    except ValueError:
        return True
    return (today - when).days >= recheck_days


def select_targets(platform: str, dead: dict[str, str], *, recheck_dead: bool,
                   limit: int | None, sample: int | None, today: _dt.date,
                   recheck_days: int = RECHECK_DAYS, seed: int = 0) -> list[str]:
    """Slugs worth probing this run.

    Unchecked slugs first, then dead marks old enough to deserve another look.
    Ordering matters when --limit bounds the run: spending the budget on slugs
    nobody has ever probed beats re-confirming what we already believe.
    """
    candidates = load_candidates(platform)
    fresh = [s for s in candidates if s not in dead]
    stale = [s for s in candidates
             if s in dead and (recheck_dead or _stale(dead[s], today, recheck_days))]
    ordered = fresh + stale
    if sample is not None and len(ordered) > sample:
        rng = random.Random(seed)
        ordered = rng.sample(ordered, sample)
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def validate_platform(platform: str, slugs: Iterable[str], *,
                      probe: Callable[[str], bool] | None = None,
                      workers: int | None = None,
                      budget_seconds: float | None = None,
                      now: Callable[[], float] = time.monotonic,
                      log: Callable[[str], None] = print) -> dict[str, int]:
    """Probe slugs concurrently. Returns counts and the resulting verdicts.

    Bounded by wall clock, not just count. Platforms differ by an order of
    magnitude in how fast they answer -- Workday needs a bodied POST and iCIMS
    two requests per slug -- so a slug budget that suits one starves the rest.
    Whatever is not reached this run is simply picked up next run: the target
    selection prefers never-probed slugs, so progress accumulates.
    """
    slugs = list(slugs)
    if probe is None:
        probe = PROBES[platform]
    if workers is None:
        workers = WORKERS.get(platform, 8)
    started = now()

    live: list[str] = []
    dead: list[str] = []
    unknown: list[str] = []
    lock = threading.Lock()

    def one(slug: str) -> None:
        try:
            ok = probe(slug)
        except Unreachable:
            with lock:
                unknown.append(slug)
            return
        except Exception:
            # Any unexpected failure is "we could not tell", never "dead":
            # marking on a bug would suppress a real company for a week.
            with lock:
                unknown.append(slug)
            return
        with lock:
            (live if ok else dead).append(slug)

    # Submitted in chunks so the budget can be checked between them. Cancelling
    # in-flight probes is not possible once submitted, so a chunk is the
    # granularity: small enough to stop promptly, large enough to keep every
    # worker busy.
    chunk = max(workers * 4, 1)
    probed = 0
    stopped_early = False
    if slugs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(slugs), chunk):
                if budget_seconds is not None and start and                         now() - started > budget_seconds:
                    stopped_early = True
                    break
                batch = slugs[start:start + chunk]
                list(pool.map(one, batch))
                probed += len(batch)

    if stopped_early:
        log(f"[validate] {platform}: budget spent, {len(slugs) - probed} slug(s) "
            "deferred to the next run")
    log(f"[validate] {platform}: probed {probed} -> "
        f"{len(live)} live, {len(dead)} dead, {len(unknown)} unknown")
    return {"probed": probed, "live": len(live), "dead": len(dead),
            "unknown": len(unknown), "deferred": len(slugs) - probed,
            "_live": live, "_dead": dead}


def apply_results(dead_map: dict[str, str], result: dict, today: _dt.date) -> dict[str, int]:
    """Fold verdicts into the dead map. Live slugs are cleared, dead ones dated.

    Unknown slugs are left exactly as they were: a timeout is not evidence.
    """
    added = revived = redated = 0
    for slug in result["_live"]:
        if dead_map.pop(slug, None) is not None:
            revived += 1
    stamp = today.isoformat()
    for slug in result["_dead"]:
        if slug in dead_map:
            redated += 1
        else:
            added += 1
        dead_map[slug] = stamp
    return {"added": added, "revived": revived, "redated": redated}


def purge_expired(dead_map: dict[str, str], today: _dt.date, ttl_days: int) -> int:
    """Drop marks older than the TTL so a long-dead slug is eventually retried."""
    cutoff = (today - _dt.timedelta(days=ttl_days)).isoformat()
    expired = [s for s, when in dead_map.items() if when < cutoff]
    for slug in expired:
        del dead_map[slug]
    return len(expired)


def audit_live_rate(platform: str, sample: int, *,
                     probe: Callable[[str], bool] | None = None,
                     workers: int | None = None,
                     seed: int = 0,
                     budget_seconds: float | None = AUDIT_BUDGET_SECONDS,
                     log: Callable[[str], None] = print) -> dict[str, int]:
    """Estimate a platform's true live rate from a uniform random sample.

    The working pass cannot answer this. It probes never-probed slugs first --
    correct for getting work done, since re-confirming known-dead companies is
    waste -- but that means it systematically samples survivors: everything the
    bot already found dead is excluded. Reading its live rate as the
    population's overstates it badly. Measured on the same data, the working
    pass reported 97.5% while a uniform sample of the same slugs was 38.8%.

    Deliberately read-only. Folding these verdicts into the dead map would let
    an audit reshape the population it is trying to measure, and the working
    pass reaches those slugs on its own soon enough.
    """
    candidates = load_candidates(platform)
    if not candidates:
        return {"sampled": 0, "live": 0, "dead": 0, "unknown": 0, "rate": 0.0}
    picks = (candidates if len(candidates) <= sample
             else random.Random(seed).sample(candidates, sample))
    # Budgeted like the working pass. Without this the audit is the one
    # unbounded network loop in the run: Paylocity probes at two workers and
    # refuses connections under load, so an audit there can outlast everything
    # it was meant to annotate. A short sample is worth less than a stalled job.
    result = validate_platform(platform, picks, probe=probe, workers=workers,
                               budget_seconds=budget_seconds,
                               log=lambda _m: None)
    decided = result["live"] + result["dead"]
    rate = round(100.0 * result["live"] / decided, 1) if decided else 0.0
    short = " (cut short by budget)" if result.get("deferred") else ""
    log(f"[validate] {platform}: audit sample {result['probed']} of "
        f"{len(candidates)} -> {rate}% live (population estimate){short}")
    return {"sampled": result["probed"], "live": result["live"],
            "dead": result["dead"], "unknown": result["unknown"],
            "rate": rate, "population": len(candidates)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--platform", action="append", dest="platforms",
                        choices=sorted(PLATFORMS), help="limit to a platform")
    parser.add_argument("--limit", type=int, default=None,
                        help="max slugs to probe per platform")
    parser.add_argument("--sample", type=int, default=None,
                        help="probe a random sample (for measuring live rates)")
    parser.add_argument("--recheck-dead", action="store_true",
                        help="re-probe every dead mark, ignoring the recheck window")
    parser.add_argument("--ttl-days", type=int, default=90,
                        help="drop dead marks older than this (default 90)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--budget-seconds", type=float, default=1200.0,
                        help="wall-clock ceiling per platform (0 disables)")
    parser.add_argument("--audit-sample", type=int, default=0,
                        help="also probe N random slugs per platform, read-only, "
                             "to estimate the true population live rate")
    parser.add_argument("--dry-run", action="store_true",
                        help="probe and report, write nothing")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    log: Callable[[str], None] = (
        (lambda m: print(m, file=sys.stderr)) if args.json else print
    )
    today = _dt.date.today()
    platforms = args.platforms or list(PLATFORMS)
    summary: dict[str, dict] = {}
    any_probed = False
    any_targets = False
    t0 = time.monotonic()

    for platform in platforms:
        dead_map = load_dead(platform)
        expired = purge_expired(dead_map, today, args.ttl_days)
        targets = select_targets(platform, dead_map, recheck_dead=args.recheck_dead,
                                 limit=args.limit, sample=args.sample, today=today)
        total = len(load_candidates(platform))
        if targets:
            any_targets = True
        if not targets:
            log(f"[validate] {platform}: nothing to probe "
                f"({total} known, {len(dead_map)} dead)")
            summary[platform] = {"probed": 0, "live": 0, "dead": 0, "unknown": 0,
                                 "known": total, "dead_total": len(dead_map),
                                 "expired": expired}
            if expired and not args.dry_run:
                # Same race as the probing path: re-read so a TTL purge does
                # not roll back marks the bot made while this was running.
                fresh = load_dead(platform)
                purge_expired(fresh, today, args.ttl_days)
                save_dead(platform, fresh)
            continue

        result = validate_platform(
            platform, targets, workers=args.workers,
            budget_seconds=args.budget_seconds or None, log=log)
        if result["probed"] and result["unknown"] < result["probed"]:
            any_probed = True
        if args.dry_run:
            changes = apply_results(dict(dead_map), result, today)
            changes["expired"] = expired
            changes["dead_total"] = len(dead_map)
        else:
            # Re-read before writing: the bot marks these same files from its
            # own scrape cycles while this runs.
            changes = save_dead_merged(platform, result, today, args.ttl_days)
            expired = changes["expired"]
            dead_map = load_dead(platform)
        rate = 100.0 * result["live"] / max(1, result["live"] + result["dead"])
        log(f"[validate] {platform}: live rate {rate:.1f}% "
            f"(+{changes['added']} dead, {changes['revived']} revived, "
            f"{expired} expired)")
        summary[platform] = {
            "probed": result["probed"], "live": result["live"],
            "dead": result["dead"], "unknown": result["unknown"],
            "live_rate": round(rate, 1), "known": total,
            "deferred": result.get("deferred", 0),
            "dead_total": len(dead_map), "expired": expired, **changes,
        }

    if args.audit_sample > 0:
        for platform in platforms:
            summary.setdefault(platform, {})["audit"] = audit_live_rate(
                platform, args.audit_sample, workers=args.workers, log=log)

    elapsed = time.monotonic() - t0
    log(f"[validate] done in {elapsed:.1f}s")
    if args.json:
        print(json.dumps({"elapsed_seconds": round(elapsed, 1),
                          "dry_run": args.dry_run, "platforms": summary}, indent=2))
    # "Nothing left to check" is a successful run -- every slug is either fresh
    # or inside its recheck window. Only a run that had work and could not do
    # any of it is a failure.
    if any_targets and not any_probed:
        print("[validate] had slugs to probe but every probe was unreachable",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
