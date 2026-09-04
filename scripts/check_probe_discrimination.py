#!/usr/bin/env python3
"""Check that every validator probe can answer both "yes" and "no".

A probe decides whether a company's job board still exists, and its answer is
written to `data/dead_slugs/`, which the bot reads. Both ways of failing are
silent, and both are destructive:

  * A probe stuck on **True** reports 100% live and marks nothing dead. That is
    indistinguishable from a healthy platform. Five probes were once in this
    state at the same time; the tell was jazzhr reporting 2,683 slugs live with
    *zero* dead -- the zero, not the rate.
  * A probe stuck on **False** is worse, and has no such tell: it marks every
    company on the platform dead, and the bot then suppresses all of them for
    the full TTL. The historical check only tested the "no" direction, so this
    half was never covered at all.

  * A probe that reads the **vendor's own hostname** as a tenant confirms
    companies that do not exist. Invented slugs cannot test this -- they 404
    everywhere -- so a probe in this state passes the other two checks
    perfectly. api.breezy.hr serves Breezy's API docs, www4.icims.com redirects
    to iCIMS's marketing site, and login.recruitee.com lands on a *different
    tenant's* board, which would file that company's jobs under "login".

So three directions are checked: invented slugs must all come back False,
known-live slugs must not come back all False, and the vendor hosts in
VENDOR_HOSTS must all come back False.

This talks to every platform, so it cannot live in the test suite (those must
stay hermetic). `assess` is pure and is tested there instead.

Usage:
    python scripts/check_probe_discrimination.py [--platform NAME] [--sample N]

Exit status is non-zero when any probe fails either direction, so it can gate a
release.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import validate_ats_slugs as v  # noqa: E402

#: Slugs no company can plausibly own. Three rather than one: a single lucky
#: 404 from an overloaded edge node would otherwise read as a working probe.
BOGUS_SLUGS: tuple[str, ...] = (
    "zzznotarealcompany7788",
    "qqqbogus99xyz",
    "xkcd404nothinghere",
)

#: Real hostnames belonging to the vendor rather than to any customer, each one
#: verified to answer 200 and redirect away from the host that was asked for.
#: This is the third direction, and it is the one invented slugs cannot test:
#: `zzznotarealcompany7788` 404s everywhere, so a probe that reads the vendor's
#: own marketing site, API docs or -- in recruitee's case -- a *different
#: tenant's* board as a live company still passes the other two checks
#: perfectly. Every entry here was a live false positive before the probes
#: learned to check where the response landed.
VENDOR_HOSTS: dict[str, tuple[str, ...]] = {
    "icims": ("www4",),            # -> www.icims.com
    "breezy": ("api",),            # -> developer.breezy.hr (API docs)
    "recruitee": ("blog", "login"),  # -> recruitee.com/blog; loginsoftware.recruitee.com
    "bamboohr": ("api2", "cdn3"),  # -> www.bamboohr.com
    "jazzhr": ("www",),            # -> www.jazzhr.com/job-seekers
}

OK = "ok"
STUCK_TRUE = "stuck-true"
STUCK_FALSE = "stuck-false"
NO_DATA = "no-data"
VENDOR_LIVE = "vendor-host-read-as-live"


def assess(bogus: Iterable[Any], live: Iterable[Any],
           vendor: Iterable[Any] = ()) -> str:
    """Verdict for one probe from its answers on invented and known-live slugs.

    `Unreachable` and exceptions are neither a yes nor a no -- an outage must
    not be read as a broken probe, which is the same distinction the validator
    itself draws when it refuses to mark dead on a refusal. They are dropped,
    and a probe that produced no usable answer at all reports NO_DATA rather
    than passing by default.
    """
    bogus_answers = [b for b in bogus if isinstance(b, bool)]
    live_answers = [l for l in live if isinstance(l, bool)]
    if not bogus_answers or not live_answers:
        return NO_DATA
    if all(bogus_answers):
        return STUCK_TRUE
    if not any(live_answers):
        return STUCK_FALSE
    # Checked last: a probe that is broken in the first two ways is broken more
    # fundamentally, and reporting the vendor result for it would be noise.
    # Unreachable answers are dropped here for the same reason as elsewhere --
    # an outage is not a verdict.
    if any(v is True for v in vendor if isinstance(v, bool)):
        return VENDOR_LIVE
    return OK


def _probe(fn: Callable[[str], bool], slug: str) -> Any:
    try:
        return fn(slug)
    except v.Unreachable:
        return None
    except Exception as exc:  # a probe that raises is not answering either
        return exc


def check_platform(name: str, sample: int, rng: random.Random) -> tuple[str, list, list]:
    fn = v.PROBES[name]
    bogus = [_probe(fn, s) for s in BOGUS_SLUGS]
    known = sorted(v.load_checked(name))
    rng.shuffle(known)
    live = [_probe(fn, s) for s in known[:sample]]
    vendor = [_probe(fn, s) for s in VENDOR_HOSTS.get(name, ())]
    return assess(bogus, live, vendor), bogus, live, vendor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", action="append", dest="platforms",
                        help="check only this platform (repeatable)")
    parser.add_argument("--sample", type=int, default=6,
                        help="known-live slugs to probe per platform (default 6)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed the sample choice, for a reproducible run")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    names = args.platforms or sorted(v.PLATFORMS)
    unknown = [n for n in names if n not in v.PROBES]
    if unknown:
        print(f"no such platform: {', '.join(unknown)}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for name in names:
        verdict, bogus, live, vendor = check_platform(name, args.sample, rng)
        live_yes = sum(1 for r in live if r is True)
        note = "" if verdict == OK else f"  <-- {verdict.upper()}"
        vendor_note = ""
        if vendor:
            wrong = [h for h, r in zip(VENDOR_HOSTS[name], vendor) if r is True]
            vendor_note = f" vendor-ok={len(vendor) - len(wrong)}/{len(vendor)}"
            if wrong:
                vendor_note += f" read-as-live={wrong}"
        print(f"{name:<16} bogus={['T' if b is True else 'F' if b is False else '?' for b in bogus]} "
              f"live={live_yes}/{len(live)}{vendor_note}{note}")
        if verdict != OK:
            failures.append(f"{name}: {verdict}")

    print()
    if failures:
        print("FAILED: " + "; ".join(failures))
        print("A stuck-true probe marks nothing dead; a stuck-false one marks "
              "every company on the platform dead; one that reads a vendor host "
              "as live will confirm companies that do not exist. Do not run a "
              "validation pass until this is fixed.")
        return 1
    print(f"all {len(names)} probes discriminate: no to invented slugs, yes to "
          f"known-live ones, no to the vendor's own hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
