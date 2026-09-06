#!/usr/bin/env python3
"""Report which ATS platforms are actually putting jobs in the archive.

A platform can be harvested, validated weekly, and confirmed live for thousands
of companies while contributing **nothing**, and nothing in the pipeline says
so. A scraper returning [] is indistinguishable from a platform with no
openings, so this failure is silent by construction -- which is exactly how it
was found: reading the archive by source showed bamboohr and paylocity absent
across a week, against 13,755 and 9,325 confirmed-live companies respectively.

The two causes were different, and both are worth catching:

  * bamboohr is switched off by `ats_bamboohr_enabled`, which defaults to False.
    Deliberate, but invisible -- a disabled platform and a broken one look the
    same from the archive.
  * paylocity had no scraper for most of its life; the companies were being
    harvested and probed the whole time.

So this compares confirmed-live companies against archived jobs per platform,
and flags any platform carrying real coverage that produced nothing.

Usage:
    python scripts/check_platform_yield.py [--days 7] [--min-live 100]

Exit status is non-zero when a platform with coverage yielded nothing, so it can
gate a deploy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKED_DIR = REPO_ROOT / "data" / "ats_checked"
DEAD_DIR = REPO_ROOT / "data" / "dead_slugs"

OK = "ok"
SILENT = "silent"
NO_COVERAGE = "no-coverage"

#: job_service matches semantically against "description", so a platform that
#: supplies none is invisible to matching rather than merely sparse. Measured
#: over one week: teamtailor, recruitee, rippling, jazzhr, workable and jobvite
#: sit at 100%, while workday, greenhouse, ashby, icims, lever and
#: smartrecruiters -- 74% of all archived ATS jobs -- sit between 1% and 6%.
#: Advisory rather than fatal: a thin field is a quality problem, while a
#: silent platform is a broken one, and collapsing the two would make the exit
#: status useless for gating.
MIN_DESCRIPTION_PCT = 20.0

#: One floor per field, because the three consequences below are not the same
#: severity and a single number silently rated them as if they were.
#:
#: Measured on the live fleet at the time these were set:
#:
#:   description  every platform sits between 19% and 100%, with greenhouse 25%,
#:                lever 26% and ashby 22% clustered near the bottom. That band
#:                is the ATS list endpoints, not this repo -- most return a
#:                title and a link and nothing else, and fetching each posting
#:                to fill it in is a request per job. So 20% flags the genuine
#:                outliers (icims 19%, smartrecruiters 9%) without declaring
#:                most of the fleet broken for behaving normally.
#:
#:   location     every platform except icims is at 100%. A blank location does
#:                not degrade a job, it removes it: _matches_location returns
#:                False, so the posting is dropped from every location-scoped
#:                search and a Canadian channel never sees it. At 90% a
#:                platform losing a tenth of its jobs that way gets named, and
#:                nothing normal trips it.
#:
#:   date_posted  the opposite direction, and the reason a single floor was
#:                wrong. _posting_age_ok returns True when the field is blank,
#:                so a job with no date is EXEMPT from "newer than N hours" and
#:                a years-old posting is shown as fresh. Nothing is lost, so it
#:                cannot be caught by looking for missing jobs -- and ashby, at
#:                22%, passed the old shared floor by two points while 78% of
#:                its postings skipped the freshness filter entirely.
FIELD_FLOORS: dict[str, float] = {
    "description": MIN_DESCRIPTION_PCT,
    "location": 90.0,
    "date_posted": 80.0,
}

#: What actually happens to a job missing each field. Spelled out because the
#: two directions are opposite, and reading one as the other sends someone
#: looking for missing jobs when the problem is stale ones being shown.
FIELD_CONSEQUENCE: dict[str, str] = {
    "description": "invisible to semantic matching, which reads this field.",
    "location": "DROPPED from every location-scoped search: an empty location "
                "matches nothing.",
    "date_posted": "EXEMPT from the 'newer than N hours' filter, so an old "
                   "posting is shown as though it were fresh.",
}


def _load_keys(path: Path) -> set[str]:
    """Slug keys from a store file, empty when it is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if isinstance(data, dict):
        return {str(k) for k in data}
    if isinstance(data, list):
        return {str(k) for k in data}
    return set()


def unvalidated(platform: str, candidates: Iterable[str],
                checked_dir: Path | None = None,
                dead_dir: Path | None = None) -> int:
    """Companies never resolved either way.

    The bot skips dead slugs but scrapes everything else, so an unvalidated slug
    costs a request on every scrape cycle until a run reaches it -- which is the
    whole reason the validator exists. Measured across the fleet it was 19%,
    concentrated in a few platforms that simply had not been run: recruitee at
    73%, applicantpro 65%, jazzhr 45%.

    Not an error on its own. It is a backlog, and the fix is a validation pass
    rather than a code change, so it is reported and never fatal.
    """
    checked_dir = CHECKED_DIR if checked_dir is None else checked_dir
    dead_dir = DEAD_DIR if dead_dir is None else dead_dir
    known = _load_keys(checked_dir / f"{platform}.json")
    known |= _load_keys(dead_dir / f"{platform}.json")
    return sum(1 for slug in candidates if slug not in known)


def confirmed_live(platform: str, checked_dir: Path | None = None) -> int:
    """How many companies this platform has that were confirmed live.

    The directory is resolved at call time rather than bound as a default. A
    default argument captures CHECKED_DIR at import, so the function kept
    reading the real store no matter what the caller set -- which made this
    report unfakeable in tests and, worse, silently correct-looking.
    """
    checked_dir = CHECKED_DIR if checked_dir is None else checked_dir
    try:
        data = json.loads((checked_dir / f"{platform}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return len(data) if isinstance(data, (dict, list)) else 0


#: Fields the pipeline acts on, and the way each one fails when it is blank.
#: They are reported separately because they are not the same failure:
#:
#:   description -- SEMANTIC_MATCH_TARGET reads it, so a blank one makes the job
#:                  invisible to semantic matching.
#:   location    -- _matches_location returns False for an empty location, so
#:                  the job is *dropped* from every location-scoped search. This
#:                  is the shape that hid 8,747 iCIMS companies.
#:   date_posted -- _posting_age_ok returns True when the field is blank ("rows
#:                  with no date_posted always pass"), so the job is *exempt*
#:                  from "newer than N hours" rather than filtered out. The
#:                  opposite direction: nothing is lost, but a years-old posting
#:                  is shown as though it were fresh.
TRACKED_FIELDS: tuple[str, ...] = ("description", "location", "date_posted")


def field_coverage(records: Iterable[dict[str, Any]],
                   platforms: Iterable[str],
                   fields: Iterable[str] = TRACKED_FIELDS,
                   ) -> dict[str, dict[str, int]]:
    """Per platform: total jobs, and how many carry each tracked field.

    One pass over the records, so the totals and the per-field counts can never
    disagree about which records belong to a platform. This replaced a separate
    count_by_platform/describe_coverage pair that computed the same totals a
    second way and could drift from these.

    Records are matched on `_source_site`, which the ATS scrapers set to the
    platform key. The watcher send path writes a presentation label instead
    ("LinkedIn"), so those simply do not match any platform and are ignored --
    this is a question about the ATS scrapers, not about every source.
    """
    names = {p.lower() for p in platforms}
    fields = tuple(fields)
    out = {p: dict({"jobs": 0}, **{f: 0 for f in fields}) for p in names}
    for record in records:
        if not isinstance(record, dict):
            continue
        site = str(record.get("_source_site") or "").strip().lower()
        if site not in out:
            continue
        out[site]["jobs"] += 1
        for field in fields:
            if str(record.get(field) or "").strip():
                out[site][field] += 1
    return out


def floor_for(field: str, override: float | None = None) -> float:
    """The advisory floor for one field.

    An override applies to every field, which is what a caller asking for one
    number means. A field with no floor of its own falls back to the
    description floor rather than to zero: a new tracked field should be
    reported like the others until someone measures what it deserves, not
    silently exempt.
    """
    if override is not None:
        return override
    return FIELD_FLOORS.get(field, MIN_DESCRIPTION_PCT)


def description_pct(jobs: int, described: int) -> float:
    """Share of a platform's jobs carrying a field, 0.0 when it has none."""
    return (100.0 * described / jobs) if jobs else 0.0


def assess(live: int, jobs: int, min_live: int) -> str:
    """Verdict for one platform.

    A platform with little coverage yielding nothing is not evidence of
    anything -- a handful of companies can genuinely have no openings. The
    signal is thousands of confirmed-live companies and total silence.
    """
    if live < min_live:
        return NO_COVERAGE
    return SILENT if jobs == 0 else OK


def _candidate_loader():
    """The validator owns the definition of "every slug the bot would scrape".

    Imported through a helper rather than at module scope so this script still
    runs (reporting no backlog) if the validator cannot be imported, which
    keeps the yield half of the report working on its own.
    """
    try:
        import validate_ats_slugs

        return validate_ats_slugs.load_candidates
    except Exception:
        return lambda _platform: []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7,
                        help="archive window to read (default 7)")
    parser.add_argument("--min-live", type=int, default=100,
                        help="confirmed-live companies below which silence "
                             "proves nothing (default 100)")
    parser.add_argument("--min-field-pct", type=float, default=None,
                        metavar="PCT",
                        help="override the per-field advisory floors with one "
                             "number for every field (defaults: "
                             + ", ".join(f"{f} {p:.0f}" for f, p in FIELD_FLOORS.items())
                             + "); reported, never fatal")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from services import ats_service
    from services.jba.merge_data import load_daily_log
    from services.job_match import window_dates

    load_candidates = _candidate_loader()

    records: list[dict[str, Any]] = []
    for date_key in window_dates(args.days):
        try:
            records.extend(load_daily_log(date_key))
        except Exception as exc:  # a corrupt zip must not hide the rest
            print(f"[yield] could not read {date_key}: {exc}", file=sys.stderr)

    coverage = field_coverage(records, ats_service.ATS_PLATFORMS)
    rows = []
    silent = []
    thin: dict[str, list[str]] = {f: [] for f in TRACKED_FIELDS}
    for platform in sorted(ats_service.ATS_PLATFORMS):
        live = confirmed_live(platform)
        counts = coverage.get(platform, {})
        jobs = counts.get("jobs", 0)
        verdict = assess(live, jobs, args.min_live)
        try:
            from services.jba import merge_data  # noqa: F401  (import cost only)
            candidates = load_candidates(platform)
        except Exception:
            candidates = []
        pending = unvalidated(platform, candidates)
        row: dict[str, Any] = {"platform": platform, "confirmed_live": live,
                               "archived_jobs": jobs, "unvalidated": pending,
                               "verdict": verdict}
        for field in TRACKED_FIELDS:
            pct = description_pct(jobs, counts.get(field, 0))
            row[f"{field}_pct"] = round(pct, 1)
            if verdict == OK and pct < floor_for(field, args.min_field_pct):
                thin[field].append(f"{platform} ({pct:.0f}%)")
        rows.append(row)
        if verdict == SILENT:
            silent.append(platform)

    if args.json:
        print(json.dumps({"days": args.days, "platforms": rows}, indent=2))
    else:
        print(f"{'platform':<16}{'live':>8}{'todo':>7}{'jobs':>7}{'desc':>7}"
              f"{'loc':>7}{'date':>7}  verdict")
        for row in rows:
            note = "" if row["verdict"] == OK else f"  <-- {row['verdict'].upper()}"
            print(f"{row['platform']:<16}{row['confirmed_live']:>8}"
                  f"{row['unvalidated']:>7}"
                  f"{row['archived_jobs']:>7}"
                  f"{row['description_pct']:>6.0f}%"
                  f"{row['location_pct']:>6.0f}%"
                  f"{row['date_posted_pct']:>6.0f}%{note}")

    for field in TRACKED_FIELDS:
        if thin[field]:
            floor = floor_for(field, args.min_field_pct)
            print()
            # The floor is on the line: "thin" means nothing without the number
            # it is thin against, and the three numbers are deliberately not
            # the same.
            print(f"Thin {field} (under {floor:.0f}%): " + ", ".join(thin[field]))
            print("  -> " + FIELD_CONSEQUENCE[field])
    if any(thin.values()):
        print()
        print("Not fatal -- unlike a silent platform, these are still "
              "delivering jobs.")

    backlog = sum(r["unvalidated"] for r in rows)
    if backlog:
        worst = sorted(rows, key=lambda r: -r["unvalidated"])[:4]
        print()
        print(f"Unvalidated: {backlog:,} companies never resolved either way, "
              "worst " + ", ".join(f"{r['platform']} ({r['unvalidated']:,})"
                                   for r in worst if r["unvalidated"]))
        print("  -> each costs a request on every scrape cycle until a "
              "validation pass reaches it. A backlog, not a fault: the fix is "
              "a run, not a change.")

    if silent:
        print()
        print("Silent: " + ", ".join(silent))
        print("Each has companies confirmed live and contributed no jobs. A "
              "scraper returning [] looks the same as a platform with no "
              "openings, so check the scraper exists and is enabled before "
              "assuming the boards are empty.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
