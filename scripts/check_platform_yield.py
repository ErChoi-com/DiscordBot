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

OK = "ok"
SILENT = "silent"
NO_COVERAGE = "no-coverage"


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


def count_by_platform(records: Iterable[dict[str, Any]],
                      platforms: Iterable[str]) -> dict[str, int]:
    """Archived jobs per ATS platform.

    Records are matched on `_source_site`, which the ATS scrapers set to the
    platform key. The watcher send path writes a presentation label instead
    ("LinkedIn"), so those simply do not match any platform and are ignored --
    this is a question about the ATS scrapers, not about every source.
    """
    names = {p.lower() for p in platforms}
    counts = {p: 0 for p in names}
    for record in records:
        if not isinstance(record, dict):
            continue
        site = str(record.get("_source_site") or "").strip().lower()
        if site in counts:
            counts[site] += 1
    return counts


def assess(live: int, jobs: int, min_live: int) -> str:
    """Verdict for one platform.

    A platform with little coverage yielding nothing is not evidence of
    anything -- a handful of companies can genuinely have no openings. The
    signal is thousands of confirmed-live companies and total silence.
    """
    if live < min_live:
        return NO_COVERAGE
    return SILENT if jobs == 0 else OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7,
                        help="archive window to read (default 7)")
    parser.add_argument("--min-live", type=int, default=100,
                        help="confirmed-live companies below which silence "
                             "proves nothing (default 100)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from services import ats_service
    from services.jba.merge_data import load_daily_log
    from services.job_match import window_dates

    records: list[dict[str, Any]] = []
    for date_key in window_dates(args.days):
        try:
            records.extend(load_daily_log(date_key))
        except Exception as exc:  # a corrupt zip must not hide the rest
            print(f"[yield] could not read {date_key}: {exc}", file=sys.stderr)

    counts = count_by_platform(records, ats_service.ATS_PLATFORMS)
    rows = []
    silent = []
    for platform in sorted(ats_service.ATS_PLATFORMS):
        live = confirmed_live(platform)
        jobs = counts.get(platform, 0)
        verdict = assess(live, jobs, args.min_live)
        rows.append({"platform": platform, "confirmed_live": live,
                     "archived_jobs": jobs, "verdict": verdict})
        if verdict == SILENT:
            silent.append(platform)

    if args.json:
        print(json.dumps({"days": args.days, "platforms": rows}, indent=2))
    else:
        print(f"{'platform':<16}{'live':>8}{'jobs':>8}  verdict")
        for row in rows:
            note = "" if row["verdict"] == OK else f"  <-- {row['verdict'].upper()}"
            print(f"{row['platform']:<16}{row['confirmed_live']:>8}"
                  f"{row['archived_jobs']:>8}{note}")

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
