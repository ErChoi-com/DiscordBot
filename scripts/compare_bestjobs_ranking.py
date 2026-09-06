"""Rank one archive window under both prefilters and diff the result.

The entry-aware prefilter scores a posting on the resume the profile could
build for it (best entry, plus whether enough entries stand behind it to fill a
page); the legacy one scores the profile as a single flat document. Both are
pure lexical passes -- no provider is contacted and no description is fetched --
so this runs offline against the archive already on disk, as many times as it
takes to tune the weights.

    python scripts/compare_bestjobs_ranking.py <profile-dir> --days 7 --top 20

The legacy ordering is reproduced by blanking `entries` on the same signal, so
the two runs differ in exactly one thing: whether the profile is treated as a
pool of selectable parts or as one blob.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_match  # noqa: E402


def _rank(signal: job_match.ProfileSignal, jobs: list, top: int) -> list[job_match.JobMatch]:
    scored = [(job, job_match.score_job(job, signal)) for job in jobs]
    scored.sort(key=lambda pair: pair[1].total, reverse=True)
    return [job_match.JobMatch(job=job, score=score) for job, score in scored[:top]]


def _line(rank: int, match: job_match.JobMatch) -> str:
    job = match.job
    where = ", ".join(part for part in (job.company, job.location) if part)
    head = f"{rank:>3}. [{match.score.total * 100:5.1f}] {job.title[:60]}"
    if where:
        head += f"  ({where[:40]})"
    return head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="path to a resumes_cache profile directory")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--end", type=str, default=None, help="ISO end date (default: today)")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    end = date.fromisoformat(args.end) if args.end else None
    signal = job_match.build_profile_signal(args.profile)
    if not signal.entries:
        print(f"{args.profile} has no [tag] entries -- both rankers would be identical.")
        return 1

    jobs, dates = job_match.load_window_jobs(args.days, end_date=end)
    if not jobs:
        print(f"No archived jobs in {dates[-1]}..{dates[0]}.")
        return 1

    legacy_signal = replace(signal, entries=())
    new = _rank(signal, jobs, args.top)
    legacy = _rank(legacy_signal, jobs, args.top)

    print(
        f"profile {signal.profile_key} | {len(signal.entries)} entries | "
        f"{len(jobs):,} jobs over {dates[-1]}..{dates[0]}"
    )
    print(f"document: {len(signal.document_text)} chars (legacy flat cut: 4000)\n")

    print("=== ENTRY-AWARE (top1 + depth) ===")
    for rank, match in enumerate(new, start=1):
        print(_line(rank, match))
        fits = ", ".join(f"{entry_id} {fit:.2f}" for entry_id, fit in match.score.entry_fits)
        print(f"      use: {fits or '-'}")

    print("\n=== LEGACY (flat blend) ===")
    for rank, match in enumerate(legacy, start=1):
        print(_line(rank, match))

    new_links = {match.job.link or match.job.title for match in new}
    legacy_links = {match.job.link or match.job.title for match in legacy}
    shared = len(new_links & legacy_links)
    print(
        f"\noverlap in top {args.top}: {shared}/{args.top} "
        f"({shared / max(1, args.top) * 100:.0f}%) -- "
        f"{args.top - shared} postings the entry-aware pass surfaced instead."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
