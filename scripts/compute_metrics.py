"""Compute operating metrics from the committed job archives.

Reads data/jba/jobs/**/*.zip (the weekly/monthly rollups the bot writes) and
emits volume + detection-latency statistics per ATS source.

Detection latency = scraped_at - date_posted, i.e. how long a posting existed
before this system saw it. Two corrections keep that number honest:

  * Backfill exclusion. The first scrape of a company returns every posting it
    already had, some of them months old. Those aren't detection latency, they
    are the cold start. Boundaries are tracked per (source, company) rather
    than per source, because companies join the slug lists continuously - a
    company first scraped in month three backfills its whole history then, and
    a source-level boundary would score all of it as slow detection.
  * Precision split. Greenhouse/Lever return full timestamps; a chunk of rows
    carry date-only values, which floors latency to midnight and biases it
    upward by up to a day. Percentiles are reported for both populations so the
    date-only bias is visible rather than baked in.

Usage:
    python scripts/compute_metrics.py                 # human-readable table
    python scripts/compute_metrics.py --json out.json # machine-readable
    python scripts/compute_metrics.py --markdown docs/metrics.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = REPO_ROOT / "data" / "jba" / "jobs"

# A posting first seen within this window of a source's earliest observation is
# treated as cold-start backfill rather than a live detection.
BACKFILL_GRACE = timedelta(days=1)

# Latencies outside this range are clock skew or bad source data, not signal.
MIN_LATENCY_H = -24.0
MAX_LATENCY_H = 24.0 * 14


def _iter_records(jobs_dir: Path, skipped: Counter[str]) -> Iterator[dict[str, Any]]:
    """Yield every job record from every archive, newest archives last.

    Monthly rollups (2026-06.zip) duplicate the weeklies they were built from,
    so records are de-duplicated by URL downstream rather than here.

    Every entry that is *not* yielded increments a counter in *skipped*, so the
    caller can reconcile the scan: nothing may disappear without a reason.
    """
    for archive in sorted(jobs_dir.rglob("*.zip")):
        try:
            zf = zipfile.ZipFile(archive)
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"  ! skipping unreadable archive {archive.name}: {exc}")
            skipped["archive_unreadable"] += 1
            continue
        with zf:
            for name in zf.namelist():
                # seen_urls.json is the dedup ledger the monthly rollup carries,
                # not job records. Parsing it as one invents a phantom source.
                if name == "seen_urls.json":
                    skipped["entry_seen_urls_ledger"] += 1
                    continue
                if not name.endswith(".json"):
                    skipped["entry_not_json"] += 1
                    continue
                try:
                    payload = json.loads(zf.read(name))
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"  ! skipping {archive.name}:{name}: {exc}")
                    skipped["entry_unparseable"] += 1
                    continue
                if not isinstance(payload, list):
                    skipped["entry_payload_not_a_list"] += 1
                    continue
                for record in payload:
                    if isinstance(record, dict):
                        yield record
                    else:
                        skipped["record_not_a_dict"] += 1


# The archives carry two record shapes. The ATS scrapers write job_url /
# _source_site / company; the channel watcher writes JobSpy results as link /
# site / site_label with no company. Reading only the first shape silently
# discarded every LinkedIn, Indeed and Glassdoor posting in the corpus.
_URL_KEYS = ("job_url", "link")
_SOURCE_KEYS = ("_source_site", "site")


def _field(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _parse_ts(value: str) -> datetime | None:
    """Parse an ISO timestamp, normalising to UTC. Returns None if unusable."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_posted(value: str) -> tuple[datetime, bool] | None:
    """Parse date_posted. Returns (timestamp, has_time_component)."""
    if not value:
        return None
    if len(value) == 10:  # date-only: floors to midnight UTC
        parsed = _parse_ts(value + "T00:00:00+00:00")
        return (parsed, False) if parsed else None
    parsed = _parse_ts(value)
    return (parsed, True) if parsed else None


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(int(fraction * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


def collect(jobs_dir: Path) -> dict[str, Any]:
    # url -> (earliest scraped_at, date_posted, source, company)
    first_seen: dict[str, tuple[datetime, str, str, str]] = {}
    rows_scanned = 0
    posted_present = 0
    per_source_rows: dict[str, int] = defaultdict(int)
    per_source_posted: dict[str, int] = defaultdict(int)

    skipped: Counter[str] = Counter()

    for record in _iter_records(jobs_dir, skipped):
        rows_scanned += 1
        posted_raw = str(record.get("date_posted") or "")
        url = _field(record, _URL_KEYS)
        if not url:
            skipped["record_no_url"] += 1
            continue
        scraped = _parse_ts(str(record.get("scraped_at") or ""))
        if scraped is None:
            skipped["record_bad_scraped_at"] += 1
            continue

        # Source names arrive with inconsistent casing from the watcher path
        # ("LinkedIn" vs "linkedin"), which would split one site into two rows.
        source = _field(record, _SOURCE_KEYS).lower() or "unknown"
        existing = first_seen.get(url)
        if existing is not None:
            # Counted whether or not it wins the earliest-scrape comparison, so
            # the reconciliation below balances.
            skipped["record_duplicate_url"] += 1
        if existing is None or scraped < existing[0]:
            company = str(record.get("company") or "")
            first_seen[url] = (scraped, posted_raw, source, company)

    # Per-source counts come from the de-duplicated set, not the raw scan. A
    # month that has both its weekly zips and its monthly rollup on disk yields
    # every posting twice, which inflated "rows" by ~19% on the real corpus.
    for _scraped, posted_raw, source, _company in first_seen.values():
        per_source_rows[source] += 1
        if posted_raw:
            posted_present += 1
            per_source_posted[source] += 1

    # Cold-start boundary per (source, company): the first moment that company
    # was observed on that board at all.
    company_start: dict[tuple[str, str], datetime] = {}
    for scraped, _posted, source, company in first_seen.values():
        key = (source, company)
        if key not in company_start or scraped < company_start[key]:
            company_start[key] = scraped

    latencies: dict[str, list[float]] = defaultdict(list)
    latencies_precise: dict[str, list[float]] = defaultdict(list)
    excluded_backfill = 0

    for scraped, posted_raw, source, company in first_seen.values():
        parsed = _parse_posted(posted_raw)
        if parsed is None:
            continue
        posted, has_time = parsed

        cutoff = company_start.get((source, company))
        if cutoff is not None and posted < cutoff + BACKFILL_GRACE:
            excluded_backfill += 1
            continue

        hours = (scraped - posted).total_seconds() / 3600.0
        if not (MIN_LATENCY_H < hours < MAX_LATENCY_H):
            continue
        latencies[source].append(hours)
        if has_time:
            latencies_precise[source].append(hours)

    sources: dict[str, Any] = {}
    for source in sorted(per_source_rows, key=lambda s: -per_source_rows[s]):
        values = sorted(latencies.get(source, []))
        precise = sorted(latencies_precise.get(source, []))
        rows = per_source_rows[source]
        sources[source] = {
            "postings": rows,
            "date_posted_fill_pct": round(100.0 * per_source_posted[source] / rows, 1),
            "latency_n": len(values),
            "latency_p50_h": round(statistics.median(values), 1) if values else None,
            "latency_p90_h": round(_percentile(values, 0.90), 1) if values else None,
            "latency_precise_n": len(precise),
            "latency_precise_p50_h": (
                round(statistics.median(precise), 1) if precise else None
            ),
        }

    all_values = sorted(v for vals in latencies.values() for v in vals)

    # Every scanned record must end up either in first_seen or in a skip
    # bucket. If this ever fails, the parser grew a silent drop path.
    accounted = (
        len(first_seen)
        + skipped["record_no_url"]
        + skipped["record_bad_scraped_at"]
        + skipped["record_duplicate_url"]
    )
    reconciles = accounted == rows_scanned
    if not reconciles:
        print(
            f"  ! accounting mismatch: {rows_scanned:,} scanned vs "
            f"{accounted:,} accounted ({rows_scanned - accounted:+,} unexplained)"
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_scanned": rows_scanned,
        "unique_urls": len(first_seen),
        "date_posted_fill_pct": (
            round(100.0 * posted_present / len(first_seen), 1) if first_seen else 0.0
        ),
        "backfill_excluded": excluded_backfill,
        "skipped": dict(sorted(skipped.items())),
        "reconciles": reconciles,
        "latency_n": len(all_values),
        "latency_p50_h": round(statistics.median(all_values), 1) if all_values else None,
        "latency_p90_h": round(_percentile(all_values, 0.90), 1) if all_values else None,
        "sources": sources,
    }


def render_table(metrics: dict[str, Any]) -> str:
    lines = [
        f"rows scanned      {metrics['rows_scanned']:>10,}  (incl. rollup duplicates)",
        f"unique postings   {metrics['unique_urls']:>10,}",
        f"date_posted fill  {metrics['date_posted_fill_pct']:>9.1f}%",
        f"backfill excluded {metrics['backfill_excluded']:>10,}",
        "",
        "record accounting" + ("" if metrics["reconciles"] else "   ** MISMATCH **"),
    ]
    for reason, count in metrics["skipped"].items():
        lines.append(f"  {reason:<28}{count:>9,}")
    lines += [
        "",
        f"{'source':<12}{'postings':>10}{'fill%':>8}{'n':>9}{'p50 h':>8}{'p90 h':>8}",
    ]
    for name, stats in metrics["sources"].items():
        p50 = stats["latency_p50_h"]
        p90 = stats["latency_p90_h"]
        lines.append(
            f"{name:<12}{stats['postings']:>10,}{stats['date_posted_fill_pct']:>7.1f}%"
            f"{stats['latency_n']:>9,}"
            f"{'n/a' if p50 is None else f'{p50:.1f}':>8}"
            f"{'n/a' if p90 is None else f'{p90:.1f}':>8}"
        )
    p50 = metrics["latency_p50_h"]
    p90 = metrics["latency_p90_h"]
    lines.append(
        f"{'ALL':<12}{metrics['unique_urls']:>10,}{'':>8}{metrics['latency_n']:>9,}"
        f"{'n/a' if p50 is None else f'{p50:.1f}':>8}"
        f"{'n/a' if p90 is None else f'{p90:.1f}':>8}"
    )
    return "\n".join(lines)


def render_markdown(metrics: dict[str, Any]) -> str:
    rows = [
        "# Operating metrics",
        "",
        f"Generated {metrics['generated_at']} from the committed job archives.",
        "",
        f"- Rows scanned: **{metrics['rows_scanned']:,}**",
        f"- Unique postings: **{metrics['unique_urls']:,}**",
        f"- `date_posted` coverage: **{metrics['date_posted_fill_pct']}%**",
        f"- Cold-start rows excluded from latency: **{metrics['backfill_excluded']:,}**",
        "",
        "## Detection latency by source",
        "",
        "Hours between a posting appearing on its source board and this system recording",
        "it. Cold-start backfill excluded. `precise p50` covers only sources returning a",
        "time component — date-only sources are floored to midnight and read high.",
        "",
        "| source | postings | `date_posted` fill | n | p50 (h) | p90 (h) | precise p50 (h) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def cell(value: float | None) -> str:
        return "—" if value is None else f"{value:.1f}"

    for name, stats in metrics["sources"].items():
        rows.append(
            f"| {name} | {stats['postings']:,} | {stats['date_posted_fill_pct']}% "
            f"| {stats['latency_n']:,} | {cell(stats['latency_p50_h'])} "
            f"| {cell(stats['latency_p90_h'])} | {cell(stats['latency_precise_p50_h'])} |"
        )
    rows.append(
        f"| **all** | {metrics['unique_urls']:,} | {metrics['date_posted_fill_pct']}% "
        f"| {metrics['latency_n']:,} | {cell(metrics['latency_p50_h'])} "
        f"| {cell(metrics['latency_p90_h'])} | |"
    )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--json", type=Path, help="write metrics as JSON to this path")
    parser.add_argument("--markdown", type=Path, help="write a Markdown report to this path")
    args = parser.parse_args()

    if not args.jobs_dir.exists():
        print(f"No archives at {args.jobs_dir} — nothing to compute.")
        return 0

    metrics = collect(args.jobs_dir)
    if not metrics["rows_scanned"]:
        print(f"No job records found under {args.jobs_dir}.")
        return 0

    print(render_table(metrics))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(metrics), encoding="utf-8")
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
