"""Metrics computed from the committed job archives.

Covers bugs where the computation silently produced a *plausible* wrong answer
rather than failing, so the assertions check real computed values against
hand-constructed fixtures rather than just exercising the call.

Dead-slug TTL flushing is covered separately in tests/test_dead_slug_ttl.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_compute_metrics():
    """Load scripts/compute_metrics.py by path.

    It is a standalone script rather than an importable package module, so
    there is no import path to it.
    """
    path = REPO_ROOT / "scripts" / "compute_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_metrics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_metrics"] = module
    spec.loader.exec_module(module)
    return module
def _write_archive(root: Path, name: str, days: dict[str, list[dict]], extra: dict | None = None):
    month_dir = root / name[:7]
    month_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(month_dir / f"{name}.zip", "w") as zf:
        for day, records in days.items():
            zf.writestr(f"{day}.json", json.dumps(records))
        for entry_name, payload in (extra or {}).items():
            zf.writestr(entry_name, json.dumps(payload))


def _job(url: str, company: str, posted: str, scraped: str, source: str = "greenhouse"):
    return {
        "job_url": url,
        "company": company,
        "date_posted": posted,
        "scraped_at": scraped,
        "_source_site": source,
    }


def test_seen_urls_ledger_is_not_counted_as_jobs(tmp_path):
    """seen_urls.json rides along in monthly rollups but holds dedup entries,
    not postings. Counting it invented a phantom 'unknown' source."""
    metrics_mod = _load_compute_metrics()

    _write_archive(
        tmp_path,
        "2026-06-w1",
        {"2026-06-02": [_job("u1", "acme", "2026-06-02T00:00:00Z", "2026-06-02T04:00:00Z")]},
        extra={
            "seen_urls.json": [
                {"url": f"https://x/{i}", "first_seen": "2026-06-01"} for i in range(500)
            ]
        },
    )

    result = metrics_mod.collect(tmp_path)

    assert result["rows_scanned"] == 1, "ledger entries leaked into the row count"
    assert "unknown" not in result["sources"]


def test_company_cold_start_excluded_but_later_postings_measured(tmp_path):
    """A company's pre-existing backlog is cold start, not detection latency.

    The boundary must be per company: 'newco' joins two weeks after 'oldco',
    so a source-level boundary would score newco's backlog as a slow detection.
    """
    metrics_mod = _load_compute_metrics()

    _write_archive(
        tmp_path,
        "2026-06-w1",
        {
            "2026-06-01": [
                # oldco's cold start: posted long before we ever saw the board.
                _job("o1", "oldco", "2026-01-15T00:00:00Z", "2026-06-01T00:00:00Z"),
            ],
        },
    )
    _write_archive(
        tmp_path,
        "2026-06-w3",
        {
            "2026-06-15": [
                # newco's cold start. Critically, this posting is dated AFTER
                # the source-level first observation (2026-06-01) but BEFORE
                # newco itself was first scraped - so a source-level boundary
                # would let it through and score it as ~10 days of latency.
                _job("n1", "newco", "2026-06-05T00:00:00Z", "2026-06-15T00:00:00Z"),
                # A genuine detection for oldco: 6h old.
                _job("o2", "oldco", "2026-06-14T18:00:00Z", "2026-06-15T00:00:00Z"),
            ],
        },
    )

    result = metrics_mod.collect(tmp_path)

    assert result["rows_scanned"] == 3
    assert result["backfill_excluded"] == 2, "both cold starts should be dropped"

    greenhouse = result["sources"]["greenhouse"]
    assert greenhouse["latency_n"] == 1, "only the real detection should be measured"
    assert greenhouse["latency_p50_h"] == pytest.approx(6.0)


def test_date_only_postings_are_split_out(tmp_path):
    """Date-only values floor to midnight and read high, so they must not
    silently inflate the precise percentile."""
    metrics_mod = _load_compute_metrics()

    base = datetime(2026, 6, 10, tzinfo=timezone.utc)
    _write_archive(
        tmp_path,
        "2026-06-w1",
        {
            "2026-06-01": [_job("seed", "acme", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z")],
            "2026-06-10": [
                # full timestamp: exactly 3h
                _job("a", "acme", "2026-06-10T00:00:00Z", (base + timedelta(hours=3)).isoformat()),
                # date-only: floors to midnight, so it reads as 20h
                _job("b", "acme", "2026-06-10", (base + timedelta(hours=20)).isoformat()),
            ],
        },
    )

    greenhouse = metrics_mod.collect(tmp_path)["sources"]["greenhouse"]

    assert greenhouse["latency_n"] == 2
    assert greenhouse["latency_precise_n"] == 1
    assert greenhouse["latency_precise_p50_h"] == pytest.approx(3.0)


def _watcher_job(link: str, posted: str, scraped: str, site: str = "linkedin"):
    """The shape the channel watcher writes for JobSpy results: link/site, no company."""
    return {
        "title": "SWE Intern",
        "link": link,
        "site": site,
        "site_label": site.title(),
        "date_posted": posted,
        "scraped_at": scraped,
        "source_url": "channel:123",
        "type": "job",
    }


def test_watcher_schema_is_read_not_dropped(tmp_path):
    """The archives carry two record shapes. Reading only the ATS one silently
    discarded every LinkedIn/Indeed/Glassdoor posting in the corpus."""
    metrics_mod = _load_compute_metrics()

    _write_archive(
        tmp_path,
        "2026-06-w1",
        {
            "2026-06-01": [
                _watcher_job("https://li/1", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"),
            ],
            "2026-06-02": [
                _watcher_job("https://li/2", "2026-06-02T00:00:00Z", "2026-06-02T05:00:00Z"),
            ],
        },
    )

    result = metrics_mod.collect(tmp_path)

    assert "linkedin" in result["sources"], "watcher records were dropped"
    assert result["sources"]["linkedin"]["postings"] == 2
    assert result["sources"]["linkedin"]["latency_p50_h"] == pytest.approx(5.0)


def test_source_names_are_case_normalised(tmp_path):
    """The watcher emits both 'LinkedIn' and 'linkedin'; un-normalised they
    split one site into two rows with half the sample each."""
    metrics_mod = _load_compute_metrics()

    _write_archive(
        tmp_path,
        "2026-06-w1",
        {
            "2026-06-01": [
                _watcher_job("https://li/seed", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"),
            ],
            "2026-06-02": [
                _watcher_job("https://li/a", "2026-06-02T00:00:00Z", "2026-06-02T02:00:00Z",
                             site="LinkedIn"),
                _watcher_job("https://li/b", "2026-06-02T00:00:00Z", "2026-06-02T02:00:00Z",
                             site="linkedin"),
            ],
        },
    )

    sources = metrics_mod.collect(tmp_path)["sources"]

    assert "LinkedIn" not in sources
    assert sources["linkedin"]["postings"] == 3


def test_every_scanned_record_is_accounted_for(tmp_path):
    """Nothing may vanish without a named reason - the reconciliation is what
    catches a future silent drop path."""
    metrics_mod = _load_compute_metrics()

    _write_archive(
        tmp_path,
        "2026-06-w1",
        {
            "2026-06-01": [
                _job("dup", "acme", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"),
                _job("dup", "acme", "2026-06-01T00:00:00Z", "2026-06-01T06:00:00Z"),
                {"title": "no url at all", "scraped_at": "2026-06-01T00:00:00Z"},
                {"title": "bad ts", "job_url": "x", "scraped_at": "not-a-timestamp"},
                "not even a dict",
            ],
        },
    )

    result = metrics_mod.collect(tmp_path)

    assert result["reconciles"] is True
    assert result["skipped"]["record_duplicate_url"] == 1
    assert result["skipped"]["record_no_url"] == 1
    assert result["skipped"]["record_bad_scraped_at"] == 1
    assert result["skipped"]["record_not_a_dict"] == 1
    # 4 dict records scanned; 1 unique survived, 3 explained away.
    assert result["rows_scanned"] == 4
    assert result["unique_urls"] == 1


def test_unreadable_archive_does_not_abort_the_run(tmp_path):
    metrics_mod = _load_compute_metrics()

    _write_archive(
        tmp_path,
        "2026-06-w1",
        {"2026-06-02": [_job("u1", "acme", "2026-06-02T00:00:00Z", "2026-06-02T04:00:00Z")]},
    )
    (tmp_path / "2026-06" / "2026-06-w2.zip").write_bytes(b"not a zip file")

    result = metrics_mod.collect(tmp_path)

    assert result["rows_scanned"] == 1
