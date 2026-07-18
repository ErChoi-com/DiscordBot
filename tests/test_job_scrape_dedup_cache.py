"""Cross-channel job-scrape dedup: two watchers with identical criteria must
share ONE external scrape per TTL window (per-channel watcher tasks have no
other coordination), while per-channel source labeling still applies and
different criteria never share."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import services.job_service as job_service
from services.job_service import scrape_job_postings


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    job_service.clear_job_scrape_cache()
    # Keep site normalization deterministic and subprocess-free.
    monkeypatch.setattr(
        job_service, "normalize_requested_sites", lambda sites, exe=None: list(sites)
    )
    yield
    job_service.clear_job_scrape_cache()


def _rows(*titles: str) -> list[dict]:
    return [
        {"title": t, "company": "ACME", "job_url": f"https://x.test/{i}", "_source_site": "indeed"}
        for i, t in enumerate(titles)
    ]


def test_identical_criteria_scrape_once_within_ttl(monkeypatch):
    calls = []

    def fake_scrape(*args, **kwargs):
        calls.append(1)
        return _rows("Dev", "QA")

    monkeypatch.setattr(job_service, "_scrape_filtered_rows_uncached", fake_scrape)

    first = scrape_job_postings(["indeed"], "python", "Toronto", source_url="channel:1")
    second = scrape_job_postings(["indeed"], "python", "Toronto", source_url="channel:2")
    assert len(calls) == 1
    assert len(first) == 2 and len(second) == 2
    # Per-channel labeling survives the shared scrape (shape_job_item stamps
    # the caller's source_url onto every item).
    assert all(item["source_url"] == "channel:1" for item in first)
    assert all(item["source_url"] == "channel:2" for item in second)


def test_different_criteria_never_share(monkeypatch):
    calls = []
    monkeypatch.setattr(
        job_service,
        "_scrape_filtered_rows_uncached",
        lambda *a, **k: calls.append(1) or _rows("Dev"),
    )
    scrape_job_postings(["indeed"], "python", "Toronto")
    scrape_job_postings(["indeed"], "python", "Vancouver")
    scrape_job_postings(["indeed"], "java", "Toronto")
    scrape_job_postings(["linkedin"], "python", "Toronto")
    assert len(calls) == 4


def test_expired_entry_rescrapes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        job_service,
        "_scrape_filtered_rows_uncached",
        lambda *a, **k: calls.append(1) or _rows("Dev"),
    )
    fake_now = [1000.0]
    monkeypatch.setattr(job_service.time, "monotonic", lambda: fake_now[0])

    scrape_job_postings(["indeed"], "python", "Toronto")
    fake_now[0] += job_service.JOB_SCRAPE_CACHE_TTL_SECONDS + 1
    scrape_job_postings(["indeed"], "python", "Toronto")
    assert len(calls) == 2


def test_simultaneous_callers_share_one_in_flight_scrape(monkeypatch):
    """Two threads hitting the same key at the same moment: the second must
    WAIT for the first scrape and reuse it, not start a duplicate."""
    calls = []
    release = threading.Event()

    def slow_scrape(*args, **kwargs):
        calls.append(1)
        assert release.wait(timeout=5)
        return _rows("Dev")

    monkeypatch.setattr(job_service, "_scrape_filtered_rows_uncached", slow_scrape)

    results = []
    threads = [
        threading.Thread(
            target=lambda i=i: results.append(
                scrape_job_postings(["indeed"], "python", "Toronto", source_url=f"channel:{i}")
            )
        )
        for i in range(2)
    ]
    for t in threads:
        t.start()
    # Let both threads reach the key lock, then release the scrape.
    release.set()
    for t in threads:
        t.join(timeout=10)
    assert len(calls) == 1
    assert len(results) == 2 and all(len(r) == 1 for r in results)


def test_cache_stores_copies_not_producer_references(monkeypatch):
    """The cache snapshot must be decoupled from the scraper's own row list —
    a scraper reusing/mutating its rows after return must not poison what a
    later channel receives."""
    rows = _rows("Dev")
    monkeypatch.setattr(
        job_service, "_scrape_filtered_rows_uncached", lambda *a, **k: rows
    )
    scrape_job_postings(["indeed"], "python", "Toronto")
    rows[0]["title"] = "MUTATED"
    second = scrape_job_postings(["indeed"], "python", "Toronto")
    assert second[0]["title"].startswith("Dev")
