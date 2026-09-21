"""iCIMS and Workday detail fetches must go through the vendor gate too.

_scrape_icims and _scrape_workday each open their own nested ThreadPoolExecutor
per slug to fetch per-job detail pages (metadata for iCIMS, date/description
for Workday), and used to submit straight to that pool -- bypassing
_enrich_gate, the per-platform Semaphore that caps in-flight requests against
one vendor at PLATFORM_WORKERS. A py-spy dump of the live bot found 215
threads inside _fetch_icims_metadata and 70 inside _fetch_workday_detail at
once: outer scheduler pool (many slugs at a time) x inner per-slug pool,
unbounded, all against one host.

These tests reproduce that shape -- several slugs' worth of _scrape_icims /
_scrape_workday calls running at once -- and assert the vendor never sees more
than PLATFORM_WORKERS in flight, the same claim tests/test_enrich_concurrency.py
makes for _enrich_rows.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as a  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_dead_slugs(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "_DEAD_SLUG_DIR", tmp_path / "dead")
    a.clear_dead_slugs()
    yield
    a.clear_dead_slugs()


@pytest.fixture(autouse=True)
def _fresh_gates(monkeypatch):
    """Gates live for the life of the process; tests must not inherit them,
    and must not inherit a size cached under the old PLATFORM_WORKERS."""
    monkeypatch.setattr(a, "_ENRICH_GATES", {})
    monkeypatch.setattr(a, "_ENRICH_GATES_LOCK", threading.Lock())


class _Tracker:
    """Counts how many detail fetches are in flight at the same moment. Same
    pattern as tests/test_enrich_concurrency.py's _Tracker."""

    def __init__(self, hold: float = 0.02) -> None:
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.calls = 0
        self.hold = hold

    def fetch(self, *args, **kwargs) -> dict[str, str]:
        with self.lock:
            self.live += 1
            self.calls += 1
            self.peak = max(self.peak, self.live)
        try:
            time.sleep(self.hold)
        finally:
            with self.lock:
                self.live -= 1
        return {"title": "t", "location": "Toronto, ON", "date_posted": "2026-01-01"}


class _Resp:
    def __init__(self, status=200, body=""):
        self.status_code = status
        self._body = body
        self.content = body.encode("utf-8") if isinstance(body, str) else body
        self.text = body if isinstance(body, str) else body.decode("utf-8")

    def json(self):
        return json.loads(self._body)


def _sitemap_xml(urls: list[str]) -> str:
    entries = "".join(
        f"<url><loc>{u}</loc><lastmod>2026-08-01</lastmod></url>" for u in urls
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{entries}</urlset>"
    )


# ── iCIMS ────────────────────────────────────────────────────────────────────

def test_icims_detail_fetches_stay_within_the_vendor_gate(monkeypatch):
    monkeypatch.setitem(a.PLATFORM_WORKERS, "icims", 3)
    tracker = _Tracker()

    job_urls = [
        f"https://careers-acme.icims.com/jobs/{i}/engineer-{i}/job"
        for i in range(1, 5)
    ]
    sitemap = _sitemap_xml(job_urls)

    def fake_get(url, headers=None, timeout=None, **kw):
        if url.endswith("/sitemap.xml"):
            return _Resp(200, sitemap)
        return _Resp(404, "")

    monkeypatch.setattr(a, "_http_get", fake_get)
    monkeypatch.setattr(a, "_fetch_icims_metadata", tracker.fetch)
    monkeypatch.setattr(a.time, "sleep", lambda *_: None)

    def run() -> list[dict]:
        return a._scrape_icims("acme", "", "", 10)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: run(), range(8)))

    for rows in results:
        assert len(rows) == 4, "not every candidate made it into the result"

    assert tracker.calls == 8 * 4, "every job should still have been fetched"
    ceiling = a.capacity.workers(3, minimum=2)
    assert tracker.peak <= ceiling, (
        f"{tracker.peak} concurrent iCIMS detail fetches against a ceiling of "
        f"{ceiling} -- the nested pool bypassed _enrich_gate"
    )


# ── Workday ──────────────────────────────────────────────────────────────────

def test_workday_detail_fetches_stay_within_the_vendor_gate(monkeypatch):
    monkeypatch.setitem(a.PLATFORM_WORKERS, "workday", 3)
    tracker = _Tracker()

    jobs = [
        {
            "title": f"Engineer {i}",
            "locationsText": "Toronto, ON",
            "externalPath": f"/job/{i}",
        }
        for i in range(4)
    ]
    body = json.dumps({"jobPostings": jobs, "total": len(jobs)})

    def fake_post(url, json=None, headers=None, timeout=None, **kw):
        return _Resp(200, body)

    def fake_get(url, headers=None, timeout=None, **kw):
        # _fetch_workday_detail is monkeypatched directly below, so nothing in
        # this test should reach _http_get -- fail loudly if it does.
        pytest.fail(f"unexpected _http_get call: {url}")

    monkeypatch.setattr(a, "_http_post", fake_post)
    monkeypatch.setattr(a, "_http_get", fake_get)
    monkeypatch.setattr(a, "_fetch_workday_detail", tracker.fetch)
    monkeypatch.setattr(a.time, "sleep", lambda *_: None)

    def run() -> list[dict]:
        return a._scrape_workday("acme|wd5|External", "", "", 10)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: run(), range(8)))

    for rows in results:
        assert len(rows) == 4, "not every candidate made it into the result"

    assert tracker.calls == 8 * 4, "every job should still have been fetched"
    ceiling = a.capacity.workers(3, minimum=2)
    assert tracker.peak <= ceiling, (
        f"{tracker.peak} concurrent Workday detail fetches against a ceiling of "
        f"{ceiling} -- the nested pool bypassed _enrich_gate"
    )
