"""Enrichment fetches a posting's page only when the archive does not already
hold that posting.

Enrichment is a page fetch per posting, and _drop_already_archived then
discards every row the archive has seen -- on a board asked every cycle,
nearly all of them. Paying for the fetch and dropping the row afterwards is
what made an icims board cost a sitemap plus hundreds of pages: icims
completed 103 boards in the 480s that let greenhouse complete 876.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as a  # noqa: E402
from services.jba import archive_index  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_gates(monkeypatch, tmp_path):
    monkeypatch.setattr(a, "_ENRICH_GATES", {})
    monkeypatch.setattr(a, "_ENRICH_GATES_LOCK", threading.Lock())
    monkeypatch.setattr(a, "_DEAD_SLUG_DIR", tmp_path / "dead")
    a.clear_dead_slugs()


def _rows(n: int, site: str = "bamboohr") -> list[dict]:
    return [
        {"title": f"Job {i}", "company": "acme", "location": "Toronto, ON",
         "job_url": f"https://acme.{site}.com/jobs/{i}", "_source_site": site}
        for i in range(n)
    ]


def _archive_holds(monkeypatch, held: set[str]) -> list:
    """Stand in for the archive: rows whose job_url is in `held` are known."""
    seen: list = []

    def _filter(rows, cutoff=None):
        seen.append([r["job_url"] for r in rows])
        kept = [r for r in rows if r["job_url"] not in held]
        return kept, len(rows) - len(kept)

    monkeypatch.setattr(archive_index, "ensure_index", lambda force=False: 0)
    monkeypatch.setattr(archive_index, "filter_new_listings", _filter)
    return seen


class _Fetch:
    def __init__(self):
        self.urls: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, url, *args):
        with self.lock:
            self.urls.append(url)
        return {"date_posted": "2026-09-01", "description": "fetched"}


# ── _enrich_rows ─────────────────────────────────────────────────────────────

def test_known_postings_are_not_fetched_and_keep_their_listing_values(monkeypatch):
    rows = _rows(4)
    _archive_holds(monkeypatch, {rows[1]["job_url"], rows[3]["job_url"]})
    fetch = _Fetch()

    a._enrich_rows(rows, fetch)

    assert sorted(fetch.urls) == sorted([rows[0]["job_url"], rows[2]["job_url"]])
    assert rows[0]["description"] == "fetched" and rows[2]["description"] == "fetched"
    assert "description" not in rows[1] and "description" not in rows[3]
    assert len(rows) == 4, "the caller's list is left whole; the archive dedup downstream drops them"


def test_the_archive_is_asked_once_per_batch_not_per_row(monkeypatch):
    rows = _rows(6)
    seen = _archive_holds(monkeypatch, set())
    a._enrich_rows(rows, _Fetch())
    assert len(seen) == 1 and len(seen[0]) == 6


def test_an_unavailable_archive_means_everything_is_enriched(monkeypatch):
    rows = _rows(3)

    def _boom(*a, **k):
        raise RuntimeError("index locked")

    monkeypatch.setattr(archive_index, "ensure_index", _boom)
    fetch = _Fetch()
    a._enrich_rows(rows, fetch)
    assert len(fetch.urls) == 3


def test_dedup_disabled_means_everything_is_enriched(monkeypatch):
    rows = _rows(3)
    _archive_holds(monkeypatch, {r["job_url"] for r in rows})
    monkeypatch.setattr(a, "ATS_ARCHIVE_DEDUP_ENABLED", False)
    fetch = _Fetch()
    a._enrich_rows(rows, fetch)
    assert len(fetch.urls) == 3


def test_a_batch_the_archive_holds_entirely_makes_no_fetch_and_no_pool(monkeypatch):
    rows = _rows(3)
    _archive_holds(monkeypatch, {r["job_url"] for r in rows})
    fetch = _Fetch()
    a._enrich_rows(rows, fetch)
    assert fetch.urls == []


# ── the two scrapers with their own detail pools ─────────────────────────────

class _Resp:
    def __init__(self, status=200, body=""):
        self.status_code = status
        self.text = body
        self.content = body.encode("utf-8")

    def json(self):
        import json
        return json.loads(self.text)


def test_icims_skips_the_page_fetch_for_postings_the_archive_holds(monkeypatch):
    urls = [f"https://careers-acme.icims.com/jobs/{i}/engineer-{i}/job" for i in range(1, 5)]
    sitemap = ('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
               + "".join(f"<url><loc>{u}</loc><lastmod>2026-08-01</lastmod></url>" for u in urls)
               + "</urlset>")
    monkeypatch.setattr(a, "_http_get", lambda url, **kw: _Resp(200, sitemap) if url.endswith("/sitemap.xml") else _Resp(404))
    monkeypatch.setattr(a.time, "sleep", lambda *_: None)
    _archive_holds(monkeypatch, {urls[0], urls[2]})
    fetched: list[str] = []

    def _meta(url):
        fetched.append(url)
        return {"title": "t", "location": "Toronto, ON", "date_posted": "2026-08-01"}

    monkeypatch.setattr(a, "_fetch_icims_metadata", _meta)

    rows = a._scrape_icims("acme", "", "", 10)

    assert sorted(fetched) == sorted([urls[1], urls[3]])
    assert sorted(r["job_url"] for r in rows) == sorted([urls[1], urls[3]]), (
        "postings the archive holds leave the result here; they were dropped after the fetch before"
    )


def test_workday_skips_the_detail_fetch_for_postings_the_archive_holds(monkeypatch):
    import json as _json

    jobs = [{"title": f"Engineer {i}", "locationsText": "Toronto, ON", "externalPath": f"/job/{i}"} for i in range(4)]
    body = _json.dumps({"jobPostings": jobs, "total": len(jobs)})
    monkeypatch.setattr(a, "_http_post", lambda url, **kw: _Resp(200, body))
    monkeypatch.setattr(a, "_http_get", lambda url, **kw: _Resp(404))
    monkeypatch.setattr(a.time, "sleep", lambda *_: None)
    fetched: list[str] = []

    def _detail(detail_url, headers):
        fetched.append(detail_url)
        return {"date_posted": "2026-08-01", "description": "d"}

    monkeypatch.setattr(a, "_fetch_workday_detail", _detail)

    # Find out what job_urls the scraper builds, then hold two of them.
    _archive_holds(monkeypatch, set())
    all_rows = a._scrape_workday("acme|wd5|External", "", "", 10)
    assert len(all_rows) == 4 and len(fetched) == 4
    held = {all_rows[0]["job_url"], all_rows[1]["job_url"]}

    fetched.clear()
    _archive_holds(monkeypatch, held)
    rows = a._scrape_workday("acme|wd5|External", "", "", 10)
    assert len(fetched) == 2
    assert {r["job_url"] for r in rows}.isdisjoint(held)
