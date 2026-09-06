"""ZipRecruiter scraper: parsing, date recovery and fingerprint rotation.

The scraper used to drive Playwright against /candidate/search. Two things
were wrong and both are pinned down here: that route no longer carries a
search (it 302s to /jobs-search and drops the query), and Cloudflare refuses
Playwright's TLS fingerprint outright, so the browser path could never return
a row. The parsing tests run against a real captured search page rather than
hand-written HTML, so a ZipRecruiter markup change fails the suite instead of
silently degrading to zero results in production.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_service
from services.job_service import (
    ZIPRECRUITER_IMPERSONATIONS,
    parse_ziprecruiter_search_html,
    scrape_ziprecruiter_postings,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ziprecruiter_search.html"


@pytest.fixture(scope="module")
def search_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_parses_real_capture_into_complete_rows(search_html):
    """Every field the pipeline consumes is really extracted from the live
    markup -- not just a non-empty list."""
    rows = parse_ziprecruiter_search_html(search_html, "Toronto, ON")

    assert len(rows) == 3
    first = rows[0]
    assert first["title"] == "Software Design Engineering (Artificial Intelligence) Co-op Student"
    assert first["company"] == "Martinrea International Inc."
    assert first["location"] == "Woodbridge, ON"
    assert first["job_url"].endswith("jid=4a32bc9610b28930")
    assert first["_source_site"] == "zip_recruiter"
    assert first["_source_sites"] == ["zip_recruiter"]

    # Company must track the card it belongs to, not leak from a neighbour.
    assert [r["company"] for r in rows] == [
        "Martinrea International Inc.", "Mercor", "Curinos Inc",
    ]
    assert [r["location"] for r in rows] == ["Woodbridge, ON", "Toronto, ON", "Toronto, ON"]
    assert len({r["job_url"] for r in rows}) == 3


def test_job_urls_are_postings_not_company_profiles(search_html):
    """The card anchors point at /co/ company pages; taking those instead of
    the ld+json URLs would produce rows that never open a real posting."""
    rows = parse_ziprecruiter_search_html(search_html, "Toronto, ON")
    for row in rows:
        assert "/Job/" in row["job_url"]
        assert "/co/" not in row["job_url"]


def test_recovers_exact_post_timestamps_matched_to_the_right_job(search_html):
    """ZipRecruiter's own filter is day-granular; these second-accurate stamps
    are what make an exact hours_old window possible. Each must land on its own
    posting -- an off-by-one pairing here would silently misdate every row."""
    rows = parse_ziprecruiter_search_html(search_html, "Toronto, ON")

    by_jid = {re.search(r"jid=([0-9a-f]+)", r["job_url"]).group(1): r["date_posted"] for r in rows}
    assert by_jid == {
        "4a32bc9610b28930": "2026-07-29T12:37:48Z",
        "2dca416f5f837e07": "2026-07-02T03:26:54Z",
        "e3bccb172c9760fd": "2026-07-16T06:43:31Z",
    }


def test_card_count_mismatch_drops_enrichment_instead_of_misattributing(search_html):
    """Positional enrichment is only valid 1:1. If a card goes missing the
    rows must lose the company, never inherit the wrong one."""
    marker = '<div class="job_result_two_pane_v2'
    head, _, tail = search_html.partition(marker)
    # Drop the first card, leaving 2 cards against 3 ItemList entries.
    _, _, rest = tail.partition(marker)
    damaged = head + marker + rest

    rows = parse_ziprecruiter_search_html(damaged, "Toronto, ON")
    assert len(rows) == 3, "postings still come from ld+json"
    assert [r["company"] for r in rows] == ["", "", ""]
    # Falls back to the requested location rather than a neighbouring card's.
    assert [r["location"] for r in rows] == ["Toronto, ON"] * 3


def test_returns_empty_without_ld_json():
    assert parse_ziprecruiter_search_html("<html><body>nothing</body></html>", "X") == []


def test_ignores_non_itemlist_ld_json(search_html):
    """Search pages also carry Organization/BreadcrumbList blocks; picking the
    wrong one must not be mistaken for a successful parse."""
    decoy = '<script type="application/ld+json">{"@type": "Organization", "name": "ZipRecruiter"}</script>'
    rows = parse_ziprecruiter_search_html(decoy + search_html, "Toronto, ON")
    assert len(rows) == 3
    assert rows[0]["company"] == "Martinrea International Inc."


def test_rows_survive_missing_metadata_blob(search_html):
    """If ZipRecruiter drops the metadata payload the jobs must still come
    back, just without dates -- dateless rows are kept, not silently filtered."""
    stripped = re.sub(r'<script id="job-metadata">.*?</script>', "", search_html, flags=re.S)
    rows = parse_ziprecruiter_search_html(stripped, "Toronto, ON")
    assert len(rows) == 3
    assert all("date_posted" not in r for r in rows)


class _FakeResponse:
    def __init__(self, status_code: int, body: str = ""):
        self.status_code = status_code
        self.content = body.encode("utf-8")


class _FakeSession:
    """Stands in for curl_cffi.requests.Session, recording fingerprints."""

    def __init__(self, impersonate: str, log: list, responder):
        self.impersonate = impersonate
        self._log = log
        self._responder = responder

    def get(self, url, **kwargs):
        self._log.append((self.impersonate, url, kwargs.get("params")))
        return self._responder(self.impersonate, url, kwargs.get("params"))


def _install_fake_curl(monkeypatch, responder) -> list:
    log: list = []

    class _Requests:
        @staticmethod
        def Session(impersonate):
            return _FakeSession(impersonate, log, responder)

    fake_module = type("curl_cffi", (), {"requests": _Requests})
    monkeypatch.setitem(sys.modules, "curl_cffi", fake_module)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", _Requests)
    monkeypatch.setattr(job_service.time, "sleep", lambda _s: None)
    return log


def _page_with_ages(search_html: str, ages_hours: list[float]) -> str:
    """Rewrite the fixture's timestamps to fixed ages relative to now, so the
    hours_old assertions stay true no matter when the suite runs."""
    now = datetime.now(timezone.utc)
    stamps = [
        (now - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ") for h in ages_hours
    ]
    it = iter(stamps)
    return re.sub(
        r'\\"rollingPostedAtUtc\\":\\"[^\\"]+\\"',
        lambda _m: '\\"rollingPostedAtUtc\\":\\"' + next(it) + '\\"',
        search_html,
    )


def test_exact_hours_old_window_trims_inside_the_day_granular_result(monkeypatch, search_html):
    """The real payoff: ZipRecruiter can only filter by whole days, so a 6h
    request returns day-old rows. Those must be dropped on the exact hour."""
    body = _page_with_ages(search_html, [2.0, 5.0, 20.0])
    _install_fake_curl(monkeypatch, lambda *a: _FakeResponse(200, body))

    rows = scrape_ziprecruiter_postings("python", "Toronto", results_wanted=10, hours_old=6)
    assert len(rows) == 2, "the 20h-old posting is outside a 6h window"

    rows = scrape_ziprecruiter_postings("python", "Toronto", results_wanted=10, hours_old=24)
    assert len(rows) == 3


def test_hours_are_converted_up_to_whole_days(monkeypatch, search_html):
    """Rounding a 30h window DOWN to 1 day would never fetch the 24-30h
    postings the caller asked for, so the request must ceil to 2 days."""
    log = _install_fake_curl(monkeypatch, lambda *a: _FakeResponse(200, search_html))

    for hours, expected_days in [(6, 1), (24, 1), (30, 2), (48, 2), (168, 7), (24 * 90, 30)]:
        log.clear()
        scrape_ziprecruiter_postings("python", "Toronto", results_wanted=1, hours_old=hours)
        params = next(entry[2] for entry in log if "jobs-search" in entry[1])
        assert params["days"] == expected_days, f"{hours}h should request {expected_days}d"


def test_rotates_fingerprints_until_one_clears(monkeypatch, search_html):
    """A 403 on the first fingerprint must be retried on the next one -- this
    is the whole reason the scraper works at all."""
    def responder(impersonate, url, params):
        if impersonate == ZIPRECRUITER_IMPERSONATIONS[0]:
            return _FakeResponse(403, "Just a moment...")
        return _FakeResponse(200, search_html)

    log = _install_fake_curl(monkeypatch, responder)
    rows = scrape_ziprecruiter_postings("software engineer", "Toronto, ON",
                                        results_wanted=3, hours_old=0)

    assert len(rows) == 3
    assert rows[0]["company"] == "Martinrea International Inc."
    used = [entry[0] for entry in log]
    assert ZIPRECRUITER_IMPERSONATIONS[0] in used
    assert ZIPRECRUITER_IMPERSONATIONS[1] in used


def test_returns_empty_when_every_fingerprint_is_blocked(monkeypatch):
    log = _install_fake_curl(monkeypatch, lambda *a: _FakeResponse(403, "Just a moment..."))
    assert scrape_ziprecruiter_postings("python", "Toronto", results_wanted=5) == []
    tried = {entry[0] for entry in log}
    assert tried == set(ZIPRECRUITER_IMPERSONATIONS), "every fingerprint gets a turn"


def test_hits_jobs_search_with_the_query_attached(monkeypatch, search_html):
    """The old /candidate/search route 302s to /jobs-search and drops the
    query string, which is why it returned generic results."""
    log = _install_fake_curl(monkeypatch, lambda *a: _FakeResponse(200, search_html))
    scrape_ziprecruiter_postings("data analyst", "Vancouver, BC", results_wanted=3,
                                 hours_old=48, radius_miles=50)

    searches = [entry for entry in log if "jobs-search" in entry[1]]
    assert searches, "must query /jobs-search"
    assert not any("candidate/search" in entry[1] for entry in log)
    params = searches[0][2]
    assert params["search"] == "data analyst"
    assert params["location"] == "Vancouver, BC"
    assert params["radius"] == 50


def test_paginates_and_dedupes_until_target_is_met(monkeypatch, search_html):
    """One page holds 20 rows; asking for more must page forward, and repeated
    postings across pages must not double-count toward the target."""
    pages_seen: list = []
    ld_blob = search_html.split('type="application/ld+json">')[1].split("</script>")[0]

    def responder(impersonate, url, params):
        if "jobs-search" not in url:
            return _FakeResponse(200, "")
        page = (params or {}).get("page", 1)
        pages_seen.append(page)
        data = json.loads(ld_blob)
        for index, entry in enumerate(data["itemListElement"]):
            # Page 2 repeats entry 0 and adds unique ones.
            if page > 1 and index > 0:
                entry["url"] = f"https://www.ziprecruiter.com/c/X/Job/p{page}-{index}?jid=ab{page}{index}"
        return _FakeResponse(200, search_html.replace(ld_blob, json.dumps(data)))

    _install_fake_curl(monkeypatch, responder)
    rows = scrape_ziprecruiter_postings("python", "Toronto", results_wanted=40, hours_old=0)

    assert pages_seen[:2] == [1, 2], "second page is actually requested"
    urls = [r["job_url"] for r in rows]
    assert len(urls) == len(set(urls)), "duplicate postings are dropped"


def test_no_browser_is_ever_launched(monkeypatch, search_html):
    """The scraper must not reach for Playwright or the browser dispatch gate
    any more -- holding that app-wide slot for a plain HTTP fetch would stall
    real browser work (resume scrapes) for nothing."""
    from services import browser_service

    def fail(*a, **k):
        raise AssertionError("browser dispatch slot must not be acquired")

    monkeypatch.setattr(browser_service, "acquire_browser_work_slot", fail)
    _install_fake_curl(monkeypatch, lambda *a: _FakeResponse(200, search_html))

    rows = scrape_ziprecruiter_postings("python", "Toronto", results_wanted=3, hours_old=0)
    assert len(rows) == 3
    assert not hasattr(job_service, "_scrape_ziprecruiter_with_playwright")


def test_missing_curl_cffi_degrades_quietly(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "curl_cffi":
            raise ImportError("no curl_cffi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert scrape_ziprecruiter_postings("python", "Toronto") == []


# --- remembering which fingerprint cleared the WAF ----------------------------
#
# The rotation always worked, so nothing here is about returning rows. It is
# about not paying for a request that is known to fail: measured 2026-09-06 the
# configured first choice draws a 403 on every attempt, and every page of every
# search opened with it, then slept a backoff, then tried the next one. Two
# live runs back to back: 3.3s, then 1.5s once the winner is remembered.


@pytest.fixture(autouse=True)
def _forget_the_remembered_fingerprint():
    """The winner is a module global, so without this the fake WAF in one test
    picks the starting fingerprint the next test observes -- and this suite
    runs in a randomised order, so that would fail intermittently and only on
    some seeds. Cleared on both sides: a test may arrive with it already set."""
    job_service._ziprecruiter_last_good = None
    yield
    job_service._ziprecruiter_last_good = None


def _search_fingerprints(log) -> list[str]:
    """Fingerprints used for the search itself, ignoring the cookie warm-up."""
    return [entry[0] for entry in log if "jobs-search" in entry[1]]


def test_the_fingerprint_that_cleared_is_tried_first_next_time(monkeypatch, search_html):
    """The point of the whole change: the second search does not re-pay for the
    fingerprint the first one already found to be blocked."""
    blocked = ZIPRECRUITER_IMPERSONATIONS[0]

    def responder(impersonate, url, params):
        if impersonate == blocked:
            return _FakeResponse(403, "Just a moment...")
        return _FakeResponse(200, search_html)

    log = _install_fake_curl(monkeypatch, responder)

    scrape_ziprecruiter_postings("python", "Toronto", results_wanted=3, hours_old=0)
    first = _search_fingerprints(log)
    assert first[0] == blocked, "the configured order is still the cold start"
    winner = first[-1]

    log.clear()
    rows = scrape_ziprecruiter_postings("python", "Toronto", results_wanted=3, hours_old=0)
    assert _search_fingerprints(log) == [winner], "one request, and it is the winner"
    assert len(rows) == 3, "and it still returns the rows"


def test_a_remembered_fingerprint_that_stops_working_is_not_sticky(monkeypatch, search_html):
    """The WAF's preference moves -- that is why this is remembered rather than
    reordered in the tuple. When the remembered one starts being refused the
    rotation must carry on past it and remember the replacement."""
    state = {"blocked": ZIPRECRUITER_IMPERSONATIONS[0]}

    def responder(impersonate, url, params):
        if impersonate == state["blocked"]:
            return _FakeResponse(403, "Just a moment...")
        return _FakeResponse(200, search_html)

    log = _install_fake_curl(monkeypatch, responder)
    scrape_ziprecruiter_postings("python", "Toronto", results_wanted=3, hours_old=0)
    winner = _search_fingerprints(log)[-1]

    state["blocked"] = winner
    log.clear()
    rows = scrape_ziprecruiter_postings("python", "Toronto", results_wanted=3, hours_old=0)
    tried = _search_fingerprints(log)
    assert tried[0] == winner, "starts from what last worked"
    assert len(tried) > 1 and len(rows) == 3, "and falls through to one that clears"

    state["blocked"] = "no fingerprint at all"
    log.clear()
    scrape_ziprecruiter_postings("python", "Toronto", results_wanted=3, hours_old=0)
    assert _search_fingerprints(log) == [tried[1]], "the replacement is remembered"


def test_every_fingerprint_still_gets_a_turn_from_any_starting_point(monkeypatch):
    """Remembering reorders the rotation; it must not shorten it. A fully
    blocked search has to try all three no matter which one it starts from,
    and try each exactly once."""
    for start in ZIPRECRUITER_IMPERSONATIONS:
        job_service._note_ziprecruiter_impersonation(start)
        log = _install_fake_curl(
            monkeypatch, lambda *a: _FakeResponse(403, "Just a moment...")
        )
        assert scrape_ziprecruiter_postings("python", "Toronto", results_wanted=5) == []
        tried = _search_fingerprints(log)
        assert tried[0] == start
        assert sorted(tried) == sorted(ZIPRECRUITER_IMPERSONATIONS), (
            f"starting from {start} must still cover the rotation exactly once"
        )


def test_a_blocked_fingerprint_is_never_recorded_as_the_winner(monkeypatch):
    """Otherwise a total outage teaches the scraper to open with whichever
    fingerprint happened to fail last, and the memory becomes noise."""
    _install_fake_curl(monkeypatch, lambda *a: _FakeResponse(403, "Just a moment..."))
    scrape_ziprecruiter_postings("python", "Toronto", results_wanted=5)
    assert job_service._ziprecruiter_last_good is None


def test_a_fingerprint_no_longer_configured_is_ignored(monkeypatch, search_html):
    """The memory outlives edits to the tuple. If an entry is dropped or
    renamed, the stale name must not be requested from curl_cffi -- which
    raises on an unknown target -- nor silently drop a real fingerprint."""
    job_service._note_ziprecruiter_impersonation("chrome99")
    log = _install_fake_curl(monkeypatch, lambda *a: _FakeResponse(200, search_html))
    scrape_ziprecruiter_postings("python", "Toronto", results_wanted=3, hours_old=0)

    tried = _search_fingerprints(log)
    assert "chrome99" not in tried
    assert tried[0] == ZIPRECRUITER_IMPERSONATIONS[0], "falls back to the configured order"
