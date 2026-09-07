"""Oracle Cloud Recruiting and Personio, the two platforms added 2026-09-05.

Oracle matters because it is Taleo's successor and carries a large share of
enterprise hiring; measured on one tenant, 2,276 open requisitions. Personio
matters because it is the dominant European SMB platform and publishes an
unauthenticated feed per tenant.

Both were verified live before being written, and both are pinned here without
touching the network: `requests.get` is replaced with a fake serving recorded
response shapes.

The Oracle paging rule is the part most worth protecting. The feed sorts newest
first and `merge_data` only archives postings inside seven days, so the scrape
can stop as soon as a page opens older than that. On the measured tenant that
was 400 rows in 2 requests instead of 12 -- the difference between a platform
with thousands of requisitions per tenant being affordable and not.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as A  # noqa: E402


def _today(offset_days: int = 0) -> str:
    return (date.today() - timedelta(days=offset_days)).isoformat()


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.content = text.encode("utf-8")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _oracle_payload(jobs):
    return {"items": [{"TotalJobsCount": 9999,
                       "requisitionList": [
                           {"Id": str(1000 + i), "Title": t,
                            "PostedDate": d,
                            "PrimaryLocation": loc,
                            "PrimaryLocationCountry": cc,
                            "ShortDescriptionStr": "desc",
                            "JobType": "Full time"}
                           for i, (t, d, loc, cc) in enumerate(jobs)]}]}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(A, "_mark_dead", lambda *a, **k: None)
    monkeypatch.setattr(A, "_mark_alive", lambda *a, **k: None)
    monkeypatch.setattr(A, "_is_dead", lambda *a, **k: False)
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    A.reset_refusal_breaker("oracle")
    A.reset_refusal_breaker("personio")
    yield
    A.reset_refusal_breaker("oracle")
    A.reset_refusal_breaker("personio")


def _serve_oracle(monkeypatch, pages):
    """Serve `pages` in order; record the URLs requested."""
    calls = []

    def _get(url, **kw):
        calls.append(url)
        i = len(calls) - 1
        if i >= len(pages):
            return _Resp(200, _oracle_payload([]))
        return _Resp(200, _oracle_payload(pages[i]))

    monkeypatch.setattr(A, "_http_get", _get)
    return calls


# ── Oracle: the paging rule ──────────────────────────────────────────────────

def test_paging_stops_once_a_page_opens_outside_the_freshness_window():
    """The whole economy of this scraper. Without it a 2,276-requisition
    tenant costs twelve requests to collect jobs the archive will discard.
    """
    fresh = [("Job A", _today(0), "Austin, TX, United States", "US")] * 200
    stale = [("Job B", _today(30), "Austin, TX, United States", "US")] * 200
    with pytest.MonkeyPatch.context() as mp:
        calls = _serve_oracle(mp, [fresh, stale, fresh])
        rows = A._scrape_oracle("host.example.com", "", "", 0)
    assert len(calls) == 2, f"kept paging past the cutoff: {len(calls)} requests"
    assert len(rows) == 400


def test_the_page_that_crosses_the_cutoff_is_still_kept():
    """It is the pages AFTER the crossing that cannot contain anything
    archivable. Dropping the crossing page would lose the jobs on it that are
    still inside the window.
    """
    mixed = ([("Fresh", _today(1), "Austin, TX", "US")] * 5
             + [("Stale", _today(40), "Austin, TX", "US")] * 5)
    with pytest.MonkeyPatch.context() as mp:
        _serve_oracle(mp, [mixed, mixed])
        rows = A._scrape_oracle("host.example.com", "", "", 0)
    assert len(rows) == 10


def test_an_all_fresh_tenant_pages_until_the_feed_runs_out():
    fresh_full = [("Job", _today(0), "Austin, TX", "US")] * 200
    short = [("Job", _today(0), "Austin, TX", "US")] * 10
    with pytest.MonkeyPatch.context() as mp:
        calls = _serve_oracle(mp, [fresh_full, fresh_full, short])
        rows = A._scrape_oracle("host.example.com", "", "", 0)
    assert len(calls) == 3
    assert len(rows) == 410


def test_paging_is_bounded_even_if_dates_never_go_stale():
    """A tenant whose PostedDate field misbehaves must not be able to walk
    thousands of pages. The date rule is the economy; this is the safety net.
    """
    full = [("Job", _today(0), "Austin, TX", "US")] * 200
    with pytest.MonkeyPatch.context() as mp:
        calls = _serve_oracle(mp, [full] * 50)
        A._scrape_oracle("host.example.com", "", "", 0)
    assert len(calls) <= A.ORACLE_MAX_PAGES


def test_an_empty_first_page_costs_one_request():
    with pytest.MonkeyPatch.context() as mp:
        calls = _serve_oracle(mp, [[]])
        rows = A._scrape_oracle("host.example.com", "", "", 0)
    assert rows == [] and len(calls) == 1


# ── Oracle: the finder syntax, which is easy to get silently wrong ──────────

def test_limit_offset_and_sort_go_inside_the_finder():
    """Putting them in the query string instead returns 200 with zero rows,
    which reads exactly like an empty tenant -- a silent, total loss of the
    platform. This was hit for real while building it.
    """
    with pytest.MonkeyPatch.context() as mp:
        calls = _serve_oracle(mp, [[("J", _today(0), "Austin, TX", "US")]])
        A._scrape_oracle("host.example.com", "", "", 0)
    url = calls[0]
    finder = url.split("finder=", 1)[1]
    assert "limit=" in finder and "sortBy=POSTING_DATES_DESC" in finder
    assert "&limit=" not in url and "&sortBy=" not in url


def test_the_slug_is_used_as_the_whole_host():
    with pytest.MonkeyPatch.context() as mp:
        calls = _serve_oracle(mp, [[("J", _today(0), "Austin, TX", "US")]])
        A._scrape_oracle("eeho.fa.us2.oraclecloud.com", "", "", 0)
    assert calls[0].startswith("https://eeho.fa.us2.oraclecloud.com/hcmRestApi/")


# ── Oracle: the location, which feeds geo priority ──────────────────────────

def test_the_iso_country_is_appended_so_geo_does_not_have_to_guess():
    """Oracle already knows the country as an ISO code. Re-deriving it from the
    display string is what produces the CA/DE ambiguities geo_priority spends
    an ambiguous-code table on.
    """
    with pytest.MonkeyPatch.context() as mp:
        _serve_oracle(mp, [[("J", _today(0), "NOIDA, UTTAR PRADESH, India", "IN")]])
        rows = A._scrape_oracle("host.example.com", "", "", 0)
    from services.jba import geo_priority as G
    assert rows[0]["location"].endswith(", IN")
    assert G.country_of(rows[0]["location"]) == "IN"


def test_a_country_already_present_is_not_duplicated():
    with pytest.MonkeyPatch.context() as mp:
        _serve_oracle(mp, [[("J", _today(0), "Austin, TX, US", "US")]])
        rows = A._scrape_oracle("host.example.com", "", "", 0)
    assert rows[0]["location"].count("US") == 1


def test_a_missing_country_still_yields_the_display_location():
    with pytest.MonkeyPatch.context() as mp:
        _serve_oracle(mp, [[("J", _today(0), "Remote", "")]])
        rows = A._scrape_oracle("host.example.com", "", "", 0)
    assert rows[0]["location"] == "Remote"


def test_a_row_without_an_id_is_dropped_rather_than_given_a_broken_url():
    payload = {"items": [{"requisitionList": [
        {"Title": "No id", "PostedDate": _today(0), "PrimaryLocation": "Austin"}]}]}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(A, "_http_get", lambda url, **kw: _Resp(200, payload))
        rows = A._scrape_oracle("host.example.com", "", "", 0)
    assert rows == []


# ── Personio ─────────────────────────────────────────────────────────────────

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<workzag-jobs>
<position>
  <id>1834171</id>
  <office>Munich</office>
  <additionalOffices><office>Berlin</office></additionalOffices>
  <name>Staff Software Engineer</name>
  <employmentType>permanent</employmentType>
</position>
<position>
  <id>99</id>
  <office>Oman</office>
  <name>Senior Process Engineer</name>
</position>
</workzag-jobs>"""


def _serve_personio(monkeypatch, status=200, text=_XML):
    calls = []

    def _get(url, **kw):
        calls.append(url)
        return _Resp(status, None, text)

    monkeypatch.setattr(A, "_http_get", _get)
    return calls


def test_positions_are_parsed_from_the_feed(monkeypatch):
    _serve_personio(monkeypatch)
    rows = A._scrape_personio("acme", "", "", 0)
    assert [r["title"] for r in rows] == ["Staff Software Engineer",
                                          "Senior Process Engineer"]


def test_additional_offices_are_kept(monkeypatch):
    """A role open in two cities is one posting in two places. Dropping the
    extras would hide the location that matters to a given channel.
    """
    _serve_personio(monkeypatch)
    rows = A._scrape_personio("acme", "", "", 0)
    assert rows[0]["location"] == "Munich, Berlin"


def test_the_job_url_is_built_from_the_tenant_and_id(monkeypatch):
    _serve_personio(monkeypatch)
    rows = A._scrape_personio("acme", "", "", 0)
    assert rows[0]["job_url"] == "https://acme.jobs.personio.de/job/1834171"


def test_a_404_marks_the_board_dead(monkeypatch):
    marked = []
    monkeypatch.setattr(A, "_mark_dead", lambda p, s: marked.append((p, s)))
    _serve_personio(monkeypatch, status=404, text="")
    assert A._scrape_personio("gone", "", "", 0) == []
    assert marked == [("personio", "gone")]


def test_a_refusal_never_marks_a_board_dead(monkeypatch):
    """403/429 is the host declining and says nothing about whether the tenant
    exists -- the same rule the shared JSON fetcher follows. Personio does not
    go through that fetcher because its feed is XML, so the rule has to be
    honoured here rather than inherited.
    """
    marked = []
    monkeypatch.setattr(A, "_mark_dead", lambda p, s: marked.append((p, s)))
    _serve_personio(monkeypatch, status=429, text="")
    A._scrape_personio("acme", "", "", 0)
    assert marked == []


def test_wholesale_refusal_trips_the_breaker(monkeypatch):
    """Not inheriting the fetcher also means not inheriting its breaker, so
    Personio would have been the one platform that could burn a whole cycle on
    a host that is refusing everything.
    """
    _serve_personio(monkeypatch, status=429, text="")
    for i in range(A._REFUSAL_TRIP + 5):
        A._scrape_personio(f"co{i}", "", "", 0)
    assert A.platform_is_refusing("personio") is True


def test_no_request_is_made_once_the_breaker_is_open(monkeypatch):
    calls = _serve_personio(monkeypatch, status=429, text="")
    for i in range(A._REFUSAL_TRIP + 40):
        A._scrape_personio(f"co{i}", "", "", 0)
    assert len(calls) <= A._REFUSAL_TRIP


def test_malformed_xml_yields_nothing_rather_than_raising(monkeypatch):
    _serve_personio(monkeypatch, text="<workzag-jobs><position>")
    assert A._scrape_personio("acme", "", "", 0) == []


def test_an_empty_but_valid_feed_is_not_an_error(monkeypatch):
    _serve_personio(monkeypatch, text="<workzag-jobs></workzag-jobs>")
    assert A._scrape_personio("acme", "", "", 0) == []


def test_a_network_failure_is_swallowed(monkeypatch):
    def _boom(url, **kw):
        raise OSError("connection reset")
    monkeypatch.setattr(A, "_http_get", _boom)
    assert A._scrape_personio("acme", "", "", 0) == []


# ── both are wired into the platform registries ─────────────────────────────

@pytest.mark.parametrize("platform", ["oracle", "personio"])
def test_the_platform_is_fully_registered(platform):
    assert platform in A.ATS_PLATFORMS
    assert platform in A._SCRAPERS
    assert platform in A.PLATFORM_WORKERS
    assert platform in A._HARVEST_FILES


@pytest.mark.parametrize("platform", ["oracle", "personio"])
def test_the_harvester_and_validator_know_it_too(platform):
    """Three separate registries in three files. A platform in one and not the
    others harvests without being scraped, or is scraped without ever being
    validated.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import harvest_ats as hc
    import validate_ats_slugs as v
    assert platform in {p.name for p in hc.PLATFORMS}
    assert platform in v.PLATFORMS
    assert platform in v.PROBES
    assert platform in v.WORKERS
