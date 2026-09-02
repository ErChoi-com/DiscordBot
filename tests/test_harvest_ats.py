"""Tests for the Common Crawl ATS slug harvester.

Hermetic per pytest.ini: every CDX response here is a fixture captured from the
live index, and the transport is injected.  The URL samples are real captures
from CC-MAIN-2026-34 and CC-MAIN-2025-38, not invented shapes -- the whole point
of this harvester is parsing someone else's URL conventions correctly, so tests
built on made-up URLs would prove nothing.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import harvest_ats as hc  # noqa: E402


def _offline_bulk(monkeypatch):
    """Stub the bulk index's own fetchers.

    They are separate from _http_get, so patching only that leaves the bulk
    path reaching data.commoncrawl.org for real -- which is how a 2-second
    suite quietly became a 62-second one.
    """
    def no_net(*args, **kwargs):
        raise urllib.error.URLError("offline in tests")

    monkeypatch.setattr(hc, "_http_get_range", no_net)
    monkeypatch.setattr(hc, "_http_head_size", no_net)


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    import io
    return urllib.error.HTTPError(
        "https://index.commoncrawl.org/x", code, "err", {}, io.BytesIO(body)
    )


# --------------------------------------------------------------------------
# Slug extraction, against URLs really present in the index
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://job-boards.greenhouse.io/0verwatch/jobs/4299368009?gh_src=t5", "0verwatch"),
    ("https://boards.greenhouse.io/100Thieves", "100thieves"),
    ("https://boards.greenhouse.io/10beauty/jobs/5162257007?utm_source=x", "10beauty"),
    # Tracking hash in the slug position - real capture, must not be harvested.
    ("https://job-boards.greenhouse.io/1945bce8d3924ece9421ba8630f57b0c", None),
    ("https://job-boards.greenhouse.io/embed/job_board?for=x", None),
    ("https://boards.greenhouse.io/robots.txt", None),
    ("https://job-boards.greenhouse.io/", None),
])
def test_greenhouse_extraction(url, expected):
    assert hc.PLATFORM_BY_NAME["greenhouse"].extract(url) == expected


@pytest.mark.parametrize("url,expected", [
    ("https://jobs.lever.co/10up-2/abc-def", "10up-2"),
    ("https://jobs.lever.co/15five/0b383a25-242f-4c83-a185-d5b59c9a202d", "15five"),
    ("https://jobs.lever.co/robots.txt", None),
])
def test_lever_extraction(url, expected):
    assert hc.PLATFORM_BY_NAME["lever"].extract(url) == expected


@pytest.mark.parametrize("url,expected", [
    ("https://jobs.ashbyhq.com/0x/01df9077-73f9-4c9b-8e87-a9de1423dcd9/application?source=y", "0x"),
    ("https://jobs.ashbyhq.com/0g/d35c9785-1912-4c23-8d09-dbbe353d4733?utm_source=z", "0g"),
])
def test_ashby_extraction(url, expected):
    assert hc.PLATFORM_BY_NAME["ashby"].extract(url) == expected


@pytest.mark.parametrize("url,expected", [
    # The canonical triple ats_service:918 rebuilds a board URL from.
    ("https://2020companies.wd1.myworkdayjobs.com/en-US/External_Careers/job/Addison-IL/X",
     "2020companies|wd1|external_careers"),
    ("https://medtronic.wd1.myworkdayjobs.com/MedtronicCareers/job/x",
     "medtronic|wd1|medtroniccareers"),
    # Staging tenants serve no public board.
    ("https://swisscom1.impl-wd103.myworkdayjobs.com/robots.txt", None),
    ("https://viscofan.impl-wd103.myworkdayjobs.com/en-US/Careers", None),
    # Locale must not be mistaken for the site name: upstream has no |en entries.
    ("https://3m.wd1.myworkdayjobs.com/en/Search", "3m|wd1|search"),
    ("https://3m.wd1.myworkdayjobs.com/es/Careers", "3m|wd1|careers"),
    # Only the first locale is skipped, so a second locale-shaped segment is
    # taken as the site. That is the deliberate trade: two locales in a row is
    # an invented shape, while locale-shaped site names are real and live
    # (abinbev|wd1|py, abinbev|wd1|hn-es), and consuming them lost the boards.
    ("https://acme.wd5.myworkdayjobs.com/en-US/en-GB/Careers", "acme|wd5|en-gb"),
    ("https://acme.wd1.myworkdayjobs.com/robots.txt", None),
    ("https://acme.wd1.myworkdayjobs.com/", None),
])
def test_workday_extraction(url, expected):
    assert hc.PLATFORM_BY_NAME["workday"].extract(url) == expected


def test_workday_site_named_jobs_is_kept():
    """``jobs`` is a real Workday site for 33 upstream tenants.

    It is also in RESERVED_SEGMENTS, which guards the path-slug namespace. If
    that set were applied to the Workday site position it would silently drop
    every one of those boards.
    """
    assert "jobs" in hc.RESERVED_SEGMENTS
    got = hc.PLATFORM_BY_NAME["workday"].extract(
        "https://acme.wd3.myworkdayjobs.com/en-US/jobs/job/1"
    )
    assert got == "acme|wd3|jobs"


@pytest.mark.parametrize("url,expected", [
    # ats_service:1110 builds f"careers-{slug}.icims.com", so the stored form is
    # the label with that prefix removed. Storing it prefixed round-trips to
    # careers-careers-gbrx.icims.com, which is why all 3,255 such upstream
    # entries are dead.
    ("https://careers-gbrx.icims.com/jobs/1", "gbrx"),
    ("https://theharrispoll.icims.com/jobs/1", "theharrispoll"),
    ("https://www.icims.com/", None),
])
def test_icims_extraction(url, expected):
    assert hc.PLATFORM_BY_NAME["icims"].extract(url) == expected


def test_icims_never_emits_double_prefixable_slug():
    for url in ("https://careers-acme.icims.com/x", "https://acme.icims.com/x"):
        slug = hc.PLATFORM_BY_NAME["icims"].extract(url)
        assert slug is not None
        assert not slug.startswith("careers-")


@pytest.mark.parametrize("url,expected", [
    ("https://100percentgroup.bamboohr.com/careers/list", "100percentgroup"),
    ("https://www.bamboohr.com/", None),
])
def test_bamboohr_extraction(url, expected):
    assert hc.PLATFORM_BY_NAME["bamboohr"].extract(url) == expected


# --------------------------------------------------------------------------
# Structural plausibility
# --------------------------------------------------------------------------

@pytest.mark.parametrize("slug,ok", [
    ("acme", True), ("10up-2", True), ("0x", True), ("a", False), ("", False),
    ("1945bce8d3924ece9421ba8630f57b0c", False),      # 32-char hex tracking id
    ("5364856uhdfnvbkldfnbhrpkdfgbdvtyhro", False),   # long, vowelless
    ("jobs", False), ("embed", False),
    ("UPPER", False),                                  # caller lowercases first
    ("a" * 101, False),
    ("my-company_name.co", True),
])
def test_looks_like_company(slug, ok):
    assert hc._looks_like_company(slug) is ok


# --------------------------------------------------------------------------
# CDX transport
# --------------------------------------------------------------------------

def test_no_captures_404_is_empty_not_error():
    """The index signals 'nothing here' with a 404 carrying a JSON message.

    Treating that as fatal would abort a sweep the first time any platform was
    absent from an older crawl - which is routine (Lever vanished from crawls
    after CC-MAIN-2025-38).
    """
    def fetch(url):
        raise _http_error(404, b'{"message": "No Captures found for: x.com"}')

    assert hc.cdx_request("u", fetch=fetch, sleep=lambda s: None) is None


def test_404_without_message_is_an_error():
    def fetch(url):
        raise _http_error(404, b"<html>not found</html>")

    with pytest.raises(hc.HarvestError):
        hc.cdx_request("u", fetch=fetch, sleep=lambda s: None)


def test_503_retries_then_succeeds():
    calls = []

    def fetch(url):
        calls.append(url)
        if len(calls) < 3:
            raise _http_error(503)
        return b'{"url": "ok"}'

    slept = []
    assert hc.cdx_request("u", fetch=fetch, sleep=slept.append) == b'{"url": "ok"}'
    assert len(calls) == 3
    assert slept == [2.0, 4.0]  # exponential


def test_incomplete_read_is_retried_not_fatal():
    """The index serves these pages chunked and truncates them in practice.

    A live run against CC-MAIN-2026-34 died here: IncompleteRead is an
    http.client.HTTPException, not an OSError, so it escaped the retry clause
    and killed the whole harvest partway through.
    """
    calls = []

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            raise http.client.IncompleteRead(b"partial")
        return b'{"url": "https://boards.greenhouse.io/acme/jobs/1"}'

    got = hc.cdx_request("u", fetch=fetch, sleep=lambda s: None)
    assert got is not None
    assert len(calls) == 2


def test_partial_body_is_never_returned():
    """Returning the truncated bytes would publish a fraction of a page as if
    it were the whole thing - silent under-harvesting, not a crash."""
    def fetch(url):
        raise http.client.IncompleteRead(b'{"url": "https://boards.greenhouse.io/a/j/1"}')

    with pytest.raises(hc.HarvestError):
        hc.cdx_request("u", fetch=fetch, sleep=lambda s: None)


def test_retries_are_bounded():
    calls = []

    def fetch(url):
        calls.append(url)
        raise _http_error(503)

    with pytest.raises(hc.HarvestError):
        hc.cdx_request("u", fetch=fetch, sleep=lambda s: None)
    assert len(calls) == hc.MAX_RETRIES


def test_iter_urls_skips_malformed_lines():
    """One bad line must not discard the other 13,000 on the page."""
    raw = (b'{"url": "https://a.com/1", "status": "200"}\n'
           b'{"url": "https://a.com/2", "status": "200"\n'      # truncated
           b'\n'
           b'not json at all\n'
           b'{"nourl": true}\n'
           b'{"url": "https://a.com/3", "status": "200"}\n')
    assert list(hc.iter_urls(raw)) == ["https://a.com/1", "https://a.com/3"]


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------

def test_workday_query_uses_domain_match():
    """A host query returns zero pages for Workday - tenants are subdomains."""
    workday = hc.PLATFORM_BY_NAME["workday"]
    # Every Workday query must be a domain match, whatever hosts are listed:
    # tenants are subdomains, so a host query returns zero pages.
    assert workday.queries, "workday must have queries"
    assert all(match == "domain" for _, match in workday.queries)
    assert ("myworkdayjobs.com", "domain") in workday.queries
    url = hc.build_query_url("CC-MAIN-2026-34", "myworkdayjobs.com", "domain", page=0)
    assert "matchType=domain" in url


def test_prefix_queries_omit_match_type():
    url = hc.build_query_url("CC-MAIN-2026-34", "jobs.lever.co/*", "prefix", page=0)
    assert "matchType" not in url
    assert "page=0" in url


def test_page_count_query_drops_field_filters():
    """showNumPages with fl/filter returns a count of the *filtered* set, which
    is not the page count the paged fetch then walks."""
    url = hc.build_query_url("CC-MAIN-2026-34", "x.com/*", "prefix", num_pages=True)
    assert "showNumPages=true" in url
    assert "filter=" not in url
    assert "fl=" not in url


def test_greenhouse_sweeps_every_board_host():
    """Greenhouse migrated hosts and runs a separate EU one; no single host
    covers the company set."""
    hosts = {q for q, _ in hc.PLATFORM_BY_NAME["greenhouse"].queries}
    assert hosts == {"boards.greenhouse.io/*", "job-boards.greenhouse.io/*",
                     "job-boards.eu.greenhouse.io/*"}


# --------------------------------------------------------------------------
# Sweep behaviour
# --------------------------------------------------------------------------

def _fake_index(pages: dict[str, list[str]], fail: set[str] | None = None):
    """Fetcher serving canned pages, keyed by the request's decoded url= param.

    Matching the exact parameter rather than a substring matters here:
    "boards.greenhouse.io/*" is a substring of "job-boards.greenhouse.io/*", so
    a substring match serves the first host's page for both queries -- which
    makes the second Greenhouse host look empty and the dedupe test pass for
    entirely the wrong reason.
    """
    import urllib.parse as _up

    fail = fail or set()

    def fetch(url: str) -> bytes:
        query = _up.parse_qs(_up.urlparse(url).query)
        target = (query.get("url") or [""])[0]
        if target not in pages:
            raise _http_error(404, b'{"message": "No Captures found"}')
        if target in fail:
            raise _http_error(500)
        if "showNumPages" in query:
            return json.dumps({"pages": 1}).encode()
        return b"\n".join(json.dumps({"url": u, "status": "200"}).encode() for u in pages[target])

    return fetch


def test_harvest_platform_collects_and_dedupes():
    gh = hc.PLATFORM_BY_NAME["greenhouse"]
    fetch = _fake_index({
        "boards.greenhouse.io/*": [
            "https://boards.greenhouse.io/acme/jobs/1",
            "https://boards.greenhouse.io/acme/jobs/2",   # same company twice
            "https://boards.greenhouse.io/embed/x",       # reserved
        ],
        "job-boards.greenhouse.io/*": [
            "https://job-boards.greenhouse.io/beta/jobs/9",
        ],
    })
    r = hc.harvest_platform(gh, "CC-MAIN-2026-34", fetch=fetch, sleep=lambda s: None,
                            delay=0, log=lambda m: None)
    assert r.slugs == {"acme", "beta"}
    assert r.records_seen == 4
    assert r.errors == []


def test_one_failing_query_does_not_lose_the_other():
    """The two Greenhouse hosts are independent sources of companies."""
    gh = hc.PLATFORM_BY_NAME["greenhouse"]
    fetch = _fake_index(
        {
            "boards.greenhouse.io/*": ["https://boards.greenhouse.io/acme/jobs/1"],
            "job-boards.greenhouse.io/*": ["https://job-boards.greenhouse.io/beta/j/9"],
        },
        fail={"boards.greenhouse.io/*"},
    )
    r = hc.harvest_platform(gh, "CC-MAIN-2026-34", fetch=fetch, sleep=lambda s: None,
                            delay=0, log=lambda m: None)
    assert r.slugs == {"beta"}
    assert r.errors  # the failure is reported, not swallowed


def test_platform_absent_from_crawl_is_not_an_error():
    lever = hc.PLATFORM_BY_NAME["lever"]
    r = hc.harvest_platform(lever, "CC-MAIN-2026-34", fetch=_fake_index({}),
                            sleep=lambda s: None, delay=0, log=lambda m: None)
    assert r.slugs == set()
    assert r.errors == []


# --------------------------------------------------------------------------
# Crawl selection
# --------------------------------------------------------------------------

def test_latest_crawls_sorts_by_year_and_week():
    body = json.dumps([
        {"id": "CC-MAIN-2025-38"},
        {"id": "CC-MAIN-2026-34"},
        {"id": "not-a-crawl"},
        {"id": "CC-MAIN-2026-05"},
        {"nope": 1},
    ]).encode()
    got = hc.latest_crawls(2, fetch=lambda u: body, sleep=lambda s: None)
    assert got == ["CC-MAIN-2026-34", "CC-MAIN-2026-05"]


def test_latest_crawls_rejects_empty_listing():
    with pytest.raises(hc.HarvestError):
        hc.latest_crawls(1, fetch=lambda u: b"[]", sleep=lambda s: None)


# --------------------------------------------------------------------------
# Output file
# --------------------------------------------------------------------------

def test_write_is_sorted_and_round_trips(tmp_path):
    path = tmp_path / "greenhouse.json"
    hc.write_slugs(path, {"zeta", "acme", "beta"})
    assert json.loads(path.read_text()) == ["acme", "beta", "zeta"]
    assert hc.load_existing(path) == {"acme", "beta", "zeta"}


def test_load_existing_tolerates_missing_and_corrupt(tmp_path):
    assert hc.load_existing(tmp_path / "nope.json") == set()
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json")
    assert hc.load_existing(bad) == set()
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"a": 1}')          # object, not list
    assert hc.load_existing(wrong) == set()


def test_write_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "x.json"
    hc.write_slugs(path, {"a"})
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_harvest_is_additive_never_removes(tmp_path, monkeypatch):
    """A harvest that sees fewer slugs than last time must not shrink the file.

    Common Crawl coverage genuinely fluctuates - Lever dropped out of the index
    entirely after CC-MAIN-2025-38. If a thin crawl overwrote the file, one bad
    week would erase years of accumulated companies.
    """
    out = tmp_path / "out"
    hc.write_slugs(out / "greenhouse.json", {"old-company", "acme"})

    fetch = _fake_index({
        "boards.greenhouse.io/*": ["https://boards.greenhouse.io/acme/jobs/1"],
        "job-boards.greenhouse.io/*": ["https://job-boards.greenhouse.io/newco/j/1"],
    })
    monkeypatch.setattr(hc, "_http_get", fetch)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)

    rc = hc.main(["--crawl", "CC-MAIN-2026-34", "--platform", "greenhouse",
                  "--out", str(out), "--index", "commoncrawl", "--delay", "0"])
    assert rc == 0
    assert hc.load_existing(out / "greenhouse.json") == {"old-company", "acme", "newco"}


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "out"
    fetch = _fake_index({
        "boards.greenhouse.io/*": ["https://boards.greenhouse.io/acme/jobs/1"],
    })
    monkeypatch.setattr(hc, "_http_get", fetch)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)
    assert hc.main(["--crawl", "CC-MAIN-2026-34", "--platform", "greenhouse",
                    "--out", str(out), "--index", "commoncrawl",
                    "--dry-run", "--delay", "0"]) == 0
    assert not (out / "greenhouse.json").exists()


def test_total_failure_exits_nonzero(tmp_path, monkeypatch):
    """Every platform yielding nothing means the index or our queries broke.

    Exiting 0 there would let the workflow publish an unchanged file and report
    success forever while harvesting nothing.
    """
    monkeypatch.setattr(hc, "_http_get", _fake_index({}))
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)
    assert hc.main(["--crawl", "CC-MAIN-2026-34", "--out", str(tmp_path / "o"),
                    "--delay", "0"]) == 1


def test_page_count_failure_falls_back_to_blind_walk():
    """A failed count must not cost the platform its data.

    showNumPages on a large domain is the most expensive query the harvester
    makes: bamboohr.com spans the entire marketing site, and its count query
    timed out on all five older crawls in a real sweep while the paged fetches
    for the same query would have worked. That run harvested zero BambooHR
    companies purely because the count failed.
    """
    pages = {"bamboohr.com": ["https://acme.bamboohr.com/careers/list",
                              "https://beta.bamboohr.com/careers/list"]}

    def fetch(url: str) -> bytes:
        import urllib.parse as _up
        query = _up.parse_qs(_up.urlparse(url).query)
        if "showNumPages" in query:
            raise _http_error(504)                     # the count times out
        page = int((query.get("page") or ["0"])[0])
        if page > 0:
            raise _http_error(404, b'{"message": "No Captures found"}')
        return b"\n".join(json.dumps({"url": u, "status": "200"}).encode()
                          for u in pages["bamboohr.com"])

    r = hc.harvest_platform(hc.PLATFORM_BY_NAME["bamboohr"], "CC-MAIN-2026-34",
                            fetch=fetch, sleep=lambda s: None, delay=0,
                            log=lambda m: None)
    assert r.slugs == {"acme", "beta"}
    assert r.errors                     # the degradation is still reported


def test_blind_walk_is_bounded():
    """An index that never returns an empty page must not loop forever."""
    def fetch(url: str) -> bytes:
        import urllib.parse as _up
        if "showNumPages" in _up.parse_qs(_up.urlparse(url).query):
            raise _http_error(504)
        return json.dumps({"url": "https://acme.bamboohr.com/careers/list",
                           "status": "200"}).encode()

    r = hc.harvest_platform(hc.PLATFORM_BY_NAME["bamboohr"], "CC-MAIN-2026-34",
                            fetch=fetch, sleep=lambda s: None, delay=0,
                            log=lambda m: None)
    assert r.pages_fetched == hc.BLIND_PAGE_LIMIT


# --------------------------------------------------------------------------
# Wayback index
# --------------------------------------------------------------------------

def test_wayback_collapse_depth_leaves_slug_characters():
    """collapse=urlkey:N groups on the first N chars of `co,lever,jobs)/acme`.

    Too shallow and distinct companies merge; too deep and the collapse stops
    saving rows. A sweep without any collapse took 200,000 rows to find 853
    companies; at the right depth 3,000 rows found 2,055.
    """
    shallow, deep = hc.wayback_collapse_depths("jobs.lever.co/*")
    prefix = len("jobs.lever.co") + 2       # `co,lever,jobs)/`
    assert shallow > prefix and deep > shallow


def test_wayback_depths_scale_with_host_length():
    short = hc.wayback_collapse_depths("jobs.lever.co/*")
    long = hc.wayback_collapse_depths("job-boards.eu.greenhouse.io/*")
    assert long[0] > short[0], "a longer host needs a deeper key prefix"


def test_wayback_wildcard_host_is_stripped_for_depth():
    """`*.icims.com/*` must measure icims.com, not the literal asterisk."""
    assert hc.wayback_collapse_depths("*.icims.com/*") == \
           hc.wayback_collapse_depths("icims.com/*")


def _wb_page(urls, resume=None):
    lines = [f"{u} co,x)/{i}" for i, u in enumerate(urls)]
    if resume:
        lines.append(resume)
    return ("\n".join(lines)).encode()


def test_wayback_salvages_a_truncated_page():
    """Wayback truncates large responses routinely. Discarding partial bodies
    loses most of a sweep; text output means every complete line is usable."""
    body = _wb_page(["https://jobs.lever.co/acme/1",
                     "https://jobs.lever.co/beta/2"])
    body += b"\nhttps://jobs.lever.co/trunc"      # half-written final line

    def fetch(url):
        raise http.client.IncompleteRead(body)

    lines, complete = hc._wayback_fetch_lines("u", fetch=fetch, sleep=lambda s: None)
    assert complete is False
    assert len(lines) == 2                        # the partial line is dropped


def test_wayback_query_collects_and_stops_at_end():
    calls = []

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            return _wb_page(["https://jobs.lever.co/acme/1"], resume="KEY2")
        return _wb_page(["https://jobs.lever.co/beta/2"])   # no resume -> done

    slugs, rows = hc.harvest_wayback_query(
        "jobs.lever.co/*", hc.PLATFORM_BY_NAME["lever"].extract, 20,
        fetch=fetch, sleep=lambda s: None, delay=0, log=lambda m: None)
    assert slugs == {"acme", "beta"}
    assert rows == 2
    assert "resumeKey=KEY2" in calls[1]


def test_wayback_query_stops_after_repeated_empty_gains():
    """Paging can run for thousands of rows inside one company's URL space.
    Without a stall cutoff a sweep spends its whole budget going nowhere."""
    def fetch(url):
        return _wb_page(["https://jobs.lever.co/acme/1"], resume="SAME")

    slugs, _ = hc.harvest_wayback_query(
        "jobs.lever.co/*", hc.PLATFORM_BY_NAME["lever"].extract, 20,
        fetch=fetch, sleep=lambda s: None, delay=0, max_pages=100,
        log=lambda m: None)
    assert slugs == {"acme"}


def test_wayback_query_is_bounded_by_max_pages():
    seen = []

    def fetch(url):
        seen.append(url)
        # Every page yields something new, so the stall cutoff never fires.
        return _wb_page([f"https://jobs.lever.co/co{len(seen)}/1"], resume=f"K{len(seen)}")

    hc.harvest_wayback_query(
        "jobs.lever.co/*", hc.PLATFORM_BY_NAME["lever"].extract, 20,
        fetch=fetch, sleep=lambda s: None, delay=0, max_pages=5,
        log=lambda m: None)
    assert len(seen) == 5


def test_wayback_one_failing_query_does_not_kill_the_platform():
    def fetch(url):
        if "job-boards.eu" in url:
            raise urllib.error.URLError("nope")
        return _wb_page(["https://boards.greenhouse.io/acme/jobs/1"])

    r = hc.harvest_platform_wayback(hc.PLATFORM_BY_NAME["greenhouse"],
                                    fetch=fetch, sleep=lambda s: None, delay=0,
                                    log=lambda m: None)
    assert "acme" in r.slugs


def test_lever_is_wayback_only_in_practice():
    """Lever blocks CCBot outright (robots.txt: User-agent CCBot / Disallow /),
    so it left Common Crawl after CC-MAIN-2025-38. Wayback is the only source
    that still carries it, and dropping that query would silently lose ~4,000
    companies."""
    assert "jobs.lever.co/*" in hc.WAYBACK_QUERIES["lever"]


def test_every_platform_has_a_wayback_query_except_icims():
    """iCIMS is deliberately empty: Wayback sorts "com,icims)" (the www
    marketing site) ahead of every "com,icims,<tenant>)", so a wildcard sweep
    pages through tens of thousands of www asset URLs and stalls before
    reaching a tenant. A real run harvested 856 entries and 0 new companies.
    Common Crawl covers the platform, so the query only burned time."""
    for platform in hc.PLATFORMS:
        queries = hc.WAYBACK_QUERIES.get(platform.name)
        assert queries is not None, platform.name
        if platform.name == "icims":
            assert queries == ()
        else:
            assert queries, platform.name


def test_wayback_query_stops_when_its_time_budget_is_spent():
    """One slow query must not consume the whole CI job.

    Wayback can degrade to the point where a single query would run for hours
    at a 180s request timeout with retries. The sweep is cumulative across
    weekly runs, so cutting a slow query short costs a little progress, not
    data.
    """
    clock = {"t": 0.0}
    seen = []

    def fetch(url):
        seen.append(url)
        clock["t"] += 100.0          # each page "takes" 100 seconds
        return _wb_page([f"https://jobs.lever.co/co{len(seen)}/1"],
                        resume=f"K{len(seen)}")

    slugs, _ = hc.harvest_wayback_query(
        "jobs.lever.co/*", hc.PLATFORM_BY_NAME["lever"].extract, 20,
        fetch=fetch, sleep=lambda s: None, delay=0, max_pages=100,
        budget_seconds=250.0, now=lambda: clock["t"], log=lambda m: None)
    # Stops once the budget is exceeded rather than running all 100 pages.
    assert 2 <= len(seen) <= 5
    assert slugs


def test_wayback_budget_never_skips_the_first_page():
    """A budget already blown by an earlier query must still fetch one page,
    otherwise a slow run silently harvests nothing at all."""
    def fetch(url):
        return _wb_page(["https://jobs.lever.co/acme/1"])

    slugs, _ = hc.harvest_wayback_query(
        "jobs.lever.co/*", hc.PLATFORM_BY_NAME["lever"].extract, 20,
        fetch=fetch, sleep=lambda s: None, delay=0,
        budget_seconds=0.0, now=lambda: 10_000.0, log=lambda m: None)
    assert slugs == {"acme"}


# --------------------------------------------------------------------------
# myworkdaysite.com - Workday's other public domain
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    # Host and tenant are the reverse of myworkdayjobs.com: the wdN host is the
    # subdomain and the tenant sits in the path after "recruiting".
    ("https://wd1.myworkdaysite.com/de-DE/recruiting/whitecase/External/job/Berlin/X",
     "whitecase|wd1|external"),
    ("https://wd503.myworkdaysite.com/recruiting/woodcountyhospital/jobs",
     "woodcountyhospital|wd503|jobs"),
    # No "recruiting" segment means the URL names no tenant at all.
    ("https://wd502.myworkdaysite.com/de-CH/jobs/job/X_JR03607", None),
    ("https://aade-wd106.myworkdaysite.com/favicon.ico", None),
    # Staging hosts serve no public board.
    ("https://dr-wd108.myworkdaysite.com/recruiting/acme/External", None),
    ("https://perf-wd102.myworkdaysite.com/recruiting/acme/External", None),
])
def test_myworkdaysite_extraction(url, expected):
    assert hc.PLATFORM_BY_NAME["workday"].extract(url) == expected


def test_myworkdaysite_triple_targets_the_api_ats_service_calls():
    """The point of harvesting this domain is that the triple it yields works
    against myworkdayjobs.com, which is what ats_service scrapes -- verified
    live for both of these. Emitting a myworkdaysite-shaped identifier instead
    would harvest companies the bot cannot use."""
    got = hc.PLATFORM_BY_NAME["workday"].extract(
        "https://wd1.myworkdaysite.com/de-DE/recruiting/whitecase/External/job/x")
    assert got == "whitecase|wd1|external"
    tenant, host, site = got.split("|")
    assert host.startswith("wd") and tenant and site


def test_workday_sweeps_both_domains():
    hosts = {q for q, _ in hc.PLATFORM_BY_NAME["workday"].queries}
    assert hosts == {"myworkdayjobs.com", "myworkdaysite.com"}


def test_greenhouse_sweeps_the_eu_host_in_both_indexes():
    """The EU board host carries companies the US hosts do not. It was in the
    Wayback queries but missing from Common Crawl, so those companies were only
    ever found by one of the two sweeps."""
    cc = {q for q, _ in hc.PLATFORM_BY_NAME["greenhouse"].queries}
    assert "job-boards.eu.greenhouse.io/*" in cc
    assert "job-boards.eu.greenhouse.io/*" in hc.WAYBACK_QUERIES["greenhouse"]


def test_every_wayback_query_host_has_a_matching_extractor():
    """A query without an extractor that recognises its host silently discards
    every row it fetches -- which is exactly what myworkdaysite.com did."""
    guid = "2eba1a0a-d60f-4fd8-95ab-90b070f1d9f2"
    probes = {
        "greenhouse": ("https://{host}/acme/jobs/1", "acme"),
        "lever": ("https://{host}/acme/abc", "acme"),
        "ashby": ("https://{host}/acme/abc", "acme"),
        "icims": ("https://acme.{host}/jobs/1", "acme"),
        "bamboohr": ("https://acme.{host}/careers/list", "acme"),
        # Paylocity's identifier is a GUID, not a name.
        "paylocity": ("https://{host}/recruiting/jobs/All/" + guid + "/X", guid),
        "workable": ("https://{host}/acme/j/abc", "acme"),
        "breezy": ("https://acme.{host}/p/abc", "acme"),
    }
    for name, queries in hc.WAYBACK_QUERIES.items():
        if name == "workday":
            continue          # covered by test_myworkdaysite_extraction
        assert name in probes, f"{name} has queries but no round-trip probe"
        template, expected = probes[name]
        extract = hc.PLATFORM_BY_NAME[name].extract
        for query in queries:
            host = query.split("/")[0].lstrip("*.")
            url = template.format(host=host)
            assert extract(url) == expected, f"{name}: {query} -> {url}"


def test_lever_sweeps_both_regions():
    """jobs.eu.lever.co is a separate region with its own companies, and it
    matters disproportionately: Lever blocks CCBot on jobs.lever.co, so the EU
    host is the only Lever board host Common Crawl still carries."""
    cc = {q for q, _ in hc.PLATFORM_BY_NAME["lever"].queries}
    assert cc == {"jobs.lever.co/*", "jobs.eu.lever.co/*"}
    assert "jobs.eu.lever.co/*" in hc.WAYBACK_QUERIES["lever"]


def test_eu_lever_urls_extract_the_bare_company_slug():
    """The region lives in the host, not the slug, so the identifier stored is
    the same shape as a US one -- ats_service picks the region by trying both
    API hosts."""
    got = hc.PLATFORM_BY_NAME["lever"].extract(
        "https://jobs.eu.lever.co/amicustherapeutics/96625890-d24b-4c98")
    assert got == "amicustherapeutics"


def test_multi_level_subdomain_queries_have_explicit_depths():
    """The depth derivation reads the slug from the path, which is wrong for a
    query whose slug lives behind two subdomain levels.

    Workday's sort key is `com,myworkdayjobs,<wdN-host>,<tenant>)/...`, so the
    derived depth cuts inside the host and collapses every tenant beneath it.
    Measured live: the derived depth 23 returned 7 rows and zero companies,
    while 40 returned 1,500 rows and 643 companies. The failure is silent --
    the sweep reports success having harvested nothing -- so it needs a guard.
    """
    for query in hc.WAYBACK_QUERIES["workday"]:
        depths = hc.wayback_collapse_depths(query)
        assert query in hc.WAYBACK_EXPLICIT_DEPTHS, query
        # Must clear the reversed registered domain plus a host label.
        domain = query.split("/")[0].lstrip("*.")
        assert min(depths) > len(domain) + 8, (query, depths)


def test_explicit_depths_are_ordered_shallow_first():
    for query, depths in hc.WAYBACK_EXPLICIT_DEPTHS.items():
        assert list(depths) == sorted(depths), query
        assert len(depths) >= 2, query


def test_explicit_depths_only_name_real_queries():
    """A typo'd key here silently reverts that query to the broken derivation."""
    every = {q for qs in hc.WAYBACK_QUERIES.values() for q in qs}
    for query in hc.WAYBACK_EXPLICIT_DEPTHS:
        assert query in every, query


@pytest.mark.parametrize("slug", ["harrison&star", "1840&company", "a+b"])
def test_ampersand_and_plus_slugs_are_kept(slug):
    """harrison&star is a live Greenhouse board; harrisonstar is not, so the
    ampersand is part of the identifier rather than noise to strip. The
    extractors take a single path segment, so these cannot be query-string
    spill."""
    assert hc._looks_like_company(slug) is True


def test_ampersand_survives_extraction():
    assert hc.PLATFORM_BY_NAME["greenhouse"].extract(
        "https://job-boards.greenhouse.io/harrison&star/jobs/123") == "harrison&star"


def test_query_string_is_still_not_treated_as_slug():
    """Widening the charset must not let a ?a&b= tail become part of the slug."""
    assert hc.PLATFORM_BY_NAME["greenhouse"].extract(
        "https://job-boards.greenhouse.io/acme?utm=x&gh_src=y") == "acme"


@pytest.mark.parametrize("url", [
    # /wday/ is the API path prefix, not a board.
    "https://acme.wd1.myworkdayjobs.com/wday/cxs/acme/careers/jobs",
    "https://acme.wd1.myworkdayjobs.com/assets/logo.png",
    "https://acme.wd1.myworkdayjobs.com/cdn-cgi/challenge",
    "https://acme.wd1.myworkdayjobs.com/ads.txt",
    "https://acme.wd1.myworkdayjobs.com/app-ads.txt",
    "https://acme.wd1.myworkdayjobs.com/shared-vendors.min.js",
    "https://acme.wd1.myworkdayjobs.com/refreshFacet",
])
def test_workday_asset_paths_are_not_sites(url):
    """These sit where a site name would but never name a board.

    They surfaced once a deeper collapse depth exposed more URL variety per
    tenant: a 200-slug sample of newly harvested Workday entries was 23% live,
    and the misses were almost entirely these. Filtering them took the same
    sample to 71.5% and removed 2,161 of 8,368 harvested entries.
    """
    assert hc.PLATFORM_BY_NAME["workday"].extract(url) is None


def test_workday_site_segment_never_contains_a_dot():
    """Zero of upstream's 12,884 Workday entries have a dot in the site
    position, while every static-asset path does -- so the dot is the reliable
    discriminator, not an ever-growing blacklist."""
    assert hc._workday_triple("acme", "wd1", "foo.txt") is None
    assert hc._workday_triple("acme", "wd1", "a.min.js") is None
    assert hc._workday_triple("acme", "wd1", "careers") == "acme|wd1|careers"


@pytest.mark.parametrize("site", ["01", "a1", "f1", "23", "jobs", "external_careers"])
def test_real_upstream_site_shapes_still_pass(site):
    """Guards the tightening: these are all real upstream site segments, and
    the short numeric ones are the easiest to reject by accident."""
    assert hc._workday_triple("acme", "wd3", site) == f"acme|wd3|{site}"


@pytest.mark.parametrize("slug,ok", [
    # Companies routinely use their own domain as the slug; all three are live.
    ("affinity.co", True), ("akasa.com", True), ("alignment.org", True),
    ("all.health", True), ("adept.ai", True),
    # Version strings look similar but carry no two-letter word. All dead.
    ("2.5", False), ("2021b-49.2", False), ("2022ae-13.3", False),
    ("u.s.a", False),
    # Root files that land in the slug position on every host.
    ("ads.txt", False), ("app-ads.txt", False),
])
def test_dotted_slugs_keep_domains_and_drop_version_strings(slug, ok):
    """Rejecting dots wholesale would delete real companies -- checked against
    live boards -- so the discriminator is a two-letter alphabetic run, which
    every TLD has and no version string does."""
    assert hc._looks_like_company(slug) is ok


@pytest.mark.parametrize("url,expected", [
    # Archived URLs pick up a sentence's full stop. All three bare names are
    # live boards, so rejecting the captured form loses real companies.
    ("https://jobs.ashbyhq.com/camber./abc", "camber"),
    ("https://jobs.ashbyhq.com/inherent./abc", "inherent"),
    ("https://jobs.lever.co/dnb./abc", "dnb"),
    # Only trailing characters are trimmed; a domain-style slug is untouched.
    ("https://jobs.ashbyhq.com/affinity.co/abc", "affinity.co"),
])
def test_trailing_punctuation_is_trimmed_not_rejected(url, expected):
    platform = "lever" if "lever" in url else "ashby"
    assert hc.PLATFORM_BY_NAME[platform].extract(url) == expected


def test_ashby_root_uuid_noise_is_dropped():
    """1,301 of 6,304 harvested Ashby entries were root.<uuid> and similar."""
    assert hc.PLATFORM_BY_NAME["ashby"].extract(
        "https://jobs.ashbyhq.com/root.00598075_84d8_48d8_bad7_5db9ef0b9e4b/x") is None


@pytest.mark.parametrize("url", [
    # www4.icims.com is real infrastructure and was harvested as a company
    # called "www4" -- the bare-name check missed every numbered variant.
    "https://www4.icims.com/jobs/1",
    "https://www.icims.com/",
    "https://api2.bamboohr.com/x",
    "https://staging.icims.com/x",
    "https://cdn3.bamboohr.com/x",
])
def test_numbered_infrastructure_subdomains_are_not_companies(url):
    platform = "icims" if "icims" in url else "bamboohr"
    assert hc.PLATFORM_BY_NAME[platform].extract(url) is None


def test_real_tenants_survive_the_infrastructure_filter():
    """Guards the widened filter: these must not be caught by it."""
    assert hc.PLATFORM_BY_NAME["icims"].extract(
        "https://careers-gbrx.icims.com/jobs/1") == "gbrx"
    assert hc.PLATFORM_BY_NAME["bamboohr"].extract(
        "https://acme.bamboohr.com/careers/list") == "acme"
    # A company whose name merely starts with an infrastructure word is fine.
    assert hc.PLATFORM_BY_NAME["bamboohr"].extract(
        "https://appleseed.bamboohr.com/careers/list") == "appleseed"


# --------------------------------------------------------------------------
# Crawl-list cache
# --------------------------------------------------------------------------

def _collinfo(*ids):
    return json.dumps([{"id": i} for i in ids]).encode()


def test_crawl_list_is_cached_on_success(tmp_path):
    cache = tmp_path / "_crawls.json"
    got = hc.latest_crawls(2, fetch=lambda u: _collinfo("CC-MAIN-2026-34",
                                                        "CC-MAIN-2026-30"),
                           sleep=lambda s: None, cache_path=cache)
    assert got == ["CC-MAIN-2026-34", "CC-MAIN-2026-30"]
    assert json.loads(cache.read_text()) == ["CC-MAIN-2026-34", "CC-MAIN-2026-30"]


def test_cached_list_is_used_when_collinfo_fails(tmp_path):
    """collinfo.json is genuinely unreliable -- two requests seconds apart
    returned 200 in 0.23s and then timed out. Losing it drops the whole Common
    Crawl half of a run for one line on stderr, and crawl ids are immutable
    once minted, so a stale list still names real crawls."""
    cache = tmp_path / "_crawls.json"
    cache.write_text(json.dumps(["CC-MAIN-2026-34", "CC-MAIN-2026-30"]))

    def fetch(url):
        raise urllib.error.URLError("timed out")

    got = hc.latest_crawls(1, fetch=fetch, sleep=lambda s: None, cache_path=cache)
    assert got == ["CC-MAIN-2026-34"]


def test_failure_with_no_cache_still_raises(tmp_path):
    """Falling back to nothing must stay an error, so the caller can decide to
    carry on with Wayback alone rather than silently harvesting zero."""
    def fetch(url):
        raise urllib.error.URLError("timed out")

    with pytest.raises(hc.HarvestError):
        hc.latest_crawls(1, fetch=fetch, sleep=lambda s: None,
                         cache_path=tmp_path / "absent.json")


def test_corrupt_cache_is_ignored(tmp_path):
    cache = tmp_path / "_crawls.json"
    cache.write_text("{ not json")

    def fetch(url):
        raise urllib.error.URLError("timed out")

    with pytest.raises(hc.HarvestError):
        hc.latest_crawls(1, fetch=fetch, sleep=lambda s: None, cache_path=cache)


def test_cache_rejects_entries_that_are_not_crawl_ids(tmp_path):
    """A cache is read back as crawl ids and pasted into request URLs, so it
    must not carry arbitrary strings."""
    cache = tmp_path / "_crawls.json"
    cache.write_text(json.dumps(["CC-MAIN-2026-34", "../../etc", "nonsense"]))

    def fetch(url):
        raise urllib.error.URLError("timed out")

    assert hc.latest_crawls(5, fetch=fetch, sleep=lambda s: None,
                            cache_path=cache) == ["CC-MAIN-2026-34"]


def test_cache_write_failure_does_not_break_a_harvest(tmp_path):
    unwritable = tmp_path / "afile" / "_crawls.json"
    (tmp_path / "afile").write_text("not a directory")
    got = hc.latest_crawls(1, fetch=lambda u: _collinfo("CC-MAIN-2026-34"),
                           sleep=lambda s: None, cache_path=unwritable)
    assert got == ["CC-MAIN-2026-34"]


# --------------------------------------------------------------------------
# Paylocity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://recruiting.paylocity.com/recruiting/jobs/All/"
     "2eba1a0a-d60f-4fd8-95ab-90b070f1d9f2/National-Registry-of-EMTs",
     "2eba1a0a-d60f-4fd8-95ab-90b070f1d9f2"),
    # The archived path casing varies; the stored form is lowercase.
    ("https://recruiting.paylocity.com/Recruiting/Jobs/All/"
     "2EBA1A0A-D60F-4FD8-95AB-90B070F1D9F2/X",
     "2eba1a0a-d60f-4fd8-95ab-90b070f1d9f2"),
    ("https://recruiting.paylocity.com/ads.txt", None),
    ("https://recruiting.paylocity.com/Recruiting/Content/citrus", None),
    ("https://recruiting.paylocity.com/recruiting/jobs/All/not-a-guid/X", None),
])
def test_paylocity_extracts_the_guid(url, expected):
    """Paylocity identifies a company by GUID rather than a name slug."""
    assert hc.PLATFORM_BY_NAME["paylocity"].extract(url) == expected


def test_paylocity_has_explicit_collapse_depths():
    """Its identifier sits behind a deep fixed path
    (`/recruiting/jobs/all/<guid>`), so the derived depth collapses every
    /recruiting/* URL into one group. Measured: derived depth 30 returned 0
    guids, depth 52 returned 775 from 900 rows."""
    query = "recruiting.paylocity.com/*"
    assert query in hc.WAYBACK_EXPLICIT_DEPTHS
    depths = hc.wayback_collapse_depths(query)
    # Must clear "com,paylocity,recruiting)/recruiting/jobs/all/".
    assert min(depths) >= 46, depths


# --------------------------------------------------------------------------
# Retroactive prune
# --------------------------------------------------------------------------

def test_prune_rewrites_rather_than_drops_when_rules_changed(tmp_path):
    """Harvest output is cumulative and published, so a slug written by an
    older revision of the filters stays forever. "al-" predates trailing
    punctuation being trimmed; a fresh harvest of the same capture now yields
    "al", so prune should converge on that rather than discard the company.
    """
    hc.write_slugs(tmp_path / "lever.json", {"acme", "al-", "beta"})
    report = hc.prune_existing(tmp_path, [hc.PLATFORM_BY_NAME["lever"]],
                               log=lambda m: None)
    kept = hc.load_existing(tmp_path / "lever.json")
    assert kept == {"acme", "al", "beta"}
    assert report["lever"]["rewritten"] == 1
    assert report["lever"]["dropped"] == 0


def test_prune_drops_what_the_rules_now_reject(tmp_path):
    hc.write_slugs(tmp_path / "workday.json",
                   {"acme|wd1|careers", "acme|wd1|assets", "acme|wd1|ads.txt"})
    hc.prune_existing(tmp_path, [hc.PLATFORM_BY_NAME["workday"]],
                      log=lambda m: None)
    assert hc.load_existing(tmp_path / "workday.json") == {"acme|wd1|careers"}


def test_prune_never_removes_what_a_fresh_harvest_would_keep(tmp_path):
    """The safety property: pruning uses the same extractor as harvesting, so
    it cannot disagree with it."""
    good = {"acme", "10up-2", "affinity.co", "harrison&star", "0x"}
    hc.write_slugs(tmp_path / "greenhouse.json", good)
    hc.prune_existing(tmp_path, [hc.PLATFORM_BY_NAME["greenhouse"]],
                      log=lambda m: None)
    assert hc.load_existing(tmp_path / "greenhouse.json") == good


def test_prune_leaves_a_clean_file_untouched(tmp_path):
    path = tmp_path / "bamboohr.json"
    hc.write_slugs(path, {"acme", "beta"})
    stamp = path.read_bytes()
    hc.prune_existing(tmp_path, [hc.PLATFORM_BY_NAME["bamboohr"]],
                      log=lambda m: None)
    assert path.read_bytes() == stamp


def test_prune_ignores_absent_files(tmp_path):
    assert hc.prune_existing(tmp_path, list(hc.PLATFORMS), log=lambda m: None) == {}


def test_prune_covers_every_platform():
    """A platform with no identifier probe would silently pass everything."""
    for platform in hc.PLATFORMS:
        assert platform.name in hc._IDENTIFIER_PROBES, platform.name


def test_prune_is_offline(tmp_path, monkeypatch):
    """Pruning must never touch an index -- it is a rules replay, and liveness
    is the dead-slug machinery's business, not its own."""
    def explode(url):
        raise AssertionError("prune made a network request")

    monkeypatch.setattr(hc, "_http_get", explode)
    hc.write_slugs(tmp_path / "lever.json", {"acme", "al-"})
    hc.prune_existing(tmp_path, [hc.PLATFORM_BY_NAME["lever"]], log=lambda m: None)


# --------------------------------------------------------------------------
# Common Crawl bulk index (data.commoncrawl.org)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("query,match,expected", [
    ("job-boards.greenhouse.io/*", "prefix", "io,greenhouse,job-boards)/"),
    ("jobs.lever.co/*", "prefix", "co,lever,jobs)/"),
    ("recruiting.paylocity.com/*", "prefix", "com,paylocity,recruiting)/"),
    # A domain match has to reach subdomains, so it stops at the comma:
    # "com,myworkdayjobs," matches com,myworkdayjobs,acme) and every tenant.
    ("myworkdayjobs.com", "domain", "com,myworkdayjobs,"),
    ("bamboohr.com", "domain", "com,bamboohr,"),
    ("*.icims.com/*", "prefix", "com,icims)/"),
])
def test_surt_prefix_translation(query, match, expected):
    assert hc._surt_prefix(query, match) == expected


def _cluster(lines):
    return ("\n".join(lines) + "\n").encode()


def test_bulk_find_blocks_selects_the_matching_run_plus_its_boundaries():
    """Every matching block, plus one either side.

    The neighbours are required, not incidental: cluster.idx keys a block by
    its first entry, so the range routinely starts inside the preceding block
    and ends inside the following one. Reading only the exactly-matching run
    cost 285 of 2,708 Ashby companies on CC-MAIN-2026-34.
    """
    body = _cluster([
        "aa,partial)/x 2026\tcdx-SKIPPED.gz\t0\t1\t0",
        "co,aaa)/x 2026\tcdx-00000.gz\t0\t100\t1",
        "io,greenhouse,job-boards)/a 2026\tcdx-00001.gz\t100\t200\t2",
        "io,greenhouse,job-boards)/m 2026\tcdx-00001.gz\t300\t250\t3",
        "io,zzz)/x 2026\tcdx-00002.gz\t550\t100\t4",
    ])
    blocks = hc.bulk_find_blocks(
        "CC-MAIN-2026-34", "io,greenhouse,job-boards)/",
        fetch_range=lambda u, a, b: body, fetch_size=lambda u: len(body),
        probe_bytes=10 ** 9)
    assert blocks == [
        ("cdx-00000.gz", 0, 100),      # boundary before
        ("cdx-00001.gz", 100, 200),
        ("cdx-00001.gz", 300, 250),
        ("cdx-00002.gz", 550, 100),    # boundary after
    ]


def test_bulk_find_blocks_skips_the_partial_first_line():
    """A byte-range read almost always starts mid-line, so the first line of a
    window is garbage and must not be parsed as an entry."""
    body = _cluster([
        "ds)/truncated-garbage 2026\tcdx-BAD.gz\t0\t1\t0",
        "io,greenhouse,job-boards)/a 2026\tcdx-00001.gz\t100\t200\t2",
    ])
    blocks = hc.bulk_find_blocks(
        "CC-MAIN-2026-34", "io,greenhouse,job-boards)/",
        fetch_range=lambda u, a, b: body, fetch_size=lambda u: len(body),
        probe_bytes=10 ** 9)
    assert blocks == [("cdx-00001.gz", 100, 200)]


def test_bulk_find_blocks_reports_an_unreachable_cluster_index():
    def boom(url):
        raise urllib.error.URLError("down")

    with pytest.raises(hc.HarvestError):
        hc.bulk_find_blocks("CC-MAIN-2026-34", "co,lever,jobs)/",
                            fetch_range=lambda u, a, b: b"", fetch_size=boom)


def _gz(text):
    import gzip as _gzip, io as _io
    buf = _io.BytesIO()
    with _gzip.GzipFile(fileobj=buf, mode="wb") as fh:
        fh.write(text.encode())
    return buf.getvalue()


def test_bulk_block_is_gunzipped_and_parsed():
    line = ('co,lever,jobs)/acme/1 20260101 '
            '{"url": "https://jobs.lever.co/acme/1", "status": "200"}')
    text = hc.bulk_fetch_block("CC-MAIN-2026-34", "cdx-0.gz", 0, 10,
                               fetch_range=lambda u, a, b: _gz(line))
    assert list(hc.iter_bulk_urls(text, "co,lever,jobs)/")) == \
        ["https://jobs.lever.co/acme/1"]


def test_bulk_keeps_redirects_and_drops_client_errors():
    """A redirect is not evidence the board is gone.

    Greenhouse answers 302 for tagged and retired job URLs while the company is
    still hiring. Filtering to 200 dropped 911 companies in one crawl, 38.6% of
    them live -- conviva surfaced it: eight captures, all 302, a live board,
    missing from the harvest entirely. 4xx stays out; none of that bucket
    resolved."""
    lines = "\n".join([
        'co,lever,jobs)/a 1 {"url": "https://jobs.lever.co/a/1", "status": "200"}',
        'co,lever,jobs)/b 1 {"url": "https://jobs.lever.co/b/1", "status": "404"}',
        'co,lever,jobs)/c 1 {"url": "https://jobs.lever.co/c/1", "status": "302"}',
        'co,lever,jobs)/d 1 {"url": "https://jobs.lever.co/d/1", "status": "500"}',
    ])
    assert list(hc.iter_bulk_urls(lines, "co,lever,jobs)/")) == \
        ["https://jobs.lever.co/a/1", "https://jobs.lever.co/c/1"]


def test_bulk_skips_malformed_lines():
    lines = "\n".join([
        'co,lever,jobs)/a 1 {"url": "https://jobs.lever.co/a/1", "status": "200"}',
        'co,lever,jobs)/b 1 {not json',
        'co,lever,jobs)/c 1',
        'co,other)/x 1 {"url": "https://x/", "status": "200"}',
        'co,lever,jobs)/d 1 {"url": "https://jobs.lever.co/d/1", "status": "200"}',
    ])
    got = list(hc.iter_bulk_urls(lines, "co,lever,jobs)/"))
    assert got == ["https://jobs.lever.co/a/1", "https://jobs.lever.co/d/1"]


def test_bulk_corrupt_block_raises_rather_than_returning_junk():
    with pytest.raises(hc.HarvestError):
        hc.bulk_fetch_block("CC-MAIN-2026-34", "cdx-0.gz", 0, 10,
                            fetch_range=lambda u, a, b: b"not gzip at all")


def test_bulk_one_bad_block_does_not_lose_the_platform():
    good = _gz('co,lever,jobs)/acme/1 1 '
               '{"url": "https://jobs.lever.co/acme/1", "status": "200"}')
    body = _cluster([
        # bulk_find_blocks discards the first line of a window by design: a
        # byte-range read starts mid-line. Without a throwaway here the fixture
        # would only ever offer one block.
        "co,lever,jobs)/partial 1\tcdx-SKIPPED.gz\t0\t1\t0",
        "co,lever,jobs)/a 1\tcdx-00000.gz\t0\t10\t1",
        "co,lever,jobs)/b 1\tcdx-00001.gz\t10\t10\t2",
    ])
    calls = {"n": 0}

    def fetch_range(url, start, end):
        if "cluster.idx" in url:
            return body
        calls["n"] += 1
        if calls["n"] == 1:
            return b"corrupt"
        return good

    r = hc.harvest_platform_bulk(
        hc.PLATFORM_BY_NAME["lever"], "CC-MAIN-2026-34",
        fetch_range=fetch_range, fetch_size=lambda u: len(body),
        log=lambda m: None)
    assert r.slugs == {"acme"}
    assert r.errors


def test_crawl_discovery_uses_the_bulk_host_not_the_api():
    """collinfo.json is the only part of a bulk run that needs the query
    service, and it is exactly the piece that goes down. Probing cluster.idx
    settles which crawls exist directly: a real one answers 206, a
    nonexistent one 404."""
    real = {"CC-MAIN-2026-34", "CC-MAIN-2026-30"}

    def head(url):
        return 206 if any(c in url for c in real) else 404

    got = hc.discover_crawls_bulk([2026], head=head, workers=2)
    assert got == ["CC-MAIN-2026-34", "CC-MAIN-2026-30"]


def test_crawl_discovery_sorts_newest_first_across_years():
    real = {"CC-MAIN-2025-51", "CC-MAIN-2026-04"}

    def head(url):
        return 206 if any(c in url for c in real) else 404

    got = hc.discover_crawls_bulk([2025, 2026], head=head, workers=2)
    assert got == ["CC-MAIN-2026-04", "CC-MAIN-2025-51"]


def test_probe_failure_is_not_read_as_absent():
    """A network error while probing means unknown, not 'this crawl does not
    exist' -- treating it as absence would silently shrink the sweep."""
    def head(url):
        raise urllib.error.URLError("flaky")

    assert hc.discover_crawls_bulk([2026], head=head, workers=2) == []


def test_latest_crawls_falls_back_to_discovery(tmp_path, monkeypatch):
    """Listing down and no cache: discovery is the last resort before failing."""
    def dead(url):
        raise urllib.error.URLError("collinfo down")

    monkeypatch.setattr(hc, "discover_crawls_bulk",
                        lambda years, **kw: ["CC-MAIN-2026-34", "CC-MAIN-2026-30"])
    cache = tmp_path / "_crawls.json"
    got = hc.latest_crawls(1, fetch=dead, sleep=lambda s: None,
                           cache_path=cache, discover_years=[2026])
    assert got == ["CC-MAIN-2026-34"]
    # and the discovery is cached so the next run does not repeat it
    assert hc._load_crawl_cache(cache) == ["CC-MAIN-2026-34", "CC-MAIN-2026-30"]


def test_discovery_is_not_attempted_when_not_asked_for(tmp_path, monkeypatch):
    def dead(url):
        raise urllib.error.URLError("collinfo down")

    monkeypatch.setattr(hc, "discover_crawls_bulk",
                        lambda *a, **k: pytest.fail("should not probe"))
    with pytest.raises(hc.HarvestError):
        hc.latest_crawls(1, fetch=dead, sleep=lambda s: None,
                         cache_path=tmp_path / "absent.json", discover_years=None)


def test_bulk_block_retries_a_throttled_read():
    """data.commoncrawl.org throttles sustained reads: a 12-crawl sweep lost 11
    blocks to 503s. Unlike a paged API there is no later request that covers
    the same ground, so a dropped block is simply a hole of ~170 companies."""
    good = _gz('co,lever,jobs)/acme/1 1 '
               '{"url": "https://jobs.lever.co/acme/1", "status": "200"}')
    calls = []

    def fetch_range(url, start, end):
        calls.append(url)
        if len(calls) < 3:
            raise _http_error(503)
        return good

    slept = []
    text = hc.bulk_fetch_block("CC-MAIN-2026-34", "cdx-0.gz", 0, 10,
                               fetch_range=fetch_range, sleep=slept.append)
    assert "acme" in text
    assert len(calls) == 3 and slept == [2.0, 4.0]


def test_bulk_block_retries_are_bounded():
    calls = []

    def fetch_range(url, start, end):
        calls.append(url)
        raise _http_error(503)

    with pytest.raises(urllib.error.HTTPError):
        hc.bulk_fetch_block("CC-MAIN-2026-34", "cdx-0.gz", 0, 10,
                            fetch_range=fetch_range, sleep=lambda s: None)
    assert len(calls) == hc.MAX_RETRIES


def test_bulk_block_does_not_retry_a_404():
    """A missing shard is not going to appear on the second ask."""
    calls = []

    def fetch_range(url, start, end):
        calls.append(url)
        raise _http_error(404)

    with pytest.raises(urllib.error.HTTPError):
        hc.bulk_fetch_block("CC-MAIN-2026-34", "cdx-0.gz", 0, 10,
                            fetch_range=fetch_range, sleep=lambda s: None)
    assert len(calls) == 1


def test_bulk_includes_the_block_before_the_first_match():
    """cluster.idx names each block by its *first* key, so a prefix beginning
    partway through a block leaves that block's key sorting below it. Skipping
    it silently drops the start of the range."""
    body = _cluster([
        "aa,skip)/x 1\tcdx-PARTIAL.gz\t0\t1\t0",
        # This block starts before the prefix but can still contain it.
        "co,lever,job)/z 1\tcdx-BEFORE.gz\t10\t10\t1",
        "co,lever,jobs)/a 1\tcdx-MATCH.gz\t20\t10\t2",
        "co,zzz)/x 1\tcdx-AFTER.gz\t30\t10\t3",
    ])
    blocks = hc.bulk_find_blocks(
        "CC-MAIN-2026-34", "co,lever,jobs)/",
        fetch_range=lambda u, a, b: body, fetch_size=lambda u: len(body),
        probe_bytes=10 ** 9)
    names = [name for name, _, _ in blocks]
    assert "cdx-BEFORE.gz" in names, "boundary block before the match was dropped"
    assert "cdx-MATCH.gz" in names


def test_bulk_includes_the_block_after_the_last_match():
    """Mirror image: the tail of the range can share the following block."""
    body = _cluster([
        "aa,skip)/x 1\tcdx-PARTIAL.gz\t0\t1\t0",
        "co,lever,jobs)/a 1\tcdx-MATCH.gz\t20\t10\t2",
        "co,zzz)/x 1\tcdx-AFTER.gz\t30\t10\t3",
    ])
    blocks = hc.bulk_find_blocks(
        "CC-MAIN-2026-34", "co,lever,jobs)/",
        fetch_range=lambda u, a, b: body, fetch_size=lambda u: len(body),
        probe_bytes=10 ** 9)
    assert "cdx-AFTER.gz" in [name for name, _, _ in blocks]


def test_bulk_extra_boundary_blocks_do_not_leak_foreign_slugs():
    """Reading neighbouring blocks is only safe because the prefix check runs
    again on their contents."""
    text = "\n".join([
        'co,lever,jobs)/acme/1 1 {"url": "https://jobs.lever.co/acme/1", "status": "200"}',
        'co,zzz,other)/x 1 {"url": "https://other.zzz/notacompany", "status": "200"}',
    ])
    assert list(hc.iter_bulk_urls(text, "co,lever,jobs)/")) == \
        ["https://jobs.lever.co/acme/1"]


@pytest.mark.parametrize("url,expected", [
    # A site name can look exactly like a locale. Both of these are live
    # boards, and consuming every locale-shaped segment walked past them to
    # "job", which is not a board at all.
    ("https://abinbev.wd1.myworkdayjobs.com/en-US/py/job/x", "abinbev|wd1|py"),
    ("https://abinbev.wd1.myworkdayjobs.com/en-US/hn-es/job/x", "abinbev|wd1|hn-es"),
    # A URL carries at most one locale prefix, so one is all that is skipped.
    ("https://3m.wd1.myworkdayjobs.com/en/Search", "3m|wd1|search"),
    ("https://2020companies.wd1.myworkdayjobs.com/en-US/External_Careers/job/X",
     "2020companies|wd1|external_careers"),
    ("https://medtronic.wd1.myworkdayjobs.com/MedtronicCareers/job/x",
     "medtronic|wd1|medtroniccareers"),
])
def test_workday_skips_at_most_one_locale(url, expected):
    assert hc.PLATFORM_BY_NAME["workday"].extract(url) == expected


@pytest.mark.parametrize("slug", [
    "abinbev|wd1|py", "abinbev|wd1|hn-es", "abinbev|wd1|hn_es",
    "2020companies|wd1|external_careers", "acme|wd3|jobs",
    "cromwell|wd3|01", "23andme|wd5|23", "medtronic|wd1|medtroniccareers",
])
def test_prune_round_trip_is_identity_for_valid_workday_ids(slug):
    """Pruning must never change an identifier a fresh harvest would produce.

    It did: the probe URL carried a locale prefix, so a locale-shaped site was
    skipped along with it and four live abinbev boards were rewritten to
    abinbev|wd1|job, which is dead. Rewriting to something wrong is worse than
    leaving an entry alone, so the round trip is asserted directly.
    """
    assert hc._current_identifier(hc.PLATFORM_BY_NAME["workday"], slug) == slug


def test_workday_probe_url_has_a_single_path_segment():
    """Anything after the site lets a locale-shaped site be mistaken for a
    locale prefix; with one segment there is nothing to skip to."""
    url = hc._workday_probe_url("acme|wd1|py")
    assert url.endswith("/py")


def test_underscore_hostnames_are_rejected():
    """vhr_fhlbdm|wd5|careers came out of the index but cannot exist: a DNS
    label has no underscore, and the host does not resolve."""
    assert hc._current_identifier(
        hc.PLATFORM_BY_NAME["workday"], "vhr_fhlbdm|wd5|careers") is None


def test_underscores_are_fine_in_a_path_slug():
    """Only hostnames are restricted. Upstream ships live Greenhouse boards
    like 2026_summer_intern_program."""
    assert hc._looks_like_company("2026_summer_intern_program") is True
    assert hc.PLATFORM_BY_NAME["greenhouse"].extract(
        "https://job-boards.greenhouse.io/2026_summer_intern_program/jobs/1"
    ) == "2026_summer_intern_program"


@pytest.mark.parametrize("site", ["job", "details", "login", "en-", "fr-"])
def test_workday_dead_path_components_are_not_sites(site):
    """Each probed across every tenant carrying it and found entirely dead:
    job 0/30 live (62 entries), details 0/25, login 0/22, and truncated
    locales like "en-" 0/24."""
    assert hc._workday_triple("acme", "wd1", site) is None


@pytest.mark.parametrize("site", ["search", "external", "careers", "jobs"])
def test_workday_real_sites_that_look_like_path_components_survive(site):
    """The guard against intuition. "search" looks exactly as much like a URL
    component as "job" does, and is a live board for 26 tenants (83.9%);
    "jobs" plural is real where "job" singular is not. Rejecting on appearance
    rather than measurement would have deleted them.
    """
    assert hc._workday_triple("acme", "wd1", site) == f"acme|wd1|{site}"


@pytest.mark.parametrize("url", [
    "https://wd1.wd1.myworkdayjobs.com/en-US/careers/job/x",
    "https://wd5.wd1.myworkdayjobs.com/simply_careers",
    "https://wd3.wd1.myworkdayjobs.com/corporate_us_theory",
])
def test_workday_host_label_in_the_tenant_slot_is_rejected(url):
    """A URL with no tenant subdomain reads as tenant "wdN", which cannot
    resolve -- there is no wd1.wd1.myworkdayjobs.com.

    Not hypothetical: 6,055 of upstream's 12,884 Workday entries (47%) have
    this shape, and a 50-slug sample of that cohort was entirely dead. Our
    harvest has none, but only because those URLs happen not to have come up.
    """
    assert hc.PLATFORM_BY_NAME["workday"].extract(url) is None


def test_real_tenants_beginning_with_wd_are_kept():
    """The guard matches a bare host label, not any name starting with "wd"."""
    assert hc.PLATFORM_BY_NAME["workday"].extract(
        "https://wdesk.wd1.myworkdayjobs.com/careers") == "wdesk|wd1|careers"
    assert hc.PLATFORM_BY_NAME["workday"].extract(
        "https://wd40company.wd5.myworkdayjobs.com/careers") == "wd40company|wd5|careers"


def test_prune_would_strip_a_host_label_tenant():
    assert hc._current_identifier(
        hc.PLATFORM_BY_NAME["workday"], "wd1|wd1|careers") is None


@pytest.mark.parametrize("status,usable", [
    ("200", True), ("201", True), ("301", True), ("302", True), ("307", True),
    ("404", False), ("410", False), ("403", False), ("500", False),
    ("503", False), ("", False),
])
def test_capture_status_rule(status, usable):
    assert hc.capture_is_usable(status) is usable


def test_both_index_paths_apply_the_same_status_rule():
    """They read the same index; if they disagreed on which captures count,
    the harvest would depend on which path happened to run."""
    api = list(hc.iter_urls(b"\n".join([
        json.dumps({"url": "https://jobs.lever.co/a/1", "status": "200"}).encode(),
        json.dumps({"url": "https://jobs.lever.co/b/1", "status": "302"}).encode(),
        json.dumps({"url": "https://jobs.lever.co/c/1", "status": "404"}).encode(),
    ])))
    bulk = list(hc.iter_bulk_urls("\n".join([
        'co,lever,jobs)/a 1 {"url": "https://jobs.lever.co/a/1", "status": "200"}',
        'co,lever,jobs)/b 1 {"url": "https://jobs.lever.co/b/1", "status": "302"}',
        'co,lever,jobs)/c 1 {"url": "https://jobs.lever.co/c/1", "status": "404"}',
    ]), "co,lever,jobs)/"))
    assert api == bulk


def test_api_query_requests_the_status_field():
    """Filtering moved client-side, so the field has to come back or every
    capture reads as status "" and nothing is harvested at all."""
    url = hc.build_query_url("CC-MAIN-2026-34", "x.com/*", "prefix", page=0)
    assert "fl=url%2Cstatus" in url
    assert "filter=" not in url


def test_progress_is_checkpointed_after_each_crawl(tmp_path, monkeypatch):
    """A run that dies partway through a platform must not lose the platform.

    Output used to be written only once every crawl for a platform had
    finished, so a 12-crawl sweep that stopped during Greenhouse -- silently,
    with no traceback -- left the file untouched and threw away everything it
    had read. The property is that the file is on disk before the run ends.
    """
    out = tmp_path / "out"
    crawls = ["CC-MAIN-2026-34", "CC-MAIN-2026-30", "CC-MAIN-2026-25"]
    on_disk_midway = {}

    def fetch_range(url, start_b, end_b):
        if "cluster.idx" in url:
            crawl = url.split("/collections/")[1].split("/")[0]
            # Record what a crash at this moment would have preserved.
            on_disk_midway[crawl] = hc.load_existing(out / "lever.json")
            return _cluster([
                "co,lever,jobs)/partial 1	cdx-SKIP.gz	0	1	0",
                "co,lever,jobs)/a 1	cdx-0.gz	0	10	1",
                "co,zzz)/x 1	cdx-1.gz	10	10	2",
            ])
        idx = len(on_disk_midway)
        return _gz(f'co,lever,jobs)/co{idx}/1 1 '
                   f'{{"url": "https://jobs.lever.co/co{idx}/1", "status": "200"}}')

    monkeypatch.setattr(hc, "_http_get_range", fetch_range)
    monkeypatch.setattr(hc, "_http_head_size", lambda u: 500)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)

    argv = ["--index", "ccbulk", "--platform", "lever", "--out", str(out),
            "--delay", "0"]
    for crawl in crawls:
        argv += ["--crawl", crawl]
    assert hc.main(argv) == 0

    # By the time the second crawl started, the first crawl's find was durable.
    assert on_disk_midway[crawls[1]], "nothing was written before the run ended"
    assert len(on_disk_midway[crawls[2]]) >= len(on_disk_midway[crawls[1]])


def test_checkpoint_only_ever_adds(tmp_path):
    """The merge is a union and the write is atomic, so an interrupted run
    leaves a smaller harvest, never a corrupt or shrunken one."""
    out = tmp_path / "out"
    hc.write_slugs(out / "lever.json", {"pre-existing"})
    args = argparse.Namespace(dry_run=False, out=out)
    hc._checkpoint(args, "lever", {"pre-existing"}, {"newly-found"})
    assert hc.load_existing(out / "lever.json") == {"pre-existing", "newly-found"}


def test_checkpoint_writes_nothing_on_a_dry_run(tmp_path):
    out = tmp_path / "out"
    args = argparse.Namespace(dry_run=True, out=out)
    hc._checkpoint(args, "lever", set(), {"found"})
    assert not (out / "lever.json").exists()


# --------------------------------------------------------------------------
# Output directory lock
# --------------------------------------------------------------------------

def test_second_harvest_on_the_same_output_is_refused(tmp_path):
    """Two harvests over one directory silently lose each other's finds.

    Both do a read-modify-write against what they loaded at their own start, so
    the later writer discards whatever the earlier added in between. Nothing
    errors; the totals just come out low.
    """
    with hc.output_lock(tmp_path):
        with pytest.raises(hc.HarvestLocked):
            with hc.output_lock(tmp_path):
                pass


def test_lock_is_released_on_the_way_out(tmp_path):
    with hc.output_lock(tmp_path):
        assert (tmp_path / hc.LOCK_NAME).exists()
    assert not (tmp_path / hc.LOCK_NAME).exists()


def test_lock_is_released_even_when_the_harvest_raises(tmp_path):
    with pytest.raises(ValueError):
        with hc.output_lock(tmp_path):
            raise ValueError("harvest blew up")
    assert not (tmp_path / hc.LOCK_NAME).exists()


def test_a_stale_lock_is_taken_over(tmp_path):
    """A process killed mid-run leaves its lock behind. Treating that as fatal
    would block every run after one crash -- which is how the harvest died in
    the first place."""
    (tmp_path / hc.LOCK_NAME).write_text("99999")
    later = time.time() + hc.LOCK_STALE_SECONDS + 1
    with hc.output_lock(tmp_path, now=lambda: later):
        pass


def test_a_fresh_lock_is_respected(tmp_path):
    (tmp_path / hc.LOCK_NAME).write_text("99999")
    with pytest.raises(hc.HarvestLocked):
        with hc.output_lock(tmp_path, now=time.time):
            pass


def test_main_exits_rather_than_racing(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / hc.LOCK_NAME).write_text("99999")
    _offline_bulk(monkeypatch)
    assert hc.main(["--index", "ccbulk", "--crawl", "CC-MAIN-2026-34",
                    "--platform", "lever", "--out", str(out)]) == 2


def test_lock_is_held_during_crawl_resolution(monkeypatch, tmp_path):
    """The lock must cover fetching the crawl listing, not start after it.

    latest_crawls can block for minutes on a slow collinfo. A lock taken
    afterwards leaves exactly that window open -- and it is the window where a
    second run is most likely to be started, because nothing has been logged
    yet and the harvest looks idle. Observed directly: a sweep sat in
    latest_crawls for over 30 seconds holding no lock.
    """
    out = tmp_path / "out"
    held: list[bool] = []

    def slow_listing(*args, **kwargs):
        held.append((out / hc.LOCK_NAME).exists())
        raise hc.HarvestError("listing unavailable")

    monkeypatch.setattr(hc, "latest_crawls", slow_listing)
    _offline_bulk(monkeypatch)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)

    hc.main(["--index", "ccbulk", "--platform", "lever", "--out", str(out)])
    assert held == [True], "crawl resolution ran without the lock held"


def test_prune_holds_the_output_lock(tmp_path, monkeypatch):
    """Pruning is a read-modify-write over the harvest files, so running it
    against a live harvest loses whichever side writes first. The workflow
    runs prune immediately before a collection, which is exactly where an
    overlapping manual run would land."""
    out = tmp_path / "out"
    out.mkdir()
    hc.write_slugs(out / "lever.json", {"acme", "al-"})
    (out / hc.LOCK_NAME).write_text("99999")
    assert hc.main(["--prune", "--out", str(out)]) == 2
    # Refused, so the file is untouched.
    assert hc.load_existing(out / "lever.json") == {"acme", "al-"}


def test_prune_releases_the_lock(tmp_path):
    out = tmp_path / "out"
    hc.write_slugs(out / "lever.json", {"acme"})
    assert hc.main(["--prune", "--out", str(out)]) == 0
    assert not (out / hc.LOCK_NAME).exists()


# --------------------------------------------------------------------------
# The crawl cache must not shrink to whatever one run looked for
# --------------------------------------------------------------------------

def test_saving_a_narrow_discovery_keeps_the_wider_cache(tmp_path):
    """Discovery is normally scoped to a year or two, so a replacing write
    would cut the cache down to that scope. The cache is what a run falls back
    on when collinfo.json is unreachable, so the loss would only surface in
    the situation it exists for."""
    cache = tmp_path / "_crawls.json"
    hc._save_crawl_cache(cache, ["CC-MAIN-2024-10", "CC-MAIN-2025-05",
                                 "CC-MAIN-2026-30"])
    hc._save_crawl_cache(cache, ["CC-MAIN-2026-34", "CC-MAIN-2026-30"])
    assert set(hc._load_crawl_cache(cache)) == {
        "CC-MAIN-2024-10", "CC-MAIN-2025-05", "CC-MAIN-2026-30",
        "CC-MAIN-2026-34"}


def test_the_cache_stays_newest_first(tmp_path):
    """Callers slice the front of this list to pick recent crawls, and recent
    crawls are where the live boards are: 12 recent crawls yielded 90.7% live
    against 11.7% for 2022-23."""
    cache = tmp_path / "_crawls.json"
    hc._save_crawl_cache(cache, ["CC-MAIN-2024-10", "CC-MAIN-2026-34"])
    hc._save_crawl_cache(cache, ["CC-MAIN-2025-05"])
    assert hc._load_crawl_cache(cache) == [
        "CC-MAIN-2026-34", "CC-MAIN-2025-05", "CC-MAIN-2024-10"]
