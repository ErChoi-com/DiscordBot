"""Tests for the Common Crawl ATS slug harvester.

Hermetic per pytest.ini: every CDX response here is a fixture captured from the
live index, and the transport is injected.  The URL samples are real captures
from CC-MAIN-2026-34 and CC-MAIN-2025-38, not invented shapes -- the whole point
of this harvester is parsing someone else's URL conventions correctly, so tests
built on made-up URLs would prove nothing.
"""

from __future__ import annotations

import http.client
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import harvest_ats as hc  # noqa: E402


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
    ("https://acme.wd5.myworkdayjobs.com/en-US/en-GB/Careers", "acme|wd5|careers"),
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
    raw = (b'{"url": "https://a.com/1"}\n'
           b'{"url": "https://a.com/2"\n'      # truncated
           b'\n'
           b'not json at all\n'
           b'{"nourl": true}\n'
           b'{"url": "https://a.com/3"}\n')
    assert list(hc.iter_urls(raw)) == ["https://a.com/1", "https://a.com/3"]


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------

def test_workday_query_uses_domain_match():
    """A host query returns zero pages for Workday - tenants are subdomains."""
    workday = hc.PLATFORM_BY_NAME["workday"]
    assert workday.queries == (("myworkdayjobs.com", "domain"),)
    url = hc.build_query_url("CC-MAIN-2026-34", "myworkdayjobs.com", "domain", page=0)
    assert "matchType=domain" in url


def test_prefix_queries_omit_match_type():
    url = hc.build_query_url("CC-MAIN-2026-34", "jobs.lever.co/*", "prefix", page=0)
    assert "matchType" not in url
    assert "filter=status%3A200" in url
    assert "page=0" in url


def test_page_count_query_drops_field_filters():
    """showNumPages with fl/filter returns a count of the *filtered* set, which
    is not the page count the paged fetch then walks."""
    url = hc.build_query_url("CC-MAIN-2026-34", "x.com/*", "prefix", num_pages=True)
    assert "showNumPages=true" in url
    assert "filter=" not in url
    assert "fl=" not in url


def test_greenhouse_sweeps_both_hosts():
    """Greenhouse migrated hosts; neither alone covers the company set."""
    hosts = {q for q, _ in hc.PLATFORM_BY_NAME["greenhouse"].queries}
    assert hosts == {"boards.greenhouse.io/*", "job-boards.greenhouse.io/*"}


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
        return b"\n".join(json.dumps({"url": u}).encode() for u in pages[target])

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

    rc = hc.main(["--crawl", "CC-MAIN-2026-34", "--platform", "greenhouse",
                  "--out", str(out), "--delay", "0"])
    assert rc == 0
    assert hc.load_existing(out / "greenhouse.json") == {"old-company", "acme", "newco"}


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "out"
    fetch = _fake_index({
        "boards.greenhouse.io/*": ["https://boards.greenhouse.io/acme/jobs/1"],
    })
    monkeypatch.setattr(hc, "_http_get", fetch)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    assert hc.main(["--crawl", "CC-MAIN-2026-34", "--platform", "greenhouse",
                    "--out", str(out), "--dry-run", "--delay", "0"]) == 0
    assert not (out / "greenhouse.json").exists()


def test_total_failure_exits_nonzero(tmp_path, monkeypatch):
    """Every platform yielding nothing means the index or our queries broke.

    Exiting 0 there would let the workflow publish an unchanged file and report
    success forever while harvesting nothing.
    """
    monkeypatch.setattr(hc, "_http_get", _fake_index({}))
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
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
        return b"\n".join(json.dumps({"url": u}).encode()
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
        return json.dumps({"url": "https://acme.bamboohr.com/careers/list"}).encode()

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


def test_every_platform_has_a_wayback_query():
    for platform in hc.PLATFORMS:
        assert hc.WAYBACK_QUERIES.get(platform.name), platform.name


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
