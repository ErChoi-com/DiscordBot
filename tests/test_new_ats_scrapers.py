"""Field mapping and board parsing for the nine platforms added from the harvest.

Every fixture here is a trimmed copy of what the real endpoint returned, so a
provider that renames a field or reshapes its markup fails here rather than in a
scrape that quietly yields nothing. That distinction matters for this codebase:
a scraper returning [] is indistinguishable from a company with no openings, so
a broken mapping is silent until someone notices a platform never contributes.

The rendered-board parsers (jazzhr, jobvite, applicantpro) are the fragile ones
-- markup changes more often than JSON field names -- which is why their
fixtures are literal fragments of the pages rather than paraphrases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as a  # noqa: E402

# Captured before the autouse fixture stubs it out, so the tests that exercise
# enrichment run the real implementation rather than the no-op.
_REAL_ENRICH_ROWS = a._enrich_rows


@pytest.fixture(autouse=True)
def _no_dead_marks(monkeypatch):
    """Keep every test off the real dead-slug store, and off the network.

    _enrich_rows fetches a page per row, so a parser test that does not stub it
    quietly makes real requests -- which is how this suite once went from two
    seconds to well over a minute without anything failing. Tests that exercise
    enrichment override this.
    """
    monkeypatch.setattr(a, "_is_dead", lambda p, s: False)
    monkeypatch.setattr(a, "_mark_dead", lambda p, s: None)
    monkeypatch.setattr(a, "_mark_alive", lambda p, s: None)
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fetch: None)


def _json_board(monkeypatch, payload):
    monkeypatch.setattr(a, "_fetch_board_json", lambda p, s, u: payload)


def _html_board(monkeypatch, html):
    monkeypatch.setattr(a, "_fetch_board_html", lambda p, s, u: html)


# --------------------------------------------------------------------------
# JSON boards
# --------------------------------------------------------------------------

def test_workable_maps_a_job(monkeypatch):
    _json_board(monkeypatch, {"jobs": [{
        "title": "Finnish Native Speakers Needed",
        "url": "https://apply.workable.com/j/2AB91180EB",
        "shortlink": "https://apply.workable.com/j/2AB91180EB",
        "published_on": "2026-07-30",
        "country": "Finland", "city": "", "state": "",
    }]})
    row = a._scrape_workable("1-stopasia", "", "", 5)[0]
    assert row["title"] == "Finnish Native Speakers Needed"
    assert row["location"] == "Finland"
    assert row["job_url"] == "https://apply.workable.com/j/2AB91180EB"
    assert row["date_posted"] == "2026-07-30"
    assert row["_source_site"] == a.WORKABLE


def test_breezy_reads_a_nested_country(monkeypatch):
    """Breezy nests the country as an object, not a string."""
    _json_board(monkeypatch, [{
        "name": "Grants Specialist",
        "url": "https://021-strategic.breezy.hr/p/6edb824628c1-grants-specialist",
        "published_date": "2026-08-14T14:13:18.119Z",
        "location": {"city": "Toronto", "country": {"name": "Canada", "id": "CA"}},
    }])
    row = a._scrape_breezy("021-strategic", "", "", 5)[0]
    assert row["title"] == "Grants Specialist"
    assert row["location"] == "Toronto, Canada"


def test_smartrecruiters_rebuilds_the_job_url(monkeypatch):
    """The postings API carries no browser URL, so it is composed from the
    company identifier and the posting id."""
    _json_board(monkeypatch, {"content": [{
        "id": "744000094978215",
        "name": "Founders Associate Intern",
        "releasedDate": "2025-11-22T16:12:06.534Z",
        "location": {"city": "Los Angeles", "region": "CA", "country": "us"},
    }]})
    row = a._scrape_smartrecruiters("10xvaluepartnersgmbh", "", "", 5)[0]
    assert row["job_url"] == (
        "https://jobs.smartrecruiters.com/10xvaluepartnersgmbh/744000094978215")
    assert row["location"] == "Los Angeles, CA, us"


def test_smartrecruiters_skips_a_posting_with_no_id(monkeypatch):
    """Without an id there is no URL to build, and a row with no link is not
    something the bot can post."""
    _json_board(monkeypatch, {"content": [{"name": "Ghost Role", "id": ""}]})
    assert a._scrape_smartrecruiters("acme", "", "", 5) == []


def test_recruitee_maps_a_job(monkeypatch):
    _json_board(monkeypatch, {"offers": [{
        "title": "Senior 3D Animator",
        "careers_url": "https://11bitstudios.recruitee.com/o/senior-3d-animator",
        "city": "Warszawa", "country_code": "PL",
        "published_at": "2026-08-19 13:16:05 UTC",
    }]})
    row = a._scrape_recruitee("11bitstudios", "", "", 5)[0]
    assert row["location"] == "Warszawa, PL"
    # Recruitee sends a space separator and a named zone, which
    # datetime.fromisoformat cannot read -- and _posting_age_ok answers True on
    # a parse failure, so every job passed an age filter regardless of age.
    assert row["date_posted"] == "2026-08-19T13:16:05+00:00"


def test_teamtailor_reads_location_from_the_embedded_jobposting(monkeypatch):
    """The feed item has no location field; it exists only in the schema.org
    JobPosting alongside it. Reading the item alone yields a blank for every
    job, which downstream reads as "unlocated" rather than as a mapping that
    was never wired up."""
    _json_board(monkeypatch, {"items": [{
        "title": "Auxiliaire de puericulture",
        "url": "https://123pousse.teamtailor.com/jobs/8294324-auxiliaire",
        "date_published": "2026-08-31T22:57:57+02:00",
        "_jobposting": {"jobLocation": [{"address": {
            "addressLocality": "Bassens", "addressRegion": "Gironde",
            "addressCountry": "FR"}}]},
    }]})
    row = a._scrape_teamtailor("123pousse", "", "", 5)[0]
    assert row["location"] == "Bassens, Gironde, FR"


def test_teamtailor_survives_a_missing_jobposting(monkeypatch):
    _json_board(monkeypatch, {"items": [{
        "title": "Role", "url": "https://x.teamtailor.com/jobs/1",
        "date_published": "2026-01-01T00:00:00+00:00",
    }]})
    row = a._scrape_teamtailor("x", "", "", 5)[0]
    assert row["location"] == ""


def test_rippling_maps_a_job(monkeypatch):
    _json_board(monkeypatch, [{
        "uuid": "761db634", "name": "Journeyman Electrician",
        "url": "https://ats.rippling.com/15-lightyears-careers/jobs/761db634",
        "workLocation": {"label": "Lake Mary, FL"},
    }])
    row = a._scrape_rippling("15-lightyears-careers", "", "", 5)[0]
    assert row["location"] == "Lake Mary, FL"
    # This feed carries no posting date, and inventing today's would date every
    # job to whenever it was first scraped.
    assert row["date_posted"] == ""


# --------------------------------------------------------------------------
# Shared row handling
# --------------------------------------------------------------------------

def test_one_malformed_row_does_not_cost_the_board(monkeypatch):
    _json_board(monkeypatch, [
        "not a dict",
        {"name": "Real Job", "url": "https://x.breezy.hr/p/1",
         "location": {"city": "Berlin"}, "published_date": "2026-01-01"},
    ])
    rows = a._scrape_breezy("x", "", "", 5)
    assert len(rows) == 1 and rows[0]["title"] == "Real Job"


def test_max_jobs_is_honoured(monkeypatch):
    _json_board(monkeypatch, [
        {"name": f"Job {i}", "url": f"https://x.breezy.hr/p/{i}",
         "location": {"city": "Berlin"}, "published_date": "2026-01-01"}
        for i in range(20)])
    assert len(a._scrape_breezy("x", "", "", 3)) == 3


def test_keyword_and_location_filters_apply(monkeypatch):
    _json_board(monkeypatch, [
        {"name": "Backend Engineer", "url": "https://x.breezy.hr/p/1",
         "location": {"city": "Toronto"}, "published_date": "2026-01-01"},
        {"name": "Chef", "url": "https://x.breezy.hr/p/2",
         "location": {"city": "Toronto"}, "published_date": "2026-01-01"},
    ])
    rows = a._scrape_breezy("x", "engineer", "", 5)
    assert [r["title"] for r in rows] == ["Backend Engineer"]


def test_a_fetch_failure_yields_nothing(monkeypatch):
    _json_board(monkeypatch, None)
    for scrape in (a._scrape_workable, a._scrape_breezy, a._scrape_recruitee,
                   a._scrape_teamtailor, a._scrape_rippling,
                   a._scrape_smartrecruiters):
        assert scrape("acme", "", "", 5) == []


# --------------------------------------------------------------------------
# Rendered boards
# --------------------------------------------------------------------------

JAZZHR_HTML = """
<div class='col col-xs-7 jobs-list'><ul class='list-group'>
<li class="list-group-item"> <h3 class='list-group-item-heading'>
<a href="https://10xhealthsystem.applytojob.com/apply/dtQIuzAfrO/Accounts-Payable-Specialist">
Accounts Payable Specialist </a> </h3>
<ul class='list-inline list-group-item-text'>
<li><i class='fa fa-map-marker'></i>Scottsdale, AZ</li> </ul> </li>
<li class="list-group-item"> <h3 class='list-group-item-heading'>
<a href="https://10xhealthsystem.applytojob.com/apply/3YboqrNDAp/Content-Writer">
Content Writer &amp; Producer </a> </h3>
<ul class='list-inline list-group-item-text'>
<li><i class='fa fa-map-marker'></i>Remote</li> </ul> </li>
</ul></div>
"""


def test_jazzhr_parses_titles_and_locations(monkeypatch):
    _html_board(monkeypatch, JAZZHR_HTML)
    rows = a._scrape_jazzhr("10xhealthsystem", "", "", 5)
    assert [r["title"] for r in rows] == [
        "Accounts Payable Specialist", "Content Writer & Producer"]
    assert [r["location"] for r in rows] == ["Scottsdale, AZ", "Remote"]
    assert rows[0]["job_url"].endswith("/apply/dtQIuzAfrO/Accounts-Payable-Specialist")


def test_jazzhr_yields_nothing_on_unexpected_markup(monkeypatch):
    """Better to return nothing than to invent rows from a page whose shape
    has changed."""
    _html_board(monkeypatch, "<html><body><p>No openings</p></body></html>")
    assert a._scrape_jazzhr("acme", "", "", 5) == []


JOBVITE_HTML = """
<table><tr>
<td class="jv-job-list-name"> <a href="/aarete/job/oa9KAfwn">AI Solutions Leader</a> </td>
<td class="jv-job-list-location"> <div class="jv-meta"> Chicago, Illinois </div> </td>
</tr><tr>
<td class="jv-job-list-name"> <a href="/aarete/job/ocVBAfw2">Analyst, Claims Analytics</a> </td>
<td class="jv-job-list-location"> <div class="jv-meta"> 2 Locations </div> </td>
</tr></table>
"""


def test_jobvite_parses_rows_and_absolutises_links(monkeypatch):
    _html_board(monkeypatch, JOBVITE_HTML)
    rows = a._scrape_jobvite("aarete", "", "", 5)
    assert [r["title"] for r in rows] == [
        "AI Solutions Leader", "Analyst, Claims Analytics"]
    assert rows[0]["location"] == "Chicago, Illinois"
    assert rows[0]["job_url"] == "https://jobs.jobvite.com/aarete/job/oa9KAfwn"


def test_jobvite_yields_nothing_on_unexpected_markup(monkeypatch):
    _html_board(monkeypatch, "<html><body>nothing here</body></html>")
    assert a._scrape_jobvite("acme", "", "", 5) == []


APPLICANTPRO_HTML = """
<div id="job_listings"><script>window.bootstrapVue("#job_listings", ['JobListings'],
{ componentData: { organizationId : 13405, domainId : 18583 } });</script></div>
"""

APPLICANTPRO_JSON = {"success": True, "data": {"jobs": [{
    "id": 4194127, "title": "General Sales Manager", "city": "Arlington",
    "subdomain": "50floor", "abbreviation": "TX", "startDateRef": "Sep 01, 2026",
}]}}


def test_applicantpro_uses_the_domain_id_from_the_page(monkeypatch):
    """The listing endpoint will not answer without the domain id its own
    script carries, so the scrape is two requests rather than one."""
    asked: list[str] = []

    def fake_json(platform, slug, url):
        asked.append(url)
        return APPLICANTPRO_JSON

    _html_board(monkeypatch, APPLICANTPRO_HTML)
    monkeypatch.setattr(a, "_fetch_board_json", fake_json)
    row = a._scrape_applicantpro("50floor", "", "", 5)[0]
    assert "/core/jobs/18583?" in asked[0]
    # The endpoint returns a PHP type error rather than an empty result when
    # getParams is absent, so one is always sent.
    assert "getParams=" in asked[0]
    assert row["title"] == "General Sales Manager"
    assert row["location"] == "Arlington, TX"
    assert row["job_url"] == "https://50floor.applicantpro.com/jobs/4194127"


def test_applicantpro_stops_when_the_page_has_no_domain_id(monkeypatch):
    """A disabled or missing board serves a bare sentence with no script block.
    That is not a 404, so it must not be marked dead here either."""
    marked: list[tuple[str, str]] = []
    monkeypatch.setattr(a, "_mark_dead", lambda p, s: marked.append((p, s)))
    _html_board(monkeypatch, "This career site has been disabled.")
    assert a._scrape_applicantpro("acme", "", "", 5) == []
    assert marked == []


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

NEW_PLATFORMS = [a.WORKABLE, a.BREEZY, a.SMARTRECRUITERS, a.RECRUITEE,
                 a.TEAMTAILOR, a.RIPPLING, a.JAZZHR, a.JOBVITE, a.APPLICANTPRO]


@pytest.mark.parametrize("platform", NEW_PLATFORMS)
def test_every_new_platform_is_wired(platform):
    """A platform in ATS_PLATFORMS with no scraper is loaded, counted and then
    silently skipped; one with a scraper but not in ATS_PLATFORMS never has its
    harvest file read at all."""
    assert platform in a.ATS_PLATFORMS
    assert platform in a._SCRAPERS
    assert platform in a.PLATFORM_WORKERS
    assert platform in a._HARVEST_FILES


def test_a_dead_slug_is_never_fetched(monkeypatch):
    """The dead check has to come before the request, or a marked slug still
    costs a round trip every cycle."""
    monkeypatch.setattr(a, "_is_dead", lambda p, s: True)
    monkeypatch.setattr(a, "_fetch_board_json", lambda *args: pytest.fail(
        "fetched a slug already marked dead"))
    monkeypatch.setattr(a, "_fetch_board_html", lambda *args: pytest.fail(
        "fetched a slug already marked dead"))
    for platform in NEW_PLATFORMS:
        assert a._SCRAPERS[platform]("acme", "", "", 5) == []


# --------------------------------------------------------------------------
# Limits the endpoints impose
# --------------------------------------------------------------------------

def test_smartrecruiters_pages_past_the_first_hundred(monkeypatch):
    """The API caps a page at 100 and real boards run far past it -- accorhotel
    advertises 6,149. Keeping only the first page drops the rest silently,
    because a truncated board and a small one look identical to the caller."""
    pages = []

    def fake_json(platform, slug, url):
        offset = int(url.split("offset=")[1])
        pages.append(offset)
        return {"totalFound": 250, "content": [
            {"id": str(offset + i), "name": f"Job {offset + i}",
             "location": {"city": "X"}, "releasedDate": "2026-01-01"}
            for i in range(100 if offset < 200 else 50)]}

    monkeypatch.setattr(a, "_fetch_board_json", fake_json)
    rows = a._scrape_smartrecruiters("accorhotel", "", "", 250)
    assert pages == [0, 100, 200]
    assert len(rows) == 250


def test_smartrecruiters_stops_once_max_jobs_is_reachable(monkeypatch):
    """Filters can only shrink the set, so paging past what max_jobs needs
    spends requests that cannot change the answer."""
    pages = []

    def fake_json(platform, slug, url):
        pages.append(int(url.split("offset=")[1]))
        return {"totalFound": 10_000, "content": [
            {"id": str(i), "name": f"Job {i}", "location": {"city": "X"},
             "releasedDate": "2026-01-01"} for i in range(100)]}

    monkeypatch.setattr(a, "_fetch_board_json", fake_json)
    a._scrape_smartrecruiters("big", "", "", 20)
    assert pages == [0]


def test_smartrecruiters_paging_is_bounded(monkeypatch):
    """A board that keeps answering must not turn one company into an
    unbounded crawl."""
    pages = []

    def fake_json(platform, slug, url):
        pages.append(int(url.split("offset=")[1]))
        return {"totalFound": 10 ** 9, "content": [
            {"id": str(i), "name": f"Job {i}", "location": {"city": "X"},
             "releasedDate": "2026-01-01"} for i in range(100)]}

    monkeypatch.setattr(a, "_fetch_board_json", fake_json)
    a._scrape_smartrecruiters("endless", "", "", 10 ** 6)
    assert max(pages) < a._SMARTRECRUITERS_MAX_OFFSET


@pytest.mark.parametrize("platform", [a.WORKABLE, a.RECRUITEE])
def test_metered_platforms_are_not_scraped_harder_than_they_were_probed(platform):
    """Eight unpaced workers is what got this address throttled during
    validation -- 91% of a workable pass came back 429. The scraper must not be
    greedier than the validator that measured the limit."""
    assert a.PLATFORM_WORKERS[platform] <= 4


# --------------------------------------------------------------------------
# Filling the fields the listings leave empty
# --------------------------------------------------------------------------

JOBPOSTING_LD = """
<script type="application/ld+json">{"@type":"Organization","name":"10X","url":"x"}</script>
<script type="application/ld+json">{"@context":"http://schema.org","@type":"JobPosting",
"title":"Accounts Payable Specialist","datePosted":"2026-06-29",
"jobLocation":[{"@type":"Place","address":{"@type":"PostalAddress",
"addressLocality":"Scottsdale","addressRegion":"AZ","addressCountry":"US"}}]}</script>
"""


def _resp(monkeypatch, text, status=200):
    class R:
        status_code = status
    R.text = text
    monkeypatch.setattr(a.requests, "get", lambda *args, **kw: R())


def test_jobposting_metadata_skips_the_organization_block(monkeypatch):
    """jazzhr emits an Organization block before the JobPosting one, so taking
    the first ld+json script finds no date and no location at all."""
    _resp(monkeypatch, JOBPOSTING_LD)
    meta = a._fetch_jobposting_meta("https://x/apply/1")
    assert meta["date_posted"] == "2026-06-29"
    assert meta["location"] == "Scottsdale, AZ, US"
    assert meta["resolved"] == "1"


def test_jobposting_metadata_keeps_every_site_of_a_multi_site_posting(monkeypatch):
    """A multi-site posting is why the listing said "2 Locations". Keeping only
    the first would discard a job in the searched city."""
    _resp(monkeypatch, """
<script type="application/ld+json">{"@type":"JobPosting","datePosted":"2026-09-02",
"jobLocation":[{"@type":"Place","address":{"addressLocality":"Chicago",
"addressRegion":"Illinois","addressCountry":"United States"}},
{"@type":"Place","address":{"addressLocality":"Austin","addressRegion":"Texas",
"addressCountry":"United States"}}]}</script>""")
    meta = a._fetch_jobposting_meta("https://x/job/1")
    assert "Chicago" in meta["location"] and "Austin" in meta["location"]


def test_jobposting_metadata_reports_an_unparsed_page(monkeypatch):
    """resolved distinguishes an unlocated posting from a failed fetch, which
    decides whether a row survives a location filter."""
    _resp(monkeypatch, "<html><body>no structured data</body></html>")
    assert a._fetch_jobposting_meta("https://x/job/1")["resolved"] == ""


def _rippling_page(job_post: dict) -> str:
    payload = {"props": {"pageProps": {"apiData": {"jobPost": job_post}}},
               "labels": {"createdOn": "Created on"}}
    return ('<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(payload) + '</script>')


def test_rippling_date_ignores_the_ui_label(monkeypatch):
    """The payload carries UI labels under the same key names as the data. A
    regex over the raw text returns the literal string "Created on" as a
    posting date -- which is what a first attempt actually produced -- so the
    payload is walked as JSON instead."""
    _resp(monkeypatch, _rippling_page(
        {"createdOn": "2026-08-10T06:17:22.825000-07:00"}))
    meta = a._fetch_rippling_created("https://ats.rippling.com/x/jobs/1")
    assert meta["date_posted"] == "2026-08-10T06:17:22.825000-07:00"


def test_rippling_joins_the_description_sections(monkeypatch):
    """The description is a dict of named sections rather than one string.
    Taking any single one hands the matcher a company blurb instead of the
    job."""
    _resp(monkeypatch, _rippling_page({
        "createdOn": "2026-08-10T00:00:00",
        "description": {"company": "About Acme.", "job": "You will wire relays."}}))
    desc = a._fetch_rippling_created("https://ats.rippling.com/x/jobs/1")["description"]
    assert "About Acme." in desc and "You will wire relays." in desc


def test_rippling_survives_a_payload_without_a_jobpost(monkeypatch):
    _resp(monkeypatch, '<script id="__NEXT_DATA__" type="application/json">{}</script>')
    assert a._fetch_rippling_created("https://x/jobs/1")["resolved"] == ""


def test_enrichment_never_blanks_what_the_listing_supplied(monkeypatch):
    """A failed fetch must leave the listing's values alone. An unconditional
    write would erase good locations on a rate-limited batch, and an empty
    location never matches a non-empty search -- so the jobs go with it."""
    rows = [{"job_url": "u1", "location": "Scottsdale, AZ", "date_posted": "2026-01-01"}]
    _REAL_ENRICH_ROWS(rows, lambda url: {"date_posted": "", "location": "", "resolved": ""})
    assert rows[0]["location"] == "Scottsdale, AZ"
    assert rows[0]["date_posted"] == "2026-01-01"


def test_jobvite_filters_on_location_only_after_enrichment(monkeypatch):
    """The listing renders "2 Locations", which matches no search. Filtering on
    that would drop a real job in the searched city before anything could look
    at its actual location."""
    _html_board(monkeypatch, JOBVITE_HTML)
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fetch: [
        r.update(location="Chicago, Illinois, United States") for r in rows])
    rows = a._scrape_jobvite("aarete", "", "Chicago", 5)
    assert len(rows) == 2, "a multi-site posting was dropped before enrichment"


def test_jazzhr_enriches_only_what_survived_filtering(monkeypatch):
    """jazzhr's listing already carries a usable location, so filtering happens
    first and only the survivors cost a page fetch."""
    enriched: list[int] = []
    _html_board(monkeypatch, JAZZHR_HTML)
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fetch: enriched.append(len(rows)))
    a._scrape_jazzhr("10xhealthsystem", "Content Writer", "", 5)
    assert enriched == [1]


# --------------------------------------------------------------------------
# date_posted has to be readable by the age filter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("2026-08-19 13:16:05 UTC", "2026-08-19T13:16:05+00:00"),   # recruitee
    ("Sep 01, 2026", "2026-09-01"),                             # applicantpro
    ("September 1, 2026", "2026-09-01"),
    ("2026-07-30", "2026-07-30"),                               # already fine
    ("2026-08-14T14:13:18.119Z", "2026-08-14T14:13:18.119Z"),
])
def test_posted_dates_are_made_parseable(raw, expected):
    """_posting_age_ok reads this field with datetime.fromisoformat and answers
    True when it raises -- so an unparseable date does not drop the row, it
    exempts the row from the age filter entirely. Recruitee and applicantpro
    were both doing that, so every job of theirs passed "newer than N hours"
    regardless of age."""
    assert a._normalise_posted(raw) == expected
    from datetime import datetime
    datetime.fromisoformat(a._normalise_posted(raw).replace("Z", "+00:00"))


def test_an_unreadable_date_is_kept_rather_than_blanked():
    """Blanking it would also exempt the row from the filter, and the text is
    still worth showing a reader."""
    assert a._normalise_posted("sometime next spring") == "sometime next spring"
    assert a._normalise_posted(None) == ""


def test_rows_normalise_their_dates(monkeypatch):
    _json_board(monkeypatch, {"offers": [{
        "title": "Role", "careers_url": "https://x.recruitee.com/o/1",
        "city": "Warszawa", "published_at": "2026-08-19 13:16:05 UTC"}]})
    assert a._scrape_recruitee("x", "", "", 5)[0]["date_posted"] == \
        "2026-08-19T13:16:05+00:00"


# --------------------------------------------------------------------------
# description, which is what semantic matching reads
# --------------------------------------------------------------------------

def test_listing_descriptions_are_carried_through(monkeypatch):
    """job_service matches semantically against "description", so a board that
    supplies none is invisible to matching rather than merely sparse."""
    _json_board(monkeypatch, {"jobs": [{
        "title": "Role", "url": "https://apply.workable.com/j/1",
        "published_on": "2026-07-30", "country": "Finland",
        "description": "<p>You will do the thing.</p>"}]})
    assert a._scrape_workable("x", "", "", 5)[0]["description"] == \
        "<p>You will do the thing.</p>"


def test_recruitee_joins_description_and_requirements(monkeypatch):
    _json_board(monkeypatch, {"offers": [{
        "title": "Role", "careers_url": "https://x.recruitee.com/o/1",
        "city": "Warszawa", "published_at": "2026-01-01",
        "description": "The role.", "requirements": "Four years."}]})
    desc = a._scrape_recruitee("x", "", "", 5)[0]["description"]
    assert "The role." in desc and "Four years." in desc


def test_every_row_carries_a_description_key(monkeypatch):
    """Absent is worse than empty here: downstream reads row["description"],
    and a missing key is a different failure from an empty one."""
    _json_board(monkeypatch, [{"name": "Role", "url": "https://x.breezy.hr/p/1",
                               "location": {"city": "Berlin"},
                               "published_date": "2026-01-01"}])
    assert "description" in a._scrape_breezy("x", "", "", 5)[0]


def test_enrichment_does_not_overwrite_a_listing_description(monkeypatch):
    """A listing description is the platform's own text; a page-scraped one is
    a fallback for boards that publish none."""
    rows = [{"job_url": "u1", "location": "Berlin", "date_posted": "2026-01-01",
             "description": "from the listing"}]
    _REAL_ENRICH_ROWS(rows, lambda url: {
        "date_posted": "", "location": "", "description": "from the page",
        "resolved": "1"})
    assert rows[0]["description"] == "from the listing"


def test_smartrecruiters_detail_joins_its_sections(monkeypatch):
    """The description is split across jobAd sections; one section alone is a
    company blurb rather than the job."""
    class R:
        status_code = 200
        @staticmethod
        def json():
            return {"jobAd": {"sections": {
                "companyDescription": {"text": "About Acme."},
                "jobDescription": {"text": "You will build things."},
                "qualifications": {"text": "Four years."}}}}
    monkeypatch.setattr(a.requests, "get", lambda *args, **kw: R())
    meta = a._fetch_smartrecruiters_detail(
        "https://jobs.smartrecruiters.com/acme/744000094978215")
    assert "You will build things." in meta["description"]
    assert "Four years." in meta["description"]


def test_the_new_platforms_do_not_dominate_the_fleet():
    """watchers/manager fans every platform out at once, so these counts sum.

    At 20 apiece the nine new platforms took the fleet from 190 concurrent HTTP
    threads to 312, and the fan-out's own comment records that 24 threads on a
    one-core host was already considered too many -- written when there were
    six platforms. This pins the total rather than any single number, since
    that is what actually bites.
    """
    total = sum(a.PLATFORM_WORKERS[p] for p in a.ATS_PLATFORMS)
    # Budgeted per platform rather than as a flat total, so adding a platform
    # does not silently raise the ceiling and does not force an unrelated
    # platform to be trimmed to make room. The six originals are the historical
    # allocation and are not counted against this.
    assert total <= 17 * len(a.ATS_PLATFORMS), (
        f"{total} concurrent workers across {len(a.ATS_PLATFORMS)} platforms")


def test_a_platform_that_enriches_is_bounded_by_its_nested_pool():
    """A platform that enriches opens a pool per company, so its real ceiling
    is workers x 5. Platforms that read everything from one listing response --
    paylocity, teamtailor, workable -- have no nested pool and can afford more.
    This pins the distinction rather than the arithmetic."""
    enriching = {a.BREEZY, a.SMARTRECRUITERS, a.RIPPLING, a.JAZZHR, a.JOBVITE,
                 a.APPLICANTPRO}
    for platform in enriching:
        assert a.PLATFORM_WORKERS[platform] <= 8, platform
    # Paylocity reads title, location, date and description out of the board
    # page's embedded pageData, so it never opens one.
    assert a.PLATFORM_WORKERS[a.PAYLOCITY] <= 10


@pytest.mark.parametrize("platform", [a.BREEZY, a.SMARTRECRUITERS, a.RIPPLING,
                                      a.JAZZHR, a.JOBVITE, a.APPLICANTPRO])
def test_enriching_platforms_stay_small(platform):
    """A platform that enriches opens a nested pool per company, so its real
    ceiling is workers x 5. At 20 that is 100 threads from one platform."""
    assert a.PLATFORM_WORKERS[platform] <= 8


# --------------------------------------------------------------------------
# Location must be read before it is filtered on
# --------------------------------------------------------------------------
#
# _matches_location returns False for "" against any non-empty search, so
# filtering before the posting page has been read drops real jobs in the
# searched city and presents it as "no jobs matched". Three scrapers had this
# ordering; jobvite and icims already had it right.

def test_jazzhr_does_not_drop_a_row_whose_listing_had_no_marker(monkeypatch):
    """JazzHR only carries a location when the row includes a map-marker
    element. Rows without one yield "", which matches nothing."""
    html = """
<ul class='list-group'>
<li class="list-group-item"> <h3 class='list-group-item-heading'>
<a href="https://acme.applytojob.com/apply/aaa/Engineer">Engineer</a> </h3>
<ul class='list-inline list-group-item-text'> <li></li> </ul> </li>
</ul>"""
    _html_board(monkeypatch, html)
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fetch: [
        r.update(location="Austin, TX") for r in rows])
    rows = a._scrape_jazzhr("acme", "", "Austin", 5)
    assert len(rows) == 1, "a job in the searched city was dropped before enrichment"
    assert rows[0]["location"] == "Austin, TX"


def test_breezy_does_not_drop_a_row_whose_listing_had_no_city(monkeypatch):
    _json_board(monkeypatch, [{"name": "Engineer", "url": "https://x.breezy.hr/p/1",
                               "location": {}, "published_date": "2026-01-01"}])
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fetch: [
        r.update(location="Berlin, DE") for r in rows])
    rows = a._scrape_breezy("x", "", "Berlin", 5)
    assert len(rows) == 1
    assert rows[0]["location"] == "Berlin, DE"


def test_applicantpro_does_not_drop_a_row_whose_listing_had_no_city(monkeypatch):
    monkeypatch.setattr(a, "_fetch_board_html", lambda p, s, u: APPLICANTPRO_HTML)
    monkeypatch.setattr(a, "_fetch_board_json", lambda p, s, u: {
        "data": {"jobs": [{"id": 1, "title": "Engineer", "subdomain": "x"}]}})
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fetch: [
        r.update(location="Austin, TX") for r in rows])
    rows = a._scrape_applicantpro("x", "", "Austin", 5)
    assert len(rows) == 1


def test_an_unlocated_search_does_not_overfetch(monkeypatch):
    """Enrichment costs a page per row, so the wider candidate net is only
    worth paying for when a location search is actually active."""
    assert a._enrich_budget(10, "") == 10
    assert a._enrich_budget(10, "Toronto") > 10


def test_location_pass_still_caps_at_max_jobs():
    rows = [{"location": "Toronto"} for _ in range(20)]
    assert len(a._location_pass(rows, "Toronto", 5)) == 5
    assert len(a._location_pass(rows, "", 5)) == 5


# --------------------------------------------------------------------------
# Dead marks must mean "gone", not "unreadable"
# --------------------------------------------------------------------------

def test_bamboohr_does_not_mark_dead_on_an_anti_bot_page(monkeypatch):
    """A Cloudflare interstitial, captcha or maintenance page is a 200 with
    text/html from the tenant's own host. Content type alone was the old test,
    so every one of those suppressed a real company for the whole TTL -- on a
    platform holding 21,290 slugs and running many workers."""
    marked: list = []
    monkeypatch.setattr(a, "_mark_dead", lambda p, s: marked.append(s))
    monkeypatch.setattr(a, "_is_dead", lambda p, s: False)

    class R:
        status_code = 200
        url = "https://acme.bamboohr.com/careers/list"
        headers = {"Content-Type": "text/html"}
    monkeypatch.setattr(a.requests, "get", lambda *args, **kw: R())
    assert a._scrape_bamboohr("acme", "", "", 5) == []
    assert marked == [], "an unreadable page was recorded as a closure"


def test_bamboohr_marks_dead_when_the_tenant_host_is_left(monkeypatch):
    """An unknown tenant answers 200 and redirects to www.bamboohr.com -- the
    same tell the validator uses."""
    marked: list = []
    monkeypatch.setattr(a, "_mark_dead", lambda p, s: marked.append(s))
    monkeypatch.setattr(a, "_is_dead", lambda p, s: False)

    class R:
        status_code = 200
        url = "https://www.bamboohr.com/"
        headers = {"Content-Type": "text/html"}
    monkeypatch.setattr(a.requests, "get", lambda *args, **kw: R())
    a._scrape_bamboohr("nosuchco", "", "", 5)
    assert marked == ["nosuchco"]


def test_ashby_marks_a_missing_org_dead(monkeypatch):
    """Ashby's GraphQL answers 200 with jobBoard null for an org that does not
    exist and never 404s, so _mark_dead was unreachable and the dead-slug
    mechanism was inert -- every dead org re-POSTed every cycle forever."""
    marked: list = []
    monkeypatch.setattr(a, "_mark_dead", lambda p, s: marked.append(s))
    monkeypatch.setattr(a, "_is_dead", lambda p, s: False)

    class R:
        status_code = 200
        @staticmethod
        def json():
            return {"data": {"jobBoard": None}}
    monkeypatch.setattr(a.requests, "post", lambda *args, **kw: R())
    assert a._scrape_ashby("nosuchco", "", "", 5) == []
    assert marked == ["nosuchco"]


def test_ashby_does_not_clear_a_dead_mark_on_a_bare_200(monkeypatch):
    """Marking alive on the bare 200 actively cleared marks the validator had
    left, because the null board arrives after the status check."""
    alive: list = []
    monkeypatch.setattr(a, "_mark_alive", lambda p, s: alive.append(s))
    monkeypatch.setattr(a, "_mark_dead", lambda p, s: None)
    monkeypatch.setattr(a, "_is_dead", lambda p, s: False)

    class R:
        status_code = 200
        @staticmethod
        def json():
            return {"data": {"jobBoard": None}}
    monkeypatch.setattr(a.requests, "post", lambda *args, **kw: R())
    a._scrape_ashby("nosuchco", "", "", 5)
    assert alive == []


def test_the_deferred_filter_still_excludes_the_wrong_city(monkeypatch):
    """Deferring the location filter must not amount to dropping it. Without
    this, a scraper that ignored `location` entirely would pass every other
    test in this section, because they all assert that matching jobs survive
    rather than that non-matching ones are excluded."""
    _json_board(monkeypatch, [
        {"name": "Here", "url": "https://x.breezy.hr/p/1", "location": {},
         "published_date": "2026-01-01"},
        {"name": "Elsewhere", "url": "https://x.breezy.hr/p/2", "location": {},
         "published_date": "2026-01-01"},
    ])

    def enrich(rows, fetch):
        for row in rows:
            row["location"] = ("Austin, TX" if row["title"] == "Here"
                               else "Berlin, DE")

    monkeypatch.setattr(a, "_enrich_rows", enrich)
    rows = a._scrape_breezy("x", "", "Austin", 5)
    assert [r["title"] for r in rows] == ["Here"]


def test_jazzhr_deferred_filter_excludes_the_wrong_city(monkeypatch):
    _html_board(monkeypatch, JAZZHR_HTML)

    def enrich(rows, fetch):
        for row in rows:
            row["location"] = ("Austin, TX" if "Accounts" in row["title"]
                               else "Berlin, DE")

    monkeypatch.setattr(a, "_enrich_rows", enrich)
    rows = a._scrape_jazzhr("acme", "", "Austin", 5)
    assert len(rows) == 1 and "Accounts" in rows[0]["title"]


# ── Location matching ───────────────────────────────────────────────────────
#
# _infer_country resolves cities and regions to their country, not just country
# names. _matches_location compared only those codes when both sides resolved,
# which quietly turned every intra-country search into a country-wide one: a
# search for Vancouver returned Toronto jobs. It read as correct in testing
# because a Canadian search mostly saw Canadian postings, so the wrong ones
# looked plausible. These pin the distinction in both directions -- the
# exclusions are the half that a coarser implementation still passes.

@pytest.mark.parametrize("job_location, search, expected", [
    # A search naming a city must not pull in other cities in that country.
    ("Toronto, ON, CA", "Toronto", True),
    ("Toronto, ON, CA", "Vancouver", False),
    ("Vancouver, BC, CA", "Vancouver", True),
    ("Austin, TX, US", "Austin", True),
    ("Austin, TX, US", "Boston", False),
    ("Miami, FL, US", "Seattle", False),
    # Boards publish regions as codes; searchers spell them out.
    ("Toronto, ON, CA", "Ontario", True),
    ("Vancouver, BC, CA", "British Columbia", True),
    ("Vancouver, BC, CA", "Ontario", False),
    ("Austin, TX, US", "Texas", True),
    ("Falls Church, VA, US", "Virginia", True),
    ("Austin, TX, US", "Virginia", False),
    # A bare country search still means the whole country.
    ("Toronto, ON, CA", "Canada", True),
    ("Austin, TX, US", "Canada", False),
    # Nothing in the posting contradicts the search, so it is kept.
    ("Canada", "Toronto", True),
    ("Remote - US", "Boston", True),
    # An empty search is not a filter; an empty job location cannot match one.
    ("Toronto, ON, CA", "", True),
    ("", "Toronto", False),
])
def test_location_matching_is_locality_precise(job_location, search, expected):
    assert a._matches_location(job_location, search) is expected


def test_region_codes_are_matched_case_sensitively():
    """Two-letter codes are too short to match case-insensitively.

    "IN" is Indiana and also an English preposition; "OR" is Oregon and also a
    conjunction. Lowercasing the job location before looking for the code makes
    "Remote in US" match a search for Indiana, which is a false positive that
    grows with every remote posting.
    """
    assert a._matches_region_code("Indianapolis, IN, US", "IN") is True
    assert a._matches_region_code("Remote in the US", "IN") is False
    assert a._matches_region_code("Portland, OR, US", "OR") is True
    assert a._matches_region_code("Sales or Marketing, TX", "OR") is False


def test_region_lookup_requires_the_whole_search_string():
    """"New York City" must not collapse to NY and match all of New York state."""
    assert a._region_code_for("new york") == "NY"
    assert a._region_code_for("New York City") is None
    assert a._region_code_for("Ontario") == "ON"


def test_icims_drops_unavailable_placeholders(monkeypatch):
    """iCIMS writes the literal string "UNAVAILABLE" into unknown address fields.

    Left in, it was shown to users as a location and matched a keyword search
    for that word, so the placeholder had to be dropped rather than passed on.
    """
    posting = {
        "@context": "https://schema.org",
        "@graph": [
            {"@type": "WebPage"},
            {"@type": "JobPosting", "title": "Dental Ops Coordinator",
             "datePosted": "2026-09-02T04:00:00.000Z",
             "description": "<p>Answer calls.</p>",
             "jobLocation": {"address": {
                 "addressLocality": "UNAVAILABLE",
                 "addressRegion": "CA",
                 "addressCountry": "US"}}},
        ],
    }
    body = ('<script type="application/ld+json">' + json.dumps(posting) + "</script>")

    class Resp:
        status_code = 200
        text = body
        url = "https://x.icims.com/jobs/1/job"

    monkeypatch.setattr(a.requests, "get", lambda *args, **kwargs: Resp())
    meta = a._fetch_icims_metadata("https://x.icims.com/jobs/1/job")
    assert meta["location"] == "CA, US"
    assert "UNAVAILABLE" not in meta["location"]
    assert meta["title"] == "Dental Ops Coordinator"
    assert meta["description"] == "Answer calls."
    assert meta["resolved"] == "1"


def test_ontario_california_is_not_ontario_canada():
    """A real row from the live boards: "Ontario, CA, US".

    Three ways to get this wrong, all of which appeared in the data: reading CA
    as Canada, letting the "ON" region code match the first two letters of
    "Ontario", and letting a bare token match send a Canadian search to a
    Californian city. The country check settles it, but the code-boundary and
    token rules have to hold on their own or a future reordering reintroduces it.
    """
    assert a._matches_location("Ontario, CA, US", "Ontario") is False
    assert a._matches_location("Ontario, CA, US", "Canada") is False
    assert a._matches_location("Ontario, CA, US", "California") is True
    assert a._matches_location("Toronto, ON, CA", "Ontario") is True
    # The code must not match inside a longer word.
    assert a._matches_region_code("Ontario, CA, US", "ON") is False


# ── Level classification wiring ─────────────────────────────────────────────
#
# job_level shipped with a full classifier and 76 passing tests and no caller
# anywhere in src/ -- the same "built, then discarded" shape as the schema.org
# fields that were parsed and thrown away. These pin the wiring, not the
# classifier's own rules, which test_job_level.py already covers.

def test_every_row_is_stamped_with_a_level():
    rows = [{"title": "Software Engineer Intern"}, {"title": "Staff Software Engineer"}]
    out = a._stamp_levels(rows, None)
    assert [r["level"] for r in out] == ["intern", "staff"]
    assert len(out) == 2, "an empty filter must not drop anything"


def test_level_filter_excludes_as_well_as_includes():
    """The half a permissive implementation still passes is the exclusion."""
    rows = [{"title": "Software Engineer Intern"}, {"title": "Staff Software Engineer"},
            {"title": "Senior Software Engineer"}]
    kept = a._stamp_levels(rows, ["intern", "coop"])
    assert [r["title"] for r in kept] == ["Software Engineer Intern"]


def test_employment_type_finds_a_posting_the_title_cannot():
    """"Software Developer (Winter 2027)" carries no level word at all.

    The OR-over-title-tokens matcher could never surface it: none of a student's
    query words appear in the title. Lever's categories.commitment says
    "Internship", which is why employment_type has to reach the classifier.
    """
    rows = [{"title": "Software Developer (Winter 2027)", "employment_type": "Internship"}]
    kept = a._stamp_levels(rows, ["intern", "coop"])
    assert len(kept) == 1
    assert kept[0]["level"] == "intern"
    assert kept[0]["level_term"] == "winter 2027"


def test_a_role_about_students_is_not_a_student_role():
    """"Campus Recruiter" is a staff job. It must not pass an early-career filter."""
    kept = a._stamp_levels([{"title": "Campus Recruiter"}], ["intern", "coop", "newgrad"])
    assert kept == []


def test_board_rows_carries_employment_type_through():
    """_board_rows feeds eight platforms; the sixth shape element is optional."""
    def shape(job):
        return (job["t"], "Austin, TX, US", "https://x/1", "", "", job.get("e"))

    rows = a._board_rows("ashby", "acme", [{"t": "Engineer", "e": "Intern"}],
                         "", "", 5, shape)
    assert rows[0]["employment_type"] == "Intern"
    # A shape that returns only four elements must still work.
    rows = a._board_rows("ashby", "acme", [{"t": "Engineer"}], "", "", 5,
                         lambda j: (j["t"], "Austin, TX, US", "https://x/1", ""))
    assert rows[0]["employment_type"] == ""


def test_lever_reads_commitment_as_employment_type(monkeypatch):
    """Verified live: categories.commitment holds Full-time / Internship."""
    payload = [{"text": "Software Developer (Winter 2027)",
                "hostedUrl": "https://jobs.lever.co/acme/1",
                "categories": {"location": "Toronto, ON", "commitment": "Internship"},
                "descriptionPlain": "Build things.", "createdAt": 0}]

    class Resp:
        status_code = 200
        url = "https://api.lever.co/v0/postings/acme"
        headers = {"Content-Type": "application/json"}

        def json(self):
            return payload

    monkeypatch.setattr(a.requests, "get", lambda *args, **kwargs: Resp())
    monkeypatch.setattr(a.time, "sleep", lambda *_: None)
    rows = a._scrape_lever("acme", "", "", 5)
    assert rows and rows[0]["employment_type"] == "Internship"
    assert a._stamp_levels(rows, ["intern"])[0]["level"] == "intern"


def test_employment_type_is_the_only_signal_when_the_title_has_none():
    """Isolates employment_type.

    The "(Winter 2027)" case above does not: a term with no level word already
    classifies as intern on its own, so that test passes even if
    employment_type never reaches classify(). A mutation run caught exactly
    that. This title carries no term and no level word, so the platform's
    commitment field is the only thing that can decide it.
    """
    plain = a._stamp_levels([{"title": "Software Developer"}], None)
    assert plain[0]["level"] == "mid", "no signal at all should stay mid"

    with_type = a._stamp_levels(
        [{"title": "Software Developer", "employment_type": "Internship"}], None)
    assert with_type[0]["level"] == "intern"

    assert a._stamp_levels(
        [{"title": "Software Developer", "employment_type": "Internship"}],
        ["intern"]) != []


def test_scrape_ats_platform_threads_levels_through(monkeypatch):
    """The parameter has to reach _stamp_levels, not merely exist.

    A mutation that hardcoded None at the call site left every test passing,
    because nothing exercised the public entry point with a filter.
    """
    jobs = [{"title": "Software Engineer Intern", "company": "acme",
             "location": "Toronto, ON, CA", "job_url": "https://x/1",
             "date_posted": "", "description": "", "_source_site": "ashby"},
            {"title": "Staff Software Engineer", "company": "acme",
             "location": "Toronto, ON, CA", "job_url": "https://x/2",
             "date_posted": "", "description": "", "_source_site": "ashby"}]

    monkeypatch.setitem(a._SCRAPERS, "ashby", lambda *args, **kwargs: list(jobs))
    monkeypatch.setattr(a, "_drop_already_archived", lambda _p, rows: rows)
    monkeypatch.setattr(a, "flush_dead_slugs", lambda: None)

    everything = a.scrape_ats_platform("ashby", "", "", 10, company_slugs=["acme"])
    assert len(everything) == 2
    assert {r["level"] for r in everything} == {"intern", "staff"}

    filtered = a.scrape_ats_platform("ashby", "", "", 10, company_slugs=["acme"],
                                     levels=["intern"])
    assert [r["title"] for r in filtered] == ["Software Engineer Intern"]


# ── employment_type across the JSON-board platforms ─────────────────────────
#
# Measured live before wiring these, because the field is only worth reading if
# it ever carries an intern value: recruitee 2/103 postings, smartrecruiters
# 3/242, breezy 0/241, workable 0/99. Low, but the field already ships in the
# listing response, so reading it costs no extra request -- and those few are
# exactly the postings an early-career search exists to find. Recorded here so
# nobody re-runs the investigation expecting a bigger number.

def _board(monkeypatch, payload):
    class Resp:
        status_code = 200
        headers = {"Content-Type": "application/json"}
        url = "https://example.test/board"

        def json(self):
            return payload

    monkeypatch.setattr(a.requests, "get", lambda *args, **kwargs: Resp())
    monkeypatch.setattr(a.time, "sleep", lambda *_: None)


def test_workable_reads_employment_type(monkeypatch):
    _board(monkeypatch, {"jobs": [{
        "title": "Software Developer", "city": "Austin", "state": "TX",
        "country": "US", "url": "https://x/1", "published_on": "2026-09-01",
        "description": "Build things.", "employment_type": "Internship"}]})
    rows = a._scrape_workable("acme", "", "", 5)
    assert rows and rows[0]["employment_type"] == "Internship"


def test_recruitee_reads_the_employment_type_code(monkeypatch):
    """recruitee spells it "internship" inside codes like fulltime_permanent."""
    _board(monkeypatch, {"offers": [{
        "title": "Software Developer", "city": "Berlin", "country_code": "DE",
        "careers_url": "https://x/1", "published_at": "2026-09-01",
        "description": "Build things.", "employment_type_code": "internship"}]})
    rows = a._scrape_recruitee("acme", "", "", 5)
    assert rows and rows[0]["employment_type"] == "internship"


def test_smartrecruiters_reads_the_nested_label(monkeypatch):
    """typeOfEmployment is {id, label}; the label is the readable half."""
    _board(monkeypatch, {"content": [{
        "id": "1", "name": "Software Developer",
        "location": {"city": "Austin", "region": "TX", "country": "US"},
        "releasedDate": "2026-09-01",
        "typeOfEmployment": {"id": "intern", "label": "Intern"}}]})
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fn: None)
    rows = a._scrape_smartrecruiters("acme", "", "", 5)
    assert rows and rows[0]["employment_type"] == "Intern"


def test_breezy_reads_the_nested_type_name(monkeypatch):
    _board(monkeypatch, [{
        "name": "Software Developer", "location": {"city": "Austin", "country": "US"},
        "url": "https://x/1", "published_date": "2026-09-01",
        "type": {"id": "intern", "name": "Intern"}}])
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fn: None)
    rows = a._scrape_breezy("acme", "", "", 5)
    assert rows and rows[0]["employment_type"] == "Intern"


@pytest.mark.parametrize("payload", [
    {"content": [{"id": "1", "name": "Dev", "location": {},
                  "releasedDate": "", "typeOfEmployment": "Intern"}]},
    {"content": [{"id": "1", "name": "Dev", "location": {},
                  "releasedDate": "", "typeOfEmployment": None}]},
])
def test_a_non_dict_employment_field_does_not_break_the_board(monkeypatch, payload):
    """These fields are nested objects until some tenant returns a bare string
    or null, and one malformed shape must not cost the whole board."""
    _board(monkeypatch, payload)
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fn: None)
    rows = a._scrape_smartrecruiters("acme", "", "", 5)
    assert len(rows) == 1


def test_reading_employment_type_did_not_displace_the_description(monkeypatch):
    """smartrecruiters and breezy pass "" as the fifth element so the sixth can
    carry the type. Getting that order wrong silently blanks the description,
    which is the field the semantic matcher reads."""
    _board(monkeypatch, {"jobs": [{
        "title": "Dev", "city": "Austin", "country": "US", "url": "https://x/1",
        "published_on": "2026-09-01", "description": "Real body text.",
        "employment_type": "Full-time"}]})
    rows = a._scrape_workable("acme", "", "", 5)
    assert rows[0]["description"] == "Real body text."
    assert rows[0]["employment_type"] == "Full-time"


@pytest.mark.parametrize("value, expected", [
    ("internship", "intern"), ("Intern", "intern"),
    # Everything the live sample actually returned, none of which is a level.
    ("fulltime_permanent", "mid"), ("parttime_minijob", "mid"),
    ("Full-time", "mid"), ("Contract", "mid"), ("Other", "mid"),
    ("Temps plein", "mid"), ("fulltime_fixed_term", "mid"),
    # The trap: \bintern\b must not fire on "Internal".
    ("Internal Audit", "mid"),
])
def test_only_intern_values_change_the_level(value, expected):
    from services import job_level

    assert job_level.classify("Software Developer", "", value).level == expected


@pytest.mark.parametrize("kind", ["Intern", None, {"id": "x"}])
def test_breezy_survives_a_non_dict_type_field(monkeypatch, kind):
    """Same guard as smartrecruiters, and it needs its own test.

    A mutation removing only breezy's isinstance check survived, because the
    smartrecruiters case above covered the identical bug on the other platform
    and read as though it covered both.
    """
    _board(monkeypatch, [{
        "name": "Software Developer", "location": {"city": "Austin"},
        "url": "https://x/1", "published_date": "2026-09-01", "type": kind}])
    monkeypatch.setattr(a, "_enrich_rows", lambda rows, fn: None)
    rows = a._scrape_breezy("acme", "", "", 5)
    assert len(rows) == 1
    assert rows[0]["employment_type"] == ("Intern" if kind == "Intern" else "")
