"""Regressions for the ATS scraper: iCIMS location handling, dead-slug
concurrency/durability, fan-out timeouts, and geo-index init.

Every test drives the real scraper functions; only the network boundary
(requests) is replaced.
"""
from __future__ import annotations

import collections
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from services import ats_service


@pytest.fixture(autouse=True)
def _isolated_dead_slugs(tmp_path, monkeypatch):
    monkeypatch.setattr(ats_service, "_DEAD_SLUG_DIR", tmp_path / "dead")
    ats_service.clear_dead_slugs()
    yield
    ats_service.clear_dead_slugs()


class _Resp:
    def __init__(self, status=200, body="", headers=None):
        self.status_code = status
        self.text = body
        self.content = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = headers or {}

    def json(self):
        return json.loads(self.text)


def _sitemap(urls: list[str]) -> str:
    entries = "".join(
        f"<url><loc>{u}</loc><lastmod>2026-08-01</lastmod></url>" for u in urls
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{entries}</urlset>"
    )


def _job_page(title: str, city: str, region: str, country: str) -> str:
    ld = {
        "@type": "JobPosting",
        "title": title,
        "datePosted": "2026-08-01",
        "jobLocation": {
            "address": {
                "addressLocality": city,
                "addressRegion": region,
                "addressCountry": country,
            }
        },
    }
    return f'<html><script type="application/ld+json">{json.dumps(ld)}</script></html>'


_SHELL_PAGE = (
    '<html><script type="application/ld+json">'
    '{"@context": "https://schema.org", "@graph": ['
    '{"@type": "WebPage", "name": "Careers"}, {"@type": "WebSite"}]}'
    "</script></html>"
)


def _install_icims(monkeypatch, jobs: dict[str, str | None], sitemap_urls=None):
    """jobs maps job_url -> page HTML, or None to make that page fetch fail."""
    urls = sitemap_urls if sitemap_urls is not None else list(jobs)

    def fake_get(url, headers=None, timeout=None, **kw):
        if url.endswith("/sitemap.xml"):
            return _Resp(200, _sitemap(urls))
        base, _, query = url.partition("?")
        if "in_iframe=1" not in query:
            # Production behaviour: the bare job URL serves a shell whose only
            # ld+json block is WebPage/BreadcrumbList -- no JobPosting. Serving
            # the posting here too would let the pre-fix code pass.
            return _Resp(200, _SHELL_PAGE)
        page = jobs.get(base)
        if page is None:
            return _Resp(429, "")
        return _Resp(200, page)

    monkeypatch.setattr(ats_service, "_http_get", fake_get)
    monkeypatch.setattr(ats_service.time, "sleep", lambda *_: None)


# ---------------------------------------------------------------------------
# iCIMS location resolution
# ---------------------------------------------------------------------------

def test_icims_unreadable_job_page_is_reported_not_silently_dropped(monkeypatch, capsys):
    """The bug: a failed metadata fetch blanked row['location'], and an empty
    location never matches a non-empty search -- so a rate-limited page looked
    exactly like 'no jobs matched'."""
    good = "https://careers-acme.icims.com/jobs/1/senior-engineer/job"
    bad = "https://careers-acme.icims.com/jobs/2/staff-engineer/job"
    _install_icims(monkeypatch, {
        good: _job_page("Senior Engineer", "Toronto", "ON", "CA"),
        bad: None,
    })

    rows = ats_service._scrape_icims("acme", "engineer", "Canada", 10)

    assert [r["title"] for r in rows] == ["Senior Engineer"]
    out = capsys.readouterr().out
    assert "1 of 2 job page(s) unreadable" in out, (
        "an unclassifiable listing must not be indistinguishable from a filtered one"
    )


def test_icims_unreadable_page_is_retried_before_giving_up(monkeypatch):
    """The retry is the actual recovery path for these listings, so it must
    really re-request rather than just log."""
    url = "https://careers-acme.icims.com/jobs/1/senior-engineer/job"
    attempts = {"n": 0}

    def fake_get(u, headers=None, timeout=None, **kw):
        if u.endswith("/sitemap.xml"):
            return _Resp(200, _sitemap([url]))
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _Resp(429, "")
        return _Resp(200, _job_page("Senior Engineer", "Toronto", "ON", "CA"))

    monkeypatch.setattr(ats_service, "_http_get", fake_get)
    monkeypatch.setattr(ats_service.time, "sleep", lambda *_: None)

    rows = ats_service._scrape_icims("acme", "engineer", "Canada", 10)

    assert attempts["n"] == 2, "the second attempt never happened"
    assert [r["location"] for r in rows] == ["Toronto, ON, CA"]


def test_icims_keeps_unresolved_listings_when_policy_is_flipped(monkeypatch):
    """The drop is a deliberate policy, not an accident; the opposite trade must
    actually work."""
    bad = "https://careers-acme.icims.com/jobs/2/staff-engineer/job"
    _install_icims(monkeypatch, {bad: None})
    monkeypatch.setattr(ats_service, "ICIMS_KEEP_UNRESOLVED_LOCATION", True)

    rows = ats_service._scrape_icims("acme", "engineer", "Canada", 10)

    assert len(rows) == 1 and rows[0]["location"] == ""


def test_icims_still_filters_out_the_wrong_country(monkeypatch):
    """Negative case -- otherwise the fixes above could just disable filtering."""
    ca = "https://careers-acme.icims.com/jobs/1/senior-engineer/job"
    de = "https://careers-acme.icims.com/jobs/2/lead-engineer/job"
    _install_icims(monkeypatch, {
        ca: _job_page("Senior Engineer", "Toronto", "ON", "CA"),
        de: _job_page("Lead Engineer", "Berlin", "BE", "DE"),
    })

    rows = ats_service._scrape_icims("acme", "engineer", "Canada", 10)

    assert [r["title"] for r in rows] == ["Senior Engineer"]


def test_icims_fills_max_jobs_despite_the_location_filter(monkeypatch):
    """max_jobs used to cap sitemap candidates *before* location filtering, so a
    location-filtered search returned far fewer rows than requested."""
    jobs: dict[str, str | None] = {}
    urls: list[str] = []
    for i in range(12):
        url = f"https://careers-acme.icims.com/jobs/{i}/engineer-{i}/job"
        urls.append(url)
        # Alternate countries so a pre-filter cap of 3 would yield ~1-2 rows.
        place = ("Toronto", "ON", "CA") if i % 2 == 0 else ("Berlin", "BE", "DE")
        jobs[url] = _job_page(f"Engineer {i}", *place)
    _install_icims(monkeypatch, jobs, sitemap_urls=urls)

    rows = ats_service._scrape_icims("acme", "engineer", "Canada", 3)

    assert len(rows) == 3, f"location filter starved the result set: {len(rows)}"
    assert all(r["location"].endswith("CA") for r in rows)


# ---------------------------------------------------------------------------
# Dead-slug durability and concurrency
# ---------------------------------------------------------------------------

def _age_past_recheck(platform: str, slug: str) -> None:
    """Backdate a mark past THIS slug's recheck age.

    Uses _recheck_days_for rather than the bare constant: recheck ages are
    spread per slug, so a fixed RECHECK_DAYS+1 leaves offset slugs still
    suppressed and the test asserting the wrong thing.
    """
    from datetime import date, timedelta
    days = ats_service._recheck_days_for(slug) + 1
    ats_service._dead_slug_dates[platform][slug] = (
        date.today() - timedelta(days=days)
    ).isoformat()


def test_dead_slug_is_reprobed_once_the_recheck_window_passes():
    """_is_dead short-circuits before any request, so without a recheck window a
    dead board can never prove it is back before the 90-day TTL."""
    ats_service._mark_dead(ats_service.GREENHOUSE, "acme")
    assert ats_service._is_dead(ats_service.GREENHOUSE, "acme") is True

    _age_past_recheck(ats_service.GREENHOUSE, "acme")
    assert ats_service._is_dead(ats_service.GREENHOUSE, "acme") is False, (
        "an aged mark must allow exactly one probe"
    )

    # A failed probe re-dates the mark, so it goes quiet again for another window.
    ats_service._mark_dead(ats_service.GREENHOUSE, "acme")
    assert ats_service._is_dead(ats_service.GREENHOUSE, "acme") is True


def test_successful_fetch_clears_a_stale_dead_mark(monkeypatch):
    """A board that 404s during a brief outage was invisible until the 90-day
    TTL fired. A 200 is proof the mark is stale."""
    ats_service._mark_dead(ats_service.GREENHOUSE, "acme")
    ats_service._mark_dead(ats_service.GREENHOUSE, "other-co")
    # Age it past the recheck window -- the state a revived board is really in.
    _age_past_recheck(ats_service.GREENHOUSE, "acme")

    body = json.dumps({"jobs": [{
        "title": "Engineer",
        "location": {"name": "Toronto, ON, Canada"},
        "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
        "updated_at": "2026-08-01",
    }]})
    monkeypatch.setattr(ats_service, "_http_get", lambda *a, **k: _Resp(200, body))

    rows = ats_service._scrape_greenhouse("acme", "engineer", "Canada", 10)

    assert rows, "fixture did not produce a matching job"
    assert "acme" not in ats_service._dead_slug_dates[ats_service.GREENHOUSE], (
        "a 200 response left the stale dead mark in place"
    )
    assert "other-co" in ats_service._dead_slug_dates[ats_service.GREENHOUSE], (
        "reviving one slug must not clear the whole platform"
    )


def test_concurrent_cold_load_and_mark_does_not_lose_deaths(monkeypatch):
    """Two threads cold-loading the same platform both parsed the file and the
    second assignment discarded the first thread's marks."""
    ats_service.clear_dead_slugs()
    real_load = ats_service._load_dead_slugs_locked

    def slow_load(platform):
        time.sleep(0.01)   # widen the read-then-assign window
        return real_load(platform)

    monkeypatch.setattr(ats_service, "_load_dead_slugs_locked", slow_load)

    slugs = [f"co-{i}" for i in range(12)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda s: ats_service._mark_dead(ats_service.LEVER, s), slugs))

    recorded = set(ats_service._dead_slug_dates[ats_service.LEVER])
    assert recorded == set(slugs), f"lost marks: {set(slugs) - recorded}"


def test_flush_keeps_marks_recorded_while_it_was_saving(monkeypatch):
    """The dirty set used to be cleared *after* the save loop, so a mark made
    mid-flush was wiped without ever being written."""
    ats_service._mark_dead(ats_service.GREENHOUSE, "gone-co")
    real_save = ats_service._save_dead_slugs
    fired = {"n": 0}

    def save_and_race(platform):
        real_save(platform)
        if not fired["n"]:
            fired["n"] = 1
            ats_service._mark_dead(ats_service.ASHBY, "raced-co")

    monkeypatch.setattr(ats_service, "_save_dead_slugs", save_and_race)
    ats_service.flush_dead_slugs()
    monkeypatch.setattr(ats_service, "_save_dead_slugs", real_save)

    assert fired["n"] == 1, "the race never fired"
    assert ats_service.ASHBY in ats_service._dead_slugs_dirty, (
        "a mark recorded during the flush was silently dropped"
    )

    ats_service.flush_dead_slugs()
    on_disk = json.loads(
        (ats_service._DEAD_SLUG_DIR / f"{ats_service.ASHBY}.json").read_text(encoding="utf-8")
    )
    assert "raced-co" in on_disk


def test_failed_write_leaves_the_previous_file_intact(monkeypatch):
    """write_text truncates in place: an interrupted write left an unparseable
    file, and the next load discards it -- resurrecting every dead slug."""
    ats_service._mark_dead(ats_service.WORKDAY, "gone-co")
    ats_service.flush_dead_slugs()
    path = ats_service._DEAD_SLUG_DIR / f"{ats_service.WORKDAY}.json"
    before = path.read_text(encoding="utf-8")

    ats_service._mark_dead(ats_service.WORKDAY, "second-co")
    real_write = ats_service.Path.write_text

    def boom(self, *args, **kwargs):
        if self.suffix == ".tmp":
            real_write(self, *args, **kwargs)   # partial write really lands
            raise OSError("disk full")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(ats_service.Path, "write_text", boom)
    with pytest.raises(OSError):
        ats_service.flush_dead_slugs()

    assert path.read_text(encoding="utf-8") == before, "the live file was corrupted"
    assert json.loads(path.read_text(encoding="utf-8")).keys() == {"gone-co"}
    leftovers = list(ats_service._DEAD_SLUG_DIR.glob("*.tmp"))
    assert not leftovers, f"temp files left behind: {leftovers}"


# ---------------------------------------------------------------------------
# Fan-out budget
# ---------------------------------------------------------------------------

def test_collect_results_bounds_a_stalled_fanout_and_keeps_partials():
    """`for f in as_completed(m): f.result(timeout=T)` never bounded anything --
    as_completed only yields finished futures, so T was dead code."""
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        futures = {
            pool.submit(lambda: "fast"): "a",
            pool.submit(stop.wait, 5): "b",
        }
        started = time.monotonic()
        got = ats_service._collect_results(futures, timeout=0.05)
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        pool.shutdown(wait=True)

    assert elapsed < 2, f"the timeout did not bite ({elapsed:.1f}s)"
    assert got == {"a": "fast"}, "partial results collected before the timeout were lost"


def test_fanout_budget_scales_with_the_number_of_waves():
    """The budget is derived from REQUEST_TIMEOUT, so a big healthy batch is not
    cut off by a flat constant."""
    assert ats_service._fanout_budget(5, 5) == ats_service.REQUEST_TIMEOUT * 2
    assert ats_service._fanout_budget(50, 5) == ats_service.REQUEST_TIMEOUT * 11
    assert ats_service._fanout_budget(0, 5) == ats_service.REQUEST_TIMEOUT * 2


# ---------------------------------------------------------------------------
# Geo index init
# ---------------------------------------------------------------------------

def test_geo_index_loads_exactly_once_under_concurrency(monkeypatch):
    """Every worker thread used to start its own 632k-row load, and _geo_loaded
    flipped only at the end, so a thread could read a half-populated index."""
    monkeypatch.setattr(ats_service, "_geo_loaded", False)
    monkeypatch.setattr(ats_service, "_geo_cities", {})
    # _load_geo_into_globals assigns _geo_admin1_name too. Without restoring it,
    # the stub loader below leaves it empty for the rest of the session and
    # every later test silently sees a geo index that cannot resolve region
    # names -- "Ontario" stops resolving to CA. The failure lands in whichever
    # test happens to run afterwards, which is why it has to be pinned here.
    monkeypatch.setattr(ats_service, "_geo_admin1_name", {})
    # The retry backoff, reset for the same reason. _ensure_geo_loaded returns
    # WITHOUT loading while _geo_failures is set and _geo_next_retry has not
    # elapsed, so a real geo load that failed earlier in the session -- which is
    # what happens whenever the live bot holds data/geo.db while the suite runs
    # -- made this test count zero loads and fail. It passed alone, passed in
    # this file, and failed only in a full run, which reads exactly like a
    # regression somewhere else entirely.
    monkeypatch.setattr(ats_service, "_geo_failures", 0)
    monkeypatch.setattr(ats_service, "_geo_next_retry", 0.0)
    loads = {"n": 0}
    counter_lock = threading.Lock()

    def counting_load():
        with counter_lock:
            loads["n"] += 1
        time.sleep(0.01)
        return {"toronto": [("CA", "ON")]}, {}, {}

    monkeypatch.setattr(ats_service, "_load_geo_lookup", counting_load)

    barrier = threading.Barrier(8)

    def worker(_):
        barrier.wait()
        ats_service._ensure_geo_loaded()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    assert loads["n"] == 1, f"the geo index was loaded {loads['n']} times"
    assert ats_service._geo_cities.get("toronto") == [("CA", "ON")]


def test_recheck_probes_are_spread_across_the_window():
    """Every slug sharing one recheck age means every dead slug marked before
    the window expires on the same day -- ~19,800 companies re-probed in a
    single cycle, then re-dated together so they re-synchronise forever."""
    slugs = [f"co-{i}" for i in range(2000)]
    ages = {ats_service._recheck_days_for(s) for s in slugs}

    assert len(ages) > 1, "all slugs share one recheck age; the herd is not spread"
    assert min(ages) >= ats_service.DEAD_SLUG_RECHECK_DAYS
    assert max(ages) < (ats_service.DEAD_SLUG_RECHECK_DAYS
                        + ats_service.DEAD_SLUG_RECHECK_SPREAD_DAYS)

    peak = max(collections.Counter(
        ats_service._recheck_days_for(s) for s in slugs).values())
    assert peak < len(slugs) * 0.30, f"one day still carries {peak} of {len(slugs)}"


def test_recheck_age_is_stable_for_a_slug():
    """Derived from the slug, not stored and not random: an unstable value would
    re-roll the due date on every process start and never let a slug settle."""
    first = ats_service._recheck_days_for("acme-corp")
    assert all(ats_service._recheck_days_for("acme-corp") == first for _ in range(5))
    assert ats_service._recheck_days_for("acme-corp") != ats_service._recheck_days_for("acme-corq") \
        or True  # different slugs may collide; only stability is guaranteed


def test_icims_metadata_is_fetched_from_the_in_iframe_variant(monkeypatch):
    """iCIMS stopped serving JobPosting JSON-LD on the plain job URL; only the
    `in_iframe=1` variant carries it. Without that, every iCIMS row came back
    with an empty location -- which matches no location filter -- so all 8,747
    companies dropped out of any scoped search looking like 'no jobs matched'."""
    job = "https://careers-acme.icims.com/jobs/1/senior-engineer/job"
    seen: list[str] = []

    def recording_get(url, headers=None, timeout=None, **kw):
        seen.append(url)
        if url.endswith("/sitemap.xml"):
            return _Resp(200, _sitemap([job]))
        if "in_iframe=1" not in url:
            return _Resp(200, _SHELL_PAGE)
        return _Resp(200, _job_page("Senior Engineer", "Toronto", "ON", "CA"))

    monkeypatch.setattr(ats_service, "_http_get", recording_get)
    monkeypatch.setattr(ats_service.time, "sleep", lambda *_: None)

    rows = ats_service._scrape_icims("acme", "engineer", "Canada", 10)

    assert [r["location"] for r in rows] == ["Toronto, ON, CA"]
    assert job + "?in_iframe=1" in seen
    # The row still links a human to the normal page, not the iframe variant.
    assert rows[0]["job_url"] == job


def test_icims_iframe_url_preserves_existing_query_and_does_not_duplicate():
    f = ats_service._icims_iframe_url
    assert f("https://x.icims.com/jobs/1/a/job") == "https://x.icims.com/jobs/1/a/job?in_iframe=1"
    assert f("https://x.icims.com/jobs/1/a/job?mobile=false") == (
        "https://x.icims.com/jobs/1/a/job?mobile=false&in_iframe=1"
    )
    assert f("https://x.icims.com/jobs/1/a/job?in_iframe=1") == (
        "https://x.icims.com/jobs/1/a/job?in_iframe=1"
    )


# ── paylocity ships its location as an object, not a string ─────────────────

def test_paylocity_location_reads_the_object_paylocity_actually_sends():
    """The field is a dict, and `job.get("LocationName") or ""` handed it
    straight to _join_location. The archive recorded locations like
    "{'LocationId': 3092273, ..., 'Country': 'USA', ...}, Remote" -- every
    paylocity row unmatchable by _matches_location, not filtered out but
    incapable of matching any search a channel could express.
    """
    from services.ats_service import _paylocity_location

    assert _paylocity_location(
        {"LocationId": 1, "City": "Toronto", "State": "ON", "Country": "CAN"}
    ) == "Toronto, ON, CAN"


def test_paylocity_location_keeps_the_country_when_the_place_fields_are_null():
    """Observed live: City and State null, Country set. A country alone is
    worth more than nothing and far more than the raw object.
    """
    from services.ats_service import _paylocity_location

    assert _paylocity_location(
        {"City": None, "State": None, "Country": "USA", "Zip": None}
    ) == "USA"


def test_paylocity_location_passes_a_plain_string_through():
    from services.ats_service import _paylocity_location

    assert _paylocity_location("Vancouver, BC") == "Vancouver, BC"


@pytest.mark.parametrize("value", [None, "", {}, {"City": None, "Country": None}])
def test_paylocity_location_yields_empty_for_nothing_usable(value):
    from services.ats_service import _paylocity_location

    assert _paylocity_location(value) == ""


def test_a_paylocity_location_never_stringifies_a_dict():
    """The failure shape, pinned directly: whatever comes back must not start
    with a brace, because that is what reached the archive for months.
    """
    from services.ats_service import _paylocity_location

    for value in ({"LocationId": 3092273, "Country": "USA"}, {"City": "X"}, {}):
        assert not _paylocity_location(value).startswith("{")
