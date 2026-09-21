"""The list-endpoint scrapers do their per-posting work only for postings the
archive does not already hold.

A GIL profile of the live bot: of 3,727 samples of whichever thread held the
interpreter, the event loop held it in 6. The holders were re.sub, json and
html.unescape inside _html_to_text, converting every posting's body to text --
including the 40,146 Greenhouse postings per cycle that _drop_already_archived
then discarded. Every one of those conversions was a slice of the GIL the
Discord heartbeat waited behind.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as a  # noqa: E402
from services.jba import archive_index  # noqa: E402


def _archive_holds(monkeypatch, held: set[str]):
    def _filter(rows, cutoff=None):
        kept = [r for r in rows if r.get("job_url") not in held]
        return kept, len(rows) - len(kept)

    monkeypatch.setattr(archive_index, "ensure_index", lambda force=False: 0)
    monkeypatch.setattr(archive_index, "filter_new_listings", _filter)


@pytest.fixture
def conversions(monkeypatch):
    """Count the expensive step."""
    seen: list[str] = []
    real = a._html_to_text

    def _counting(raw):
        seen.append(str(raw or ""))
        return real(raw)

    monkeypatch.setattr(a, "_html_to_text", _counting)
    return seen


# ── the helper ───────────────────────────────────────────────────────────────

def test_fresh_only_drops_held_urls_keeps_order_and_unreadable_items(monkeypatch):
    _archive_holds(monkeypatch, {"https://x/2"})
    items = [{"u": "https://x/1"}, {"u": "https://x/2"}, {"u": ""}, {"u": "https://x/3"}, "not-a-dict"]
    kept = a._fresh_only("greenhouse", "acme", items, lambda i: i["u"])
    assert kept == [{"u": "https://x/1"}, {"u": ""}, {"u": "https://x/3"}, "not-a-dict"]


def test_fresh_only_carries_the_posting_date_into_the_identity(monkeypatch):
    """A stub without a date collides with any prior sighting; the row it
    stands in for would not. The stub must be exactly as strict as the row."""
    seen: list[dict] = []

    def _filter(rows, cutoff=None):
        seen.extend(rows)
        return rows, 0

    monkeypatch.setattr(archive_index, "ensure_index", lambda force=False: 0)
    monkeypatch.setattr(archive_index, "filter_new_listings", _filter)
    a._fresh_only("greenhouse", "acme", [{"u": "https://x/1", "d": "2026-09-01T00:00:00Z"}],
                  lambda i: i["u"], lambda i: i["d"])
    assert seen and seen[0]["date_posted"] == a._normalise_posted("2026-09-01T00:00:00Z")
    assert seen[0]["_source_site"] == "greenhouse" and seen[0]["company"] == "acme"


def test_fresh_only_is_a_passthrough_when_dedup_is_off_or_the_archive_fails(monkeypatch):
    items = [{"u": "https://x/1"}]
    monkeypatch.setattr(a, "ATS_ARCHIVE_DEDUP_ENABLED", False)
    assert a._fresh_only("greenhouse", "acme", items, lambda i: i["u"]) == items
    monkeypatch.setattr(a, "ATS_ARCHIVE_DEDUP_ENABLED", True)

    def _boom(*args, **kwargs):
        raise RuntimeError("index locked")

    monkeypatch.setattr(archive_index, "ensure_index", _boom)
    assert a._fresh_only("greenhouse", "acme", items, lambda i: i["u"]) == items


# ── greenhouse: the conversion is skipped for held postings ──────────────────

class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def _gh_job(i: int) -> dict:
    return {"title": f"Engineer {i}", "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{i}",
            "location": {"name": "Toronto, ON"}, "first_published": "2026-09-01T00:00:00Z",
            "content": f"&lt;p&gt;Body &amp; details {i}&lt;/p&gt;"}


def test_greenhouse_converts_only_the_bodies_of_new_postings(monkeypatch, conversions):
    jobs = [_gh_job(i) for i in range(1, 5)]
    monkeypatch.setattr(a, "_http_get", lambda url, **kw: _Resp({"jobs": jobs}))
    monkeypatch.setattr(a, "_is_dead", lambda p, s: False)
    monkeypatch.setattr(a, "_mark_alive", lambda p, s: None)
    _archive_holds(monkeypatch, {jobs[0]["absolute_url"], jobs[2]["absolute_url"]})

    rows = a._scrape_greenhouse("acme", "", "", 10)

    assert [r["job_url"] for r in rows] == [jobs[1]["absolute_url"], jobs[3]["absolute_url"]]
    assert len(conversions) == 2, f"converted {len(conversions)} bodies for 2 new postings"
    assert rows[0]["description"] == "Body & details 2"


def test_greenhouse_with_an_empty_archive_is_unchanged(monkeypatch, conversions):
    jobs = [_gh_job(i) for i in range(1, 4)]
    monkeypatch.setattr(a, "_http_get", lambda url, **kw: _Resp({"jobs": jobs}))
    monkeypatch.setattr(a, "_is_dead", lambda p, s: False)
    monkeypatch.setattr(a, "_mark_alive", lambda p, s: None)
    _archive_holds(monkeypatch, set())
    rows = a._scrape_greenhouse("acme", "", "", 10)
    assert len(rows) == 3 and len(conversions) == 3


# ── _board_rows: a deferred description is resolved only for kept rows ───────

def test_board_rows_resolves_deferred_descriptions_only_for_new_postings(monkeypatch):
    resolved: list[str] = []

    def shape(job):
        def body():
            resolved.append(job["url"])
            return f"text {job['url']}"
        return (job["title"], "Toronto, ON", job["url"], "2026-09-01", body)

    jobs = [{"title": f"J{i}", "url": f"https://x/{i}"} for i in range(4)]
    _archive_holds(monkeypatch, {"https://x/1", "https://x/2"})

    rows = a._board_rows("ashby", "acme", jobs, "", "", 10, shape)

    assert [r["job_url"] for r in rows] == ["https://x/0", "https://x/3"]
    assert resolved == ["https://x/0", "https://x/3"]
    assert rows[0]["description"] == "text https://x/0"


def test_board_rows_still_takes_a_plain_string_description(monkeypatch):
    _archive_holds(monkeypatch, set())
    rows = a._board_rows("ashby", "acme", [{"title": "J", "url": "https://x/1"}], "", "", 10,
                         lambda job: (job["title"], "Toronto, ON", job["url"], "2026-09-01", "plain body"))
    assert rows[0]["description"] == "plain body"


def test_a_deferred_description_that_fails_costs_the_body_not_the_row(monkeypatch):
    _archive_holds(monkeypatch, set())

    def shape(job):
        def body():
            raise ValueError("bad html")
        return (job["title"], "Toronto, ON", job["url"], "2026-09-01", body)

    rows = a._board_rows("ashby", "acme", [{"title": "J", "url": "https://x/1"}], "", "", 10, shape)
    assert len(rows) == 1 and rows[0]["description"] == ""


def test_ashby_and_paylocity_shapes_defer_the_conversion(monkeypatch, conversions):
    """The two shapes that convert HTML hand _board_rows a callable."""
    _archive_holds(monkeypatch, {"https://jobs.ashbyhq.com/acme/1"})
    monkeypatch.setattr(a, "_is_dead", lambda p, s: False)
    monkeypatch.setattr(a, "_fetch_board_json", lambda p, s, u: {"jobs": [
        {"title": "Held", "location": "Toronto", "jobUrl": "https://jobs.ashbyhq.com/acme/1",
         "publishedAt": "2026-09-01", "descriptionHtml": "<p>held</p>"},
        {"title": "New", "location": "Toronto", "jobUrl": "https://jobs.ashbyhq.com/acme/2",
         "publishedAt": "2026-09-01", "descriptionHtml": "<p>new</p>"},
    ]})
    rows = a._scrape_ashby("acme", "", "", 10)
    assert [r["title"] for r in rows] == ["New"]
    assert conversions == ["<p>new</p>"]


# ── _html_to_text ────────────────────────────────────────────────────────────

def test_html_to_text_still_unescapes_then_strips():
    assert a._html_to_text("&lt;p&gt;Hello &amp; welcome&lt;/p&gt;") == "Hello & welcome"
    assert a._html_to_text("<b>Bold</b>  text\n\nhere") == "Bold text here"
    assert a._html_to_text(None) == "" and a._html_to_text("") == ""


def test_html_to_text_skips_the_entity_pass_for_plain_text(monkeypatch):
    calls = {"n": 0}
    real = a.html.unescape

    def _counting(s):
        calls["n"] += 1
        return real(s)

    monkeypatch.setattr(a.html, "unescape", _counting)
    assert a._html_to_text("plain   body of text") == "plain body of text"
    assert calls["n"] == 0
    a._html_to_text("with &amp; entity")
    assert calls["n"] == 1
