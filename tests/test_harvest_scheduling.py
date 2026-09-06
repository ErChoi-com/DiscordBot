"""What makes an unconditional harvest schedule affordable and safe to stop.

Discovery had no trigger that fires. The weekly cron in ats-harvest.yml cannot
run, because GitHub reads `schedule:` only from the default branch and the
workflow lives on a feature branch -- so the only thing that ever started a
harvest was a person. These two flags are what let something on a fixed
schedule call the harvester unconditionally:

  --only-new-crawls      answer "nothing new" in milliseconds and no requests,
                         which is the answer on twenty-nine days out of thirty
  --total-budget-seconds stop between platforms so a scheduled run cannot
                         delay whatever is queued behind it

The pairing is what makes them safe. A bounded run can stop half way through
the fleet, and the record of what has been swept is what makes the next run
skip -- so a truncated run that recorded its crawl would retire it with most
platforms never harvested, and no later run would go back for them. Every test
below that involves the budget therefore also asserts on the record.
"""
from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import harvest_ats as hc  # noqa: E402


def _offline_bulk(monkeypatch):
    """Stub the bulk index's own fetchers, which are not _http_get."""
    def no_net(*args, **kwargs):
        raise urllib.error.URLError("offline in tests")

    monkeypatch.setattr(hc, "_http_get_range", no_net)
    monkeypatch.setattr(hc, "_http_head_size", no_net)


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    import io
    return urllib.error.HTTPError(
        "https://index.commoncrawl.org/x", code, "err", {}, io.BytesIO(body)
    )


def _fake_index(pages: dict[str, list[str]], *, on_fetch=None):
    """Fetcher serving canned pages, keyed by the request's decoded url= param.

    `on_fetch` runs on every call, which is how the budget tests advance a fake
    clock: the time a harvest spends is time spent fetching, so driving the
    clock from the fetcher is the honest place to do it.
    """
    import urllib.parse as _up

    def fetch(url: str) -> bytes:
        if on_fetch is not None:
            on_fetch()
        query = _up.parse_qs(_up.urlparse(url).query)
        target = (query.get("url") or [""])[0]
        if target not in pages:
            raise _http_error(404, b'{"message": "No Captures found"}')
        if "showNumPages" in query:
            return json.dumps({"pages": 1}).encode()
        return b"\n".join(
            json.dumps({"url": u, "status": "200"}).encode() for u in pages[target]
        )

    return fetch


_GREENHOUSE_PAGES = {
    "boards.greenhouse.io/*": ["https://boards.greenhouse.io/acme/jobs/1"],
    "job-boards.greenhouse.io/*": ["https://job-boards.greenhouse.io/newco/j/1"],
    "job-boards.eu.greenhouse.io/*": [],
}
_LEVER_PAGES = {
    "jobs.lever.co/*": ["https://jobs.lever.co/globex/abc-def"],
    "jobs.eu.lever.co/*": [],
}


# --------------------------------------------------------------------------
# The swept record itself
# --------------------------------------------------------------------------

def test_an_unreadable_record_means_sweep_it_again(tmp_path):
    """Fail open, always. Skipping is the destructive direction.

    Re-sweeping a crawl costs time. Wrongly skipping one costs a month of
    companies nobody discovers, and because the skip is silent nothing would
    ever report it.
    """
    path = tmp_path / "_swept.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert hc._load_swept(path) == set()


def test_a_missing_record_means_nothing_has_been_swept(tmp_path):
    assert hc._load_swept(tmp_path / "nope.json") == set()


def test_a_record_of_the_wrong_shape_is_ignored_rather_than_trusted(tmp_path):
    path = tmp_path / "_swept.json"
    path.write_text(json.dumps({"crawls": "CC-MAIN-2026-34"}), encoding="utf-8")
    assert hc._load_swept(path) == set()


def test_saving_merges_rather_than_replacing(tmp_path):
    """Two runs sweeping different crawls must not erase each other."""
    path = tmp_path / "_swept.json"
    hc._save_swept(path, ["CC-MAIN-2026-30"])
    hc._save_swept(path, ["CC-MAIN-2026-34"])
    assert hc._load_swept(path) == {"CC-MAIN-2026-30", "CC-MAIN-2026-34"}


# --------------------------------------------------------------------------
# --only-new-crawls
# --------------------------------------------------------------------------

def test_an_already_swept_crawl_costs_no_request_at_all(tmp_path, monkeypatch):
    """The whole point: the common case must be free.

    A daily caller meets the same newest crawl for about a month. If finding
    that out cost a sweep, the schedule would be unaffordable and would go back
    to being run by hand, which is the defect this exists to remove.
    """
    out = tmp_path / "out"
    out.mkdir()
    hc._save_swept(out / "_swept.json", ["CC-MAIN-2026-34"])

    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        raise AssertionError("no request may be made for an already-swept crawl")

    monkeypatch.setattr(hc, "_http_get", fetch)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)

    rc = hc.main(["--crawl", "CC-MAIN-2026-34", "--out", str(out),
                  "--index", "commoncrawl", "--only-new-crawls", "--delay", "0"])
    assert rc == 0
    assert calls == []


def test_nothing_new_exits_zero_not_the_empty_harvest_failure(tmp_path, monkeypatch):
    """A run with nothing to do succeeded; it did not fail to harvest.

    main() exits 1 when every platform yielded nothing, because that means the
    index or the queries broke. Skipping has the same shape -- no platform
    produced anything -- and must not be reported the same way, or a scheduled
    caller would log a failure every day between Common Crawl releases.
    """
    out = tmp_path / "out"
    out.mkdir()
    hc._save_swept(out / "_swept.json", ["CC-MAIN-2026-34"])
    monkeypatch.setattr(hc, "_http_get", _fake_index({}))
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)

    assert hc.main(["--crawl", "CC-MAIN-2026-34", "--out", str(out),
                    "--index", "commoncrawl", "--only-new-crawls",
                    "--delay", "0"]) == 0


def test_an_unswept_crawl_is_still_harvested(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    hc._save_swept(out / "_swept.json", ["CC-MAIN-2026-30"])
    monkeypatch.setattr(hc, "_http_get", _fake_index(_GREENHOUSE_PAGES))
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)

    rc = hc.main(["--crawl", "CC-MAIN-2026-34", "--platform", "greenhouse",
                  "--out", str(out), "--index", "commoncrawl",
                  "--only-new-crawls", "--delay", "0"])
    assert rc == 0
    assert hc.load_existing(out / "greenhouse.json") == {"acme", "newco"}


def test_without_the_flag_a_swept_crawl_is_harvested_again(tmp_path, monkeypatch):
    """The flag is opt-in. CI passes --crawls 6 and means all six."""
    out = tmp_path / "out"
    out.mkdir()
    hc._save_swept(out / "_swept.json", ["CC-MAIN-2026-34"])
    monkeypatch.setattr(hc, "_http_get", _fake_index(_GREENHOUSE_PAGES))
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)

    rc = hc.main(["--crawl", "CC-MAIN-2026-34", "--platform", "greenhouse",
                  "--out", str(out), "--index", "commoncrawl", "--delay", "0"])
    assert rc == 0
    assert hc.load_existing(out / "greenhouse.json") == {"acme", "newco"}


# --------------------------------------------------------------------------
# What gets recorded, and what deliberately does not
# --------------------------------------------------------------------------

def test_a_completed_run_records_the_crawl(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(hc, "_http_get", _fake_index(_GREENHOUSE_PAGES))
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)

    assert hc.main(["--crawl", "CC-MAIN-2026-34", "--platform", "greenhouse",
                    "--out", str(out), "--index", "commoncrawl",
                    "--delay", "0"]) == 0
    assert hc._load_swept(out / "_swept.json") == {"CC-MAIN-2026-34"}


def test_a_dry_run_records_nothing(tmp_path, monkeypatch):
    """It wrote no slugs, so it must not claim the crawl has been read."""
    out = tmp_path / "out"
    monkeypatch.setattr(hc, "_http_get", _fake_index(_GREENHOUSE_PAGES))
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)

    assert hc.main(["--crawl", "CC-MAIN-2026-34", "--platform", "greenhouse",
                    "--out", str(out), "--index", "commoncrawl",
                    "--dry-run", "--delay", "0"]) == 0
    assert hc._load_swept(out / "_swept.json") == set()


def test_a_run_that_found_nothing_records_nothing(tmp_path, monkeypatch):
    """Finding nothing means the index or the queries broke, not that the
    crawl holds no companies. Retiring it would make that outage permanent."""
    out = tmp_path / "out"
    monkeypatch.setattr(hc, "_http_get", _fake_index({}))
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    _offline_bulk(monkeypatch)

    assert hc.main(["--crawl", "CC-MAIN-2026-34", "--platform", "greenhouse",
                    "--out", str(out), "--index", "commoncrawl",
                    "--delay", "0"]) == 1
    assert hc._load_swept(out / "_swept.json") == set()


# --------------------------------------------------------------------------
# --total-budget-seconds
# --------------------------------------------------------------------------

class _Clock:
    """A monotonic clock that only moves when the harvest does work."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += self.step


def test_the_budget_stops_before_starting_another_platform(tmp_path, monkeypatch):
    """Greenhouse runs and keeps its slugs; lever is never started.

    The check sits between platforms rather than inside one on purpose: a
    platform cut off mid-sweep would still write what it found, and its
    per-run counts are what the yield guard reads to decide whether a platform
    has gone silent.
    """
    out = tmp_path / "out"
    clock = _Clock(step=10.0)
    monkeypatch.setattr(hc.time, "monotonic", clock)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        hc, "_http_get",
        _fake_index({**_GREENHOUSE_PAGES, **_LEVER_PAGES}, on_fetch=clock.tick),
    )
    _offline_bulk(monkeypatch)

    rc = hc.main(["--crawl", "CC-MAIN-2026-34",
                  "--platform", "greenhouse", "--platform", "lever",
                  "--out", str(out), "--index", "commoncrawl",
                  "--total-budget-seconds", "5", "--delay", "0"])
    assert rc == 0
    assert hc.load_existing(out / "greenhouse.json") == {"acme", "newco"}
    assert not (out / "lever.json").exists()


def test_a_truncated_run_does_not_retire_the_crawl(tmp_path, monkeypatch):
    """The one that matters.

    Recording a crawl is what makes every later run skip it. A truncated run
    that recorded its crawl would retire it having harvested only the
    platforms it reached, and nothing would ever go back for the rest -- the
    budget would silently convert a bounded run into permanent lost coverage.
    """
    out = tmp_path / "out"
    clock = _Clock(step=10.0)
    monkeypatch.setattr(hc.time, "monotonic", clock)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        hc, "_http_get",
        _fake_index({**_GREENHOUSE_PAGES, **_LEVER_PAGES}, on_fetch=clock.tick),
    )
    _offline_bulk(monkeypatch)

    hc.main(["--crawl", "CC-MAIN-2026-34",
             "--platform", "greenhouse", "--platform", "lever",
             "--out", str(out), "--index", "commoncrawl",
             "--total-budget-seconds", "5", "--delay", "0"])
    assert hc._load_swept(out / "_swept.json") == set()


def test_the_next_run_resumes_the_crawl_the_budget_cut_short(tmp_path, monkeypatch):
    """Truncation defers work; it does not lose it.

    Proves the pairing end to end: a bounded first run leaves the crawl
    unrecorded, so a second run with --only-new-crawls still sees it as new and
    harvests the platform the first run never reached.
    """
    out = tmp_path / "out"
    clock = _Clock(step=10.0)
    monkeypatch.setattr(hc.time, "monotonic", clock)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        hc, "_http_get",
        _fake_index({**_GREENHOUSE_PAGES, **_LEVER_PAGES}, on_fetch=clock.tick),
    )
    _offline_bulk(monkeypatch)

    args = ["--crawl", "CC-MAIN-2026-34",
            "--platform", "greenhouse", "--platform", "lever",
            "--out", str(out), "--index", "commoncrawl",
            "--only-new-crawls", "--delay", "0"]
    hc.main(args + ["--total-budget-seconds", "5"])
    assert not (out / "lever.json").exists()

    clock.now = 0.0
    assert hc.main(args) == 0
    assert hc.load_existing(out / "lever.json") == {"globex"}
    assert hc._load_swept(out / "_swept.json") == {"CC-MAIN-2026-34"}


def test_no_budget_means_no_ceiling(tmp_path, monkeypatch):
    """Absent the flag, a long harvest is not silently truncated."""
    out = tmp_path / "out"
    clock = _Clock(step=1000.0)
    monkeypatch.setattr(hc.time, "monotonic", clock)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        hc, "_http_get",
        _fake_index({**_GREENHOUSE_PAGES, **_LEVER_PAGES}, on_fetch=clock.tick),
    )
    _offline_bulk(monkeypatch)

    rc = hc.main(["--crawl", "CC-MAIN-2026-34",
                  "--platform", "greenhouse", "--platform", "lever",
                  "--out", str(out), "--index", "commoncrawl", "--delay", "0"])
    assert rc == 0
    assert hc.load_existing(out / "lever.json") == {"globex"}
