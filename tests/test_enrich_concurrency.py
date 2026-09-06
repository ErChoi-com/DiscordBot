"""PLATFORM_WORKERS has to be the real rate against a vendor, not a factor of it.

_scrape_ats_platform fans out PLATFORM_WORKERS slugs at once and every one of
those slugs then enriched its rows through its own pool. The pools nested, so
the actual number of concurrent requests to one vendor was the product --
thirty slugs times five enrichers against bamboohr.com, under a ceiling this
repo documents as measured and says to scale down only.

That is not a tidiness point. bamboohr's detail endpoint serves a posting date
and a full description reliably when asked one at a time (three of three
probed, ~2.9s each); 4 of 212 archived rows carried either, and the same 4
carried both. icims and oracle, the other two platforms that must enrich, sit
at 26% and 18%, while greenhouse and lever report 100% because their list
endpoints carry the fields and they never enrich at all.

So the tests that matter here run several enrichments at once and count what is
actually in flight. A test that enriched one batch would pass against the old
code, because one batch was never the problem.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as a  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_gates(monkeypatch):
    """Gates live for the life of the process; tests must not inherit them."""
    monkeypatch.setattr(a, "_ENRICH_GATES", {})
    monkeypatch.setattr(a, "_ENRICH_GATES_LOCK", threading.Lock())


class _Tracker:
    """Counts how many fetches are inside the gate at the same moment."""

    def __init__(self, hold: float = 0.02) -> None:
        self.lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.calls = 0
        self.hold = hold

    def fetch(self, url: str) -> dict[str, str]:
        with self.lock:
            self.live += 1
            self.calls += 1
            self.peak = max(self.peak, self.live)
        try:
            time.sleep(self.hold)
        finally:
            with self.lock:
                self.live -= 1
        return {"date_posted": "2026-01-01", "description": "d"}


def _rows(platform: str, n: int) -> list[dict]:
    return [{"title": f"t{i}", "company": "c", "location": "",
             "job_url": f"https://{platform}.example/{i}",
             "date_posted": "", "_source_site": platform} for i in range(n)]


# -- the gate itself --------------------------------------------------------

def test_the_gate_is_sized_from_that_platforms_ceiling(monkeypatch):
    monkeypatch.setitem(a.PLATFORM_WORKERS, "bamboohr", 30)
    monkeypatch.setitem(a.PLATFORM_WORKERS, "oracle", 8)
    assert a._enrich_gate("bamboohr")._value == a.capacity.workers(30, minimum=2)
    assert a._enrich_gate("oracle")._value == a.capacity.workers(8, minimum=2)


def test_one_gate_per_platform_shared_across_calls():
    """A gate rebuilt per call would bound nothing: the whole point is that
    every slug of a platform draws from the same allowance."""
    assert a._enrich_gate("bamboohr") is a._enrich_gate("bamboohr")
    assert a._enrich_gate("bamboohr") is not a._enrich_gate("icims")


def test_an_unknown_platform_still_gets_a_gate():
    """Rows without a source must not crash enrichment, and must not run
    unbounded either -- the default ceiling applies."""
    gate = a._enrich_gate("")
    assert gate._value == a.capacity.workers(10, minimum=2)


# -- what the gate is for ---------------------------------------------------

def test_concurrent_slugs_share_one_vendor_allowance(monkeypatch):
    """The regression, stated directly.

    Eight slugs enriching at once used to mean eight independent pools. Here
    the ceiling is 3, so no more than 3 requests may be inside the vendor at
    any moment no matter how many slugs are in flight.
    """
    monkeypatch.setitem(a.PLATFORM_WORKERS, "bamboohr", 3)
    tracker = _Tracker()

    threads = [threading.Thread(target=a._enrich_rows,
                                args=(_rows("bamboohr", 4), tracker.fetch))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert tracker.calls == 32, "every row should still have been fetched"
    assert tracker.peak <= a.capacity.workers(3, minimum=2), (
        f"{tracker.peak} concurrent requests against a ceiling of "
        f"{a.capacity.workers(3, minimum=2)}"
    )


def test_a_slow_vendor_does_not_hold_up_a_different_one(monkeypatch):
    """Gates are per platform, so one vendor's ceiling must not serialise
    another's -- otherwise the fix would trade thin fields for a slow fleet."""
    monkeypatch.setitem(a.PLATFORM_WORKERS, "bamboohr", 2)
    monkeypatch.setitem(a.PLATFORM_WORKERS, "icims", 2)
    seen: list[str] = []
    lock = threading.Lock()

    def fetch(url: str) -> dict[str, str]:
        with lock:
            seen.append(url.split("//")[1].split(".")[0])
        time.sleep(0.05)
        return {"date_posted": "2026-01-01"}

    threads = [
        threading.Thread(target=a._enrich_rows, args=(_rows("bamboohr", 4), fetch)),
        threading.Thread(target=a._enrich_rows, args=(_rows("icims", 4), fetch)),
    ]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    elapsed = time.monotonic() - t0

    assert len(seen) == 8
    assert {"bamboohr", "icims"} <= set(seen)
    # Serialised through one shared gate this would be ~8 * 0.05 = 0.4s; run as
    # two independent ceilings of 2 it is ~0.1s. The bound is loose enough to
    # survive a loaded machine and still fail a single global gate.
    assert elapsed < 0.35, f"platforms appear to share a gate ({elapsed:.2f}s)"


# -- the gate must not cost what it was added to buy ------------------------

def test_enrichment_still_fills_the_fields(monkeypatch):
    monkeypatch.setitem(a.PLATFORM_WORKERS, "bamboohr", 4)
    rows = _rows("bamboohr", 3)
    a._enrich_rows(rows, lambda url: {"date_posted": "2026-02-03",
                                      "description": "real text",
                                      "location": "Toronto"})
    for row in rows:
        assert row["date_posted"] == a._normalise_posted("2026-02-03")
        assert row["description"] == "real text"
        assert row["location"] == "Toronto"


def test_a_failing_fetch_still_leaves_the_listing_values(monkeypatch):
    """Unchanged behaviour, pinned because the gate now wraps the fetch: a
    refusal must not blank what the listing already gave us."""
    monkeypatch.setitem(a.PLATFORM_WORKERS, "bamboohr", 4)
    rows = _rows("bamboohr", 2)
    for row in rows:
        row["location"] = "Ottawa"

    def boom(url: str):
        raise RuntimeError("refused")

    a._enrich_rows(rows, boom)
    for row in rows:
        assert row["location"] == "Ottawa"
        assert row["date_posted"] == ""


def test_no_rows_makes_no_gate_and_no_call():
    a._enrich_rows([], lambda url: pytest.fail("must not fetch"))
    assert a._ENRICH_GATES == {}
