"""A platform that refuses everything must stop being asked, this cycle.

Measured on workable, 2026-09-05: 18 of 18 sampled requests answered HTTP 429,
with no 200 at all. The fan-out cannot see that -- each slug is handled
independently, so a wholesale block is indistinguishable from N separate
unlucky boards, and the retry path treats 429 as worth retrying because for a
transient limit it is.

The bill: 9,268 non-dead workable slugs at three attempts each is 27,804
requests that cannot succeed, and roughly 4.5s of backoff sleep per slug. Across
its four workers that is about 2.9 hours of sleeping against a 600s
per-platform budget -- so workable burned its whole allowance every cycle,
returned nothing, and spent wall-clock the other platforms were sharing.

Nothing here touches the network: `requests.get` is replaced with a fake that
returns the status codes under test.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as A  # noqa: E402


class _Resp:
    def __init__(self, status):
        self.status_code = status

    def json(self):
        return {"jobs": []}


@pytest.fixture(autouse=True)
def _no_network_no_sleep(monkeypatch):
    """Count requests, never make one, and never actually sleep."""
    calls = {"n": 0, "urls": []}
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    monkeypatch.setattr(A, "_mark_dead", lambda *a, **k: None)
    monkeypatch.setattr(A, "_mark_alive", lambda *a, **k: None)
    A.reset_refusal_breaker("testplat")
    yield calls
    A.reset_refusal_breaker("testplat")


def _serve(monkeypatch, calls, status):
    """Answer every request with `status` (int) or the next of a sequence."""
    seq = iter(status) if isinstance(status, (list, tuple)) else None

    def _get(url, **kw):
        calls["n"] += 1
        calls["urls"].append(url)
        return _Resp(next(seq) if seq else status)

    monkeypatch.setattr(A.requests, "get", _get)


def _fetch(n=1, platform="testplat"):
    for i in range(n):
        A._fetch_board_json(platform, f"co{i}", f"https://x/{i}")


# ── the breaker opens ────────────────────────────────────────────────────────

def test_a_wholesale_refusal_stops_the_platform(monkeypatch, _no_network_no_sleep):
    calls = _no_network_no_sleep
    _serve(monkeypatch, calls, 429)

    _fetch(500)
    assert A.platform_is_refusing("testplat") is True
    # 500 slugs would have been 1,500 requests. It stops at the trip point,
    # and the retries stop with it.
    assert calls["n"] < 100, f"still made {calls['n']} requests after refusal"


def test_403_trips_it_too(monkeypatch, _no_network_no_sleep):
    """A host blocking by IP answers 403 rather than 429, and the waste is
    identical."""
    _serve(monkeypatch, _no_network_no_sleep, 403)
    _fetch(200)
    assert A.platform_is_refusing("testplat") is True


def test_an_in_flight_call_abandons_its_retries_when_another_thread_trips_it(
        monkeypatch, _no_network_no_sleep):
    """The fan-out is 30-50 threads. A call that got its 429 just before some
    other thread opened the breaker must not go on to sleep through two
    backoffs and two more doomed requests -- that is the exact cost being
    removed, paid once per worker in flight.
    """
    calls = _no_network_no_sleep

    def _get(url, **kw):
        calls["n"] += 1
        # Something else trips the breaker while this call is mid-flight.
        for i in range(A._REFUSAL_TRIP):
            A._note_board_refused("testplat")
        return _Resp(429)

    monkeypatch.setattr(A.requests, "get", _get)
    A._fetch_board_json("testplat", "co", "https://x/co")

    assert calls["n"] == 1, f"retried {calls['n'] - 1} times into an open breaker"


def test_no_request_is_made_once_it_is_open(monkeypatch, _no_network_no_sleep):
    calls = _no_network_no_sleep
    _serve(monkeypatch, calls, 429)
    _fetch(200)
    opened_at = calls["n"]

    _fetch(200)
    assert calls["n"] == opened_at, "kept requesting after the breaker opened"


def test_it_takes_a_real_run_not_one_unlucky_board(monkeypatch, _no_network_no_sleep):
    """A burst of rate-limiting on a healthy platform must ride through. One
    refusal closing the platform would be far worse than the waste it saves.
    """
    _serve(monkeypatch, _no_network_no_sleep, 429)
    _fetch(3)
    assert A.platform_is_refusing("testplat") is False


# ── the breaker must not open on a working platform ─────────────────────────

def test_a_platform_that_answers_never_trips(monkeypatch, _no_network_no_sleep):
    _serve(monkeypatch, _no_network_no_sleep, 200)
    _fetch(300)
    assert A.platform_is_refusing("testplat") is False


def test_one_success_immunises_the_rest_of_the_cycle(monkeypatch, _no_network_no_sleep):
    """Once something has answered, later refusals are rate-limiting, which is
    exactly what the retry path exists for -- not a block.
    """
    _serve(monkeypatch, _no_network_no_sleep, [200] + [429] * 2000)
    _fetch(400)
    assert A.platform_is_refusing("testplat") is False


def test_a_refusal_streak_is_broken_by_a_success(monkeypatch, _no_network_no_sleep):
    _serve(monkeypatch, _no_network_no_sleep, [429] * 20 + [200] + [429] * 2000)
    _fetch(30)
    assert A.platform_is_refusing("testplat") is False


def test_a_404_proves_the_platform_is_answering_us(monkeypatch, _no_network_no_sleep):
    """A deleted board is a real answer about a real company, so it arms the
    same "this platform talks to us" flag a 200 does. Without that, a fleet
    whose head happens to be all-deleted, hitting rate limiting afterwards,
    would read as a wholesale block and close a healthy platform.

    Asserted by what happens *after* the 404s: refusals that would otherwise
    trip the breaker must not, because the platform already answered.
    """
    _serve(monkeypatch, _no_network_no_sleep, [404] * 5 + [429] * 3000)
    _fetch(400)
    assert A.platform_is_refusing("testplat") is False


def test_a_503_is_not_a_refusal(monkeypatch, _no_network_no_sleep):
    """503 is the host being broken, not declining us. It is already retried,
    and closing the platform on it would suppress a recovering service.
    """
    _serve(monkeypatch, _no_network_no_sleep, 503)
    _fetch(200)
    assert A.platform_is_refusing("testplat") is False


# ── it must never look like a dead company ──────────────────────────────────

def test_a_refusal_never_dead_marks_a_board(monkeypatch, _no_network_no_sleep):
    """403/429 is the host declining and says nothing about whether a company
    exists. Marking on refusal would suppress live boards for the whole TTL.
    """
    marked = []
    monkeypatch.setattr(A, "_mark_dead", lambda p, s: marked.append((p, s)))
    _serve(monkeypatch, _no_network_no_sleep, 429)
    _fetch(300)
    assert marked == []


# ── it is a within-cycle economy, never a persistent state ──────────────────

def test_a_new_cycle_gets_a_clean_try(monkeypatch, _no_network_no_sleep):
    """A platform blocked six hours ago may answer now. The breaker must never
    outlive the fan-out that opened it, or one bad cycle would suppress a
    platform until the bot restarts.
    """
    _serve(monkeypatch, _no_network_no_sleep, 429)
    _fetch(200)
    assert A.platform_is_refusing("testplat") is True

    A.reset_refusal_breaker("testplat")
    assert A.platform_is_refusing("testplat") is False


def test_the_fan_out_resets_it(monkeypatch):
    """Pinned at the source: the reset is worthless if nothing calls it, and
    'the piece exists but nothing calls it' is the recurring bug in this repo.
    """
    import inspect
    compact = " ".join(inspect.getsource(A.scrape_ats_platform).split())
    assert "reset_refusal_breaker(platform)" in compact


def test_one_platform_refusing_does_not_stop_another(monkeypatch, _no_network_no_sleep):
    """All ~16 platforms fan out concurrently. A shared counter would let
    workable's block close greenhouse.
    """
    A.reset_refusal_breaker("other")
    _serve(monkeypatch, _no_network_no_sleep, 429)
    _fetch(200, platform="testplat")

    assert A.platform_is_refusing("testplat") is True
    assert A.platform_is_refusing("other") is False
    A.reset_refusal_breaker("other")


# ── reporting ────────────────────────────────────────────────────────────────

def test_an_open_breaker_is_reportable(monkeypatch, _no_network_no_sleep):
    """Silence is what this whole subsystem fights. A platform cut off must be
    visible, not just quieter.
    """
    _serve(monkeypatch, _no_network_no_sleep, 429)
    _fetch(200)
    assert A.refusal_report().get("testplat", 0) >= A._REFUSAL_TRIP


def test_nothing_refusing_reports_nothing(_no_network_no_sleep):
    assert "testplat" not in A.refusal_report()


# ── concurrency, because the fan-out is 30-50 threads per platform ──────────

def test_the_breaker_is_safe_under_concurrent_workers(monkeypatch, _no_network_no_sleep):
    """Every scraper runs inside a per-platform thread pool. An unlocked
    read-modify-write on the streak counter would trip late, or not at all.
    """
    _serve(monkeypatch, _no_network_no_sleep, 429)

    threads = [threading.Thread(target=_fetch, args=(40,)) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert A.platform_is_refusing("testplat") is True
