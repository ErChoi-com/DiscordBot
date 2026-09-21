"""One requests.Session per worker thread, not one per call.

A py-spy dump of the live bot found 166 of 541 threads inside
ssl_wrap_socket. Every bare `requests.get(...)` call built a fresh Session,
adapter, and SSLContext -- reloading the CA bundle -- and then threw them
away. `_http()` gives each thread its own Session, created once and reused
for the life of the thread; `_http_get`/`_http_post` are the only two call
sites that are allowed to reach the network, and the only seam tests stub.

Nothing here touches the network.
"""
from __future__ import annotations

import inspect
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as a  # noqa: E402


def test_same_thread_gets_the_same_session():
    first = a._http()
    second = a._http()
    assert first is second


def test_two_threads_get_two_distinct_sessions(monkeypatch):
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())
    # Keep the Session objects themselves alive (not just their ids): a
    # thread's threading.local storage is torn down when the thread exits,
    # and a freed object's id can be reused by the next allocation -- which
    # would make two genuinely distinct sessions compare equal by id.
    sessions: dict[int, object] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=10)

    def grab(idx: int) -> None:
        session = a._http()
        barrier.wait()  # hold both threads alive until both have fetched
        with lock:
            sessions[idx] = session

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1], "two threads shared one Session"


def test_the_session_survives_across_calls_in_the_same_thread(monkeypatch):
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())
    result: dict[str, object] = {}

    def worker() -> None:
        first = a._http()
        second = a._http()
        result["same"] = first is second
        result["id"] = id(first)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)

    assert result["same"] is True


def test_http_get_really_uses_the_calling_threads_session(monkeypatch):
    """The one seam tests stub: `_http_get` must go through `_http()`, not
    through a module-level `requests.get`."""
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())

    calls: list[tuple[str, dict]] = []

    class _FakeSession:
        def mount(self, prefix, adapter):
            pass

        def get(self, url, **kw):
            calls.append((url, kw))
            return "resp"

    monkeypatch.setattr(a.requests, "Session", _FakeSession)

    result = a._http_get("https://x", headers={"a": "b"}, timeout=3)

    assert result == "resp"
    assert calls == [("https://x", {"headers": {"a": "b"}, "timeout": 3})]


def test_http_get_reuses_the_same_fake_session_on_a_second_call(monkeypatch):
    """Confirms the fake from the previous test is actually the thread-local
    session, not a fresh one built each call."""
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())

    build_count = {"n": 0}

    class _FakeSession:
        def mount(self, prefix, adapter):
            pass

        def __init__(self):
            build_count["n"] += 1

        def get(self, url, **kw):
            return "resp"

    monkeypatch.setattr(a.requests, "Session", _FakeSession)

    a._http_get("https://x")
    a._http_get("https://y")

    assert build_count["n"] == 1, "a new Session was built on the second call"


def test_no_bare_requests_get_or_post_remain_in_the_module():
    """Regression guard: every network call in ats_service.py must go through
    _http_get/_http_post so it picks up the thread-local Session. The two
    helpers themselves call `_http().get(...)` / `_http().post(...)`, which
    does not match this substring, so this holds once every other call site
    has been converted."""
    source = inspect.getsource(a)
    assert source.count("requests.get(") == 0, (
        "a bare requests.get( is still present outside _http_get"
    )
    assert source.count("requests.post(") == 0, (
        "a bare requests.post( is still present outside _http_post"
    )
