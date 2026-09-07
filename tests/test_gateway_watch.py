"""The receive-side watch separates a silent gateway from an unacked one.

discord.py prints one number for both -- "websocket is 41.5s behind" is
`ack_time - _last_send`, and says nothing about whether anything else was
arriving at the time. Every warning in the last run was between 41.4s and
43.2s against a 41.25s heartbeat interval, which is one missed beat rather
than accumulated lag, and `Heartbeat blocked` never fired, so the send was
never the problem. These tests cover the instrument that tells the two apart.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import gateway_watch as gw  # noqa: E402


class _KeepAlive:
    """The three stamps discord.py's KeepAliveHandler actually keeps."""

    def __init__(self, last_recv=0.0, last_send=0.0, last_ack=0.0, interval=41.25):
        self._last_recv = last_recv
        self._last_send = last_send
        self._last_ack = last_ack
        self.interval = interval


class _WS:
    def __init__(self, keepalive):
        self._keep_alive = keepalive


class _Client:
    def __init__(self, ws=None):
        self.ws = ws


def _watch(client=None, **kwargs):
    lines: list[str] = []
    watch = gw.GatewayWatch(
        client if client is not None else _Client(),
        log=lines.append,
        locate=lambda: "select (windows_events.py:446)",
        **kwargs,
    )
    return watch, lines


# ── reading the handler ──────────────────────────────────────────────────────

def test_the_handler_is_found_through_the_live_websocket():
    keepalive = _KeepAlive()
    assert gw.keepalive_of(_Client(_WS(keepalive))) is keepalive


@pytest.mark.parametrize("client", [
    _Client(None),                       # not connected yet
    _Client(_WS(None)),                  # connected, no keep-alive yet
    object(),                            # not a client at all
])
def test_an_absent_handler_reads_as_nothing_rather_than_raising(client):
    assert gw.keepalive_of(client) is None


def test_a_handler_missing_a_stamp_is_a_shape_we_do_not_trust():
    """A future discord.py that renames a stamp must turn this into a no-op,
    not into an exception on a daemon thread nobody is watching."""
    keepalive = _KeepAlive()
    del keepalive._last_ack
    assert gw.keepalive_of(_Client(_WS(keepalive))) is None


def test_the_handler_is_refetched_so_a_reconnect_is_not_read_through_a_dead_socket():
    """The handler is replaced on every reconnect. Holding one would report
    the stamps of a socket that no longer exists."""
    client = _Client(_WS(_KeepAlive(last_recv=1.0)))
    first = gw.keepalive_of(client)
    client.ws = _WS(_KeepAlive(last_recv=500.0))
    assert gw.keepalive_of(client) is not first
    assert gw.sample(gw.keepalive_of(client), 501.0)["recv_gap"] == pytest.approx(1.0)


# ── what a sample says ───────────────────────────────────────────────────────

def test_a_beat_already_acked_is_not_outstanding():
    reading = gw.sample(_KeepAlive(last_recv=99.0, last_send=90.0, last_ack=90.2), now=100.0)
    assert reading["outstanding"] == 0.0
    assert reading["recv_gap"] == pytest.approx(1.0)


def test_a_beat_sent_after_the_last_ack_is_outstanding_from_the_send():
    reading = gw.sample(_KeepAlive(last_recv=99.0, last_send=95.0, last_ack=90.0), now=100.0)
    assert reading["outstanding"] == pytest.approx(5.0)


def test_the_two_halves_move_independently():
    """The whole point: traffic can flow while the ack does not arrive."""
    live_but_unacked = gw.sample(_KeepAlive(last_recv=99.9, last_send=60.0, last_ack=59.0), now=100.0)
    assert live_but_unacked["recv_gap"] < 1.0
    assert live_but_unacked["outstanding"] == pytest.approx(40.0)

    silent = gw.sample(_KeepAlive(last_recv=60.0, last_send=60.0, last_ack=59.0), now=100.0)
    assert silent["recv_gap"] == pytest.approx(40.0)


def test_sample_reports_nothing_for_no_handler_and_survives_a_hostile_one():
    assert gw.sample(None, now=1.0) is None

    class _Exploding:
        @property
        def _last_recv(self):
            raise RuntimeError("socket closed under us")
        _last_send = 0.0
        _last_ack = 0.0

    assert gw.sample(_Exploding(), now=1.0) is None


# ── what counts as silence on this connection ────────────────────────────────

def test_a_quiet_shard_is_not_reported_as_silent_every_minute():
    """The regression this threshold exists for. tick() is called for every
    frame including the ack, so a shard with no other traffic refreshes the
    stamp once per interval and nothing more. Against a fixed few-second
    threshold that shard reads as permanently silent, and the report becomes
    a line a minute that means nothing."""
    watch, lines = _watch()
    interval = 41.25
    now = 0.0
    for _ in range(10):                      # ten heartbeats' worth of samples
        for gap in range(0, int(interval)):
            now += 1.0
            # The healthy shape of one cycle: nothing arrives until the ack,
            # so the receive gap climbs the whole interval -- but the beat is
            # acked within a sample of being sent, so it is outstanding for
            # one sample and no longer.
            watch.step(now, {"recv_gap": float(gap),
                             "outstanding": 1.0 if gap == 0 else 0.0,
                             "interval": interval})
    assert lines == [], f"a quiet but healthy shard produced {lines}"
    assert watch.silences == []


def test_a_healthy_cycle_read_from_real_stamps_leaves_no_beat_outstanding():
    """The previous test hand-feeds readings, so this one checks the readings
    themselves: with the stamps a healthy handler actually keeps -- an ack
    landing just after its send -- nothing is outstanding for the rest of the
    interval, which is why the window floor is never approached in normal
    operation."""
    keepalive = _KeepAlive(last_recv=100.05, last_send=100.0, last_ack=100.05)
    for elapsed in (1.0, 10.0, 40.0):
        reading = gw.sample(keepalive, now=100.05 + elapsed)
        assert reading["outstanding"] == 0.0
        # ...while the receive gap grows across the whole interval, which is
        # exactly why silence cannot be judged on a fixed few seconds.
        assert reading["recv_gap"] == pytest.approx(elapsed)


def test_the_threshold_follows_the_interval_the_gateway_gave_us():
    assert gw.silence_threshold(41.25, floor=60.0) == pytest.approx(41.25 * gw.SILENCE_MARGIN)
    # A shorter interval must tighten the threshold, not sit on the floor.
    assert gw.silence_threshold(10.0, floor=60.0) == pytest.approx(11.5)


def test_an_unknown_interval_falls_back_to_the_floor():
    """Before IDENTIFY there is no interval. Erring long means saying nothing
    rather than saying something wrong."""
    assert gw.silence_threshold(0.0, floor=60.0) == 60.0
    assert gw.silence_threshold(None or 0, floor=60.0) == 60.0


def test_a_missed_beat_crosses_the_threshold_a_healthy_one_does_not():
    interval = 41.25
    healthy = gw.silence_threshold(interval, 60.0) > interval
    missed = gw.silence_threshold(interval, 60.0) < interval * 2
    assert healthy and missed, "one interval must be fine and two must not"


# ── silence is opened, measured and closed ───────────────────────────────────

def test_a_silence_is_reported_once_with_the_loop_location_not_once_per_second():
    watch, lines = _watch()
    interval = 41.25
    for second in range(48, 60):
        watch.step(100.0 + second, {"recv_gap": float(second), "outstanding": 0.0,
                                    "interval": interval})
    assert len(lines) == 1
    assert "nothing received from the gateway for 48.0s" in lines[0]
    assert "windows_events.py:446" in lines[0], "a silence must say where the loop was"


def test_the_silence_report_carries_how_long_the_beat_has_been_outstanding():
    """Which is what says whether the missing frame is the ack or everything.

    A beat outstanding as long as the silence means the ack is simply one of
    the frames that did not arrive; a beat outstanding while frames keep
    coming means the ack alone went missing, which is not ours to fix."""
    watch, lines = _watch()
    watch.step(100.0, {"recv_gap": 60.0, "outstanding": 59.0, "interval": 41.25})
    assert "a beat has been outstanding 59.0s" in lines[0]


def test_a_silence_that_ends_is_measured_from_when_it_started():
    watch, lines = _watch()
    interval = 41.25
    watch.step(150.0, {"recv_gap": 50.0, "outstanding": 49.0, "interval": interval})  # quiet since 100.0
    watch.step(180.0, {"recv_gap": 80.0, "outstanding": 79.0, "interval": interval})  # still quiet
    watch.step(181.0, {"recv_gap": 0.5, "outstanding": 0.0, "interval": interval})    # spoke at 180.5
    assert watch.silences == [(pytest.approx(80.5), pytest.approx(49.0))]
    assert "talking again after 80.5s of silence" in lines[-1]


def test_two_separate_silences_are_two_records():
    watch, _ = _watch()
    interval = 41.25
    for now, gap in [(100.0, 50.0), (101.0, 0.1), (300.0, 60.0), (301.0, 0.1)]:
        watch.step(now, {"recv_gap": gap, "outstanding": 0.0, "interval": interval})
    assert len(watch.silences) == 2


# ── the per-window summary ───────────────────────────────────────────────────

def test_a_window_reports_its_peaks_and_then_starts_over():
    watch, lines = _watch()
    watch.step(0.0, {"recv_gap": 0.0, "outstanding": 0.0, "interval": 41.25})
    watch.step(30.0, {"recv_gap": 12.0, "outstanding": 41.5, "interval": 41.25})
    watch.step(60.0, {"recv_gap": 1.0, "outstanding": 1.0, "interval": 41.25})
    assert len(lines) == 1
    assert "longest gap with nothing received was 12.0s" in lines[0]
    assert "longest a beat went unacked was 41.5s" in lines[0]

    # The next window is measured on its own, not against the first.
    watch.step(90.0, {"recv_gap": 3.0, "outstanding": 20.0, "interval": 41.25})
    watch.step(121.0, {"recv_gap": 0.0, "outstanding": 0.0, "interval": 41.25})
    assert "was 3.0s" in lines[-1] and "unacked was 20.0s" in lines[-1]


def test_a_window_is_printed_for_the_lag_discord_would_warn_about_and_no_other():
    """discord.py warns when the ack latency passes ten seconds. A window that
    holds no such beat has nothing to add to the log, however long the gaps
    between frames were on an idle shard."""
    watch, lines = _watch()
    watch.step(0.0, {"recv_gap": 0.0, "outstanding": 0.0, "interval": 41.25})
    watch.step(61.0, {"recv_gap": 40.0, "outstanding": 9.9, "interval": 41.25})
    assert lines == [], "an idle shard with no laggy beat must stay quiet"

    watch.step(122.0, {"recv_gap": 0.5, "outstanding": 10.1, "interval": 41.25})
    assert len(lines) == 1 and "unacked was 10.1s" in lines[0]


def test_a_window_with_no_readings_at_all_is_silent_and_does_not_crash():
    """Before the client connects there is no handler to sample."""
    watch, lines = _watch()
    watch.step(0.0, None)
    watch.step(61.0, None)
    assert lines == []


# ── the thread ───────────────────────────────────────────────────────────────

def test_start_runs_a_daemon_thread_that_samples_the_client_and_stops(monkeypatch):
    monkeypatch.setattr(gw, "_SAMPLE_S", 0.01)
    keepalive = _KeepAlive(last_recv=0.0, last_send=0.0, last_ack=0.0)
    watch, _ = _watch(_Client(_WS(keepalive)))
    seen = threading.Event()
    readings: list[dict] = []

    def _record(now, reading):
        if reading is not None:
            readings.append(reading)
            seen.set()

    monkeypatch.setattr(watch, "step", _record)
    try:
        watch.start()
        assert seen.wait(3.0), "the watch never sampled the client"
        # It read the live handler, not a placeholder.
        assert readings[0]["interval"] == pytest.approx(keepalive.interval)
        thread = watch._thread
        assert thread is not None and thread.daemon
        watch.start()
        assert watch._thread is thread, "start() must not open a second thread"
    finally:
        watch.stop()
    thread.join(2.0)
    assert not thread.is_alive(), "stop() must actually end the thread"


def test_the_module_level_start_is_idempotent(monkeypatch):
    monkeypatch.setattr(gw, "_active", None)
    monkeypatch.setattr(gw.GatewayWatch, "start", lambda self: None)
    first = gw.start(_Client())
    assert gw.start(_Client()) is first
    monkeypatch.setattr(gw, "_active", None)


def test_the_loop_location_falls_back_when_no_loop_watch_is_running(monkeypatch):
    from services import loop_watch

    monkeypatch.setattr(loop_watch, "_active", None)
    assert gw._loop_location() == "(unknown)"
