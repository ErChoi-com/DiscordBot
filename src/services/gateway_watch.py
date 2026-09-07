"""Say whether a gateway that is "behind" went silent or merely went unacked.

discord.py reports gateway trouble as one number: `ack_time - _last_send`,
printed as "Can't keep up, websocket is 41.5s behind". That number conflates
two unrelated failures. Either nothing at all was arriving on the socket --
in which case the ack is late because *everything* was late -- or traffic kept
flowing and only the heartbeat's ack did not come. The first is ours to fix
and the second is not, and the warning cannot tell them apart.

This run's warnings are the reason the distinction matters. All of them fall
inside the ATS fan-out windows, all report between 41.4s and 43.2s, and the
gateway's heartbeat interval is 41.25s -- so each one is almost exactly one
missed beat rather than lag that accumulated. Meanwhile discord.py's
`Heartbeat blocked` message, which fires when the *send* cannot be scheduled
within ten seconds, never appeared once, and loop_watch recorded no stall over
five seconds. The loop was running and the send went out on time. That leaves
only the receive side, and nothing in the process was watching it.

So this watches it. A daemon thread samples the keep-alive handler's own
stamps once a second -- `_last_recv`, updated on every frame the gateway
delivers, and `_last_send`/`_last_ack` for the outstanding beat -- and says
which of the two failures is happening while it is happening. A silence is
reported with the loop's location, because a loop that is running while the
socket is quiet means the bytes are not arriving, and a loop that is elsewhere
means they arrived and were not read.

It reads private attributes of a third-party class, so every read is guarded
and a shape it does not recognise turns the watch into a no-op rather than an
exception. It costs one wakeup a second.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

# What counts as silence has to be measured against the heartbeat interval,
# not against a fixed number of seconds. `tick()` is called for every frame
# the gateway delivers *including the ack itself*, so on a shard with no other
# traffic the only thing that ever refreshes the stamp is the ack, once every
# interval. Against a fixed five-second threshold such a shard would be
# reported as silent almost continuously, and the report would be worthless.
# One interval plus a margin is the first duration that means something went
# missing rather than nothing was due.
SILENCE_MARGIN = 1.15
# Used only when the interval cannot be read -- before IDENTIFY, or on a shape
# this does not understand. Longer than any interval Discord has issued, so an
# unknown interval errs towards saying nothing.
DEFAULT_SILENT_AFTER_S = 60.0
DEFAULT_REPORT_EVERY_S = 60.0
# A window is worth printing when a beat went unacked for longer than this.
# Deliberately discord.py's own threshold for "Can't keep up" (latency > 10),
# so the window summary covers exactly the events that produce a warning and
# stays quiet the rest of the time.
WINDOW_FLOOR_S = 10.0
_SAMPLE_S = 1.0


def silence_threshold(interval: float, floor: float) -> float:
    """How long nothing may arrive before that is news, on this connection."""
    if interval and interval > 0:
        return max(1.0, interval * SILENCE_MARGIN)
    return floor


def keepalive_of(client: Any) -> Any | None:
    """The live websocket's keep-alive handler, or None.

    Fetched per sample rather than held: the handler is replaced on every
    reconnect, and a stale one reports the stamps of a socket that is gone.
    """
    try:
        ws = getattr(client, "ws", None)
        keepalive = getattr(ws, "_keep_alive", None)
    except Exception:
        return None
    if keepalive is None:
        return None
    # Anything missing these is a shape this does not understand.
    for name in ("_last_recv", "_last_send", "_last_ack"):
        if not isinstance(getattr(keepalive, name, None), (int, float)):
            return None
    return keepalive


def sample(keepalive: Any, now: float) -> dict[str, float] | None:
    """What the gateway's clocks say, or None if they cannot be read.

    `recv_gap` is the time since any frame at all arrived; `outstanding` is
    the age of a heartbeat that has been sent and not yet acked, and is zero
    when the last beat is already acked. Together they separate a silent
    socket from a live one whose ack is missing.
    """
    if keepalive is None:
        return None
    try:
        last_recv = float(keepalive._last_recv)
        last_send = float(keepalive._last_send)
        last_ack = float(keepalive._last_ack)
        interval = getattr(keepalive, "interval", None)
    except Exception:
        return None
    return {
        "recv_gap": max(0.0, now - last_recv),
        # A send newer than its ack is a beat still in flight.
        "outstanding": max(0.0, now - last_send) if last_send > last_ack else 0.0,
        "interval": float(interval) if isinstance(interval, (int, float)) else 0.0,
    }


class GatewayWatch:
    """One client's receive-side watch. `start()` once the client exists."""

    def __init__(
        self,
        client: Any,
        silent_after: float = DEFAULT_SILENT_AFTER_S,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.perf_counter,
        locate: Callable[[], str] | None = None,
    ) -> None:
        self.client = client
        self.silent_after = max(1.0, float(silent_after))
        self.report_every = DEFAULT_REPORT_EVERY_S
        self._log = log
        # perf_counter, because that is the clock discord.py stamps with.
        self._clock = clock
        self._locate = locate or _loop_location
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.silences: list[tuple[float, float]] = []  # (seconds, outstanding at start)
        self._silent_since: float | None = None
        self._silent_outstanding = 0.0
        self._peak_recv = 0.0
        self._peak_outstanding = 0.0
        self._window_started: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._watch, name="gateway-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        while not self._stop.wait(_SAMPLE_S):
            now = self._clock()
            self.step(now, sample(keepalive_of(self.client), now))

    # One sample's worth of work, so that the whole watch can be driven a
    # tick at a time rather than by sleeping through it.
    def step(self, now: float, reading: dict[str, float] | None) -> None:
        if self._window_started is None:
            self._window_started = now
        if reading is not None:
            self._peak_recv = max(self._peak_recv, reading["recv_gap"])
            self._peak_outstanding = max(self._peak_outstanding, reading["outstanding"])
            self._note_silence(reading, now)
        if now - self._window_started >= self.report_every:
            if self._peak_outstanding >= WINDOW_FLOOR_S:
                self._log(
                    f"[gateway-watch] in the last {int(now - self._window_started)}s the longest "
                    f"gap with nothing received was {self._peak_recv:.1f}s and the longest a beat "
                    f"went unacked was {self._peak_outstanding:.1f}s"
                )
            self._peak_recv = 0.0
            self._peak_outstanding = 0.0
            self._window_started = now

    def _note_silence(self, reading: dict[str, float], now: float) -> None:
        """Open a silence when the socket goes quiet, close it when it speaks."""
        threshold = silence_threshold(reading["interval"], self.silent_after)
        if self._silent_since is None and reading["recv_gap"] > threshold:
            self._log(
                f"[gateway-watch] nothing received from the gateway for "
                f"{reading['recv_gap']:.1f}s (a beat has been outstanding "
                f"{reading['outstanding']:.1f}s); the loop is at: {self._locate()}"
            )
            self._silent_since = now - reading["recv_gap"]
            self._silent_outstanding = reading["outstanding"]
        elif self._silent_since is not None and reading["recv_gap"] <= threshold:
            length = (now - reading["recv_gap"]) - self._silent_since
            self.silences.append((length, self._silent_outstanding))
            self._log(
                f"[gateway-watch] the gateway is talking again after {length:.1f}s of silence"
            )
            self._silent_since = None
            self._silent_outstanding = 0.0


def _loop_location() -> str:
    """Where the event loop is, borrowed from the loop watch if it is running.

    A silent socket with the loop parked in its selector means the bytes never
    arrived; a silent socket with the loop deep in someone else's call means
    they arrived and nothing read them.
    """
    try:
        from services import loop_watch

        watch = getattr(loop_watch, "_active", None)
        if watch is not None:
            return watch.loop_location()
    except Exception:
        pass
    return "(unknown)"


_active: GatewayWatch | None = None


def start(client: Any, silent_after: float = DEFAULT_SILENT_AFTER_S, log: Callable[[str], None] = print) -> GatewayWatch:
    """Start watching this client's gateway; idempotent per process."""
    global _active
    if _active is None:
        _active = GatewayWatch(client, silent_after=silent_after, log=log)
        _active.start()
    return _active
