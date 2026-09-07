"""Say what the event loop was doing when it stopped ticking.

The Discord gateway reports a stalled loop only as an effect -- "Can't keep
up, websocket is 41.5s behind", the heartbeat ack latency -- and only when the
stall straddles a heartbeat. discord.py prints the loop thread's stack only
for the narrower case where the heartbeat *send* itself cannot run. Everything
else that blocks the loop is invisible: it shows up as latency on commands and
as gateway warnings that name no cause. Four such warnings in two runs, each
~42s, and an external profiler that cannot read a 400-thread process reliably,
is what this exists to replace.

A coroutine on the loop stamps a shared clock once a second. A daemon thread
watches the stamp; when it goes stale by more than `stale_after` seconds the
thread reads the loop thread's frame through sys._current_frames -- which is
exactly what discord.py does -- and prints it, once per stall, with the
stall's length when the loop comes back. It costs one wakeup a second on each
side and nothing else.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import traceback
from typing import Callable

DEFAULT_STALE_AFTER_S = 5.0
_TICK_S = 1.0


class LoopWatch:
    """One loop's stall detector. `start()` from the loop it is to watch."""

    def __init__(
        self,
        stale_after: float = DEFAULT_STALE_AFTER_S,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stale_after = max(0.5, float(stale_after))
        self._log = log
        self._clock = clock
        self._last_tick = clock()
        self._loop_thread_id: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._task: asyncio.Task | None = None
        self.stalls: list[tuple[float, str]] = []   # (seconds, first frame line), newest last

    # ── the loop side ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin ticking on the running loop and watching from a thread."""
        loop = asyncio.get_running_loop()
        self._loop_thread_id = threading.get_ident()
        self._last_tick = self._clock()
        self._task = loop.create_task(self._tick_forever(), name="loop-watch-tick")
        self._thread = threading.Thread(target=self._watch, name="loop-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()

    async def _tick_forever(self) -> None:
        try:
            while not self._stop.is_set():
                self._last_tick = self._clock()
                await asyncio.sleep(_TICK_S)
        except asyncio.CancelledError:
            pass

    # ── the thread side ──────────────────────────────────────────────────────

    def _watch(self) -> None:
        stalled_since: float | None = None
        while not self._stop.wait(_TICK_S):
            age = self._clock() - self._last_tick
            if stalled_since is None and age > self.stale_after:
                stalled_since = self._last_tick
                self._log(
                    f"[loop-watch] event loop has not ticked for {age:.1f}s; "
                    f"loop thread is at:\n{self.loop_stack()}"
                )
            elif stalled_since is not None and age <= self.stale_after:
                length = self._last_tick - stalled_since
                self.stalls.append((length, self._first_frame_line))
                self._log(f"[loop-watch] event loop is back after a {length:.1f}s stall")
                stalled_since = None

    _first_frame_line: str = ""

    def loop_stack(self) -> str:
        """The loop thread's current Python stack, innermost frame last."""
        if self._loop_thread_id is None:
            return "  (loop not started)"
        frame = sys._current_frames().get(self._loop_thread_id)
        if frame is None:
            return "  (loop thread not found)"
        lines = traceback.format_stack(frame)
        self._first_frame_line = lines[-1].strip().splitlines()[0] if lines else ""
        return "".join(lines).rstrip()


_active: LoopWatch | None = None


def start(stale_after: float = DEFAULT_STALE_AFTER_S, log: Callable[[str], None] = print) -> LoopWatch:
    """Start watching the running loop; idempotent per process."""
    global _active
    if _active is None:
        _active = LoopWatch(stale_after=stale_after, log=log)
        _active.start()
    return _active
