from __future__ import annotations

import asyncio
import heapq
import itertools
import os
import statistics
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from services import capacity

_T = TypeVar("_T")

# Two priority tiers: interactive (user-triggered Discord commands like
# .resumebuild) always preempts background (job/reddit/ATS watcher work),
# except when aging has promoted a long-waiting background task.
INTERACTIVE = 0
BACKGROUND = 1

# A background task that has waited this long gets promoted to the
# interactive tier so a steady stream of interactive submissions can't
# starve it forever.
_AGING_THRESHOLD_SECONDS = 20.0
_AGING_SWEEP_INTERVAL_SECONDS = 1.0

# Dynamic per-label cost is the median of every measured execution duration
# for that label within this trailing window. Two days so day/night and
# weekday/weekend load patterns both stay represented in the sample.
_COST_HISTORY_WINDOW_SECONDS = 2 * 24 * 3600.0

# Cost for a label's first submission, before any duration has been measured.
# Deliberately cheap: an unmeasured task shouldn't be assumed expensive and
# starve everything else, and running it promptly is exactly what produces
# the first real sample.
_DEFAULT_COST_UNTIL_MEASURED = 1.0


class _DurationHistory:
    """Rolling-window duration samples for one task label, used to derive
    that label's scheduling cost from what it actually measured, not a
    guess.

    Samples are never subsampled or capped by count -- only pruned by age --
    so the median is computed over the full two-day sample instead of some
    arbitrarily shrunk tail that would throw away signal (e.g. a fixed
    last-N ring buffer would under-represent a label that runs often and
    over-represent one that runs rarely).
    """

    def __init__(self, window_seconds: float = _COST_HISTORY_WINDOW_SECONDS) -> None:
        self._window_seconds = window_seconds
        self._samples: deque[tuple[float, float]] = deque()  # (recorded_at, duration_seconds)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def record(self, duration_seconds: float) -> None:
        now = time.time()
        with self._lock:
            self._samples.append((now, duration_seconds))
            self._prune(now)

    def median(self, default: float) -> float:
        with self._lock:
            self._prune(time.time())
            if not self._samples:
                return default
            return statistics.median(duration for _, duration in self._samples)

    def sample_count(self) -> int:
        with self._lock:
            self._prune(time.time())
            return len(self._samples)


@dataclass(order=True)
class _QueuedTask:
    sort_key: tuple = field(compare=True)
    seq: int = field(compare=False)
    tier: int = field(compare=False)
    cost: float = field(compare=False)
    label: str | None = field(compare=False)
    submitted_at: float = field(compare=False)
    fn: Callable[..., Any] = field(compare=False)
    args: tuple = field(compare=False)
    kwargs: dict = field(compare=False)
    future: Future = field(compare=False)


class PriorityWorkScheduler:
    """Thread pool that dispatches queued blocking work by (tier, cost)
    instead of FIFO submission order.

    Interactive work (Discord commands the user is actively waiting on, e.g.
    .resumebuild) is always dispatched ahead of background work (job/reddit
    watcher scraping, ATS scraping, semantic-match filtering) of the same or
    lower priority. Within a tier, cheaper/shorter jobs go first
    (shortest-job-first minimizes total wait across same-priority work).

    A single instance is meant to be shared across the whole job-watcher
    domain (WatcherManager's background loops and CommandRouter's
    interactive command handlers) -- the arbitration only works if
    everything that competes for the same CPU actually goes through the
    same queue.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        # cpu_count() reports the HOST's cores inside a cgroup-limited container,
        # so capacity.cpu_limit() is what actually reflects available CPU there.
        # Capped: these are Python threads, so past a couple of dozen the GIL
        # means extra workers buy queue depth rather than throughput.
        self._max_workers = max_workers or max(2, min(32, round(capacity.cpu_limit())))
        self._cv = threading.Condition()
        self._heap: list[_QueuedTask] = []
        self._seq_counter = itertools.count()
        self._shutdown = False
        # Woken by shutdown so the aging thread stops immediately instead of
        # sleeping out its sweep interval. Deliberately NOT the shared
        # condition variable: workers notify that on every submit, which would
        # wake the aging thread on each one and turn a once-a-second sweep into
        # a per-task one on a busy queue.
        self._stopping = threading.Event()
        self._active_count = 0
        self._completed_count = 0
        self._promoted_count = 0
        self._history: dict[str, _DurationHistory] = {}
        self._history_lock = threading.Lock()
        self._workers: list[threading.Thread] = [
            threading.Thread(target=self._worker_loop, name=f"priority-work-{i}", daemon=True)
            for i in range(self._max_workers)
        ]
        for worker in self._workers:
            worker.start()
        self._aging_thread = threading.Thread(target=self._aging_loop, name="priority-work-aging", daemon=True)
        self._aging_thread.start()

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while not self._heap and not self._shutdown:
                    self._cv.wait()
                if self._shutdown and not self._heap:
                    return
                task = heapq.heappop(self._heap)
                self._active_count += 1

            # set_running_or_notify_cancel() atomically checks for a prior cancel()
            # and, on success, moves the future to RUNNING -- after which cancel()
            # can no longer succeed (concurrent.futures.Future.cancel() only
            # transitions PENDING->CANCELLED). Without this call the future stays
            # PENDING for the whole run, so a caller's cancel() (e.g. asyncio task
            # cancellation via asyncio.wrap_future) can succeed at any point up to
            # set_result/set_exception, racing it -- set_result/set_exception then
            # raises InvalidStateError, which is unhandled here and kills this
            # worker thread permanently.
            if not task.future.set_running_or_notify_cancel():
                with self._cv:
                    self._active_count -= 1
                continue

            started = time.monotonic()
            try:
                result = task.fn(*task.args, **task.kwargs)
            except BaseException as exc:  # noqa: BLE001 - propagate to the awaiting caller
                task.future.set_exception(exc)
            else:
                task.future.set_result(result)
            finally:
                if task.label is not None:
                    self._history_for(task.label).record(time.monotonic() - started)
                with self._cv:
                    self._active_count -= 1
                    self._completed_count += 1

    def _history_for(self, label: str) -> _DurationHistory:
        with self._history_lock:
            history = self._history.get(label)
            if history is None:
                history = _DurationHistory()
                self._history[label] = history
            return history

    def _aging_loop(self) -> None:
        while True:
            # A bare sleep here made shutdown(wait=True) take a full sweep
            # interval -- about a second -- because the thread could not be
            # woken. That is why the test suite's teardown fixture had to use
            # wait=False, which leaves worker threads from one test still
            # running during the next: three separate test files were flaking
            # intermittently on scheduler assertions because of it.
            if self._stopping.wait(timeout=_AGING_SWEEP_INTERVAL_SECONDS):
                return
            with self._cv:
                if self._shutdown:
                    return
                if not self._heap:
                    continue
                now = time.monotonic()
                promoted = False
                for task in self._heap:
                    if task.tier != INTERACTIVE and (now - task.submitted_at) >= _AGING_THRESHOLD_SECONDS:
                        task.tier = INTERACTIVE
                        task.sort_key = (INTERACTIVE, task.cost, task.seq)
                        promoted = True
                        self._promoted_count += 1
                if promoted:
                    heapq.heapify(self._heap)
                    self._cv.notify_all()

    def estimated_cost(self, label: str, default: float = _DEFAULT_COST_UNTIL_MEASURED) -> float:
        """Median measured duration (seconds) for `label` over the trailing
        two-day window, or `default` if nothing has been measured yet."""
        return self._history_for(label).median(default)

    def sample_count(self, label: str) -> int:
        return self._history_for(label).sample_count()

    def submit(
        self,
        fn: Callable[..., _T],
        *args: Any,
        tier: int = BACKGROUND,
        cost: float | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> Future:
        """Queue `fn` for execution. `cost` is the scheduling cost within
        `tier` (cheaper goes first); if omitted and `label` is given, cost is
        derived from that label's measured median duration (see
        `estimated_cost`) and the actual duration of this run feeds back into
        that label's history once it completes -- costs calibrate themselves
        from real measurements instead of staying fixed at a guess. If
        neither is given, cost defaults to 1.0."""
        if cost is None:
            cost = self.estimated_cost(label) if label is not None else _DEFAULT_COST_UNTIL_MEASURED
        future: Future = Future()
        seq = next(self._seq_counter)
        task = _QueuedTask(
            sort_key=(tier, cost, seq),
            seq=seq,
            tier=tier,
            cost=cost,
            label=label,
            submitted_at=time.monotonic(),
            fn=fn,
            args=args,
            kwargs=kwargs,
            future=future,
        )
        with self._cv:
            if self._shutdown:
                raise RuntimeError("PriorityWorkScheduler is shut down")
            heapq.heappush(self._heap, task)
            self._cv.notify()
        return future

    async def run(
        self,
        fn: Callable[..., _T],
        *args: Any,
        tier: int = BACKGROUND,
        cost: float | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> _T:
        future = self.submit(fn, *args, tier=tier, cost=cost, label=label, **kwargs)
        return await asyncio.wrap_future(future)

    def stats(self) -> dict[str, Any]:
        with self._cv:
            queued = len(self._heap)
            queued_interactive = sum(1 for t in self._heap if t.tier == INTERACTIVE)
            result = {
                "workers": self._max_workers,
                "active": self._active_count,
                "completed": self._completed_count,
                "promoted": self._promoted_count,
                "queued": queued,
                "queued_interactive": queued_interactive,
                "queued_background": queued - queued_interactive,
            }
        with self._history_lock:
            labels = list(self._history.items())
        result["label_costs"] = {
            label: {"median_seconds": round(history.median(0.0), 3), "samples": history.sample_count()}
            for label, history in labels
        }
        return result

    def shutdown(self, wait: bool = True) -> None:
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        self._stopping.set()
        if wait:
            for worker in self._workers:
                worker.join(timeout=5)
            self._aging_thread.join(timeout=2)
