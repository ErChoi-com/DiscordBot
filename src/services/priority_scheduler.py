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

from services import capacity, platform_support

_T = TypeVar("_T")

# Three priority tiers. Interactive (user-triggered Discord commands like
# .resumebuild) outranks background (job/reddit/ATS watcher work). PROMOTED
# sits between them and exists solely as aging's destination.
#
# Aging used to promote straight to INTERACTIVE, which erased the distinction
# the reserved workers below depend on: a flooded queue meant every background
# task became "interactive" within 20s, and a real command then had no
# priority left to exercise. Aging still lifts a starved task above its peers;
# it just can no longer disguise it as user-facing work.
INTERACTIVE = 0
PROMOTED = 1
BACKGROUND = 2

# A background task that has waited this long is lifted to the PROMOTED tier
# so a steady stream of interactive submissions can't starve it forever.
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

    def __init__(self, max_workers: int | None = None, reserved_interactive: int | None = None) -> None:
        # cpu_count() reports the HOST's cores inside a cgroup-limited container,
        # so capacity.cpu_limit() is what actually reflects available CPU there.
        # Capped: these are Python threads, so past a couple of dozen the GIL
        # means extra workers buy queue depth rather than throughput.
        self._max_workers = max_workers or max(2, min(32, round(capacity.cpu_limit())))
        # Workers that accept INTERACTIVE tasks only. Ordering alone could not
        # keep a command responsive: priority decides who goes next, not who
        # gets a thread, so a pool fully occupied by hours-long scrapes left
        # .resumebuild waiting on work that had no deadline it could enforce.
        # A floor of dedicated workers is what makes the guarantee real. It is
        # a floor and not a partition -- interactive work still spills into
        # idle general workers when there is more of it than the floor holds.
        if reserved_interactive is None:
            reserved_interactive = max(1, self._max_workers // 6)
        self._reserved_interactive = max(0, min(reserved_interactive, self._max_workers - 1))
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
        # A task whose caller cancelled it while it sat queued is discarded on
        # pickup. That path used to increment nothing at all, so the discards
        # were invisible and the counters read as self-contradictory: work
        # demonstrably submitted, nothing completed, nothing failed.
        self._dropped_count = 0
        # seq -> (label, started_at). What is holding a worker right now, which
        # `active` alone cannot answer -- and the first question worth asking
        # when the pool is full.
        self._in_flight: dict[int, tuple[str | None, float]] = {}
        self._history: dict[str, _DurationHistory] = {}
        self._history_lock = threading.Lock()
        self._workers: list[threading.Thread] = [
            threading.Thread(
                target=self._worker_loop,
                args=(i < self._reserved_interactive,),
                name=(
                    f"priority-work-{i}-interactive"
                    if i < self._reserved_interactive
                    else f"priority-work-{i}"
                ),
                daemon=True,
            )
            for i in range(self._max_workers)
        ]
        for worker in self._workers:
            worker.start()
        self._aging_thread = threading.Thread(target=self._aging_loop, name="priority-work-aging", daemon=True)
        self._aging_thread.start()

    def _claimable(self, reserved: bool) -> bool:
        """Whether this worker may take the task currently at the queue head.

        Called with `self._cv` held. The heap is ordered by (tier, cost, seq),
        so the best INTERACTIVE task is at the head whenever one exists at all
        -- a reserved worker only has to inspect the head, never scan.
        """
        if not self._heap:
            return False
        return not reserved or self._heap[0].tier == INTERACTIVE

    def _worker_loop(self, reserved: bool = False) -> None:
        while True:
            with self._cv:
                while not self._claimable(reserved) and not self._shutdown:
                    self._cv.wait()
                if self._shutdown and not self._claimable(reserved):
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
                    self._dropped_count += 1
                continue

            started = time.monotonic()
            with self._cv:
                self._in_flight[task.seq] = (task.label, started)
            # Background work runs below normal OS priority for as long as it
            # holds this thread. The reservation above decides which task gets
            # a worker; this decides which thread gets a core when all of them
            # want one, which is the case a busy scrape creates and the one an
            # interactive command and the event loop actually lose on. The
            # reserved workers only ever carry interactive work and are never
            # lowered, so on POSIX -- where a thread cannot be raised back --
            # they keep their priority for good.
            lowered = (
                not reserved
                and task.tier != INTERACTIVE
                and platform_support.set_current_thread_background(True)
            )
            try:
                result = task.fn(*task.args, **task.kwargs)
            except BaseException as exc:  # noqa: BLE001 - propagate to the awaiting caller
                task.future.set_exception(exc)
            else:
                task.future.set_result(result)
            finally:
                if lowered:
                    platform_support.set_current_thread_background(False)
                if task.label is not None:
                    self._history_for(task.label).record(time.monotonic() - started)
                with self._cv:
                    self._in_flight.pop(task.seq, None)
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
                    if task.tier == BACKGROUND and (now - task.submitted_at) >= _AGING_THRESHOLD_SECONDS:
                        task.tier = PROMOTED
                        task.sort_key = (PROMOTED, task.cost, task.seq)
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
            # notify_all, not notify: a lone wakeup can land on a reserved
            # worker that is not allowed to take this task, which would then
            # go back to waiting and leave the task sitting there while
            # general workers sat idle.
            self._cv.notify_all()
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
        now = time.monotonic()
        with self._cv:
            queued = len(self._heap)
            queued_interactive = sum(1 for t in self._heap if t.tier == INTERACTIVE)
            queued_promoted = sum(1 for t in self._heap if t.tier == PROMOTED)
            in_flight = sorted(
                (
                    {"label": label, "age_seconds": round(now - started, 1)}
                    for label, started in self._in_flight.values()
                ),
                key=lambda entry: entry["age_seconds"],
                reverse=True,
            )
            result = {
                "workers": self._max_workers,
                "reserved_interactive": self._reserved_interactive,
                "active": self._active_count,
                "completed": self._completed_count,
                "dropped": self._dropped_count,
                "promoted": self._promoted_count,
                "queued": queued,
                "queued_interactive": queued_interactive,
                "queued_promoted": queued_promoted,
                # Anything not interactive or promoted is still plain
                # background; keeping the old key's meaning intact.
                "queued_background": queued - queued_interactive - queued_promoted,
                "in_flight": in_flight,
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
